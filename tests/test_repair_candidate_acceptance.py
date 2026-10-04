from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from ccg.repairs.acceptance_candidates import (
    CONDITIONS,
    DOSE_CELLS,
    EVAL_SPLITS,
    SEEDS,
    validate_candidate_chain,
)


SOURCE_KEYS = (
    ("RPN", "phase1f_frozen_candidate_archive", "hard5"),
    ("RPN", "phase1f_frozen_candidate_archive", "hard10"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m0"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m2"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m4"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m8"),
    ("RPN", "v2_p1_frozen_hard_k5_predictions", "hard5"),
    ("DETR", "v2_p1_frozen_hard_k5_predictions", "hard5"),
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _repo_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _formal(seed: int = 0) -> dict:
    return {"n_replicates": 5000, "seed": seed, "ci_level": 0.95,
            "resample_unit": "image_cluster", "method": "percentile"}


def _group_names_and_map():
    groups = set()
    mapping = {}
    dose_source = ("RPN", "phase1f_frozen_candidate_archive")
    for family, source, cell in SOURCE_KEYS:
        if cell in DOSE_CELLS:
            continue
        for split in EVAL_SPLITS:
            name = f"{family}__{source}__{cell}__{split}"
            groups.add(name)
            mapping[(family, source, cell, split)] = name
    for split in EVAL_SPLITS:
        name = f"{dose_source[0]}__{dose_source[1]}__dose_joint__{split}"
        groups.add(name)
        for cell in DOSE_CELLS:
            mapping[(*dose_source, cell, split)] = name
    return groups, mapping


def _forward_arrays(key: tuple[str, str, str]) -> dict[str, np.ndarray]:
    family, source, cell = key
    n_rows = 10
    if cell in DOSE_CELLS:
        id_base = 30000
    else:
        id_base = 10000 * (SOURCE_KEYS.index(key) + 1)
    sentence = id_base + np.arange(n_rows, dtype=np.int64)
    images = id_base + np.repeat(np.arange(5, dtype=np.int64), 2)
    k = int(cell[-1]) if cell.startswith("hard") else 10
    arrays = {
        "sentence_id": sentence,
        "ref_id": sentence + 100000,
        "image_id": images,
        "ann_id": sentence + 200000,
        "eval_split": np.asarray(["testA"] * 4 + ["testB"] * 6),
        "cell": np.asarray(cell),
        "k": np.asarray(k, dtype=np.int32),
        "family": np.asarray(family),
        "candidate_source": np.asarray(source),
        "condition_names": np.asarray(CONDITIONS),
        "seed_names": np.asarray(SEEDS),
    }
    for condition in CONDITIONS:
        for seed in SEEDS:
            arrays[f"{condition}__{seed}__scores"] = np.zeros((n_rows, k + 1), dtype=np.float32)
            arrays[f"{condition}__{seed}__correct"] = np.zeros(n_rows, dtype=np.bool_)
            for head in ("msp", "stats", "e1b"):
                arrays[f"{condition}__{seed}__conf_{head}"] = np.zeros(n_rows, dtype=np.float64)
    return arrays


def _make_valid_chain(root: Path) -> dict[str, Path]:
    out = root / "results" / "research_repair_v1"
    candidates = out / "candidates"
    run_dir = candidates / "audit_run_fixture"
    forward_dir = run_dir / "forward"
    sens_dir = forward_dir / "sensitivity_bootstrap"
    (run_dir / "candidate_indices.npz").parent.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "candidate_indices.npz"
    index_path.write_bytes(b"versioned-candidate-index-fixture")
    index_hash = _sha(index_path)

    source_ids = [
        {"family": family, "candidate_source": source, "cell": cell,
         "source_path": f"source/{i}.npz", "source_rows": 10,
         "source_rows_covered_by_category_audit": 10,
         "source_sentence_ids_unique": True, "measured_cell_applicable": True}
        for i, (family, source, cell) in enumerate(SOURCE_KEYS)
    ]
    audit_summary_path = run_dir / "summary.json"
    audit_summary = {
        "schema": "research-repair-v1-candidate-summary-v1",
        "audit_run_id": "fixture", "status": "AUDIT_COMPLETE_MISMATCH_FOUND",
        "candidate_variant": "true_target_category_v1", "candidate_variant_constructed": True,
        "mismatch": {"total_expressions": 80, "total_mismatches": 2},
        "historical_candidate_source_scope": {"sources": source_ids},
    }
    _write_json(audit_summary_path, audit_summary)
    audit_hash = _sha(audit_summary_path)
    pointer_path = candidates / "current_audit.json"
    pointer = {
        "schema": "research-repair-v1-current-audit-pointer-v1",
        "status": audit_summary["status"], "audit_run_id": "fixture",
        "summary_path": _repo_path(root, audit_summary_path), "summary_sha256": audit_hash,
    }
    _write_json(pointer_path, pointer)

    forward_entries = []
    contract_entries = []
    identities = {}
    for key in SOURCE_KEYS:
        arrays = _forward_arrays(key)
        identities[key] = arrays
        family, source, cell = key
        output_path = forward_dir / f"{family}_{source}_{cell}_all_seeds.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        k = int(arrays["k"])
        n_rows = arrays["sentence_id"].size
        n_images = np.unique(arrays["image_id"]).size
        confidence_status = "PASS_1E-9" if source == "phase1f_frozen_candidate_archive" else "PRESERVED_SOURCE_CONFIDENCE_NO_FULL_LOGITS"
        anchors = {
            seed: {"passed": True, "rows": n_rows, "confidence_anchor_status": confidence_status}
            for seed in SEEDS
        }
        entry = {
            "family": family, "candidate_source": source, "cell": cell,
            "status": "FORWARDED_WITH_OLD_ANCHOR_PASS", "k": k,
            "n_rows": n_rows, "n_images": int(n_images),
            "candidate_source_rows": n_rows, "old_new_intersection_rows": n_rows,
            "rows_lost_under_true_category": 0,
            "canonical_order": "ascending sentence_id; rows are not joined or deduplicated by ref_id",
            "anchor_checks": anchors,
            "output": {"path": _repo_path(root, output_path), "size_bytes": output_path.stat().st_size,
                       "sha256": _sha(output_path)},
        }
        forward_entries.append(entry)
        contract_entries.append({k2: entry[k2] for k2 in ("family", "candidate_source", "cell", "status", "k", "n_rows", "n_images")})

    manifest_path = forward_dir / "predictions_manifest.json"
    manifest = {
        "schema": "research-repair-v1-candidate-forward-manifest-v1",
        "status": "COMPLETE", "preflight_rows_per_source": None,
        "audit_run_id": "fixture", "candidate_audit_summary_sha256": audit_hash,
        "archive_schema": "canonical-row-three-seed-old-random-corrected-v1",
        "row_fields": ["sentence_id", "ref_id", "image_id", "ann_id", "eval_split", "cell", "k"],
        "condition_names": list(CONDITIONS),
        "prediction_key_template": "{condition}__{seed}__{scores|correct|conf_msp|conf_stats|conf_e1b}",
        "canonical_row_order": "ascending sentence_id; each archive holds one source/family/cell old-new availability intersection",
        "entries": forward_entries,
    }
    _write_json(manifest_path, manifest)
    manifest_hash = _sha(manifest_path)
    forward_path = forward_dir / "summary.json"
    forward = {
        "schema": "research-repair-v1-candidate-forward-summary-v1",
        "status": "COMPLETE", "preflight_only": False, "audit_run_id": "fixture",
        "candidate_audit_pointer": {
            "path": _repo_path(root, pointer_path),
            "summary_path": _repo_path(root, audit_summary_path),
            "summary_sha256": audit_hash,
            "candidate_indices_path": _repo_path(root, index_path),
            "candidate_indices_sha256": index_hash,
        },
        "cohort_contract": {
            "source_cohorts_remain_separate": True,
            "row_order": "ascending sentence_id; no ref_id joins or deduplication",
            "families_and_cells": contract_entries,
        },
        "predictions_manifest": {
            "path": _repo_path(root, manifest_path), "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_hash, "schema": manifest["schema"],
        },
    }
    _write_json(forward_path, forward)
    forward_hash = _sha(forward_path)

    group_names, source_split_map = _group_names_and_map()
    group_rows = []
    expected_source_rows = []
    all_estimate_rows = []
    for key in SOURCE_KEYS:
        family, source, cell = key
        for split in EVAL_SPLITS:
            dose = cell in DOSE_CELLS
            group_name = source_split_map[(*key, split)]
            expected_source_rows.append({
                "family": family, "candidate_source": source, "cell": cell,
                "eval_split": split, "group": group_name,
                "status": "COMPONENT_INCLUDED_IN_DOSE_JOINT_GROUP_COMPLETE" if dose else "COMPLETE",
            })

    for group_index, group_name in enumerate(sorted(group_names)):
        dose_joint = "__dose_joint__" in group_name
        family, source, _, split = group_name.split("__", 3)
        n_rows = 10 if split == "pooled_testA_testB" else (4 if split == "testA" else 6)
        n_images = 5 if split == "pooled_testA_testB" else (2 if split == "testA" else 3)
        n_estimates = 260 if dose_joint else 52
        if dose_joint:
            group_source_keys = [list((*((family, source)), cell)) for cell in DOSE_CELLS]
        else:
            cell = group_name.split("__")[-2]
            group_source_keys = [[family, source, cell]]
        if dose_joint:
            sentence_ids = np.asarray(identities[(family, source, DOSE_CELLS[0])]["sentence_id"], dtype=np.int64)
            split_ids = np.asarray(identities[(family, source, DOSE_CELLS[0])]["eval_split"]).astype(str)
            common_ids = sentence_ids[np.isin(split_ids, ("testA", "testB")) if split == "pooled_testA_testB" else split_ids == split]
            cohort = {
                "family": family, "candidate_source": source, "eval_split": split,
                "common_sentence_ids_sha256": hashlib.sha256(common_ids.astype("<i8", copy=False).tobytes()).hexdigest(),
                "n_rows": n_rows, "n_images": n_images,
                "source_cohort_keys": group_source_keys,
            }
        else:
            cell = group_name.split("__")[-2]
            arrays = identities[(family, source, cell)]
            sentence_ids = np.asarray(arrays["sentence_id"], dtype=np.int64)
            split_ids = np.asarray(arrays["eval_split"]).astype(str)
            selected = sentence_ids[np.isin(split_ids, ("testA", "testB")) if split == "pooled_testA_testB" else split_ids == split]
            cohort = {
                "family": family, "candidate_source": source, "cell": cell,
                "eval_split": split, "source_rows": 10, "n_rows": n_rows, "n_images": n_images,
                "canonical_sentence_id_sha256": hashlib.sha256(selected.astype("<i8", copy=False).tobytes()).hexdigest(),
                "source_cohort_keys": group_source_keys,
            }

        if dose_joint:
            macro_metric_heads = [("accuracy", "msp")]
            macro_metric_heads.extend(
                (metric, head)
                for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
                for head in ("msp", "stats", "e1b")
            )
            macro_names = [
                f"dose_macro_old_minus_new__dose_macro__{head}__{metric}"
                for metric, head in macro_metric_heads
            ]
            estimate_names = macro_names + [
                f"estimate_{i:03d}" for i in range(n_estimates - len(macro_names))
            ]
        else:
            estimate_names = [f"estimate_{i:03d}" for i in range(n_estimates)]
        raw_path = sens_dir / f"raw_{group_index:02d}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, np.zeros(5000, dtype=np.float64), allow_pickle=False)
        npy_payload = buffer.getvalue()
        with zipfile.ZipFile(raw_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for estimate_name in estimate_names:
                for key_name in (f"bootstrap__mean__{estimate_name}",
                                 *(f"bootstrap__{seed}__{estimate_name}" for seed in SEEDS)):
                    archive.writestr(f"{key_name}.npy", npy_payload)
        raw_hash = _sha(raw_path)

        group_summary_path = sens_dir / f"group_{group_index:02d}_summary.json"
        group_summary = {
            "schema": "research-repair-v1-candidate-sensitivity-group-v1",
            "status": "COMPLETE", "formal": True, "group": group_name,
            "cohort": cohort, "bootstrap_contract": _formal(),
            "estimates": {name: {"point": 0.0} for name in estimate_names},
        }
        _write_json(group_summary_path, group_summary)
        group_rows.append({
            **cohort, "group": group_name,
            "summary_path": group_summary_path.relative_to(sens_dir).as_posix(),
            "summary_sha256": _sha(group_summary_path),
            "raw_replicates_path": raw_path.relative_to(sens_dir).as_posix(),
            "raw_replicates_sha256": raw_hash,
            "n_estimates": n_estimates, "status": "COMPLETE",
        })
        for estimate_name in estimate_names:
            is_macro = estimate_name.startswith("dose_macro_old_minus_new__dose_macro__")
            cell_value = (
                "dose_macro" if is_macro else
                (DOSE_CELLS[int(estimate_name.split("_")[1]) % len(DOSE_CELLS)]
                 if dose_joint else cohort["cell"])
            )
            metric_value = estimate_name.split("__")[-1] if is_macro else "accuracy"
            head_value = estimate_name.split("__")[-2] if is_macro else "msp"
            all_estimate_rows.append({
                "group": group_name, "family": family, "candidate_source": source,
                "cell": cell_value, "eval_split": split, "metric": metric_value,
                "contrast": "dose_macro_old_minus_new" if is_macro else "old_minus_new",
                "formula": "unweighted dose macro" if is_macro else "old - corrected",
                "units": "proportion difference", "confidence_head": head_value,
                "n_rows": n_rows, "n_images": n_images,
                "n_replicates": 5000, "estimate": estimate_name, "point": 0.0,
                "raw_replicates_path": raw_path.relative_to(sens_dir).as_posix(),
                "raw_replicates_sha256": raw_hash,
                "raw_replicates_key": f"bootstrap__mean__{estimate_name}",
                "per_seed_raw_replicates_keys_json": json.dumps(
                    [f"bootstrap__{seed}__{estimate_name}" for seed in SEEDS]
                ),
            })

    estimates_csv = sens_dir / "estimates.csv"
    fields = list(all_estimate_rows[0])
    with estimates_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_estimate_rows)
    sens_summary = {
        "schema": "research-repair-v1-candidate-sensitivity-summary-v1",
        "status": "COMPLETE", "formal": True, "audit_run_id": "fixture",
        "candidate_audit_summary_sha256": audit_hash,
        "candidate_forward_summary_sha256": forward_hash,
        "predictions_manifest": {"path": _repo_path(root, manifest_path), "sha256": manifest_hash},
        "bootstrap": _formal(), "bootstrap_contract": _formal(),
        "forwarded_source_cohorts": [list(key) for key in SOURCE_KEYS],
        "nonforwardable_empty_source_cohorts": [],
        "expected_source_cohort_split_groups": expected_source_rows,
        "groups": group_rows, "estimates_csv": "estimates.csv", "n_estimates": len(all_estimate_rows),
    }
    _write_json(sens_dir / "summary.json", sens_summary)
    return {"root": root, "out": out, "pointer": pointer_path, "audit": audit_summary_path,
            "forward": forward_path, "manifest": manifest_path, "sensitivity": sens_dir / "summary.json",
            "estimates": estimates_csv, "sensitivity_dir": sens_dir, "groups": group_rows}


def test_valid_candidate_chain_binds_eight_sources_and_fifteen_formal_groups(tmp_path) -> None:
    paths = _make_valid_chain(tmp_path)
    result = validate_candidate_chain(paths["root"], paths["out"])

    assert result["failures"] == []
    assert result["status"] == "PASS_METADATA_RAW_LENGTHS_DEFERRED"
    assert result["forward_source_count"] == 8
    assert result["sensitivity_group_count"] == 15
    assert result["source_split_mapping_count"] == 24
    assert result["estimate_row_count"] == 1404
    assert result["raw_group_sha256_checked"] is True
    assert result["raw_seed_keys_checked"] is True
    assert result["raw_array_lengths_checked"] is False
    assert result["raw_array_length_validation"] == "DEFERRED_TO_ROOT_RAW_REPLICATE_VERIFIER"
    assert len(result["source_sha256"]) >= 40


def test_missing_pointer_returns_failure_instead_of_raising(tmp_path) -> None:
    out = tmp_path / "results" / "research_repair_v1"
    result = validate_candidate_chain(tmp_path, out)
    assert result["status"] == "INCOMPLETE"
    assert any("missing JSON" in failure for failure in result["failures"])
    assert isinstance(result["source_sha256"], dict)


def test_zero_mismatches_explicitly_skips_sensitivity(tmp_path) -> None:
    paths = _make_valid_chain(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["status"] = "AUDIT_COMPLETE_NO_MISMATCH"
    audit["mismatch"]["total_mismatches"] = 0
    audit.pop("candidate_variant", None)
    audit.pop("candidate_variant_constructed", None)
    _write_json(paths["audit"], audit)
    pointer = json.loads(paths["pointer"].read_text(encoding="utf-8"))
    pointer["status"] = audit["status"]
    pointer["summary_sha256"] = _sha(paths["audit"])
    _write_json(paths["pointer"], pointer)

    result = validate_candidate_chain(paths["root"], paths["out"])

    assert result["failures"] == []
    assert result["status"] == "PASS_METADATA_RAW_LENGTHS_DEFERRED"
    assert result["mismatch_count"] == 0
    assert result["sensitivity_required"] is False
    assert result["candidate_sensitivity"] == "NOT_REQUIRED_ZERO_MISMATCH"


def test_candidate_chain_rejects_forward_from_a_different_audit(tmp_path) -> None:
    paths = _make_valid_chain(tmp_path)
    forward = json.loads(paths["forward"].read_text(encoding="utf-8"))
    forward["audit_run_id"] = "stale_audit"
    _write_json(paths["forward"], forward)

    result = validate_candidate_chain(paths["root"], paths["out"])

    assert any("Candidate forward audit_run_id differs from current audit" in failure
               for failure in result["failures"])


def test_candidate_chain_rejects_pilot_or_wrong_seed_sensitivity(tmp_path) -> None:
    paths = _make_valid_chain(tmp_path)
    original = paths["sensitivity"].read_bytes()
    payload = json.loads(original)
    payload["bootstrap"]["seed"] = 1
    _write_json(paths["sensitivity"], payload)
    result = validate_candidate_chain(paths["root"], paths["out"])
    assert any("Candidate sensitivity bootstrap:" in failure and "seed=1" in failure
               for failure in result["failures"])

    payload["bootstrap"]["seed"] = 0
    payload["formal"] = False
    payload["status"] = "PILOT"
    _write_json(paths["sensitivity"], payload)
    result = validate_candidate_chain(paths["root"], paths["out"])
    assert any("not a completed formal run" in failure for failure in result["failures"])


def test_candidate_chain_rejects_manifest_and_raw_key_source_mismatch(tmp_path) -> None:
    paths = _make_valid_chain(tmp_path)
    original = paths["sensitivity"].read_bytes()
    payload = json.loads(original)
    payload["predictions_manifest"]["path"] = "results/research_repair_v1/candidates/other/manifest.json"
    _write_json(paths["sensitivity"], payload)
    result = validate_candidate_chain(paths["root"], paths["out"])
    assert any("sensitivity predictions manifest path" in failure or "different predictions manifest" in failure
               for failure in result["failures"])

    paths = _make_valid_chain(tmp_path / "source_identity")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["entries"][0]["family"] = "GDINO"
    _write_json(paths["manifest"], manifest)
    result = validate_candidate_chain(paths["root"], paths["out"])
    assert any("Candidate forward source identities differ from the frozen 8-source acceptance scope"
               in failure for failure in result["failures"])

    paths = _make_valid_chain(tmp_path / "raw_seed_keys")
    with paths["estimates"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["per_seed_raw_replicates_keys_json"] = json.dumps([
        f"bootstrap__{SEEDS[1]}__{rows[0]['estimate']}",
        f"bootstrap__{SEEDS[0]}__{rows[0]['estimate']}",
        f"bootstrap__{SEEDS[2]}__{rows[0]['estimate']}",
    ])
    with paths["estimates"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = validate_candidate_chain(paths["root"], paths["out"])
    assert any("per-seed raw keys are missing, reordered, or mismatched" in failure
               for failure in result["failures"])


def test_candidate_chain_rejects_missing_mismatch_count_and_bad_group_sha(tmp_path) -> None:
    paths = _make_valid_chain(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit.pop("mismatch")
    _write_json(paths["audit"], audit)
    pointer = json.loads(paths["pointer"].read_text(encoding="utf-8"))
    pointer["summary_sha256"] = _sha(paths["audit"])
    _write_json(paths["pointer"], pointer)
    result = validate_candidate_chain(paths["root"], paths["out"])
    assert any("mismatch count is missing" in failure for failure in result["failures"])

    # Restore a valid chain, then tamper with one group's bytes without changing
    # the indexed hash. The checker must notice the content-address mismatch.
    paths = _make_valid_chain(tmp_path / "second")
    raw_path = paths["sensitivity_dir"] / paths["groups"][0]["raw_replicates_path"]
    with raw_path.open("ab") as handle:
        handle.write(b"tamper")
    result = validate_candidate_chain(paths["root"], paths["out"])
    assert any("raw replicates: SHA256 mismatch" in failure for failure in result["failures"])
