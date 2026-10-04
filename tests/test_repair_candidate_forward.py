from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccg.repairs import candidate_forward as forward
from ccg.repairs import candidates


def _prediction_rows(n: int = 2, k: int = 5) -> dict[str, np.ndarray]:
    ids = np.arange(10, 10 + n, dtype=np.int64)
    return {
        "sentence_id": ids,
        "ref_id": ids + 100,
        "image_id": ids + 1000,
        "ann_id": ids + 2000,
    }


def _seed_predictions(ids: dict[str, np.ndarray], k: int = 5) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for seed in forward.SEED_NAMES:
        scores = np.tile(np.linspace(0.5, 0.1, k, dtype=np.float32), (len(ids["sentence_id"]), 1))
        result[seed] = {
            "scores": scores,
            "correct": np.ones(len(scores), dtype=bool),
            "conf_msp": np.full(len(scores), 0.5, dtype=np.float64),
            "conf_stats": np.full(len(scores), 0.6, dtype=np.float64),
            "conf_e1b": np.full(len(scores), 0.7, dtype=np.float64),
        }
    return result


def test_prediction_archive_is_canonical_and_shared_across_seeds_and_conditions() -> None:
    ids = _prediction_rows()
    candidates_row = {
        **ids,
        "old_indices": np.tile(np.arange(5, dtype=np.int64), (2, 1)),
        "new_indices": np.tile(np.asarray([0, 2, 3, 4, 5], dtype=np.int64), (2, 1)),
    }
    predictions = _seed_predictions(ids)
    archive = forward.build_prediction_archive(
        family="RPN", source="phase1f_frozen_candidate_archive", cell="hard5",
        candidate=candidates_row, eval_split=np.asarray(["testA", "testB"]),
        old_result=predictions, new_result=predictions,
    )
    assert archive["sentence_id"].tolist() == [10, 11]
    assert archive["condition_names"].tolist() == ["old", "true_target_category_v1"]
    assert archive["seed_names"].tolist() == list(forward.SEED_NAMES)
    for seed in forward.SEED_NAMES:
        assert archive[f"old__{seed}__correct"].tolist() == [True, True]
        assert archive[f"true_target_category_v1__{seed}__conf_stats"].tolist() == [0.6, 0.6]
        assert archive[f"old__{seed}__scores"].shape == (2, 5)


def test_prediction_archive_carries_paired_random_control_with_target_alignment() -> None:
    ids = _prediction_rows()
    candidates_row = {
        **ids,
        "old_indices": np.tile(np.arange(5, dtype=np.int64), (2, 1)),
        "new_indices": np.tile(np.asarray([0, 2, 3, 4, 5], dtype=np.int64), (2, 1)),
    }
    predictions = _seed_predictions(ids)
    random_indices = np.tile(np.asarray([0, 5, 6, 7, 8], dtype=np.int64), (2, 1))
    archive = forward.build_prediction_archive(
        family="RPN", source="phase1f_frozen_candidate_archive", cell="hard5",
        candidate=candidates_row, eval_split=np.asarray(["testA", "testB"]),
        old_result=predictions, new_result=predictions, random_result=predictions,
        random_indices=random_indices,
    )
    assert archive["condition_names"].tolist() == ["old", "random", "true_target_category_v1"]
    assert np.array_equal(archive["random_candidate_indices"], random_indices)
    for seed in forward.SEED_NAMES:
        assert archive[f"random__{seed}__conf_e1b"].tolist() == [0.7, 0.7]


def test_old_forward_anchor_checks_logits_and_loaded_reliability_tolerances() -> None:
    ids = _prediction_rows()
    actual = _seed_predictions(ids)
    refs = _seed_predictions(ids)
    for seed in forward.SEED_NAMES:
        refs[seed] = {**refs[seed], "sentence_id": ids["sentence_id"],
                      "ref_id": ids["ref_id"], "image_id": ids["image_id"]}
    checked = forward._anchor_old_predictions(
        source="phase1f_frozen_candidate_archive", family="RPN", cell="hard5",
        observed={seed: {**{key: ids[key] for key in ("sentence_id", "ref_id", "image_id")}, **value}
                  for seed, value in actual.items()},
        references=refs,
    )
    assert all(row["passed"] for row in checked.values())
    assert checked["b3_seed1"]["max_abs_deltas"]["raw_scores_max_abs"] == 0

    perturbed = {key: dict(value) for key, value in actual.items()}
    perturbed["b3_seed2"]["conf_e1b"] = np.asarray([0.7 + 1e-7, 0.7])
    with pytest.raises(forward.CandidateForwardError, match="reliability anchor failed"):
        forward._anchor_old_predictions(
            source="phase1f_frozen_candidate_archive", family="RPN", cell="hard5",
            observed={seed: {**{key: ids[key] for key in ("sentence_id", "ref_id", "image_id")}, **value}
                      for seed, value in perturbed.items()},
            references=refs,
        )


def test_phase1f_anchor_separates_fresh_replay_residual_from_exact_saved_logit_route() -> None:
    ids = _prediction_rows()
    actual = _seed_predictions(ids)
    refs = _seed_predictions(ids)
    for seed in forward.SEED_NAMES:
        ref = refs[seed]
        ref.update({key: ids[key] for key in ("sentence_id", "ref_id", "image_id")})
        # The independent scorer replay can differ within STOP tolerance while
        # reliability is reconstructed from exact archived float32 logits.
        actual[seed]["scores"] = actual[seed]["scores"] + np.float32(1e-5)
        actual[seed]["fresh_conf_msp"] = ref["conf_msp"] + 2e-7
        actual[seed]["fresh_conf_stats"] = ref["conf_stats"] + 2e-7
        actual[seed]["fresh_conf_e1b"] = ref["conf_e1b"] + 2e-7
    checks = forward._anchor_old_predictions(
        source="phase1f_frozen_candidate_archive", family="RPN", cell="hard5",
        observed={seed: {**{key: ids[key] for key in ("sentence_id", "ref_id", "image_id")}, **value}
                  for seed, value in actual.items()},
        references=refs,
    )
    assert checks["b3_seed1"]["confidence_anchor_status"] == "PASS_1E-9"
    assert checks["b3_seed1"]["full_stable_ranking_identical"] is True
    assert checks["b3_seed1"]["max_abs_deltas"]["conf_stats_exact_saved_logits_max_abs"] == 0
    assert checks["b3_seed1"]["max_abs_deltas"]["fresh_conf_stats_max_abs"] == pytest.approx(2e-7)


def test_v2_anchor_preserves_archived_confidence_when_full_logits_are_absent() -> None:
    ids = _prediction_rows()
    actual = _seed_predictions(ids)
    refs = _seed_predictions(ids)
    for seed in forward.SEED_NAMES:
        ref = refs[seed]
        ref.update({key: ids[key] for key in ("sentence_id", "ref_id", "image_id")})
        ordered = np.sort(ref["scores"].astype(np.float64), axis=1)[:, ::-1]
        ref["raw_top1"] = ordered[:, 0]
        ref["raw_margin12"] = ordered[:, 0] - ordered[:, 1]
        del ref["scores"]
        actual[seed]["fresh_conf_stats"] = ref["conf_stats"] + 2e-7
        actual[seed]["fresh_conf_msp"] = ref["conf_msp"] + 2e-7
        actual[seed]["fresh_conf_e1b"] = ref["conf_e1b"] + 2e-7
    checks = forward._anchor_old_predictions(
        source="v2_p1_frozen_hard_k5_predictions", family="RPN", cell="hard5",
        observed={seed: {**{key: ids[key] for key in ("sentence_id", "ref_id", "image_id")}, **value}
                  for seed, value in actual.items()},
        references=refs,
    )
    assert checks["b3_seed1"]["confidence_anchor_status"] == "PRESERVED_SOURCE_CONFIDENCE_NO_FULL_LOGITS"
    assert checks["b3_seed1"]["full_stable_ranking_identical"] is None
    assert checks["b3_seed1"]["max_abs_deltas"]["fresh_conf_stats_max_abs"] == pytest.approx(2e-7)


def test_candidate_cell_must_equal_reported_old_new_intersection() -> None:
    ids = _prediction_rows()
    prefix = "RPN__phase1f_frozen_candidate_archive__hard5"
    archive = {
        f"{prefix}__sentence_id": ids["sentence_id"],
        f"{prefix}__ref_id": ids["ref_id"],
        f"{prefix}__image_id": ids["image_id"],
        f"{prefix}__ann_id": ids["ann_id"],
        f"{prefix}__old_indices": np.tile(np.arange(5, dtype=np.int64), (2, 1)),
        f"{prefix}__true_target_category_v1_indices": np.tile(
            np.asarray([0, 2, 3, 4, 5], dtype=np.int64), (2, 1)
        ),
    }
    key = ("RPN", "phase1f_frozen_candidate_archive", "hard5")
    availability = {key: {
        10: {"in_old_new_common_intersection": "1", "historical_candidate_available": "1",
             "corrected_candidate_available": "1"},
        11: {"in_old_new_common_intersection": "1", "historical_candidate_available": "1",
             "corrected_candidate_available": "1"},
        12: {"in_old_new_common_intersection": "0", "historical_candidate_available": "1",
             "corrected_candidate_available": "0"},
    }}
    result = forward._read_candidate_cell(archive, availability, family=key[0], source=key[1], cell=key[2])
    assert result["sentence_id"].tolist() == [10, 11]
    availability[key][12]["in_old_new_common_intersection"] = "1"
    with pytest.raises(forward.CandidateForwardError, match="common flag conflicts"):
        forward._read_candidate_cell(archive, availability, family=key[0], source=key[1], cell=key[2])


def test_phase1f_dose_control_uses_versioned_m0_candidates_and_identity_alignment() -> None:
    family, source = "RPN", "phase1f_frozen_candidate_archive"
    ids = {
        "sentence_id": np.asarray([10, 12], dtype=np.int64),
        "ref_id": np.asarray([20, 22], dtype=np.int64),
        "image_id": np.asarray([30, 32], dtype=np.int64),
        "ann_id": np.asarray([40, 42], dtype=np.int64),
    }
    old_m0 = np.tile(np.arange(10, dtype=np.int64), (2, 1))
    old_m2 = old_m0.copy()
    old_m2[:, 1:3] = np.asarray([[10, 11], [12, 13]])
    archive = {}
    availability = {}
    for cell, old in (("expb_m0", old_m0), ("expb_m2", old_m2)):
        prefix = f"{family}__{source}__{cell}"
        for field in ("sentence_id", "ref_id", "image_id", "ann_id"):
            archive[f"{prefix}__{field}"] = ids[field]
        archive[f"{prefix}__old_indices"] = old
        archive[f"{prefix}__true_target_category_v1_indices"] = old.copy()
        availability[(family, source, cell)] = {
            int(sid): {"in_old_new_common_intersection": "1", "historical_candidate_available": "1",
                       "corrected_candidate_available": "1"}
            for sid in ids["sentence_id"]
        }
    selected = forward._dose_fixed_m0_indices(
        archive, availability, family=family, source=source, ids=ids,
        current_old_indices=old_m2,
    )
    assert np.array_equal(selected, old_m0)
    assert forward._random_control_cell(source, "expb_m8") == "expb_m0"
    assert forward._random_control_cell(source, "hard10") == "rand10"

    archive["RPN__phase1f_frozen_candidate_archive__expb_m0__image_id"] = np.asarray([31, 32])
    with pytest.raises(forward.CandidateForwardError, match="image_id identities"):
        forward._dose_fixed_m0_indices(
            archive, availability, family=family, source=source, ids=ids,
            current_old_indices=old_m2,
        )


def test_current_audit_pointer_rejects_changed_candidate_artifact(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    candidate_root = repo / "results/research_repair_v1/candidates"
    run_dir = candidate_root / "audit_run_test"
    run_dir.mkdir(parents=True)
    candidate_path = run_dir / "candidate_sets.npz"
    np.savez_compressed(candidate_path, sentence_id=np.asarray([1], dtype=np.int64))
    summary = {
        "schema": "research-repair-v1-candidate-summary-v1",
        "status": "AUDIT_COMPLETE_MISMATCH_FOUND",
        "audit_run_id": "test", "candidate_variant": candidates.VARIANT,
        "candidate_variant_constructed": True,
        "mismatch": {"total_mismatches": 1},
        "output_files": {"versioned_candidate_indices": candidates._rel(candidate_path) if hasattr(candidates, "_rel") else ""},
        "output_artifacts": {"versioned_candidate_indices": {
            "path": "results/research_repair_v1/candidates/audit_run_test/candidate_sets.npz",
            "size_bytes": candidate_path.stat().st_size,
            "sha256": candidates._sha256(candidate_path),
        }},
    }
    summary["output_files"]["versioned_candidate_indices"] = summary["output_artifacts"]["versioned_candidate_indices"]["path"]
    summary_path = run_dir / "summary.json"
    candidates._write_json(summary_path, summary)
    pointer = {
        "schema": "research-repair-v1-current-audit-pointer-v1",
        "status": summary["status"], "audit_run_id": "test",
        "summary_path": "results/research_repair_v1/candidates/audit_run_test/summary.json",
        "summary_sha256": candidates._sha256(summary_path),
    }
    pointer_path = candidate_root / "current_audit.json"
    candidates._write_json(pointer_path, pointer)
    monkeypatch.setattr(forward, "ROOT", repo)
    monkeypatch.setattr(forward, "CANDIDATE_OUT_DIR", candidate_root)
    monkeypatch.setattr(forward, "CURRENT_AUDIT_PATH", pointer_path)
    resolved = forward.resolve_current_audit(pointer_path=pointer_path)
    assert resolved["candidate_indices_path"] == candidate_path.resolve()
    assert resolved["candidate_indices_sha256"] == candidates._sha256(candidate_path)

    candidate_path.write_bytes(candidate_path.read_bytes() + b"tamper")
    with pytest.raises(forward.CandidateForwardError, match="size differs"):
        forward.resolve_current_audit(pointer_path=pointer_path)


def test_detr_feature_manifest_is_immutable_and_covers_all_files(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    features = repo / "cache/features_detr_r50"
    features.mkdir(parents=True)
    for name in ("metadata.json", "region_features.h5", "text_features.h5", "text_index.csv", "invalid_crops.csv"):
        (features / name).write_bytes(name.encode())
    baseline = repo / "results/research_repair_v1/input_manifest.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(forward, "ROOT", repo)
    monkeypatch.setattr(forward, "INPUT_MANIFEST_PATH", baseline)
    out_dir = repo / "results/research_repair_v1/candidates/audit_run_x/forward"
    first = forward._feature_tree_manifest(features, out_dir, "baseline-sha")
    manifest_bytes = (out_dir / "supplemental_input_manifest.json").read_bytes()
    assert len(first["inputs"]) == 5
    assert (out_dir / "supplemental_input_manifest.json").exists()
    assert forward._feature_tree_manifest(features, out_dir, "baseline-sha") == first
    assert (out_dir / "supplemental_input_manifest.json").read_bytes() == manifest_bytes
    (features / "text_index.csv").write_bytes(b"changed")
    with pytest.raises(forward.CandidateForwardError, match="changed after"):
        forward._feature_tree_manifest(features, out_dir, "baseline-sha")
