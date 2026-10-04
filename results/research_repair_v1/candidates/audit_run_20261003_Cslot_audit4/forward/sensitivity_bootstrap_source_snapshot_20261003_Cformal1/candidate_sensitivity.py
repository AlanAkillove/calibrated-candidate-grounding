"""Paired shared-image bootstrap for corrected-candidate sensitivity predictions."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ccg.repairs.bootstrap import _summarize_replicates
from ccg.repairs.candidate_forward import (
    CandidateForwardError,
    CURRENT_AUDIT_PATH,
    _read_json,
    _safe_repo_path,
    _sha256,
    _write_json,
    _write_npz,
    resolve_current_audit,
)
from ccg.repairs.statistics import METRIC_NAMES, joint_prediction_bootstrap

FORMAL_REPLICATES = 5000
FORMAL_SEED = 0
FORMAL_CI = 0.95
HEADS = ("msp", "stats", "e1b")
SENSITIVITY_METRICS = ("accuracy", "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
DOSE_LEVELS = ("expb_m0", "expb_m2", "expb_m4", "expb_m8")
EVAL_SPLITS = (("pooled_testA_testB", None), ("testA", "testA"), ("testB", "testB"))
VARIANT = "true_target_category_v1"


class CandidateSensitivityError(RuntimeError):
    """A forward archive or paired-bootstrap invariant failed."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if key not in ("replicates", "per_seed_replicates")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    from ccg.repairs.atomic import replace_with_retry
    replace_with_retry(temp, path)


def _forward_archives(pointer_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve one completed forward exclusively through the current audit pointer."""
    audit = resolve_current_audit(pointer_path)
    forward_dir = audit["forward_dir"].resolve()
    forward_summary_path = forward_dir / "summary.json"
    manifest_path = forward_dir / "predictions_manifest.json"
    if not forward_summary_path.is_file() or not manifest_path.is_file():
        raise CandidateSensitivityError("the pointed candidate forward has no completed summary and predictions manifest")
    forward_summary = _read_json(forward_summary_path)
    if forward_summary.get("schema") != "research-repair-v1-candidate-forward-summary-v1":
        raise CandidateSensitivityError("candidate-forward summary schema mismatch")
    if forward_summary.get("status") != "COMPLETE":
        raise CandidateSensitivityError("candidate forward is not complete")
    if forward_summary.get("audit_run_id") != audit["pointer"].get("audit_run_id"):
        raise CandidateSensitivityError("forward and current audit run IDs differ")
    pointer_contract = forward_summary.get("candidate_audit_pointer", {})
    if pointer_contract.get("summary_sha256") != audit["summary_sha256"]:
        raise CandidateSensitivityError("forward is not anchored to the exact current audit summary")
    manifest_contract = forward_summary.get("predictions_manifest", {})
    expected_repo_path = manifest_contract.get("path")
    if not expected_repo_path or _safe_repo_path(str(expected_repo_path)) != manifest_path.resolve():
        raise CandidateSensitivityError("forward summary manifest path is not its sibling predictions manifest")
    if _sha256(manifest_path) != manifest_contract.get("sha256"):
        raise CandidateSensitivityError("predictions manifest hash differs from the forward summary")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "research-repair-v1-candidate-forward-manifest-v1":
        raise CandidateSensitivityError("predictions manifest schema mismatch")
    if manifest.get("audit_run_id") != audit["pointer"].get("audit_run_id"):
        raise CandidateSensitivityError("predictions manifest audit run ID mismatch")
    if manifest.get("candidate_audit_summary_sha256") != audit["summary_sha256"]:
        raise CandidateSensitivityError("predictions manifest is not bound to the current audit summary")
    if manifest.get("condition_names") != ["old", "random", VARIANT]:
        raise CandidateSensitivityError("forward archive conditions are not old/random/corrected")

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in manifest.get("entries", []):
        key = (str(entry.get("family")), str(entry.get("candidate_source")), str(entry.get("cell")))
        if key in seen:
            raise CandidateSensitivityError(f"predictions manifest duplicates cohort {key}")
        seen.add(key)
        status = str(entry.get("status"))
        row = {**entry, "_key": key, "_status": status}
        if status == "FORWARDED_WITH_OLD_ANCHOR_PASS":
            output = entry.get("output")
            if not isinstance(output, Mapping):
                raise CandidateSensitivityError(f"{key}: forwarded source has no artifact metadata")
            path = _safe_repo_path(str(output.get("path", "")))
            try:
                path.relative_to(forward_dir)
            except ValueError as exc:
                raise CandidateSensitivityError(f"{key}: prediction archive is outside the pointed forward run") from exc
            if not path.is_file() or path.stat().st_size != int(output.get("size_bytes", -1)):
                raise CandidateSensitivityError(f"{key}: prediction archive missing or size mismatch")
            if _sha256(path) != output.get("sha256"):
                raise CandidateSensitivityError(f"{key}: prediction archive SHA256 mismatch")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            _validate_prediction_archive(key, arrays)
            row["_path"] = path
            row["_arrays"] = arrays
        elif status == "NOT_FORWARDABLE_EMPTY_OLD_NEW_INTERSECTION":
            row["_path"] = None
            row["_arrays"] = None
        else:
            raise CandidateSensitivityError(f"{key}: unsupported forward status {status!r}")
        entries.append(row)
    if not entries:
        raise CandidateSensitivityError("predictions manifest contains no source cohorts")
    summary_keys = {
        (str(row.get("family")), str(row.get("candidate_source")), str(row.get("cell")))
        for row in forward_summary.get("cohort_contract", {}).get("families_and_cells", [])
    }
    if seen != summary_keys:
        raise CandidateSensitivityError("forward summary and predictions manifest source cohorts differ")
    return {"audit": audit, "forward_summary": forward_summary, "manifest_path": manifest_path,
            "manifest": manifest}, entries


def _validate_prediction_archive(key: tuple[str, str, str], arrays: Mapping[str, np.ndarray]) -> None:
    required = ("sentence_id", "ref_id", "image_id", "ann_id", "eval_split", "cell", "k",
                "condition_names", "seed_names", "old_candidate_indices", "random_candidate_indices",
                f"{VARIANT}_candidate_indices")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise CandidateSensitivityError(f"{key}: prediction archive lacks required fields {missing}")
    ids = np.asarray(arrays["sentence_id"], dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.unique(ids).size != ids.size or not np.array_equal(ids, np.sort(ids)):
        raise CandidateSensitivityError(f"{key}: sentence IDs are empty, duplicated, or noncanonical")
    for field in ("ref_id", "image_id", "ann_id", "eval_split"):
        if np.asarray(arrays[field]).reshape(-1).size != ids.size:
            raise CandidateSensitivityError(f"{key}: {field} row count differs from sentence IDs")
    if np.asarray(arrays["condition_names"]).tolist() != ["old", "random", VARIANT]:
        raise CandidateSensitivityError(f"{key}: condition names differ from forward manifest")
    if np.asarray(arrays["seed_names"]).tolist() != ["b3_seed1", "b3_seed2", "b3_seed3"]:
        raise CandidateSensitivityError(f"{key}: seed names differ from fixed three-seed bundle")
    candidate_shapes = [np.asarray(arrays[name]).shape for name in (
        "old_candidate_indices", "random_candidate_indices", f"{VARIANT}_candidate_indices")]
    if len(set(candidate_shapes)) != 1 or candidate_shapes[0][0] != ids.size:
        raise CandidateSensitivityError(f"{key}: candidate index matrices are not aligned")
    old_indices = np.asarray(arrays["old_candidate_indices"], dtype=np.int64)
    random_indices = np.asarray(arrays["random_candidate_indices"], dtype=np.int64)
    corrected_indices = np.asarray(arrays[f"{VARIANT}_candidate_indices"], dtype=np.int64)
    if not np.array_equal(old_indices[:, 0], random_indices[:, 0]) or not np.array_equal(old_indices[:, 0], corrected_indices[:, 0]):
        raise CandidateSensitivityError(f"{key}: old, random, and corrected target proposals are not aligned")
    for condition in ("old", "random", VARIANT):
        for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
            for field in ("correct", "conf_msp", "conf_stats", "conf_e1b"):
                name = f"{condition}__{seed}__{field}"
                if name not in arrays or np.asarray(arrays[name]).reshape(-1).size != ids.size:
                    raise CandidateSensitivityError(f"{key}: missing/misaligned prediction field {name}")
            correct = np.asarray(arrays[f"{condition}__{seed}__correct"], dtype=bool).reshape(-1)
            if np.any((correct != 0) & (correct != 1)):
                raise CandidateSensitivityError(f"{key}: {condition}/{seed} correctness is not binary")


def _conditions(arrays: Mapping[str, np.ndarray], *, rows: np.ndarray | None = None) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    conditions_by_seed: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    row_index = np.arange(np.asarray(arrays["sentence_id"]).size, dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
        conditions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for condition in ("old", "random", VARIANT):
            correctness = np.asarray(arrays[f"{condition}__{seed}__correct"], dtype=np.float64)[row_index]
            for head in HEADS:
                confidence = np.asarray(arrays[f"{condition}__{seed}__conf_{head}"], dtype=np.float64)[row_index]
                conditions[f"{condition}_{head}"] = (confidence, correctness)
        conditions_by_seed[seed] = conditions
    return conditions_by_seed


def _specs_for_cell(cell_tag: str, *, include_dose_interactions: Sequence[str] = ()) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    dose_mode = bool(include_dose_interactions)
    levels = tuple(include_dose_interactions) or (None,)
    for level in levels:
        label_tag = str(level) if level is not None else cell_tag
        suffix = f"_{level}" if level is not None else ""
        for metric in SENSITIVITY_METRICS:
            heads = ("msp",) if metric == "accuracy" else HEADS
            for head in heads:
                old = f"old{suffix}_{head}"
                random = f"random{suffix}_{head}"
                corrected = f"{VARIANT}{suffix}_{head}"
                contrasts = (
                    (("old_minus_new", old, corrected),
                     ("old_dose_minus_old_m0", old, f"old_expb_m0_{head}"),
                     ("corrected_dose_minus_corrected_m0", corrected, f"{VARIANT}_expb_m0_{head}"),
                     ("corrected_dose_minus_fixed_original_m0", corrected, random))
                    if dose_mode else
                    (("old_minus_new", old, corrected),
                     ("old_hard_minus_random", old, random),
                     ("corrected_hard_minus_random", corrected, random))
                )
                for label, left, right in contrasts:
                    specs.append({
                        "name": f"{label}__{label_tag}__{head}__{metric}",
                        "operation": "difference", "a": left, "b": right, "metric": metric,
                    })
                if not dose_mode:
                    # The same fixed random control is subtracted from both arms;
                    # this accounting contrast is algebraically old-minus-new.
                    specs.append({
                        "name": f"hard_random_interaction__{label_tag}__{head}__{metric}",
                        "operation": "difference_of_differences",
                        "a": old, "b": random, "c": corrected, "d": random, "metric": metric,
                    })
    for level in (() if not include_dose_interactions else include_dose_interactions[1:]):
        for metric in SENSITIVITY_METRICS:
            heads = ("msp",) if metric == "accuracy" else HEADS
            for head in heads:
                specs.append({
                    "name": f"dose_vs_m0_interaction__{level}__{head}__{metric}",
                    "operation": "difference_of_differences",
                    "a": f"old_{level}_{head}", "b": f"{VARIANT}_{level}_{head}",
                    "c": "old_expb_m0_" + head, "d": VARIANT + "_expb_m0_" + head,
                    "metric": metric,
                })
    return specs


def _dose_seed_cells(entries: Mapping[str, Mapping[str, Any]], common_ids: np.ndarray) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], np.ndarray, dict[str, dict[str, int]]]:
    seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        seed: {} for seed in ("b3_seed1", "b3_seed2", "b3_seed3")
    }
    positions_by_level: dict[str, np.ndarray] = {}
    for level in DOSE_LEVELS:
        arrays = entries[level]["_arrays"]
        ids = np.asarray(arrays["sentence_id"], dtype=np.int64)
        positions = np.searchsorted(ids, common_ids)
        if np.any(positions >= ids.size) or not np.array_equal(ids[positions], common_ids):
            raise CandidateSensitivityError(f"dose common IDs are absent from {level}")
        positions_by_level[level] = positions
    base_arrays = entries[DOSE_LEVELS[0]]["_arrays"]
    base_positions = positions_by_level[DOSE_LEVELS[0]]
    common_image_ids = np.asarray(base_arrays["image_id"], dtype=np.int64)[base_positions]
    for level in DOSE_LEVELS[1:]:
        arrays, positions = entries[level]["_arrays"], positions_by_level[level]
        for field in ("ref_id", "image_id", "ann_id", "eval_split"):
            if not np.array_equal(np.asarray(base_arrays[field])[base_positions], np.asarray(arrays[field])[positions]):
                raise CandidateSensitivityError(f"dose common sentence rows disagree on {field} at {level}")
        for seed in seed_cells:
            for field in ("correct", "conf_msp", "conf_stats", "conf_e1b"):
                name = f"random__{seed}__{field}"
                if not np.array_equal(np.asarray(base_arrays[name])[base_positions], np.asarray(arrays[name])[positions]):
                    raise CandidateSensitivityError(f"dose common random control differs for {seed}/{field} at {level}")
    for level in DOSE_LEVELS:
        arrays, positions = entries[level]["_arrays"], positions_by_level[level]
        for seed in seed_cells:
            for condition in ("old", "random", VARIANT):
                correct = np.asarray(arrays[f"{condition}__{seed}__correct"], dtype=np.float64)[positions]
                for head in HEADS:
                    conf = np.asarray(arrays[f"{condition}__{seed}__conf_{head}"], dtype=np.float64)[positions]
                    seed_cells[seed][f"{condition}_{level}_{head}"] = (conf, correct)
    attrition = {
        level: {
            "source_common_rows": int(np.asarray(entries[level]["_arrays"]["sentence_id"]).size),
            "dose_all_level_common_rows_in_selected_split": int(common_ids.size),
            "rows_lost_to_intersection_and_split_filter": int(
                np.asarray(entries[level]["_arrays"]["sentence_id"]).size - common_ids.size
            ),
        }
        for level in DOSE_LEVELS
    }
    return seed_cells, common_image_ids, attrition


def _add_dose_macros(result: dict[str, Any], *, cell_tag: str = "dose_macro") -> list[dict[str, Any]]:
    """Summarize macros from per-seed, per-draw effects in the same shared bootstrap."""
    rows: list[dict[str, Any]] = []
    table = result["estimates"]
    for metric in SENSITIVITY_METRICS:
        heads = ("msp",) if metric == "accuracy" else HEADS
        for head in heads:
            components = [f"old_minus_new__{level}__{head}__{metric}" for level in DOSE_LEVELS]
            seed_names = ("b3_seed1", "b3_seed2", "b3_seed3")
            seed_points: dict[str, float] = {}
            seed_raw: dict[str, np.ndarray] = {}
            for seed in seed_names:
                points = [float(table[name]["seed_estimates"][seed]) for name in components]
                seed_points[seed] = float(np.mean(points)) if np.all(np.isfinite(points)) else float("nan")
                reps = np.stack([
                    np.asarray(table[name]["per_seed_replicates"][seed], dtype=np.float64)
                    for name in components
                ], axis=0)
                raw = np.full(reps.shape[1], np.nan, dtype=np.float64)
                valid = np.all(np.isfinite(reps), axis=0)
                raw[valid] = reps[:, valid].mean(axis=0)
                seed_raw[seed] = raw
            aggregate = np.full(result["n_replicates"], np.nan, dtype=np.float64)
            stacked = np.stack([seed_raw[seed] for seed in seed_names], axis=0)
            valid_aggregate = np.all(np.isfinite(stacked), axis=0)
            aggregate[valid_aggregate] = stacked[:, valid_aggregate].mean(axis=0)
            point = float(np.mean(list(seed_points.values()))) if all(np.isfinite(v) for v in seed_points.values()) else float("nan")
            aggregate_summary = _summarize_replicates(aggregate, point, result["ci_level"], 0.95)
            per_seed_summary = {
                seed: _summarize_replicates(seed_raw[seed], seed_points[seed], result["ci_level"], 0.95)
                for seed in seed_names
            }
            name = f"dose_macro_old_minus_new__{cell_tag}__{head}__{metric}"
            table[name] = {
                **aggregate_summary,
                "seed_standard_deviation": float(np.std(list(seed_points.values()), ddof=1)),
                "seed_estimates": seed_points,
                "per_seed": per_seed_summary,
                "per_seed_replicates": seed_raw,
            }
            rows.append({"name": name, **table[name]})
    return rows


def _extract_output(result: Mapping[str, Any], *, output_prefix: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for name, estimate in result["estimates"].items():
        arrays[f"{output_prefix}__mean__{name}"] = np.asarray(estimate["replicates"], dtype=np.float64)
        per_seed_reps = estimate["per_seed_replicates"]
        for seed, values in per_seed_reps.items():
            arrays[f"{output_prefix}__{seed}__{name}"] = np.asarray(values, dtype=np.float64)
        safe = _json_safe({key: value for key, value in estimate.items() if key not in ("replicates", "per_seed_replicates")})
        description = _estimate_description(name)
        rows.append({
            "estimate": name, **description,
            "point": safe.get("point"), "ci_low": safe.get("ci_low"), "ci_high": safe.get("ci_high"),
            "bootstrap_std": safe.get("bootstrap_std"), "seed_standard_deviation": safe.get("seed_standard_deviation"),
            "valid_replicates": safe.get("valid_replicates"), "invalid_replicates": safe.get("invalid_replicates"),
            "valid_fraction": safe.get("valid_fraction"), "interval_status": safe.get("interval_status"),
            "uncertainty_status": safe.get("uncertainty_status"),
            "per_seed_json": json.dumps(safe.get("per_seed", {}), ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            "raw_replicates_key": f"{output_prefix}__mean__{name}",
        })
    return arrays, rows


def _estimate_description(name: str) -> dict[str, str]:
    parts = name.split("__")
    contrast = parts[0]
    cell = parts[1]
    head = parts[2]
    metric = parts[3]
    formulas = {
        "old_minus_new": "metric(old hard candidate) - metric(corrected true-target-category candidate)",
        "old_hard_minus_random": "metric(old hard candidate) - metric(matched frozen random candidate)",
        "corrected_hard_minus_random": "metric(corrected true-target-category candidate) - metric(matched frozen random candidate)",
        "hard_random_interaction": "[metric(old hard) - metric(random)] - [metric(corrected hard) - metric(random)]",
        "old_dose_minus_old_m0": "metric(old Phase 1F dose-m candidate) - metric(old Phase 1F expb_m0 candidate)",
        "corrected_dose_minus_corrected_m0": "metric(corrected true-target-category dose-m candidate) - metric(corrected true-target-category expb_m0 candidate)",
        "corrected_dose_minus_fixed_original_m0": "metric(corrected true-target-category dose-m candidate) - metric(fixed original Phase 1F expb_m0 candidate); descriptive fixed-control contrast, not corrected baseline",
        "dose_vs_m0_interaction": "[metric(old dose) - metric(corrected dose)] - [metric(old m0) - metric(corrected m0)]; version change in the dose effect",
        "dose_macro_old_minus_new": "unweighted mean across m0/m2/m4/m8 of [metric(old) - metric(corrected)] within each seed and draw",
    }
    units = {
        "accuracy": "proportion difference",
        "auroc_correct": "AUROC difference",
        "e_aurc": "eAURC difference",
        "rer_at_50": "RER ratio difference",
        "rer_at_80": "RER ratio difference",
    }
    if contrast not in formulas or metric not in units:
        raise CandidateSensitivityError(f"unknown estimate naming contract: {name}")
    return {"cell": cell, "contrast": contrast, "formula": formulas[contrast], "metric": metric,
            "confidence_head": head, "units": units[metric]}


def _run_one_group(seed_cells: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]], image_ids: np.ndarray,
                   *, cell_tag: str, dose_levels: Sequence[str] = (), n_replicates: int,
                   seed: int, ci: float) -> dict[str, Any]:
    specs = _specs_for_cell(cell_tag, include_dose_interactions=dose_levels)
    metrics = tuple(dict.fromkeys(SENSITIVITY_METRICS))
    result = joint_prediction_bootstrap(
        image_ids, seed_cells, estimates=specs, metrics=metrics,
        n_replicates=n_replicates, seed=seed, ci=ci,
    )
    if dose_levels:
        _add_dose_macros(result)
    return result


def _source_groups_complete(
    expected_groups: Sequence[Mapping[str, Any]],
    status_by_source_group: Mapping[tuple[str, str, str, str], str],
    accepted_prefixes: tuple[str, ...],
) -> bool:
    """Check final mapped statuses; expected-group declarations start as PENDING."""
    return all(
        str(status_by_source_group.get((
            str(row["family"]), str(row["candidate_source"]),
            str(row["cell"]), str(row["eval_split"]),
        ), "")).startswith(accepted_prefixes)
        for row in expected_groups
    )


def run_candidate_sensitivity_bootstrap(
    *, pointer_path: Path = CURRENT_AUDIT_PATH, n_replicates: int = FORMAL_REPLICATES,
    seed: int = FORMAL_SEED, ci: float = FORMAL_CI, output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run paired candidate sensitivity inference; nonformal runs are isolated PILOTs."""
    if tuple(METRIC_NAMES) == ():
        raise CandidateSensitivityError("repair metric registry is unexpectedly empty")
    formal = int(n_replicates) == FORMAL_REPLICATES and int(seed) == FORMAL_SEED and float(ci) == FORMAL_CI
    resolved, entries = _forward_archives(pointer_path)
    forward_dir = resolved["audit"]["forward_dir"]
    if output_dir is None:
        if formal:
            output_dir = forward_dir / "sensitivity_bootstrap"
        else:
            output_dir = forward_dir / f"pilot_sensitivity_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    output_dir = Path(output_dir).resolve()
    try:
        output_dir.relative_to(forward_dir.resolve())
    except ValueError as exc:
        raise CandidateSensitivityError("bootstrap output must remain under the pointed forward run") from exc
    if output_dir.exists():
        raise FileExistsError(f"candidate sensitivity output exists; refusing overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {
        "schema": "research-repair-v1-candidate-sensitivity-progress-v1",
        "status": "RUNNING" if formal else "PILOT_RUNNING",
        "formal": formal,
        "n_replicates": int(n_replicates), "seed": int(seed), "ci": float(ci),
        "completed_groups": [], "n_groups": 0,
    }
    _write_json(output_dir / "progress.json", progress)

    forwarded = [entry for entry in entries if entry["_status"] == "FORWARDED_WITH_OLD_ANCHOR_PASS"]
    regular = [entry for entry in entries if str(entry["cell"]) not in DOSE_LEVELS]
    dose_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for entry in entries:
        if str(entry["cell"]) in DOSE_LEVELS:
            dose_groups.setdefault((str(entry["family"]), str(entry["candidate_source"])), {})[str(entry["cell"])] = entry
    groups: list[dict[str, Any]] = []
    source_group_for_split: dict[tuple[str, str, str, str], str] = {}
    for row in regular:
        family, source, cell = map(str, row["_key"])
        for split_name, split_filter in EVAL_SPLITS:
            group_name = f"{family}__{source}__{cell}__{split_name}"
            groups.append({"name": group_name, "entries": [row], "is_dose": False,
                           "eval_split": split_name, "split_filter": split_filter})
            source_group_for_split[(family, source, cell, split_name)] = group_name
    for (family, source), rows in sorted(dose_groups.items()):
        ordered = [rows[level] for level in DOSE_LEVELS if level in rows]
        for split_name, split_filter in EVAL_SPLITS:
            group_name = f"{family}__{source}__dose_joint__{split_name}"
            groups.append({"name": group_name, "entries": ordered, "is_dose": True,
                           "eval_split": split_name, "split_filter": split_filter})
            for level in DOSE_LEVELS:
                if level in rows:
                    source_group_for_split[(family, source, level, split_name)] = group_name
    expected_source_cohort_splits = []
    for row in entries:
        family, source, cell = map(str, row["_key"])
        for split_name, _ in EVAL_SPLITS:
            expected_source_cohort_splits.append({
                "family": family, "candidate_source": source, "cell": cell,
                "eval_split": split_name,
                "group": source_group_for_split.get((family, source, cell, split_name)),
                "status": "PENDING",
            })
    progress["n_groups"] = len(groups)
    _write_json(output_dir / "progress.json", progress)

    all_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    status_by_source_group: dict[tuple[str, str, str], str] = {}
    for group in groups:
        group_name = str(group["name"])
        group_entries = group["entries"]
        is_dose = bool(group["is_dose"])
        eval_split_name = str(group["eval_split"])
        split_filter = group["split_filter"]
        if is_dose:
            levels = {str(row["cell"]): row for row in group_entries if row["_arrays"] is not None}
            family, source = group_name.split("__", 2)[:2]
            source_keys = [(family, source, level) for level in DOSE_LEVELS]
            if tuple(level for level in DOSE_LEVELS if level in levels) != DOSE_LEVELS:
                status = "UNAVAILABLE_INCOMPLETE_DOSE_SOURCE_COVERAGE"
                group_summaries.append({
                    "group": group_name, "status": status, "eval_split": eval_split_name,
                    "source_cohort_keys": [list(key) for key in source_keys],
                    "available_dose_levels": sorted(levels), "required_dose_levels": list(DOSE_LEVELS),
                })
                for key in source_keys:
                    status_by_source_group[(*key, eval_split_name)] = status
                progress["completed_groups"].append(group_name)
                progress["status"] = "PARTIAL" if formal else "PILOT_PARTIAL"
                _write_json(output_dir / "progress.json", progress)
                continue
            ids_by_level = {
                level: np.asarray(levels[level]["_arrays"]["sentence_id"], dtype=np.int64)
                for level in DOSE_LEVELS
            }
            common = set(ids_by_level[DOSE_LEVELS[0]].tolist())
            for level in DOSE_LEVELS[1:]:
                common.intersection_update(ids_by_level[level].tolist())
            common_all_ids = np.asarray(sorted(common), dtype=np.int64)
            base_arrays = levels[DOSE_LEVELS[0]]["_arrays"]
            base_positions = np.searchsorted(ids_by_level[DOSE_LEVELS[0]], common_all_ids)
            split_values = np.asarray(base_arrays["eval_split"]).astype(str)[base_positions]
            if split_filter is None:
                keep = np.isin(split_values, ("testA", "testB"))
            else:
                keep = split_values == str(split_filter)
            common_ids = common_all_ids[keep]
            if common_ids.size == 0:
                status = "UNAVAILABLE_EMPTY_DOSE_SPLIT_INTERSECTION"
                group_summaries.append({
                    "group": group_name, "status": status, "eval_split": eval_split_name,
                    "source_cohort_keys": [list(key) for key in source_keys],
                    "per_level_rows": {level: int(ids_by_level[level].size) for level in DOSE_LEVELS},
                    "all_level_common_rows_before_split": int(common_all_ids.size), "common_rows": 0,
                    "excluded_non_testA_testB_rows": int(np.count_nonzero(~np.isin(split_values, ("testA", "testB")))),
                })
                for key in source_keys:
                    status_by_source_group[(*key, eval_split_name)] = status
                progress["completed_groups"].append(group_name)
                progress["status"] = "PARTIAL" if formal else "PILOT_PARTIAL"
                _write_json(output_dir / "progress.json", progress)
                continue
            seed_cells, image_ids, attrition = _dose_seed_cells(levels, common_ids)
            for level in DOSE_LEVELS:
                attrition[level].update({
                    "all_level_common_rows_pooled": int(common_all_ids.size),
                    "rows_lost_to_pooled_all_level_intersection": int(ids_by_level[level].size - common_all_ids.size),
                    "rows_lost_to_eval_split_filter": int(common_all_ids.size - common_ids.size),
                })
            result = _run_one_group(
                seed_cells, image_ids, cell_tag="dose", dose_levels=DOSE_LEVELS,
                n_replicates=n_replicates, seed=seed, ci=ci,
            )
            cohort = {
                "family": family, "candidate_source": source,
                "eval_split": eval_split_name,
                "common_sentence_ids_sha256": __import__("hashlib").sha256(common_ids.astype("<i8", copy=False).tobytes()).hexdigest(),
                "n_rows": int(common_ids.size),
                "n_images": int(np.unique(image_ids).size),
                "all_level_common_rows_before_split": int(common_all_ids.size),
                "excluded_non_testA_testB_rows": int(np.count_nonzero(~np.isin(split_values, ("testA", "testB")))),
                "source_cohort_keys": [list(key) for key in source_keys],
                "dose_intersection_attrition": attrition,
            }
            result_name = group_name
            for key in source_keys:
                status_by_source_group[(*key, eval_split_name)] = (
                    "COMPONENT_INCLUDED_IN_DOSE_JOINT_GROUP_COMPLETE"
                    if formal else "COMPONENT_INCLUDED_IN_DOSE_JOINT_GROUP_PILOT"
                )
        else:
            entry = group_entries[0]
            family, source, cell = map(str, entry["_key"])
            if entry["_arrays"] is None:
                status = "NOT_FORWARDABLE_EMPTY_OLD_NEW_INTERSECTION"
                group_summaries.append({
                    "group": group_name, "status": status, "eval_split": eval_split_name,
                    "source_cohort_keys": [list(entry["_key"])],
                    "reason": "candidate audit reported no old/new common rows",
                })
                status_by_source_group[(*entry["_key"], eval_split_name)] = status
                progress["completed_groups"].append(group_name)
                progress["status"] = "PARTIAL" if formal else "PILOT_PARTIAL"
                _write_json(output_dir / "progress.json", progress)
                continue
            arrays = entry["_arrays"]
            ids = np.asarray(arrays["sentence_id"], dtype=np.int64)
            split_values = np.asarray(arrays["eval_split"]).astype(str)
            if split_filter is None:
                keep = np.isin(split_values, ("testA", "testB"))
            else:
                keep = split_values == str(split_filter)
            selected_rows = np.flatnonzero(keep)
            if selected_rows.size == 0:
                status = "UNAVAILABLE_EMPTY_EVAL_SPLIT"
                group_summaries.append({
                    "group": group_name, "status": status, "eval_split": eval_split_name,
                    "source_cohort_keys": [list(entry["_key"])],
                    "source_rows": int(ids.size), "n_rows": 0,
                    "excluded_non_testA_testB_rows": int(np.count_nonzero(~np.isin(split_values, ("testA", "testB")))),
                })
                status_by_source_group[(*entry["_key"], eval_split_name)] = status
                progress["completed_groups"].append(group_name)
                progress["status"] = "PARTIAL" if formal else "PILOT_PARTIAL"
                _write_json(output_dir / "progress.json", progress)
                continue
            image_ids = np.asarray(arrays["image_id"], dtype=np.int64)[selected_rows]
            if np.unique(image_ids).size < 2:
                status = "UNAVAILABLE_FEWER_THAN_TWO_IMAGE_CLUSTERS"
                group_summaries.append({
                    "group": group_name, "status": status, "eval_split": eval_split_name,
                    "source_cohort_keys": [list(entry["_key"])],
                    "source_rows": int(ids.size), "n_rows": int(selected_rows.size),
                    "n_images": int(np.unique(image_ids).size),
                })
                status_by_source_group[(*entry["_key"], eval_split_name)] = status
                progress["completed_groups"].append(group_name)
                progress["status"] = "PARTIAL" if formal else "PILOT_PARTIAL"
                _write_json(output_dir / "progress.json", progress)
                continue
            seed_cells = _conditions(arrays, rows=selected_rows)
            result = _run_one_group(seed_cells, image_ids, cell_tag=str(entry["cell"]),
                                    n_replicates=n_replicates, seed=seed, ci=ci)
            cohort = {
                "family": family, "candidate_source": source, "cell": cell,
                "eval_split": eval_split_name, "source_rows": int(ids.size),
                "n_rows": int(selected_rows.size), "n_images": int(np.unique(image_ids).size),
                "n_testA_rows": int(np.count_nonzero(split_values == "testA")),
                "n_testB_rows": int(np.count_nonzero(split_values == "testB")),
                "excluded_non_testA_testB_rows": int(np.count_nonzero(~np.isin(split_values, ("testA", "testB")))),
                "canonical_sentence_id_sha256": __import__("hashlib").sha256(ids[selected_rows].astype("<i8", copy=False).tobytes()).hexdigest(),
                "source_cohort_keys": [list(entry["_key"])],
            }
            result_name = group_name
            status_by_source_group[(*entry["_key"], eval_split_name)] = "COMPLETE" if formal else "PILOT"

        raw_arrays, estimate_rows = _extract_output(result, output_prefix="bootstrap")
        # Full source-cohort labels make Windows output paths too long when
        # repeated in both the group artifact and its atomic-write temp file.
        # The complete, human-readable group identity remains in every JSON/CSV
        # record; this deterministic short stem is only a filesystem key.
        artifact_stem = f"group_{hashlib.sha256(result_name.encode('utf-8')).hexdigest()[:16]}"
        raw_path = output_dir / f"{artifact_stem}_raw.npz"
        _write_npz(raw_path, raw_arrays)
        group_summary_path = output_dir / f"{artifact_stem}_summary.json"
        group_summary = {
            "schema": "research-repair-v1-candidate-sensitivity-group-v1",
            "status": "COMPLETE" if formal else "PILOT",
            "formal": formal, "group": result_name, "cohort": cohort,
            "bootstrap_contract": {
                "resample_unit": result["resample_unit"], "n_replicates": result["n_replicates"],
                "seed": result["seed"], "ci_level": result["ci_level"], "method": result["method"],
                "derived_effects_calculated_within_each_seed_and_shared_draw": True,
                "seed_aggregation": "mean over fixed B3 seeds within the same image-cluster draw",
                "rer_zero_error_draws": "NaN and counted invalid; point uses legacy zero convention",
            },
            "estimates": _json_safe(result["estimates"]),
            "cohort_interpretation": (
                "Accuracy measures candidate-induced outcome harm and is not treated as an explanation for AUROC differences."
            ),
            "matched_control_interpretation": (
                "For hard5/hard10 groups, the control is the fixed matched random-manifest construction. For dose groups, the original Phase 1F expb_m0 construction ([target] plus the frozen level-rest shuffle) is aligned by sentence_id; it is not canonical rand10. Report old_m minus old_m0 and corrected_m minus corrected_m0 as the two arm-wise dose contrasts. The four-corner dose-vs-m0 difference-in-differences is (old_m - corrected_m) - (old_m0 - corrected_m0), the version change in the dose effect. Corrected_m minus fixed original old_m0 is retained only as a separate descriptive fixed-control contrast, not as the corrected arm's baseline. Accuracy harm is not used to explain AUROC gaps."
            ),
        }
        _write_json(group_summary_path, group_summary)
        row_record = {
            **cohort,
            "group": result_name,
            "summary_path": group_summary_path.relative_to(output_dir).as_posix(),
            "summary_sha256": _sha256(group_summary_path),
            "raw_replicates_path": raw_path.relative_to(output_dir).as_posix(),
            "raw_replicates_sha256": _sha256(raw_path),
            "n_estimates": len(estimate_rows),
        }
        group_summaries.append({**row_record, "status": "COMPLETE" if formal else "PILOT"})
        for row in estimate_rows:
            row.update({
                "family": cohort["family"], "candidate_source": cohort["candidate_source"],
                "cell": row["cell"], "eval_split": eval_split_name,
                "n_rows": cohort["n_rows"], "n_images": cohort["n_images"],
                "n_replicates": result["n_replicates"],
                "raw_replicates_path": raw_path.relative_to(output_dir).as_posix(),
                "raw_replicates_sha256": _sha256(raw_path),
                "per_seed_raw_replicates_keys_json": json.dumps([
                    f"bootstrap__{seed_key}__{row['estimate']}"
                    for seed_key in ("b3_seed1", "b3_seed2", "b3_seed3")
                ], separators=(",", ":")),
            })
        all_rows.extend({"group": result_name, **row} for row in estimate_rows)
        progress["completed_groups"].append(result_name)
        progress["status"] = "PARTIAL" if formal else "PILOT_PARTIAL"
        _write_json(output_dir / "progress.json", progress)
        _write_csv(output_dir / "estimates.csv", all_rows, fields=(
            "group", "family", "candidate_source", "cell", "eval_split", "metric", "contrast",
            "formula", "units", "confidence_head", "n_rows", "n_images", "n_replicates",
            "estimate", "point", "ci_low", "ci_high", "bootstrap_std", "seed_standard_deviation",
            "valid_replicates", "invalid_replicates", "valid_fraction", "interval_status",
            "uncertainty_status", "raw_replicates_path", "raw_replicates_key",
            "raw_replicates_sha256", "per_seed_raw_replicates_keys_json", "per_seed_json",
        ))

    total_expected = len(groups)
    expected_status_prefixes = (
        ("COMPLETE", "COMPONENT_INCLUDED_IN_DOSE_JOINT_GROUP_COMPLETE")
        if formal else ("PILOT", "COMPONENT_INCLUDED_IN_DOSE_JOINT_GROUP_PILOT")
    )
    all_completed = (
        len(progress["completed_groups"]) == total_expected
        and all(str(row.get("status", "")).startswith(("COMPLETE", "PILOT")) for row in group_summaries)
        and _source_groups_complete(
            expected_source_cohort_splits, status_by_source_group, expected_status_prefixes
        )
    )
    report = {
        "schema": "research-repair-v1-candidate-sensitivity-summary-v1",
        "status": ("COMPLETE" if formal and all_completed else "PARTIAL") if formal else ("PILOT" if all_completed else "PILOT_PARTIAL"),
        "formal": formal,
        "audit_run_id": resolved["audit"]["pointer"].get("audit_run_id"),
        "candidate_audit_summary_sha256": resolved["audit"]["summary_sha256"],
        "candidate_forward_summary_sha256": _sha256(forward_dir / "summary.json"),
        "predictions_manifest": {
            "path": str(resolved["forward_summary"].get("predictions_manifest", {}).get("path")),
            "sha256": _sha256(resolved["manifest_path"]),
        },
        "bootstrap_contract": {
            "resample_unit": "image_cluster", "n_replicates": int(n_replicates),
            "seed": int(seed), "ci_level": float(ci), "method": "percentile",
            "estimand_orientation": "old minus corrected candidate; fractions and RER ratio remain unscaled",
            "no_percentage_share_claims": True,
        },
        "bootstrap": {
            "resample_unit": "image_cluster", "n_replicates": int(n_replicates),
            "seed": int(seed), "ci_level": float(ci), "method": "percentile",
        },
        "dose_estimand_contract": {
            "old_arm": "metric(old_m) - metric(old_m0)",
            "corrected_arm": "metric(corrected_m) - metric(corrected_m0)",
            "version_change": "[metric(old_m) - metric(corrected_m)] - [metric(old_m0) - metric(corrected_m0)]",
            "fixed_original_m0_contrast": "metric(corrected_m) - metric(original old_m0); descriptive only, not corrected baseline",
        },
        "forwarded_source_cohorts": [row["_key"] for row in forwarded],
        "nonforwardable_empty_source_cohorts": [row["_key"] for row in entries if row["_status"] == "NOT_FORWARDABLE_EMPTY_OLD_NEW_INTERSECTION"],
        "expected_source_cohort_split_groups": [
            {**row, "status": status_by_source_group.get(
                (row["family"], row["candidate_source"], row["cell"], row["eval_split"]),
                "NOT_COMPLETED",
            )}
            for row in expected_source_cohort_splits
        ],
        "groups": group_summaries,
        "estimates_csv": "estimates.csv" if all_rows else None,
        "n_estimates": int(len(all_rows)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output_dir / "summary.json", report)
    progress["status"] = report["status"]
    progress["summary"] = "summary.json"
    _write_json(output_dir / "progress.json", progress)
    return report


__all__ = [
    "CandidateSensitivityError", "DOSE_LEVELS", "FORMAL_CI", "FORMAL_REPLICATES",
    "FORMAL_SEED", "SENSITIVITY_METRICS", "run_candidate_sensitivity_bootstrap",
]
