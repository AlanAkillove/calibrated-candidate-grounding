"""Formal B image-cluster bootstrap; run only after the statistics slot is granted.

Loads the B point-prediction CSVs, checks sentence/image row alignment, and
produces shared 5,000-draw CIs across seeds and all requested comparisons.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccg.repairs.bootstrap import assert_row_alignment
from ccg.repairs.atomic import replace_with_retry
from ccg.repairs.statistics import joint_prediction_bootstrap

OUT_DEFAULT = ROOT / "results/research_repair_v1/information"
SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
SPLITS = ("testA", "testB", "__pooled_test__")
METRICS = ("accuracy", "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _new_temp_path(path)
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _new_temp_path(path)
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temp, path)


def _new_temp_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    return Path(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PROCESSING_NODES = {
    "ROOT", "SRC", "OUT_DEFAULT", "SEEDS", "SPLITS", "METRICS",
    "_read_table", "_key", "_verify_alignment", "_append_absolute_and_effects",
    "_run_call", "_add_condition", "_load_conditions", "_effect_rules",
    "_named_contrast", "_scope_specs",
}


def _processing_signature(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    nodes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _PROCESSING_NODES:
            nodes.append(ast.get_source_segment(source, node) or "")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_compute_bootstrap":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "plans" for target in child.targets
                ):
                    nodes.append(ast.get_source_segment(source, child) or "")
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & _PROCESSING_NODES:
                nodes.append(ast.get_source_segment(source, node) or "")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in _PROCESSING_NODES:
            nodes.append(ast.get_source_segment(source, node) or "")
        elif isinstance(node, ast.ImportFrom) and node.module in {"ccg.repairs.bootstrap", "ccg.repairs.statistics"}:
            nodes.append(ast.get_source_segment(source, node) or "")
    if not nodes or not {"_run_call", "_scope_specs"} <= {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }:
        raise ValueError(f"could not identify the B processing functions in {path}")
    return hashlib.sha256("\n\n".join(nodes).encode("utf-8")).hexdigest()


def _summary_from_saved_draws(values: np.ndarray, point: float) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    valid = samples[np.isfinite(samples)]
    invalid = int(samples.size - valid.size)
    minimum_fraction = 0.95
    minimum_valid = int(np.ceil(minimum_fraction * samples.size))
    if valid.size >= 2:
        low, high = np.quantile(valid, [0.025, 0.975])
        bootstrap_std = float(np.std(valid, ddof=1))
        ci_low: float | None = float(low)
        ci_high: float | None = float(high)
        interval_status = "AVAILABLE_VALID_REPLICATES"
    else:
        bootstrap_std = float("nan")
        ci_low = ci_high = None
        interval_status = "UNAVAILABLE_TOO_FEW_VALID_REPLICATES"
    valid_fraction = float(valid.size / samples.size) if samples.size else 0.0
    if valid.size == samples.size and samples.size > 0:
        uncertainty_status = "SUFFICIENT_ALL_PLANNED_REPLICATES_VALID"
        gate_eligible = True
    elif valid.size >= minimum_valid and valid.size >= 2:
        uncertainty_status = "CONDITIONAL_ON_VALID_DRAWS_NOT_GATE_ELIGIBLE"
        gate_eligible = False
    else:
        uncertainty_status = "UNCERTAINTY_INSUFFICIENT"
        gate_eligible = False
    return {
        "point": float(point), "ci_low": ci_low, "ci_high": ci_high,
        "bootstrap_std": bootstrap_std, "valid_replicates": int(valid.size),
        "invalid_replicates": invalid, "interval_status": interval_status,
        "uncertainty_status": uncertainty_status, "valid_fraction": valid_fraction,
        "minimum_valid_fraction_for_exploratory_ci": minimum_fraction,
        "minimum_valid_replicates_for_exploratory_ci": minimum_valid,
        "gate_eligible": gate_eligible,
        "ci_conditioning": "all_planned_draws_valid" if valid.size == samples.size else "conditional_on_finite_draws",
    }


def _refresh_call_summary_from_draws(
    rows: list[dict[str, Any]], call: dict[str, Any], raw_arrays: Mapping[str, np.ndarray],
) -> None:
    """Rebuild compact bootstrap summaries from a complete saved replicate archive."""
    for row in rows:
        mean_key = str(row["raw_replicates_key"])
        summary = _summary_from_saved_draws(raw_arrays[mean_key], float(row["point"]))
        row["n_replicates"] = int(raw_arrays[mean_key].shape[0])
        for field in ("ci_low", "ci_high", "valid_replicates", "invalid_replicates", "interval_status"):
            row[field] = summary[field]
        seed_points = json.loads(row["per_seed_point_json"])
        seed_keys = json.loads(row["per_seed_evidence_path"].split("::", 1)[1])
        seed_summary = {}
        for seed, key in seed_keys.items():
            current = _summary_from_saved_draws(raw_arrays[key], float(seed_points[seed]))
            seed_summary[seed] = {
                "point": current["point"], "ci_low": current["ci_low"], "ci_high": current["ci_high"],
                "valid_replicates": current["valid_replicates"], "invalid_replicates": current["invalid_replicates"],
            }
        row["per_seed_ci_json"] = json.dumps(seed_summary, ensure_ascii=False)

    for estimate in call["estimates"].values():
        mean_key = str(estimate["raw_replicates_key"])
        aggregate = _summary_from_saved_draws(raw_arrays[mean_key], float(estimate["point"]))
        for field in (
            "ci_low", "ci_high", "bootstrap_std", "valid_replicates", "invalid_replicates",
            "interval_status", "uncertainty_status", "valid_fraction",
            "minimum_valid_fraction_for_exploratory_ci", "minimum_valid_replicates_for_exploratory_ci",
            "gate_eligible", "ci_conditioning",
        ):
            estimate[field] = aggregate[field]
        estimate["per_seed"] = {}
        for seed, key in estimate["per_seed_raw_replicates_keys"].items():
            seed_point = float(estimate["seed_estimates"][seed])
            seed_summary = _summary_from_saved_draws(raw_arrays[key], seed_point)
            estimate["per_seed"][seed] = {
                "point": seed_point, "ci_low": seed_summary["ci_low"], "ci_high": seed_summary["ci_high"],
                "valid_replicates": seed_summary["valid_replicates"],
                "invalid_replicates": seed_summary["invalid_replicates"],
            }
    call["n_replicates"] = int(next(iter(raw_arrays.values())).shape[0])


def _protocol_payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("source_sha256", None)
    value.pop("bootstrap_preflight", None)
    return value


def _recover_orphan_replicates(
    *,
    checkpoint_dir: Path,
    call_id: str,
    cache_metadata: Mapping[str, Any],
    source_fingerprint: Mapping[str, str],
    scope: str,
    cell: str,
    k: str,
    split_name: str,
    reference: Mapping[str, np.ndarray],
    per_seed_conditions: Mapping[str, Mapping[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]]],
    absolute_fields: Mapping[str, Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    n_replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray], dict[str, Any]] | None:
    """Recover a complete flushed NPZ left orphaned by a failed Windows replace.

    Recovery is allowed only when the producer's frozen runner/protocol snapshots,
    all input and metric implementation hashes, estimator-function signature,
    cohort identity, config, and every raw-array key/shape match exactly.
    """
    log_path = checkpoint_dir / "call_log.jsonl"
    old_runner = checkpoint_dir / "attempt1_runner_source.py"
    old_protocol = checkpoint_dir / "attempt1_protocol.json"
    if not log_path.exists() or not old_runner.exists() or not old_protocol.exists():
        return None
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed_index = next((index for index in range(len(events) - 1, -1, -1)
                         if events[index].get("event") == "FAILED" and events[index].get("call_id") == call_id), None)
    if failed_index is None:
        return None
    failed = events[failed_index]
    start = next((event for event in reversed(events[:failed_index])
                  if event.get("event") == "STARTED" and event.get("call_id") == call_id), None)
    if start is None:
        return None
    old_source = start.get("source_sha256", {})
    if not old_source or failed.get("raw_temp_sha256") is None:
        return None
    current_metadata = dict(cache_metadata)
    current_metadata["source_sha256"] = old_source
    old_token = hashlib.sha256(
        json.dumps(current_metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    slug = call_id.replace("/", "_").replace("\\", "_")
    expected_name = f"{slug}.{old_token}.npz.tmp"
    raw_path = Path(str(failed.get("raw_temp_path", ""))).resolve()
    if raw_path.parent != checkpoint_dir.resolve() or raw_path.name != expected_name or not raw_path.is_file():
        return None
    if _sha256_file(raw_path) != failed["raw_temp_sha256"]:
        raise ValueError(f"{call_id}: orphan raw archive hash differs from the preserved failed-attempt record")
    if _sha256_file(old_runner) != old_source.get("runner_sha256"):
        raise ValueError(f"{call_id}: frozen attempt runner snapshot does not match its STARTED source hash")
    if _sha256_file(old_protocol) != old_source.get("protocol_sha256"):
        raise ValueError(f"{call_id}: frozen attempt protocol snapshot does not match its STARTED source hash")
    if failed.get("attempt1_runner_snapshot_sha256") != old_source.get("runner_sha256"):
        raise ValueError(f"{call_id}: failed-attempt runner snapshot evidence is inconsistent")
    if failed.get("attempt1_protocol_snapshot_sha256") != old_source.get("protocol_sha256"):
        raise ValueError(f"{call_id}: failed-attempt protocol snapshot evidence is inconsistent")
    for key in ("shared_bootstrap_sha256", "statistics_sha256", "semantic_predictions_sha256", "cross_k_predictions_sha256"):
        if old_source.get(key) != source_fingerprint.get(key):
            raise ValueError(f"{call_id}: orphan raw source fingerprint changed for {key}")
    if start.get("cohort_counts") != cache_metadata.get("cohort_counts"):
        raise ValueError(f"{call_id}: orphan raw cohort rows/images differ from the current canonical cohort")
    if _protocol_payload(old_protocol) != _protocol_payload(OUT_DEFAULT / "protocol.json"):
        raise ValueError(f"{call_id}: frozen protocol parameters differ from the current protocol")
    old_processing = _processing_signature(old_runner)
    current_processing = _processing_signature(Path(__file__).resolve())
    if old_processing != current_processing:
        raise ValueError(f"{call_id}: estimator processing code changed since the orphan raw archive was computed")

    try:
        with np.load(raw_path, allow_pickle=False) as archive:
            raw_arrays = {name: archive[name].copy() for name in archive.files}
    except Exception as exc:
        raise ValueError(f"{call_id}: orphan raw archive is not a complete readable NPZ") from exc
    if not raw_arrays or any(array.shape != (n_replicates,) or array.dtype != np.float64 for array in raw_arrays.values()):
        raise ValueError(f"{call_id}: orphan raw arrays have unexpected names, shapes, or dtypes")
    if any(np.any(np.isinf(array)) for array in raw_arrays.values()):
        raise ValueError(f"{call_id}: orphan raw arrays contain infinity; expected finite values or NaNs")

    rows, call, point_probe_arrays = _run_call(
        out=Path("."), scope=scope, cell=cell, k=k, split_name=split_name,
        reference=reference, per_seed_conditions=per_seed_conditions,
        absolute_fields=absolute_fields, contrasts=contrasts, n_replicates=1,
    )
    if set(raw_arrays) != set(point_probe_arrays):
        raise ValueError(f"{call_id}: orphan raw estimate keys do not match the current estimator spec")
    _refresh_call_summary_from_draws(rows, call, raw_arrays)
    recovery = {
        "recovered": True,
        "recovered_from_call_id": call_id,
        "original_raw_temp_path": str(raw_path),
        "original_raw_temp_bytes": raw_path.stat().st_size,
        "original_raw_temp_sha256": failed["raw_temp_sha256"],
        "producer_runner_sha256": old_source["runner_sha256"],
        "producer_protocol_sha256": old_source["protocol_sha256"],
        "current_runner_sha256": source_fingerprint["runner_sha256"],
        "current_protocol_sha256": source_fingerprint["protocol_sha256"],
        "original_attempt_started_utc": start.get("started_utc"),
        "original_attempt_failed_utc": failed.get("finished_utc"),
        "original_attempt_elapsed_seconds": failed.get("elapsed_seconds"),
        "estimator_processing_signature": current_processing,
        "validated": [
            "raw NPZ archive and checksum", "all raw keys and 5,000-row float64 shapes",
            "input predictions, metric implementation, and shared bootstrap source hashes",
            "frozen attempt runner/protocol snapshots", "unchanged estimator processing signature",
            "canonical sentence/image cohort order, cell set, and bootstrap parameters",
            "one separate nonformal draw was generated only to verify current point/spec/raw-key assembly and was discarded",
            "the compact points are recomputed from canonical predictions; formal CI arrays are exclusively the preserved 5,000 original draws",
        ],
        "recovery_diagnostic_draws_generated_and_discarded": 1,
        "formal_replicates_adopted_from_original_archive": int(n_replicates),
        "diagnostic_draws_mixed_into_formal_replicates": 0,
        "original_failure": failed.get("error"),
    }
    return rows, call, raw_arrays, recovery


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _new_temp_path(path)
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temp, path)


def _append_status(out: Path, stage: str, state: str, detail: str) -> None:
    path = out / "STATUS.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    stages = payload.setdefault("stages", {})
    node = stages.setdefault(stage, {})
    if state == "RUNNING" and not node.get("started_utc"):
        node["started_utc"] = _utc()
    node.update({"state": state, "updated_utc": _utc(), "detail": detail})
    if state in {"COMPLETE", "FAILED"}:
        node["finished_utc"] = _utc()
    if any(value.get("state") == "RUNNING" for value in stages.values()):
        payload["status"] = "RUNNING"
    else:
        payload["status"] = "COMPLETE" if state == "COMPLETE" else state
        payload["current_stage"] = stage
    payload["updated_utc"] = _utc()
    _write_json(path, payload)


def _read_table(path: Path, *, has_training_regime: bool) -> dict[tuple[str, str, int, str, str], dict[str, np.ndarray]]:
    """Stream only testA/testB rows into compact arrays, preserving file order."""
    parts: dict[tuple[str, str, int, str, str], dict[str, list[Any]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["eval_split"]
            if split not in ("testA", "testB"):
                continue
            key = (
                str(row["seed"]), str(row["cell"]), int(row["K"]), str(row["model"]),
                str(row["training_regime"]) if has_training_regime else "smallK5_10",
            )
            node = parts.setdefault(key, {"sentence_id": [], "ref_id": [], "image_id": [], "eval_split": [], "K": [], "grounding_correct": [], "confidence": []})
            node["sentence_id"].append(int(row["sentence_id"]))
            node["ref_id"].append(int(row["ref_id"]))
            node["image_id"].append(int(row["image_id"]))
            node["eval_split"].append(0 if split == "testA" else 1)
            node["K"].append(int(row["K"]))
            node["grounding_correct"].append(int(row["grounding_correct"]))
            node["confidence"].append(float(row["probability"]))
    arrays: dict[tuple[str, str, int, str, str], dict[str, np.ndarray]] = {}
    for key, node in parts.items():
        arrays[key] = {
            "sentence_id": np.asarray(node["sentence_id"], dtype=np.int64),
            "ref_id": np.asarray(node["ref_id"], dtype=np.int64),
            "image_id": np.asarray(node["image_id"], dtype=np.int64),
            "eval_split": np.asarray(node["eval_split"], dtype=np.int8),
            "K": np.asarray(node["K"], dtype=np.int16),
            "grounding_correct": np.asarray(node["grounding_correct"], dtype=np.int8),
            "correct": np.asarray(node["grounding_correct"], dtype=np.int8),
            "confidence": np.asarray(node["confidence"], dtype=np.float64),
        }
    return arrays


def _key(seed: str, cell: str, k: int, model: str, regime: str = "smallK5_10") -> tuple[str, str, int, str, str]:
    return seed, cell, int(k), model, regime


def _verify_alignment(
    reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray], *,
    label: str, expected_k: int | None = None, same_correctness: bool = False,
) -> None:
    assert_row_alignment(
        reference, candidate,
        required_fields=("ref_id", "image_id", "eval_split"),
        reference_name=f"canonical-{label}", candidate_name=f"candidate-{label}",
    )
    if expected_k is not None and not np.all(np.asarray(candidate["K"]) == int(expected_k)):
        raise ValueError(f"{label}: K metadata does not equal expected K={expected_k}")
    if same_correctness and not np.array_equal(reference["grounding_correct"], candidate["grounding_correct"]):
        raise ValueError(f"{label}: grounding correctness differs within a same-seed, same-cell reliability comparison")


def _append_absolute_and_effects(
    conditions: Mapping[str, tuple[Any, Any]],
    meta: dict[str, dict[str, Any]],
    *,
    scope: str,
    cell: str,
    k: str,
    eval_split: str,
    absolute_fields: Mapping[str, Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for condition, info in absolute_fields.items():
        for metric in METRICS:
            name = f"abs::{condition}::{metric}"
            specs.append({"name": name, "operation": "condition", "condition": condition, "metric": metric})
            meta[name] = {
                "scope": scope, "cell": info.get("cell", cell), "K": str(info.get("K", k)), "eval_split": eval_split,
                "contrast": "", "model": info.get("model", ""),
                "training_regime": info.get("training_regime", ""), "metric": metric,
            }
    for contrast in contrasts:
        label = str(contrast["label"])
        for metric, rule in contrast["metrics"].items():
            output_metric = str(rule["output_metric"])
            name = f"effect::{contrast.get('name_token', label)}::{output_metric}"
            spec = {"name": name, "operation": rule.get("operation", "difference"), "metric": metric, "scale": float(rule.get("scale", 1.0))}
            for field in ("a", "b", "c", "d"):
                if field in rule:
                    spec[field] = rule[field]
            specs.append(spec)
            meta[name] = {
                "scope": scope,
                "cell": contrast.get("cell", cell),
                "K": contrast.get("K", k),
                "eval_split": eval_split,
                "contrast": label,
                "model": (
                    f"({rule['a']} - {rule['b']}) - ({rule['c']} - {rule['d']})"
                    if rule.get("operation") == "difference_of_differences"
                    else f"{rule['a']} - {rule['b']}"
                ),
                "training_regime": contrast.get("training_regime", "paired"),
                "metric": output_metric,
            }
    return specs


def _run_call(
    *,
    out: Path,
    scope: str,
    cell: str,
    k: str,
    split_name: str,
    reference: Mapping[str, np.ndarray],
    per_seed_conditions: Mapping[str, Mapping[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]]],
    absolute_fields: Mapping[str, Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    n_replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    split_code = {"testA": 0, "testB": 1}
    select = np.ones(reference["sentence_id"].size, dtype=bool) if split_name == "__pooled_test__" else reference["eval_split"] == split_code[split_name]
    if int(select.sum()) < 2:
        raise ValueError(f"{scope}/{cell}/{split_name}: fewer than two test rows")
    image_ids = reference["image_id"][select]
    seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for seed in SEEDS:
        current: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for condition, (identity, values) in per_seed_conditions[seed].items():
            _verify_alignment(reference, identity, label=f"{scope}-{cell}-{seed}-{condition}")
            current[condition] = (np.asarray(values["confidence"])[select], np.asarray(values["correct"])[select])
        seed_cells[seed] = current
    estimates_meta: dict[str, dict[str, Any]] = {}
    specs = _append_absolute_and_effects(
        seed_cells[SEEDS[0]], estimates_meta, scope=scope, cell=cell, k=k,
        eval_split=split_name, absolute_fields=absolute_fields, contrasts=contrasts,
    )
    result = joint_prediction_bootstrap(
        image_ids,
        seed_cells,
        estimates=specs,
        metrics=METRICS,
        n_replicates=n_replicates,
        seed=0,
        ci=0.95,
    )
    npz_values: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    compact_estimates: dict[str, Any] = {}
    for name, estimate in result["estimates"].items():
        key_suffix = hashlib.sha1(f"{scope}|{cell}|{split_name}|{name}".encode()).hexdigest()[:16]
        mean_key = f"mean_{key_suffix}"
        npz_values[mean_key] = np.asarray(estimate["replicates"], dtype=np.float64)
        per_seed_summary: dict[str, Any] = {}
        per_seed_keys: dict[str, str] = {}
        for seed, summary in estimate["per_seed"].items():
            seed_key = f"seed_{seed[-1]}_{key_suffix}"
            npz_values[seed_key] = np.asarray(estimate["per_seed_replicates"][seed], dtype=np.float64)
            per_seed_keys[seed] = seed_key
            per_seed_summary[seed] = {
                "point": summary["point"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
                "valid_replicates": summary["valid_replicates"], "invalid_replicates": summary["invalid_replicates"],
            }
        info = estimates_meta[name]
        rows.append({
            **info,
            "point": estimate["point"], "ci_low": estimate["ci_low"], "ci_high": estimate["ci_high"],
            "n_replicates": result["n_replicates"], "valid_replicates": estimate["valid_replicates"],
            "invalid_replicates": estimate["invalid_replicates"],
            "seed_standard_deviation": estimate["seed_standard_deviation"],
            "per_seed_point_json": json.dumps(estimate["seed_estimates"], ensure_ascii=False),
            "per_seed_ci_json": json.dumps(per_seed_summary, ensure_ascii=False),
            "per_seed_evidence_path": f"information_bootstrap_raw.npz::{json.dumps(per_seed_keys, ensure_ascii=False)}",
            "raw_replicates_key": mean_key,
            "interval_status": estimate["interval_status"],
        })
        compact_estimates[name] = {
            "metadata": info,
            "point": estimate["point"], "ci_low": estimate["ci_low"], "ci_high": estimate["ci_high"],
            "valid_replicates": estimate["valid_replicates"], "invalid_replicates": estimate["invalid_replicates"],
            "seed_standard_deviation": estimate["seed_standard_deviation"],
            "seed_estimates": estimate["seed_estimates"], "per_seed": per_seed_summary,
            "raw_replicates_key": mean_key, "per_seed_raw_replicates_keys": per_seed_keys,
        }
    call_payload = {
        "scope": scope, "cell": cell, "K": k, "eval_split": split_name,
        "resample_unit": result["resample_unit"], "n_rows": result["n_rows"], "n_images": result["n_images"],
        "n_replicates": result["n_replicates"], "bootstrap_seed": 0, "ci_level": 0.95,
        "method": "percentile", "estimates": compact_estimates,
    }
    return rows, call_payload, npz_values


def _add_condition(
    conditions: dict[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]],
    info: dict[str, dict[str, Any]],
    condition: str,
    data: Mapping[str, np.ndarray],
    *,
    model: str,
    regime: str,
    cell: str,
    k: int,
) -> None:
    conditions[condition] = (data, {"confidence": data["confidence"], "correct": data["grounding_correct"]})
    info[condition] = {"model": model, "training_regime": regime, "cell": cell, "K": int(k)}


def _load_conditions(
    semantic: Mapping[Any, Any], cross: Mapping[Any, Any], seed: str,
    canonical: Mapping[str, np.ndarray], scope: str, cell: str = "",
) -> tuple[dict[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]], dict[str, dict[str, Any]]]:
    conditions: dict[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]] = {}
    info: dict[str, dict[str, Any]] = {}
    if scope == "random":
        for k in (5, 10, 20, 50):
            random_cell = f"randomK{k}"
            semantic_s = semantic[_key(seed, random_cell, k, "S")]
            _verify_alignment(canonical, semantic_s, label=f"random-{seed}-{random_cell}-S", expected_k=k)
            for group in ("S", "S+Q", "S+V", "Full"):
                data = semantic[_key(seed, random_cell, k, group)]
                _verify_alignment(
                    semantic_s, data, label=f"random-{seed}-{random_cell}-{group}",
                    expected_k=k, same_correctness=(group != "S"),
                )
                _verify_alignment(canonical, data, label=f"random-canonical-{seed}-{random_cell}-{group}", expected_k=k)
                condition = f"semantic::{group}::K{k}"
                _add_condition(conditions, info, condition, data, model=group, regime="smallK5_10", cell=random_cell, k=k)
            for model, regime in (
                ("StatsLogistic_smallK5_10", "smallK5_10"),
                ("StatsLogistic_largeK20_50", "largeK20_50"),
                ("ScoreDeepSets_smallK5_10", "smallK5_10"),
                ("ScoreDeepSets_largeK20_50", "largeK20_50"),
                ("corrected_score_MSP", "score_only"),
            ):
                data = cross[_key(seed, random_cell, k, model, regime)]
                _verify_alignment(semantic_s, data, label=f"random-{seed}-{random_cell}-{model}", expected_k=k, same_correctness=True)
                _verify_alignment(canonical, data, label=f"random-canonical-{seed}-{random_cell}-{model}", expected_k=k)
                condition = f"cross::{model}::K{k}"
                _add_condition(conditions, info, condition, data, model=model, regime=regime, cell=random_cell, k=k)
    elif scope in ("matched_same4", "dose_same8"):
        cells = ("rand5", "hard5") if scope == "matched_same4" else ("expb_m0", "expb_m2", "expb_m4", "expb_m8")
        k = 5 if scope == "matched_same4" else 10
        for cell_name in cells:
            semantic_s = semantic[_key(seed, cell_name, k, "S")]
            _verify_alignment(canonical, semantic_s, label=f"{scope}-{seed}-{cell_name}-S", expected_k=k)
            for group in ("S", "S+Q", "S+V", "Full"):
                data = semantic[_key(seed, cell_name, k, group)]
                _verify_alignment(semantic_s, data, label=f"{scope}-{seed}-{cell_name}-{group}", expected_k=k, same_correctness=(group != "S"))
                _verify_alignment(canonical, data, label=f"{scope}-canonical-{seed}-{cell_name}-{group}", expected_k=k)
                condition = f"{cell_name}::{group}"
                _add_condition(conditions, info, condition, data, model=group, regime=f"smallK5_10/{cell_name}", cell=cell_name, k=k)
    else:
        raise ValueError(scope)
    return conditions, info


def _effect_rules(a: str, b: str, *, include_accuracy: bool = True) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    if include_accuracy:
        rules["accuracy"] = {"output_metric": "accuracy_diff_pp", "a": a, "b": b, "scale": 100.0}
    rules["auroc_correct"] = {"output_metric": "auroc_correct", "a": a, "b": b}
    # Positive values mean the second (improved) condition reduces E-AURC or RER.
    rules["e_aurc"] = {"output_metric": "e_aurc_reduction", "a": b, "b": a}
    # RER=(baseline risk - selective risk)/baseline risk is a skill score:
    # larger is better. Thus the named A-minus-B gain is RER(A)-RER(B).
    rules["rer_at_50"] = {"output_metric": "rer_at_50_gain_pp", "a": a, "b": b, "scale": 100.0}
    rules["rer_at_80"] = {"output_metric": "rer_at_80_gain_pp", "a": a, "b": b, "scale": 100.0}
    return rules


def _named_contrast(
    label: str, rules: Mapping[str, Mapping[str, Any]], *, name_token: str,
    cell: str, k: str, training_regime: str,
) -> dict[str, Any]:
    return {"label": label, "name_token": name_token, "metrics": dict(rules), "cell": cell, "K": k, "training_regime": training_regime}


def _scope_specs(
    scope: str, cell: str, k: str, info: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    absolute = dict(info)
    contrasts: list[dict[str, Any]] = []
    if scope in ("random", "matched_same4", "dose_same8"):
        if scope == "random":
            group_cells = ("randomK5", "randomK10", "randomK20", "randomK50")
        elif scope == "matched_same4":
            group_cells = ("rand5", "hard5")
        else:
            group_cells = ("expb_m0", "expb_m2", "expb_m4", "expb_m8")
        for group_cell in group_cells:
            group_k = int(group_cell.removeprefix("randomK")) if scope == "random" else int(k)
            if scope == "random":
                prefixes = {group: f"semantic::{group}::K{group_k}" for group in ("S", "S+Q", "S+V", "Full")}
            else:
                prefixes = {group: f"{group_cell}::{group}" for group in ("S", "S+Q", "S+V", "Full")}
            for label, condition_a, condition_b in (
                ("Full_minus_S_plus_Q", prefixes["Full"], prefixes["S+Q"]),
                ("S_plus_V_minus_S", prefixes["S+V"], prefixes["S"]),
                ("Full_minus_S", prefixes["Full"], prefixes["S"]),
            ):
                contrasts.append(_named_contrast(
                    label, _effect_rules(condition_a, condition_b), name_token=f"{label}__{group_cell}",
                    cell=group_cell, k=str(group_k), training_regime="smallK5_10",
                ))
        if scope == "matched_same4":
            for group in ("S", "S+Q", "S+V", "Full"):
                rand = f"rand5::{group}"
                hard = f"hard5::{group}"
                contrasts.append(_named_contrast(
                    f"Hard_minus_Rand_{group}", _effect_rules(hard, rand),
                    name_token=f"Hard_minus_Rand_{group}", cell="hard5_vs_rand5", k="5",
                    training_regime="matched_same_seed",
                ))
        if scope == "dose_same8":
            for group in ("S", "S+Q", "S+V", "Full"):
                baseline = f"expb_m0::{group}"
                for dose_cell in ("expb_m2", "expb_m4", "expb_m8"):
                    dose = f"{dose_cell}::{group}"
                    contrasts.append(_named_contrast(
                        f"{dose_cell}_minus_m0_{group}", _effect_rules(dose, baseline),
                        name_token=f"{dose_cell}_minus_m0_{group}", cell=dose_cell, k="10",
                        training_regime="dose_shift_same_seed",
                    ))
    if scope == "random":
        for model in ("StatsLogistic", "ScoreDeepSets"):
            small_regime, large_regime = "smallK5_10", "largeK20_50"
            # Same-test-K comparison of models trained on different K regimes.
            for k_value in (5, 10, 20, 50):
                small = f"cross::{model}_{small_regime}::K{k_value}"
                large = f"cross::{model}_{large_regime}::K{k_value}"
                training_rules = {
                    "accuracy": {"output_metric": "accuracy_diff_pp", "a": large, "b": small, "scale": 100.0},
                    "auroc_correct": {"output_metric": "auroc_difference", "a": large, "b": small},
                    "e_aurc": {"output_metric": "e_aurc_difference", "a": large, "b": small},
                    "rer_at_50": {"output_metric": "rer_at_50_difference_pp", "a": large, "b": small, "scale": 100.0},
                    "rer_at_80": {"output_metric": "rer_at_80_difference_pp", "a": large, "b": small, "scale": 100.0},
                }
                contrasts.append(_named_contrast(
                    f"large_minus_small_{model}_K{k_value}", training_rules,
                    name_token=f"large_minus_small_{model}_K{k_value}",
                    cell=f"randomK{k_value}", k=str(k_value), training_regime="largeK20_50-minus-smallK5_10",
                ))

            for regime in (small_regime, large_regime):
                for k_hi in (20, 50):
                    low = f"cross::{model}_{regime}::K5"
                    high = f"cross::{model}_{regime}::K{k_hi}"
                    kshift = {
                        "accuracy": {"output_metric": "accuracy_change_pp", "a": high, "b": low, "scale": 100.0},
                        "auroc_correct": {"output_metric": "auroc_drop", "a": low, "b": high},
                        "e_aurc": {"output_metric": "e_aurc_worsening", "a": high, "b": low},
                        "rer_at_50": {"output_metric": "rer_at_50_drop_pp", "a": low, "b": high, "scale": 100.0},
                        "rer_at_80": {"output_metric": "rer_at_80_drop_pp", "a": low, "b": high, "scale": 100.0},
                    }
                    contrasts.append(_named_contrast(
                        f"K5_to_K{k_hi}_{model}_{regime}", kshift,
                        name_token=f"K5_to_K{k_hi}_{model}_{regime}",
                        cell=f"randomK5_to_randomK{k_hi}", k=f"5->{k_hi}", training_regime=regime,
                    ))

            # Difference-in-differences: (K50-K5 under large training) -
            # (K50-K5 under small training), formed inside each bootstrap draw.
            s5, s50 = f"cross::{model}_{small_regime}::K5", f"cross::{model}_{small_regime}::K50"
            l5, l50 = f"cross::{model}_{large_regime}::K5", f"cross::{model}_{large_regime}::K50"
            dod_rules: dict[str, dict[str, Any]] = {}
            for metric, output, scale in (
                ("accuracy", "accuracy_change_DoD_pp", 100.0),
                ("auroc_correct", "auroc_change_DoD", 1.0),
                ("e_aurc", "e_aurc_change_DoD", 1.0),
                ("rer_at_50", "rer_at_50_change_DoD_pp", 100.0),
                ("rer_at_80", "rer_at_80_change_DoD_pp", 100.0),
            ):
                dod_rules[metric] = {
                    "operation": "difference_of_differences", "output_metric": output,
                    "a": l50, "b": l5, "c": s50, "d": s5, "scale": scale,
                }
            contrasts.append(_named_contrast(
                f"K50_vs_K5_regime_DoD_{model}", dod_rules,
                name_token=f"K50_vs_K5_regime_DoD_{model}", cell="randomK50_vs_randomK5",
                k="(50-5)_large-(50-5)_small", training_regime="largeK20_50-vs-smallK5_10",
            ))
    return absolute, contrasts


def _compute_bootstrap(
    semantic: Mapping[Any, Any], cross: Mapping[Any, Any], *, n_replicates: int,
    evidence_path: str = "information_bootstrap_raw.npz",
    checkpoint_dir: Path | None = None,
    source_fingerprint: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    flat_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    raw_npz: dict[str, np.ndarray] = {}
    plans = (
        ("random", "random_allK", "all"),
        ("matched_same4", "matched_rand5_hard5", "5"),
        ("dose_same8", "dose_m0_m2_m4_m8", "10"),
    )
    for scope, cell, k in plans:
        if scope == "random":
            reference = semantic[_key(SEEDS[0], "randomK5", 5, "S")]
        elif scope == "matched_same4":
            reference = semantic[_key(SEEDS[0], "rand5", 5, "S")]
        else:
            reference = semantic[_key(SEEDS[0], "expb_m0", int(k), "S")]
        seed_conditions: dict[str, dict[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]]] = {}
        info_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            current, info = _load_conditions(semantic, cross, seed, reference, scope, cell)
            for condition, (identity, _values) in current.items():
                _verify_alignment(reference, identity, label=f"{scope}-{cell}-{seed}-{condition}")
            seed_conditions[seed] = current
            info_by_seed[seed] = info
        absolute, contrasts = _scope_specs(scope, cell, k, info_by_seed[SEEDS[0]])
        if any(tuple(info_by_seed[seed]) != tuple(info_by_seed[SEEDS[0]]) for seed in SEEDS[1:]):
            raise ValueError(f"{scope}/{cell}: condition order differs across seeds")
        for split_name in SPLITS:
            split_code = {"testA": 0, "testB": 1}
            selected = (
                np.ones(reference["sentence_id"].size, dtype=bool)
                if split_name == "__pooled_test__"
                else reference["eval_split"] == split_code[split_name]
            )
            selected_ids = reference["sentence_id"][selected]
            selected_images = reference["image_id"][selected]
            condition_info = info_by_seed[SEEDS[0]]
            condition_counts = [
                {
                    "name": name,
                    "cell": str(details.get("cell", "")),
                    "K": str(details.get("K", "")),
                    "training_regime": str(details.get("training_regime", "")),
                    "n_rows": int(selected.sum()),
                    "n_images": int(np.unique(selected_images).size),
                }
                for name, details in condition_info.items()
            ]
            cohort_counts = {
                "n_rows": int(selected.sum()),
                "n_images": int(np.unique(selected_images).size),
                "n_conditions_per_seed": {seed: len(seed_conditions[seed]) for seed in SEEDS},
                "conditions": condition_counts,
                "sentence_id_order_sha256": hashlib.sha256(np.asarray(selected_ids, dtype="<i8").tobytes()).hexdigest(),
                "image_id_order_sha256": hashlib.sha256(np.asarray(selected_images, dtype="<i8").tobytes()).hexdigest(),
            }
            call_id = f"{scope}__{cell}__{split_name}"
            cache_metadata = {
                "schema_version": 1,
                "call_id": call_id,
                "scope": scope,
                "cell": cell,
                "K": k,
                "eval_split": split_name,
                "n_replicates": int(n_replicates),
                "bootstrap_seed": 0,
                "ci_level": 0.95,
                "resample_unit": "image_cluster",
                "method": "percentile",
                "cohort_counts": cohort_counts,
                "source_sha256": dict(source_fingerprint or {}),
            }
            cache_token = hashlib.sha256(
                json.dumps(cache_metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            cache_slug = call_id.replace("/", "_").replace("\\", "_")
            cache_json = cache_dir / f"{cache_slug}.{cache_token}.json" if (cache_dir := checkpoint_dir) else None
            cache_npz = cache_dir / f"{cache_slug}.{cache_token}.npz" if cache_dir is not None else None
            started_utc = _utc()
            start_clock = time.perf_counter()
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                _append_jsonl(cache_dir / "call_log.jsonl", {
                    "event": "STARTED", "call_id": call_id, "started_utc": started_utc,
                    "cohort_counts": cohort_counts, "source_sha256": dict(source_fingerprint or {}),
                    "n_replicates": n_replicates, "bootstrap_seed": 0, "ci_level": 0.95,
                    "resample_unit": "image_cluster", "method": "percentile",
                })

            reused = bool(cache_json is not None and cache_npz is not None and cache_json.exists() and cache_npz.exists())
            if reused:
                saved = json.loads(cache_json.read_text(encoding="utf-8"))
                if saved.get("metadata") != cache_metadata:
                    raise ValueError(f"{call_id}: checkpoint metadata differs from current inputs/config")
                if _sha256_file(cache_npz) != saved.get("raw_npz_sha256"):
                    raise ValueError(f"{call_id}: checkpoint raw-replicate checksum mismatch")
                with np.load(cache_npz, allow_pickle=False) as archive:
                    arrays = {name: archive[name].copy() for name in archive.files}
                rows = saved["rows"]
                call = saved["call"]
                elapsed_seconds = time.perf_counter() - start_clock
                if cache_dir is not None:
                    _append_jsonl(cache_dir / "call_log.jsonl", {
                        "event": "COMPLETED", "call_id": call_id, "started_utc": started_utc,
                        "finished_utc": _utc(), "elapsed_seconds": elapsed_seconds,
                        "completion_mode": "CHECKPOINT_REUSE", "cohort_counts": cohort_counts,
                        "raw_npz_sha256": saved.get("raw_npz_sha256"),
                    })
            else:
                recovery = None
                try:
                    recovered = None
                    if cache_dir is not None and source_fingerprint:
                        recovered = _recover_orphan_replicates(
                            checkpoint_dir=cache_dir, call_id=call_id, cache_metadata=cache_metadata,
                            source_fingerprint=source_fingerprint, scope=scope, cell=cell, k=k,
                            split_name=split_name, reference=reference,
                            per_seed_conditions=seed_conditions, absolute_fields=absolute,
                            contrasts=contrasts, n_replicates=n_replicates,
                        )
                    if recovered is not None:
                        rows, call, arrays, recovery = recovered
                        finished_utc = _utc()
                        recovery_elapsed = time.perf_counter() - start_clock
                        original_elapsed = float(recovery.get("original_attempt_elapsed_seconds") or 0.0)
                        elapsed_seconds = original_elapsed + recovery_elapsed
                        call.update({
                            "call_id": call_id,
                            "started_utc": recovery.get("original_attempt_started_utc", started_utc),
                            "finished_utc": finished_utc,
                            "elapsed_seconds": elapsed_seconds,
                            "recovery_elapsed_seconds": recovery_elapsed,
                            "cohort_counts": cohort_counts,
                            "raw_recovery": recovery,
                        })
                    else:
                        rows, call, arrays = _run_call(
                            out=Path("."), scope=scope, cell=cell, k=k, split_name=split_name,
                            reference=reference, per_seed_conditions=seed_conditions,
                            absolute_fields=absolute, contrasts=contrasts,
                            n_replicates=n_replicates,
                        )
                        elapsed_seconds = time.perf_counter() - start_clock
                        finished_utc = _utc()
                        call.update({
                            "call_id": call_id, "started_utc": started_utc, "finished_utc": finished_utc,
                            "elapsed_seconds": elapsed_seconds, "cohort_counts": cohort_counts,
                        })
                    if cache_dir is not None and cache_json is not None and cache_npz is not None:
                        _write_npz_atomic(cache_npz, arrays)
                        saved = {
                            "metadata": cache_metadata,
                            "call": call,
                            "rows": rows,
                            "raw_npz_file": cache_npz.name,
                            "raw_npz_sha256": _sha256_file(cache_npz),
                        }
                        if recovery is not None:
                            saved["raw_recovery"] = recovery
                        _write_json(cache_json, saved)
                        _append_jsonl(cache_dir / "call_log.jsonl", {
                            "event": "COMPLETED", "call_id": call_id, "started_utc": started_utc,
                            "finished_utc": finished_utc, "elapsed_seconds": elapsed_seconds,
                            "completion_mode": "ORPHAN_RAW_RECOVERY" if recovery is not None else "NEW_BOOTSTRAP",
                            "cohort_counts": cohort_counts, "checkpoint_json": cache_json.name,
                            "checkpoint_npz": cache_npz.name, "raw_npz_sha256": saved["raw_npz_sha256"],
                            "raw_recovery": recovery,
                        })
                except Exception as exc:
                    if cache_dir is not None:
                        _append_jsonl(cache_dir / "call_log.jsonl", {
                            "event": "FAILED", "call_id": call_id, "started_utc": started_utc,
                            "finished_utc": _utc(), "elapsed_seconds": time.perf_counter() - start_clock,
                            "cohort_counts": cohort_counts, "error": f"{type(exc).__name__}: {exc}",
                        })
                    raise
            call.setdefault("call_id", call_id)
            call.setdefault("cohort_counts", cohort_counts)
            # Raw evidence keys are resolved relative to the artifact file.
            for row in rows:
                row["per_seed_evidence_path"] = f"{evidence_path}::{row['per_seed_evidence_path'].split('::', 1)[1]}"
            flat_rows.extend(rows)
            calls.append(call)
            raw_npz.update(arrays)
    return flat_rows, calls, raw_npz


def run(out: Path, n_replicates: int) -> None:
    fixed = OUT_DEFAULT.resolve()
    if out.resolve() != fixed:
        raise ValueError(f"bootstrap output is restricted to {fixed}")
    if n_replicates != 5000:
        raise ValueError("the frozen B protocol requires exactly 5,000 replicates")
    for required in (out / "semantic_ablation_predictions.csv.gz", out / "cross_k_predictions.csv.gz", out / "summary.json"):
        if not required.exists():
            raise FileNotFoundError(f"required B point artifact is missing: {required}")
    run_started = time.perf_counter()
    source_fingerprint = {
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "atomic_replace_sha256": _sha256_file(SRC / "ccg/repairs/atomic.py"),
        "shared_bootstrap_sha256": _sha256_file(SRC / "ccg/repairs/bootstrap.py"),
        "statistics_sha256": _sha256_file(SRC / "ccg/repairs/statistics.py"),
        "protocol_sha256": _sha256_file(out / "protocol.json"),
        "semantic_predictions_sha256": _sha256_file(out / "semantic_ablation_predictions.csv.gz"),
        "cross_k_predictions_sha256": _sha256_file(out / "cross_k_predictions.csv.gz"),
    }
    _append_status(out, "formal_bootstrap", "RUNNING", "shared 5,000-replicate image-cluster bootstrap; this is the separately scheduled heavy statistics stage")
    try:
        semantic = _read_table(out / "semantic_ablation_predictions.csv.gz", has_training_regime=True)
        cross = _read_table(out / "cross_k_predictions.csv.gz", has_training_regime=True)
        flat_rows, calls, raw_npz = _compute_bootstrap(
            semantic, cross, n_replicates=n_replicates,
            checkpoint_dir=out / "information_bootstrap_call_checkpoints",
            source_fingerprint=source_fingerprint,
        )
    except Exception as exc:
        _append_status(out, "formal_bootstrap", "FAILED", f"formal call failed; completed per-call checkpoints are preserved: {type(exc).__name__}: {exc}")
        raise
    _write_npz_atomic(out / "information_bootstrap_raw.npz", raw_npz)
    flat_fields = (
        "scope", "cell", "K", "eval_split", "contrast", "model", "training_regime", "metric",
        "point", "ci_low", "ci_high", "n_replicates", "valid_replicates", "invalid_replicates",
        "seed_standard_deviation", "per_seed_point_json", "per_seed_ci_json",
        "per_seed_evidence_path", "raw_replicates_key", "interval_status",
    )
    _write_csv(out / "information_bootstrap.csv", flat_rows, flat_fields)
    bootstrap_json = {
        "status": "COMPLETE",
        "method": "shared_image_cluster_bootstrap via joint_prediction_bootstrap",
        "resample_unit": "image_cluster",
        "n_replicates": n_replicates,
        "bootstrap_seed": 0,
        "ci_level": 0.95,
        "endpoint_nan_policy": "One-class AUROC and zero-error-denominator RER draws are NaN and count as invalid; scalar legacy zero convention applies only to point estimates.",
        "raw_replicates_file": "information_bootstrap_raw.npz",
        "flat_csv": "information_bootstrap.csv",
        "call_checkpoint_dir": "information_bootstrap_call_checkpoints",
        "call_log": "information_bootstrap_call_checkpoints/call_log.jsonl",
        "source_sha256": source_fingerprint,
        "elapsed_seconds": time.perf_counter() - run_started,
        "calls": calls,
    }
    _write_json(out / "information_bootstrap.json", bootstrap_json)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    main_rows = [row for row in flat_rows if row["contrast"] == "Full_minus_S_plus_Q" and row["metric"] == "auroc_correct" and row["eval_split"] in SPLITS]
    summary["status"] = "COMPLETE"
    summary["formal_replicates"] = n_replicates
    summary["formal_bootstrap_complete"] = True
    summary["main_effect"] = {
        "contrast": "Full - (S+Q)", "endpoint": "AUROC_correct",
        "per_cell_point_and_ci": main_rows,
        "point_direction": "reported per test cell; no gate-derived conclusion substituted for the effect estimate",
        "replicates": n_replicates,
    }
    summary["bootstrap_files"] = {
        "formal_csv": "information_bootstrap.csv",
        "formal_json": "information_bootstrap.json",
        "raw_replicates": "information_bootstrap_raw.npz",
        "call_checkpoints": "information_bootstrap_call_checkpoints",
        "call_log": "information_bootstrap_call_checkpoints/call_log.jsonl",
    }
    summary["formal_source_sha256"] = source_fingerprint
    summary["formal_bootstrap_elapsed_seconds"] = time.perf_counter() - run_started
    summary["scientific_conclusion"] = "Interpret the fixed-seed mean contrasts and their shared image-cluster percentile intervals by endpoint and test cell. The three fixed models' test-sampling uncertainty is separated from seed-to-seed training variability; RER intervals with zero-error draws report invalid replicate counts."
    _write_json(summary_path, summary)
    _append_status(out, "formal_bootstrap", "COMPLETE", f"completed {len(calls)} shared-bootstrap calls, each with {n_replicates} image-cluster replicates")
    status_path = out / "STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["stages"]["overall"] = {
        "state": "COMPLETE",
        "updated_utc": _utc(),
        "detail": "all B training, point evaluation, diagnostics, and formal shared image-cluster bootstrap calls are complete",
    }
    _write_json(status_path, status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args(argv)
    run(args.out, args.replicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
