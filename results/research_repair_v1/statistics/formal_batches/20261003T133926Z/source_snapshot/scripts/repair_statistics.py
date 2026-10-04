"""Run Research Repair v1 image-cluster statistics on frozen prediction artifacts.

All generated files are restricted to ``results/research_repair_v1/statistics``.
Formal 5000-draw execution requires the resource-slot acknowledgement flag;
light input verification and ``--preflight`` never bootstrap.
"""
from __future__ import annotations

import os

# Set before NumPy / sklearn are imported. This is required even when the
# caller's shell forgot to constrain BLAS.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccg.repairs.bootstrap import assert_row_alignment
from ccg.repairs.statistics import _estimate_value, joint_prediction_bootstrap, metric_bundle
from ccg.repairs.atomic import replace_with_retry
from ccg.v2.semantic_features import MANIPULATION_METRICS, V2_PRIMARY_SEMANTIC_NAMES

OUT = ROOT / "results" / "research_repair_v1" / "statistics"
BASELINE_MANIFEST = ROOT / "results" / "research_repair_v1" / "input_manifest.json"
SUMMARY_SCHEMA = "ccg.research_repair_v1.statistics.summary.v1"
CI_METRICS = ("accuracy", "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80", "rer_at_90",
              "ece_adaptive", "brier_binary", "nll_binary")
ANCHOR_TOLERANCE = 1e-6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_temp_path(path: Path) -> Path:
    """Use a short same-directory unique temp name and preserve it on failure."""
    key = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:10]
    return path.with_name(f".{key}.{os.getpid()}.tmp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _short_temp_path(path)
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replace_with_retry(tmp, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write one repair-owned prediction archive atomically and return its hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _short_temp_path(path)
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    replace_with_retry(tmp, path)
    return sha256(path)


def _csv_write(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _short_temp_path(path)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    replace_with_retry(tmp, path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return ""
        return float(value)
    return value


def _load_manifest() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not BASELINE_MANIFEST.is_file():
        raise RuntimeError(f"Missing frozen baseline manifest: {BASELINE_MANIFEST}")
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    baseline = {record["path"]: record for record in manifest["inputs"]}
    supplemental: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "results" / "research_repair_v1").rglob("supplemental_input_manifest.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        for record in item["inputs"]:
            if record["path"] in supplemental and supplemental[record["path"]] != record:
                raise RuntimeError(f"Conflicting supplemental snapshots for {record['path']}")
            supplemental[record["path"]] = record
    return manifest, baseline, supplemental


_MANIFEST, _BASELINE, _SUPPLEMENTAL = _load_manifest()
_SOURCE_HASHES: dict[str, str] = {}


def frozen_source(path: Path) -> dict[str, str | int]:
    """Verify one source file against an immutable manifest before parsing it."""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Source outside workspace: {path}")
    rel = resolved.relative_to(ROOT.resolve()).as_posix()
    record = _BASELINE.get(rel) or _SUPPLEMENTAL.get(rel)
    if record is None:
        raise RuntimeError(f"Refusing to read unmanifested source: {rel}; freeze it in supplemental_input_manifest.json first")
    if rel not in _SOURCE_HASHES:
        if not resolved.is_file() or resolved.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"Frozen source missing or size changed: {rel}")
        observed = sha256(resolved)
        if observed != record["sha256"]:
            raise RuntimeError(f"Frozen source hash changed: {rel}")
        _SOURCE_HASHES[rel] = observed
    return {"path": rel, "size_bytes": int(record["size_bytes"]), "sha256": _SOURCE_HASHES[rel]}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    frozen_source(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _completed_recovery_job(job_id: str, seeds: tuple[int, ...] = (1, 2, 3)) -> list[dict[str, Any]]:
    """Load an immutable recovery archive set after verifying its manifest hashes."""
    from ccg.repairs.legacy_recovery import load_prediction_archive

    recovery_root = OUT / "recovery"
    manifest_paths = sorted((recovery_root / "runs").glob("*/manifest.json"),
                            key=lambda path: path.stat().st_mtime_ns, reverse=True)
    expected = set(int(seed) for seed in seeds)
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        matched = [row for row in manifest.get("jobs", [])
                   if row.get("job_id") == job_id and row.get("status") == "COMPLETE"]
        by_seed = {int(row["seed"]): row for row in matched if row.get("seed") is not None}
        if set(by_seed) != expected:
            continue
        recovered = []
        for seed in sorted(expected):
            record = by_seed[seed]
            archive_record, sidecar_record = record.get("archive", {}), record.get("sidecar", {})
            archive_path = ROOT / str(archive_record.get("path", ""))
            sidecar_path = ROOT / str(sidecar_record.get("path", ""))
            if not archive_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(f"{job_id}/seed{seed}: recovery archive or sidecar missing")
            if sha256(archive_path) != archive_record.get("sha256"):
                raise RuntimeError(f"{job_id}/seed{seed}: archive hash differs from recovery manifest")
            if sha256(sidecar_path) != sidecar_record.get("sha256"):
                raise RuntimeError(f"{job_id}/seed{seed}: sidecar hash differs from recovery manifest")
            arrays, sidecar = load_prediction_archive(archive_path, sidecar_path)
            if sidecar.get("job_id") != job_id or int(sidecar.get("seed", -1)) != seed:
                raise RuntimeError(f"{job_id}/seed{seed}: archive identity differs from recovery manifest")
            anchor_checks = sidecar.get("anchor_checks", [])
            if not anchor_checks or any(item.get("status") != "PASS" for item in anchor_checks):
                raise RuntimeError(f"{job_id}/seed{seed}: recovery anchors are absent or not all PASS")
            recovered.append({
                "seed": seed, "arrays": arrays, "sidecar": sidecar,
                "archive_path": archive_path, "sidecar_path": sidecar_path,
                "manifest_path": manifest_path, "manifest": manifest,
                "manifest_job": record,
            })
        return recovered
    raise FileNotFoundError(
        f"No single recovery run has a complete, anchor-passing {job_id} archive for seeds {sorted(expected)}"
    )


def _recovery_cell(record: dict[str, Any], cell_name: str) -> dict[str, Any]:
    from ccg.repairs.legacy_recovery import get_cell

    return get_cell(record["arrays"], record["sidecar"], cell_name)


def _canonicalize_recovery_cell(cell: dict[str, Any], *, label: str) -> tuple[dict[str, Any], np.ndarray]:
    """Sort archived rows by native sentence ID and return the point-order bridge."""
    sentence_ids = np.asarray(cell["sentence_id"], dtype=np.int64).reshape(-1)
    if sentence_ids.size == 0 or np.unique(sentence_ids).size != sentence_ids.size:
        raise ValueError(f"{label}: sentence_id must be non-empty and unique")
    sort_rows = np.argsort(sentence_ids, kind="stable")
    inverse_order = np.argsort(sort_rows, kind="stable").astype(np.int64, copy=False)
    output: dict[str, Any] = {}
    for field in ("sentence_id", "image_id", "ref_id", "eval_split", "correct"):
        if field not in cell:
            raise ValueError(f"{label}: recovery cell lacks {field}")
        output[field] = np.asarray(cell[field]).reshape(-1)[sort_rows]
    output["predictions"] = {
        str(name): np.asarray(values, dtype=np.float64).reshape(-1)[sort_rows]
        for name, values in cell.get("predictions", {}).items()
    }
    if not output["predictions"]:
        raise ValueError(f"{label}: recovery cell has no confidence arrays")
    return output, inverse_order


def _recovery_source_paths(records: list[dict[str, Any]], extra_paths: tuple[Path, ...] = ()) -> list[Path]:
    paths = list(extra_paths)
    for record in records:
        for source in record["sidecar"].get("source_files", []):
            # Recovery code snapshots are provenance, not statistical input
            # files. They are already immutably recorded in the recovery run
            # manifest; frozen_source() is reserved for baseline/supplemental
            # data and model artifacts.
            if source.get("status") != "SUPPLEMENTAL_SOURCE_SNAPSHOT":
                paths.append(ROOT / str(source["path"]))
        # The archive and sidecar are repair-owned immutable prediction
        # inputs. Their hashes are also checked against the recovery manifest
        # before they are added here.
        paths.extend((record["archive_path"], record["sidecar_path"]))
    return list(dict.fromkeys(path.resolve() for path in paths))


def read_csv(path: Path):
    import pandas as pd
    frozen_source(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    frozen_source(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def _canonical_indices(raw_ids: np.ndarray, canonical_ids: np.ndarray, *, label: str) -> np.ndarray:
    """Map unique native row IDs to a sorted canonical table without ref deduplication."""
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    target = np.asarray(canonical_ids, dtype=np.int64).reshape(-1)
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{label}: duplicate sentence IDs")
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    pos = np.searchsorted(sorted_ids, target)
    if np.any(pos >= sorted_ids.size) or not np.array_equal(sorted_ids[pos], target):
        raise ValueError(f"{label}: canonical sentence IDs absent from source rows")
    return order[pos]


def _split_metadata_from_text_index(sentence_ids: np.ndarray, ref_ids: np.ndarray,
                                    image_ids: np.ndarray) -> dict[str, int]:
    """Verify canonical native IDs against the frozen text-index split table."""
    path = ROOT / "cache" / "features" / "text_index.csv"
    frame = read_csv(path)
    if frame["sentence_id"].duplicated().any():
        raise ValueError("frozen text_index.csv sentence_id must be unique")
    indexed = frame.set_index("sentence_id", drop=False)
    ids = np.asarray(sentence_ids, dtype=np.int64)
    missing = [int(value) for value in ids if value not in indexed.index]
    if missing:
        raise ValueError(f"prediction sentence IDs absent from frozen text_index.csv: {missing[:5]}")
    rows = indexed.loc[ids]
    if not np.array_equal(rows["ref_id"].to_numpy(dtype=np.int64), np.asarray(ref_ids, dtype=np.int64)):
        raise ValueError("prediction ref_id differs from frozen text_index.csv sentence bridge")
    if not np.array_equal(rows["image_id"].to_numpy(dtype=np.int64), np.asarray(image_ids, dtype=np.int64)):
        raise ValueError("prediction image_id differs from frozen text_index.csv sentence bridge")
    labels, counts = np.unique(rows["split"].astype(str).to_numpy(), return_counts=True)
    return {str(name): int(count) for name, count in zip(labels, counts)}


def _cohort_metadata_from_counts(
    split_counts: dict[str, int], *, historical_pool_identity: str,
    heldout_test_only: bool | None = None,
) -> dict[str, Any]:
    """Describe the exact split mixture attached to a statistics job."""
    counts = {str(name): int(count) for name, count in split_counts.items() if int(count) > 0}
    names = set(counts)
    contains_train = any(name.lower().startswith("train") for name in names)
    contains_validation = any(name.lower().startswith(("val", "valid")) for name in names)
    inferred_test_only = bool(names) and names <= {"testA", "testB"}
    return {
        "split_counts": counts,
        "contains_train": contains_train,
        "contains_validation": contains_validation,
        "heldout_test_only": inferred_test_only if heldout_test_only is None else bool(heldout_test_only),
        "historical_pool_identity": historical_pool_identity,
    }


def _cohort_metadata_from_labels(
    labels: np.ndarray, *, historical_pool_identity: str,
    heldout_test_only: bool | None = None,
) -> dict[str, Any]:
    names, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
    return _cohort_metadata_from_counts(
        {str(name): int(count) for name, count in zip(names, counts)},
        historical_pool_identity=historical_pool_identity,
        heldout_test_only=heldout_test_only,
    )


def _check_anchor(row: dict[str, Any], expected: float, *, atol: float = ANCHOR_TOLERANCE) -> None:
    observed = float(row["point"])
    if np.isfinite(expected) != np.isfinite(observed):
        raise ValueError(f"point anchor finite-status mismatch: old={expected}, new={observed}")
    if np.isfinite(expected) and not math.isclose(observed, expected, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"point anchor mismatch: old={expected:.12g}, new={observed:.12g}, abs_error={abs(observed-expected):.3g}, atol={atol}")


def _legacy_anchor_observed(item: dict[str, Any], seed_key: str) -> float:
    """Resolve per-seed and historical seed-mean anchor labels explicitly."""
    if seed_key in {"b3_mean", "__aggregate__", "seed_mean"}:
        return float(item["point"])
    if seed_key not in item.get("per_seed", {}):
        raise KeyError(seed_key)
    return float(item["per_seed"][seed_key]["point"])


def _write_raw(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _short_temp_path(path)
    with tmp.open("wb") as handle:
        arrays: dict[str, np.ndarray] = {}
        for name, estimate in result["estimates"].items():
            arrays[f"aggregate__{name}"] = estimate["replicates"]
            for seed_key, values in estimate["per_seed_replicates"].items():
                arrays[f"seed_{seed_key}__{name}"] = values
        np.savez(handle, **arrays)
    replace_with_retry(tmp, path)


def _estimate_spec_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    """Keep the exact operation, scale, formula, and units beside every estimate."""
    operation = str(spec.get("operation", "condition"))
    metric = str(spec.get("metric", "unknown"))
    scale = float(spec.get("scale", 1.0))
    if metric == "mean":
        base_unit = "feature_native_units"
    elif metric == "nll_binary":
        base_unit = "nats"
    else:
        base_unit = "fraction"
    if operation == "condition":
        formula = f"{metric}({spec['condition']})"
        unit = base_unit
    elif operation == "difference":
        formula = f"{scale:g} * ({metric}({spec['a']}) - {metric}({spec['b']}))"
        unit = ("percentage_points" if scale == 100.0 and metric != "mean" else base_unit)
    elif operation == "relative_contrast":
        formula = (f"{scale:g} * ({metric}({spec['contrast_positive']}) - "
                   f"{metric}({spec['contrast_negative']})) / "
                   f"{metric}({spec['denominator_condition']})")
        unit = "relative_fraction" if scale == 1.0 else "relative_percent"
    elif operation == "difference_of_differences":
        formula = (f"{scale:g} * (({metric}({spec['a']}) - {metric}({spec['b']})) - "
                   f"({metric}({spec['c']}) - {metric}({spec['d']})))")
        unit = "percentage_points" if scale == 100.0 and metric != "mean" else base_unit
    elif operation == "difference_of_relative_contrasts":
        formula = (f"{scale:g} * (({metric}({spec['positive_a']}) - {metric}({spec['negative_a']})) / "
                   f"{metric}({spec['denominator_a']}) - ({metric}({spec['positive_b']}) - "
                   f"{metric}({spec['negative_b']})) / {metric}({spec['denominator_b']}))")
        unit = "relative_fraction" if scale == 1.0 else "relative_percent"
    elif operation == "relative_contrast_of_macros":
        positive = list(spec["contrast_positive_conditions"])
        negative = list(spec["contrast_negative_conditions"])
        if not positive or len(positive) != len(negative):
            raise ValueError("relative_contrast_of_macros requires paired non-empty condition lists")
        positive_macro = f"mean({metric} over [{', '.join(positive)}])"
        negative_macro = f"mean({metric} over [{', '.join(negative)}])"
        formula = (
            f"{scale:g} * (({positive_macro} - {negative_macro}) / {positive_macro})"
        )
        unit = "relative_fraction" if scale == 1.0 else "relative_percent"
    elif operation == "macro_mean":
        conditions = list(spec["conditions"])
        formula = f"mean({', '.join(f'{metric}({condition})' for condition in conditions)})"
        unit = base_unit
    elif operation == "macro_difference":
        left = list(spec["conditions_a"])
        right = list(spec["conditions_b"])
        formula = (f"{scale:g} * mean(pairwise({metric}({', '.join(left)}) - "
                   f"{metric}({', '.join(right)})))")
        unit = "percentage_points" if scale == 100.0 and metric != "mean" else base_unit
    else:
        raise ValueError(f"cannot describe unsupported estimate operation {operation!r}")
    return {"estimate_spec": dict(spec), "operation": operation, "scale": scale,
            "formula": formula, "unit": unit}


def _job_summary(job: dict[str, Any], result: dict[str, Any], sources: list[dict[str, Any]], raw_rel: str) -> dict[str, Any]:
    legacy = job.get("legacy_points", {})
    old_ci_map = job.get("old_ci", {})
    estimate_specs = {str(spec["name"]): spec for spec in list(job.get("estimates", [])) + list(job.get("auxiliary_estimates", []))}
    estimates = []
    for name, item in result["estimates"].items():
        if name not in estimate_specs:
            raise KeyError(f"estimate spec missing while summarizing {job['job_id']}/{name}")
        old_seed_points = legacy.get(name, {})
        if not isinstance(old_seed_points, dict):
            old_seed_points = {"legacy": old_seed_points}
        old_seed_points = {str(key): _float_or_nan(value) for key, value in old_seed_points.items()}
        for seed_key, old_value in old_seed_points.items():
            # A non-finite value in a legacy table means that the old method did
            # not report a point for that seed/metric. It is not evidence that a
            # new finite prediction is wrong; record it as unanchored instead.
            if not np.isfinite(old_value):
                continue
            try:
                _check_anchor({"point": _legacy_anchor_observed(item, seed_key)}, old_value)
            except ValueError as exc:
                raise ValueError(f"anchor failed job={job['job_id']} estimate={name} seed={seed_key}: {exc}") from exc
            except KeyError as exc:
                raise ValueError(f"legacy anchor seed {seed_key!r} is absent from {name}") from exc
        old_point = float(np.mean(list(old_seed_points.values()))) if old_seed_points and all(np.isfinite(v) for v in old_seed_points.values()) else None
        nonfinite_old_seeds = sorted(seed for seed, value in old_seed_points.items() if not np.isfinite(value))
        unavailable_for_estimate = [entry for entry in job.get("legacy_anchor_unavailable", [])
                                    if entry.get("estimate") == name]
        raw_old_ci = old_ci_map.get(name, {})
        if isinstance(raw_old_ci, dict):
            low_values = [_float_or_nan(value[0]) for value in raw_old_ci.values() if value and value[0] is not None]
            high_values = [_float_or_nan(value[1]) for value in raw_old_ci.values() if value and value[1] is not None]
        else:
            low_values = [_float_or_nan(raw_old_ci[0])] if raw_old_ci else []
            high_values = [_float_or_nan(raw_old_ci[1])] if raw_old_ci else []
        old_ci_low = float(np.mean(low_values)) if low_values and all(np.isfinite(v) for v in low_values) else None
        old_ci_high = float(np.mean(high_values)) if high_values and all(np.isfinite(v) for v in high_values) else None
        estimates.append({
            "name": name,
            "scope": job["scope"],
            "cohort": job["cohort"],
            "metric": job.get("metric_by_name", {}).get(name, name.rsplit("::", 1)[-1]),
            **_estimate_spec_metadata(estimate_specs[name]),
            "point": item["point"] if np.isfinite(item["point"]) else None,
            "ci_low": item["ci_low"],
            "ci_high": item["ci_high"],
            "bootstrap_std": item["bootstrap_std"] if np.isfinite(item["bootstrap_std"]) else None,
            "seed_standard_deviation": item["seed_standard_deviation"] if np.isfinite(item["seed_standard_deviation"]) else None,
            "seed_estimates": {str(seed): (float(v) if np.isfinite(v) else None) for seed, v in item["seed_estimates"].items()},
            "per_seed": {
                str(seed): {
                    "point": stats["point"] if np.isfinite(stats["point"]) else None,
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "bootstrap_std": stats["bootstrap_std"] if np.isfinite(stats["bootstrap_std"]) else None,
                    "valid_replicates": stats["valid_replicates"],
                    "invalid_replicates": stats["invalid_replicates"],
                    "uncertainty_status": stats["uncertainty_status"],
                    "valid_fraction": stats["valid_fraction"],
                    "gate_eligible": stats["gate_eligible"],
                    "ci_conditioning": stats["ci_conditioning"],
                }
                for seed, stats in item["per_seed"].items()
            },
            "valid_replicates": item["valid_replicates"],
            "invalid_replicates": item["invalid_replicates"],
            "interval_status": item["interval_status"],
            "uncertainty_status": item["uncertainty_status"],
            "valid_fraction": item["valid_fraction"],
            "gate_eligible": item["gate_eligible"],
            "ci_conditioning": item["ci_conditioning"],
            "minimum_valid_replicates_for_exploratory_ci": item["minimum_valid_replicates_for_exploratory_ci"],
            "old_point": old_point,
            "old_seed_points": {key: (float(value) if np.isfinite(value) else None) for key, value in old_seed_points.items()},
            "old_ci_low": old_ci_low,
            "old_ci_high": old_ci_high,
            "old_ci_method": (
                "historical_mean_of_seed_endpoints_not_a_seed_mean_95CI" if low_values and len(old_seed_points) > 1 else
                "historical_source_interval" if low_values else None
            ),
            "anchor_tolerance": ANCHOR_TOLERANCE if any(np.isfinite(v) for v in old_seed_points.values()) else None,
            "anchor_status": (
                "PASS" if old_seed_points and not nonfinite_old_seeds else
                "PARTIAL_PASS_OLD_NONFINITE_UNANCHORED" if any(np.isfinite(v) for v in old_seed_points.values()) else
                "LEGACY_POINT_UNAVAILABLE" if unavailable_for_estimate else
                "OLD_POINT_NONFINITE_UNANCHORED" if old_seed_points else
                "NO_LEGACY_POINT_ANCHOR"
            ),
            "nonfinite_old_seed_points": nonfinite_old_seeds,
            "nonfinite_anchor_reason": (
                "Legacy point was non-finite or absent and therefore cannot anchor a finite-status comparison."
                if nonfinite_old_seeds else ""
            ),
            "legacy_anchor_unavailable": unavailable_for_estimate,
            "n_rows": result["n_rows"],
            "n_images": result["n_images"],
            "n_replicates": result["n_replicates"],
            "resample_unit": result["resample_unit"],
            "ci_level": result["ci_level"],
            "raw_replicates": raw_rel,
            "source_artifacts": sources,
            "recovery_status": job.get("recovery_status", "READ_ONLY_PREDICTION_ARTIFACT"),
            "sentence_id_status": job.get("sentence_id_status", "VERIFIED_NATIVE_ID"),
            "evidence": job.get("evidence", ""),
            "model_identity_by_seed": job.get("model_identity_by_seed", {}),
            "prediction_source_precision": job.get("prediction_source_precision", "unspecified"),
            "recovery_output_path": job.get("recovery_output_path", ""),
            "recovery_output_sha256": job.get("recovery_output_sha256", ""),
        })
    return {
        "job_id": job["job_id"],
        "scope": job["scope"],
        "cohort": job["cohort"],
        "old_gate": job.get("old_gate", ""),
        "thresholds": job.get("thresholds", {}),
        "new_gate": job.get("new_gate", "THRESHOLD_REVIEW_REQUIRED"),
        "n_seeds": len(next(iter(result["estimates"].values()))["seed_estimates"]) if result["estimates"] else 0,
        "n_conditions": len(next(iter(job["seed_cells"].values()))) if job.get("seed_cells") else 0,
        "cohort_construction": job.get("cohort_construction", "historical_eval_split"),
        "legacy_pooled_anchor": job.get("legacy_pooled_anchor", False),
        "legacy_cohort_classification": job.get("legacy_cohort_classification", "HISTORICAL_SPLIT"),
        "legacy_cohort_note": job.get("legacy_cohort_note", ""),
        "legacy_pool_status": job.get("legacy_pool_status", "NOT_APPLICABLE"),
        "cohort_metadata": job.get("cohort_metadata", {}),
        "point_estimate_ordering": job.get("point_estimate_ordering", "canonical_row_order"),
        "status": "RECOMPUTED_5000" if result["n_replicates"] == 5000 else "PILOT_ONLY",
        "source_artifacts": sources,
        "model_identity_by_seed": job.get("model_identity_by_seed", {}),
        "prediction_source_precision": job.get("prediction_source_precision", "unspecified"),
        "recovery_output_path": job.get("recovery_output_path", ""),
        "recovery_output_sha256": job.get("recovery_output_sha256", ""),
        "nonfinite_legacy_anchors": job.get("nonfinite_legacy_anchors", []),
        "legacy_anchor_unavailable": job.get("legacy_anchor_unavailable", []),
        "result": {key: value for key, value in result.items() if key != "estimates"},
        "estimates": estimates,
    }


def _default_coverage_row(summary: dict[str, Any], estimate: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": summary["scope"],
        "endpoint_or_gate": f"{summary['cohort']}::{estimate['name']}",
        "source_artifacts": ";".join(f"{x['path']}#{x['sha256']}" for x in estimate["source_artifacts"]),
        "source_hashes": ";".join(f"{x['path']}={x['sha256']}" for x in estimate["source_artifacts"]),
        "old_point": estimate["old_point"],
        "old_ci_low": estimate["old_ci_low"],
        "old_ci_high": estimate["old_ci_high"],
        "new_point": estimate["point"],
        "new_ci_low": estimate["ci_low"],
        "new_ci_high": estimate["ci_high"],
        "operation": estimate.get("operation", ""),
        "scale": estimate.get("scale", 1.0),
        "formula": estimate.get("formula", ""),
        "unit": estimate.get("unit", ""),
        "old_gate": summary.get("old_gate", ""),
        "new_gate": "THRESHOLD_REVIEW_REQUIRED",
        "thresholds": json.dumps(summary.get("thresholds", {}), ensure_ascii=False) if summary.get("thresholds") else "",
        "n_rows": estimate["n_rows"],
        "n_images": estimate["n_images"],
        "n_seeds": summary.get("n_seeds", 0),
        "n_replicates": estimate["n_replicates"],
        "valid_replicates": estimate["valid_replicates"],
        "invalid_replicates": estimate["invalid_replicates"],
        "valid_fraction": estimate["valid_fraction"],
        "uncertainty_status": estimate["uncertainty_status"],
        "gate_eligible": estimate["gate_eligible"],
        "ci_conditioning": estimate["ci_conditioning"],
        "minimum_valid_replicates_for_exploratory_ci": estimate["minimum_valid_replicates_for_exploratory_ci"],
        "recovery_status": estimate["recovery_status"],
        "cohort_construction": summary.get("cohort_construction", "historical_eval_split"),
        "point_estimate_ordering": summary.get("point_estimate_ordering", "canonical_row_order"),
        "legacy_pooled_anchor": summary.get("legacy_pooled_anchor", False),
        "legacy_cohort_classification": summary.get("legacy_cohort_classification", "HISTORICAL_SPLIT"),
        "legacy_cohort_note": summary.get("legacy_cohort_note", ""),
        "legacy_pool_status": summary.get("legacy_pool_status", "NOT_APPLICABLE"),
        "anchor_status": estimate["anchor_status"],
        "anchor_tolerance": estimate["anchor_tolerance"],
        "nonfinite_old_seed_points": ";".join(estimate.get("nonfinite_old_seed_points", [])),
        "nonfinite_anchor_reason": estimate.get("nonfinite_anchor_reason", ""),
        "legacy_anchor_unavailable": json.dumps(estimate.get("legacy_anchor_unavailable", []), ensure_ascii=False),
        "sentence_id_status": estimate["sentence_id_status"],
        "model_identity_by_seed": json.dumps(estimate.get("model_identity_by_seed", {}), ensure_ascii=False),
        "prediction_source_precision": estimate.get("prediction_source_precision", "unspecified"),
        "recovery_output_path": estimate.get("recovery_output_path", ""),
        "recovery_output_sha256": estimate.get("recovery_output_sha256", ""),
        "evidence": estimate["evidence"],
        "status": summary["status"],
        "reason": "" if estimate["gate_eligible"] else "CI is conditional on valid draws or has insufficient valid replicates; do not use as a scientific gate.",
        "output_path": estimate["raw_replicates"],
    }


def _refresh_persisted_job(summary: dict[str, Any], job: dict[str, Any]) -> None:
    """Upgrade metadata/anchors for a previously completed job without rerunning CI."""
    _derive_legacy_points(job)
    stored = next((item for item in summary["jobs"] if item["job_id"] == job["job_id"]), None)
    if stored is None:
        raise KeyError(f"cannot refresh absent completed job {job['job_id']}")
    sources = [frozen_source(Path(path)) for path in job["source_paths"]]
    stored.update({
        "scope": job["scope"],
        "cohort": job["cohort"],
        "old_gate": job.get("old_gate", ""),
        "thresholds": job.get("thresholds", {}),
        "cohort_construction": job.get("cohort_construction", "historical_eval_split"),
        "legacy_pooled_anchor": job.get("legacy_pooled_anchor", False),
        "legacy_cohort_classification": job.get("legacy_cohort_classification", "HISTORICAL_SPLIT"),
        "legacy_cohort_note": job.get("legacy_cohort_note", ""),
        "legacy_pool_status": job.get("legacy_pool_status", "NOT_APPLICABLE"),
        "cohort_metadata": job.get("cohort_metadata", stored.get("cohort_metadata", {})),
        "n_seeds": len(job["seed_cells"]),
        "n_conditions": len(next(iter(job["seed_cells"].values()))),
        "source_artifacts": sources,
        "model_identity_by_seed": job.get("model_identity_by_seed", {}),
        "prediction_source_precision": job.get("prediction_source_precision", "unspecified"),
        "recovery_output_path": job.get("recovery_output_path", ""),
        "recovery_output_sha256": job.get("recovery_output_sha256", ""),
    })
    old_ci_map = job.get("old_ci", {})
    estimate_specs = {str(spec["name"]): spec for spec in list(job.get("estimates", [])) + list(job.get("auxiliary_estimates", []))}
    for estimate in stored["estimates"]:
        name = estimate["name"]
        if name not in estimate_specs:
            raise KeyError(f"estimate spec missing while refreshing {job['job_id']}/{name}")
        legacy = job.get("legacy_points", {}).get(name, {})
        if not isinstance(legacy, dict):
            legacy = {"legacy": legacy}
        legacy = {str(key): _float_or_nan(value) for key, value in legacy.items()}
        for seed_key, old_value in legacy.items():
            if not np.isfinite(old_value):
                continue
            try:
                _check_anchor({"point": _legacy_anchor_observed(estimate, seed_key)}, old_value)
            except ValueError as exc:
                raise ValueError(f"anchor failed job={job['job_id']} estimate={name} seed={seed_key}: {exc}") from exc
            except KeyError as exc:
                raise ValueError(f"legacy anchor seed {seed_key!r} is absent from {job['job_id']}/{name}") from exc
        old_point = (
            float(np.mean(list(legacy.values())))
            if legacy and all(np.isfinite(value) for value in legacy.values()) else None
        )
        raw_old_ci = old_ci_map.get(name, {})
        if isinstance(raw_old_ci, dict):
            low_values = [_float_or_nan(value[0]) for value in raw_old_ci.values() if value and value[0] is not None]
            high_values = [_float_or_nan(value[1]) for value in raw_old_ci.values() if value and value[1] is not None]
        else:
            low_values = [_float_or_nan(raw_old_ci[0])] if raw_old_ci else []
            high_values = [_float_or_nan(raw_old_ci[1])] if raw_old_ci else []
        nonfinite_old_seeds = sorted(seed for seed, value in legacy.items() if not np.isfinite(value))
        unavailable_for_estimate = [entry for entry in job.get("legacy_anchor_unavailable", [])
                                    if entry.get("estimate") == name]
        estimate.update({
            **_estimate_spec_metadata(estimate_specs[name]),
            "old_point": old_point,
            "old_seed_points": {key: (float(value) if np.isfinite(value) else None) for key, value in legacy.items()},
            "old_ci_low": float(np.mean(low_values)) if low_values and all(np.isfinite(v) for v in low_values) else None,
            "old_ci_high": float(np.mean(high_values)) if high_values and all(np.isfinite(v) for v in high_values) else None,
            "old_ci_method": (
                "historical_mean_of_seed_endpoints_not_a_seed_mean_95CI" if low_values and len(legacy) > 1 else
                "historical_source_interval" if low_values else None
            ),
            "anchor_tolerance": ANCHOR_TOLERANCE if any(np.isfinite(v) for v in legacy.values()) else None,
            "anchor_status": (
                "PASS" if legacy and not nonfinite_old_seeds else
                "PARTIAL_PASS_OLD_NONFINITE_UNANCHORED" if any(np.isfinite(v) for v in legacy.values()) else
                "LEGACY_POINT_UNAVAILABLE" if unavailable_for_estimate else
                "OLD_POINT_NONFINITE_UNANCHORED" if legacy else
                "NO_LEGACY_POINT_ANCHOR"
            ),
            "nonfinite_old_seed_points": nonfinite_old_seeds,
            "nonfinite_anchor_reason": (
                "Legacy point was non-finite or absent and therefore cannot anchor a finite-status comparison."
                if nonfinite_old_seeds else ""
            ),
            "source_artifacts": sources,
            "recovery_status": job.get("recovery_status", "READ_ONLY_PREDICTION_ARTIFACT"),
            "sentence_id_status": job.get("sentence_id_status", "VERIFIED_NATIVE_ID"),
            "evidence": job.get("evidence", ""),
            "legacy_anchor_unavailable": unavailable_for_estimate,
            "model_identity_by_seed": job.get("model_identity_by_seed", {}),
            "prediction_source_precision": job.get("prediction_source_precision", "unspecified"),
            "recovery_output_path": job.get("recovery_output_path", ""),
            "recovery_output_sha256": job.get("recovery_output_sha256", ""),
        })
        n_replicates = int(estimate.get("n_replicates", stored.get("result", {}).get("n_replicates", 0)))
        valid_replicates = int(estimate.get("valid_replicates", 0))
        minimum_valid = int(np.ceil(0.95 * n_replicates)) if n_replicates else 0
        if n_replicates and valid_replicates == n_replicates:
            uncertainty_status, gate_eligible = "SUFFICIENT_ALL_PLANNED_REPLICATES_VALID", True
        elif valid_replicates >= minimum_valid and valid_replicates >= 2:
            uncertainty_status, gate_eligible = "CONDITIONAL_ON_VALID_DRAWS_NOT_GATE_ELIGIBLE", False
        else:
            uncertainty_status, gate_eligible = "UNCERTAINTY_INSUFFICIENT", False
        estimate.update({
            "uncertainty_status": uncertainty_status,
            "valid_fraction": float(valid_replicates / n_replicates) if n_replicates else 0.0,
            "gate_eligible": gate_eligible,
            "ci_conditioning": "all_planned_draws_valid" if valid_replicates == n_replicates else "conditional_on_finite_draws",
            "minimum_valid_replicates_for_exploratory_ci": minimum_valid,
        })
        for seed_stats in estimate.get("per_seed", {}).values():
            seed_replicates = int(seed_stats.get("valid_replicates", 0)) + int(seed_stats.get("invalid_replicates", 0))
            seed_valid = int(seed_stats.get("valid_replicates", 0))
            seed_minimum = int(np.ceil(0.95 * seed_replicates)) if seed_replicates else 0
            if seed_replicates and seed_valid == seed_replicates:
                seed_status, seed_gate_eligible = "SUFFICIENT_ALL_PLANNED_REPLICATES_VALID", True
            elif seed_valid >= seed_minimum and seed_valid >= 2:
                seed_status, seed_gate_eligible = "CONDITIONAL_ON_VALID_DRAWS_NOT_GATE_ELIGIBLE", False
            else:
                seed_status, seed_gate_eligible = "UNCERTAINTY_INSUFFICIENT", False
            seed_stats.update({
                "uncertainty_status": seed_status,
                "valid_fraction": float(seed_valid / seed_replicates) if seed_replicates else 0.0,
                "gate_eligible": seed_gate_eligible,
                "ci_conditioning": "all_planned_draws_valid" if seed_valid == seed_replicates else "conditional_on_finite_draws",
            })
    summary["estimates"] = [estimate for stored_job in summary["jobs"] for estimate in stored_job["estimates"]]


def _read_b0_legacy() -> dict[tuple[str, int, str, str], float]:
    root = ROOT / "results" / "phase0a_corrected"
    metrics = read_csv(root / "normalized_selective_metrics.csv")
    cal = read_csv(root / "calibration_metrics_corrected.csv")
    auc = read_csv(root / "correctness_auroc.csv")
    merged = metrics.merge(cal, on=["eval_split", "K", "variant", "n"], suffixes=("", "_cal"))
    merged = merged.merge(auc[["eval_split", "K", "variant", "auroc_correct", "auprc_correct"]],
                          on=["eval_split", "K", "variant"], suffixes=("", "_auc"))
    output: dict[tuple[str, int, str, str], float] = {}
    for row in merged.to_dict("records"):
        variant = row["variant"]
        if variant == "oracle_T_K":
            variant = "validation_perK_diagnostic"
        for metric in CI_METRICS:
            key = metric + "_auc" if metric in ("auroc_correct",) and not np.isfinite(_float_or_nan(row.get(metric))) else metric
            value = row.get(metric, row.get(metric + "_auc"))
            if metric == "auroc_correct":
                value = row.get("auroc_correct_auc")
            output[(str(row["eval_split"]), int(row["K"]), variant, metric)] = _float_or_nan(value)
    return output


def _read_b0_legacy_ci(split: str) -> dict[str, dict[str, list[float]]]:
    """Read only historical intervals whose estimand matches the repair effect."""
    root = ROOT / "results" / "phase0a_corrected"
    legacy_split = "__pooled__" if split == "__all_common__" else split
    if split == "__pooled_test__":
        return {}
    aliases = {"native": "native", "global_T": "global_T_corrected", "validation_perK_diagnostic": "oracle_T_K"}
    output: dict[str, dict[str, list[float]]] = {}
    auroc = read_csv(root / "auroc_bootstrap.csv")
    eaurc = read_csv(root / "eaurc_bootstrap.csv")
    rer = read_csv(root / "rer_bootstrap.csv")
    for variant, old_variant in aliases.items():
        row = auroc[(auroc.eval_split == legacy_split) & (auroc.variant == old_variant)
                    & (auroc.K_a == 5) & (auroc.K_b == 50)]
        if len(row) == 1:
            item = row.iloc[0]
            output.setdefault(f"crossK_K5_minus_K50__{variant}::auroc_correct", {})["B0"] = [
                float(item.ci_low), float(item.ci_high),
            ]
        row = eaurc[(eaurc.eval_split == legacy_split) & (eaurc.variant == old_variant)
                    & (eaurc.K_a == 5) & (eaurc.K_b == 50)]
        if len(row) == 1:
            item = row.iloc[0]
            output.setdefault(f"crossK_K50_minus_K5__{variant}::e_aurc", {})["B0"] = [
                float(item.ci_low_abs), float(item.ci_high_abs),
            ]
            # The legacy relative effect is (E-AURC_K50 - E-AURC_K5) / E-AURC_K5,
            # the same signed denominator used by this repair.
            output.setdefault(f"crossK_relative_eaurc_worsening__{variant}", {})["B0"] = [
                float(item.ci_low_relative), float(item.ci_high_relative),
            ]
        row = rer[(rer.eval_split == legacy_split) & (rer.variant == old_variant)
                  & (rer.coverage == 0.5) & (rer.K_a == 5) & (rer.K_b == 50)]
        if len(row) == 1:
            item = row.iloc[0]
            output.setdefault(f"crossK_K5_minus_K50__{variant}::rer_at_50", {})["B0"] = [
                float(item.ci_low), float(item.ci_high),
            ]
    return output


def _derive_legacy_points(job: dict[str, Any]) -> None:
    """Derive historical paired points from anchored condition-level points."""
    points = job.setdefault("legacy_points", {})
    for spec in job.get("estimates", []):
        name = str(spec["name"])
        if name in points:
            continue
        operation = str(spec.get("operation", "condition"))
        metric = str(spec["metric"])
        if operation == "condition":
            continue
        if operation == "difference":
            operands = (f"{spec['a']}::{metric}", f"{spec['b']}::{metric}")
        elif operation == "relative_contrast":
            operands = (
                f"{spec['contrast_positive']}::{metric}",
                f"{spec['contrast_negative']}::{metric}",
                f"{spec['denominator_condition']}::{metric}",
            )
        elif operation == "difference_of_differences":
            operands = tuple(f"{spec[key]}::{metric}" for key in ("a", "b", "c", "d"))
        elif operation == "difference_of_relative_contrasts":
            operands = tuple(
                f"{spec[key]}::{metric}"
                for key in ("positive_a", "negative_a", "denominator_a", "positive_b", "negative_b", "denominator_b")
            )
        elif operation == "macro_mean":
            operands = tuple(f"{condition}::{metric}" for condition in spec["conditions"])
        elif operation == "macro_difference":
            operands = tuple(f"{condition}::{metric}" for condition in (*spec["conditions_a"], *spec["conditions_b"]))
        else:
            continue
        operand_points = [points.get(operand, {}) for operand in operands]
        if not operand_points or not all(isinstance(values, dict) for values in operand_points):
            continue
        common_seeds = set(operand_points[0])
        for values in operand_points[1:]:
            common_seeds.intersection_update(values)
        derived: dict[str, float] = {}
        for seed in sorted(common_seeds, key=str):
            values = [float(item[seed]) for item in operand_points]
            if operation == "difference":
                value = float(spec.get("scale", 1.0)) * (values[0] - values[1])
            elif operation == "relative_contrast":
                denominator = values[2]
                value = (float(spec.get("scale", 1.0)) * (values[0] - values[1]) / denominator
                         if denominator != 0.0 else float("nan"))
            elif operation == "difference_of_differences":
                value = float(spec.get("scale", 1.0)) * ((values[0] - values[1]) - (values[2] - values[3]))
            elif operation == "difference_of_relative_contrasts":
                numerator_a = values[0] - values[1]
                denominator_a = values[2]
                numerator_b = values[3] - values[4]
                denominator_b = values[5]
                if denominator_a == 0.0 or denominator_b == 0.0:
                    value = float("nan")
                else:
                    value = float(spec.get("scale", 1.0)) * (
                        numerator_a / denominator_a - numerator_b / denominator_b
                    )
            elif operation == "macro_difference":
                n_pairs = len(spec["conditions_a"])
                left, right = values[:n_pairs], values[n_pairs:]
                value = (float(spec.get("scale", 1.0)) * float(np.mean(np.asarray(left) - np.asarray(right)))
                         if len(left) == len(right) and left and np.all(np.isfinite(values)) else float("nan"))
            else:
                value = float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan")
            derived[str(seed)] = float(value)
        if derived:
            points[name] = derived


def _read_phase0b_legacy_ci(seed_number: int, split: str) -> dict[str, list[float]]:
    """Read legacy Phase0B CIs only for effects with the same direction/denominator."""
    path = ROOT / "results" / "phase0b_independent" / f"seed_{seed_number}" / "bootstrap.csv"
    if split == "__pooled_test__":
        return {}
    frame = read_csv(path)
    legacy_split = "__pooled__" if split == "__all_common__" else split
    aliases = {"native": "native", "global_T": "global_T_corrected", "validation_perK_diagnostic": "oracle_T_K"}
    output: dict[str, list[float]] = {}
    for variant, old_variant in aliases.items():
        for metric, effect, reverse in (
            ("auroc_correct", "absolute", False),
            ("rer_at_50", "absolute", False),
            ("e_aurc", "absolute", True),
        ):
            row = frame[(frame.eval_split == legacy_split) & (frame.variant == old_variant)
                        & (frame.K_a == 5) & (frame.K_b == 50)
                        & (frame.metric == metric) & (frame.diff_kind == effect)]
            if len(row) == 1:
                item = row.iloc[0]
                low, high = float(item.ci_low), float(item.ci_high)
                if reverse:
                    low, high = -high, -low
                name = (
                    f"crossK_K50_minus_K5__{variant}::e_aurc" if reverse else
                    f"crossK_K5_minus_K50__{variant}::{metric}"
                )
                output[name] = [low, high]
    return output


def _split_masks(split_values: np.ndarray, *, pooled_alias: str) -> list[tuple[str, np.ndarray]]:
    available = [str(value) for value in np.unique(split_values)]
    result = [(split, split_values == split) for split in available]
    if "testA" in available and "testB" in available:
        result.append(("__pooled_test__", np.isin(split_values, ("testA", "testB"))))
        result.append(("__all_common__", np.ones(split_values.shape, dtype=np.bool_)))
    if pooled_alias != "__pooled_test__":
        return [(split, mask) for split, mask in result if split != pooled_alias]
    return result


def build_phase0a_jobs() -> list[dict[str, Any]]:
    root = ROOT / "results" / "phase0a_corrected"
    source = root / "per_sentence_predictions.npz"
    data = load_npz(source)
    legacy = _read_b0_legacy()
    aliases = {
        "native": "native",
        "global_T": "global_T_corrected",
        "validation_perK_diagnostic": "oracle_T_K",
    }
    jobs = []
    for split, mask in _split_masks(data["eval_split"], pooled_alias="__pooled__"):
        row_idx = np.flatnonzero(mask)
        legacy_split = "__pooled__" if split == "__all_common__" else (None if split == "__pooled_test__" else split)
        point_order = np.argsort(data["sentence_id"][row_idx], kind="stable")
        image_ids = data["image_id"][row_idx]
        cells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {"B0": cells}
        specs: list[dict[str, Any]] = []
        old_points: dict[str, dict[str, float]] = {}
        for k in (5, 10, 20, 50):
            correct = data[f"correct_K{k}"][row_idx]
            variants = {
                "native": data[f"confidence_native_K{k}"][row_idx],
                "global_T": data[f"confidence_global_T_K{k}"][row_idx],
                "validation_perK_diagnostic": data[f"confidence_oracle_T_K{k}"][row_idx],
            }
            for variant, confidence in variants.items():
                condition = f"K{k}__{variant}"
                cells[condition] = (confidence, correct)
                for metric in CI_METRICS:
                    name = f"{condition}::{metric}"
                    specs.append({"name": name, "metric": metric, "operation": "condition", "condition": condition})
                    old_value = legacy.get((legacy_split, k, aliases[variant], metric)) if legacy_split is not None else None
                    if old_value is not None:
                        old_points[name] = {"B0": old_value}
            for variant in ("global_T", "validation_perK_diagnostic"):
                condition = f"K{k}__{variant}"
                native = f"K{k}__native"
                for metric in ("auroc_correct", "e_aurc", "rer_at_50", "ece_adaptive"):
                    name = f"{condition}_minus_native::{metric}"
                    specs.append({"name": name, "metric": metric, "operation": "difference", "a": condition, "b": native})
        for variant in ("native", "global_T", "validation_perK_diagnostic"):
            for metric, a, b in (
                ("auroc_correct", "K5", "K50"),
                ("e_aurc", "K50", "K5"),
                ("rer_at_50", "K5", "K50"),
            ):
                left, right = f"{a}__{variant}", f"{b}__{variant}"
                name = f"crossK_{a}_minus_{b}__{variant}::{metric}"
                specs.append({"name": name, "metric": metric, "operation": "difference", "a": left, "b": right})
            name = f"crossK_relative_eaurc_worsening__{variant}"
            specs.append({
                "name": name,
                "metric": "e_aurc",
                "operation": "relative_contrast",
                "contrast_positive": f"K50__{variant}",
                "contrast_negative": f"K5__{variant}",
                "denominator_condition": f"K5__{variant}",
            })
        pooled = split == "__pooled_test__"
        all_common = split == "__all_common__"
        cohort_metadata = _cohort_metadata_from_labels(
            np.asarray(data["eval_split"])[row_idx],
            historical_pool_identity=(
                "new testA+testB-only pool; historical __pooled__ is __all_common__ including validation"
                if pooled else
                "historical B0 all-common pool across val_select/val_calib/testA/testB (no train rows)"
                if all_common else f"historical B0 split {split}"
            ),
        )
        legacy_ci = _read_b0_legacy_ci(split)
        jobs.append({
            "job_id": f"phase0a_{split}", "scope": "Phase0A_B0_temperature", "cohort": str(split),
            "image_ids": image_ids, "seed_cells": seed_cells, "estimates": specs, "metrics": CI_METRICS,
            "legacy_points": old_points,
            "point_order": point_order,
            "point_estimate_ordering": "sentence_id_ascending_to_match_historical_stable_tie_order",
            "source_paths": [
                source, root / "normalized_selective_metrics.csv", root / "calibration_metrics_corrected.csv",
                root / "correctness_auroc.csv", root / "temperature_fit.json", root / "auroc_bootstrap.csv",
                root / "eaurc_bootstrap.csv", root / "rer_bootstrap.csv",
            ],
            "old_ci": legacy_ci,
            "cohort_construction": (
                "testA_plus_testB_new_test_cohort" if pooled else
                "all_available_validation_and_test_rows" if all_common else
                "historical_eval_split"
            ),
            "legacy_pooled_anchor": all_common,
            "legacy_cohort_classification": (
                "NEW_TESTA_TESTB_POOL_NO_HISTORICAL_POOL_ANCHOR" if pooled else
                "ALL_COMMON_INCLUDES_VALIDATION_NO_TRAIN" if all_common else
                "HISTORICAL_SPLIT"
            ),
            "legacy_pool_status": "NEW_SUPPLEMENTAL_NO_MATCHED_LEGACY_TEST_POOL" if pooled else ("EXISTING_LEGACY_ALL_COMMON" if all_common else "NOT_APPLICABLE"),
            "legacy_cohort_note": (
                "Legacy __pooled__ is the all-common 20,799-row validation+test cohort; this new testA+testB pool has 10,286 rows/1,490 images and must not use the legacy point as an anchor or test-only evidence."
                if pooled else
                "Historical __pooled__ is reconstructed over canonical val_select/val_calib/testA/testB rows (no train rows) and retained as an appendix-only all-common analysis; it is not an independent test cohort."
                if all_common else ""
            ),
            "cohort_metadata": cohort_metadata,
            "sentence_id_status": "VERIFIED_NATIVE_ID", "evidence": "per_sentence_predictions.npz carries sentence_id/ref_id/image_id/eval_split",
            "recovery_status": "READ_ONLY_PREDICTION_ARTIFACT",
        })
    return jobs


def build_phase0b_jobs() -> list[dict[str, Any]]:
    sources = [ROOT / "results" / "phase0b_independent" / f"seed_{seed}" / "per_sentence_predictions.npz" for seed in (1, 2, 3)]
    arrays = {f"b3_seed{seed}": load_npz(path) for seed, path in zip((1, 2, 3), sources, strict=True)}
    ref = arrays["b3_seed1"]
    canonical = {"sentence_id": ref["sentence_ids"], "ref_id": ref["ref_ids"], "image_id": ref["image_ids"], "eval_split": ref["eval_split"]}
    for seed, candidate in arrays.items():
        candidate_anchors = {"sentence_id": candidate["sentence_ids"], "ref_id": candidate["ref_ids"], "image_id": candidate["image_ids"], "eval_split": candidate["eval_split"]}
        assert_row_alignment(canonical, candidate_anchors, required_fields=("ref_id", "image_id", "eval_split"), candidate_name=seed)

    jobs = []
    for split, mask in _split_masks(ref["eval_split"], pooled_alias="__pooled__"):
        row_idx = np.flatnonzero(mask)
        legacy_split = "__pooled__" if split == "__all_common__" else (None if split == "__pooled_test__" else split)
        point_order = np.argsort(ref["sentence_ids"][row_idx], kind="stable")
        image_ids = ref["image_ids"][row_idx]
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        legacy_points: dict[str, dict[str, float]] = {}
        for seed, data in arrays.items():
            seed_cells[seed] = {}
            seed_dir = f"seed_{seed[-1]}"
            point_path = ROOT / "results" / "phase0b_independent" / seed_dir / "normalized_selective_metrics.csv"
            cal_path = ROOT / "results" / "phase0b_independent" / seed_dir / "calibration_metrics.csv"
            old_metric = read_csv(point_path)
            old_cal = read_csv(cal_path)
            old = old_metric.merge(old_cal, on=["eval_split", "K", "variant", "n"], suffixes=("", "_cal"))
            for k in (5, 10, 20, 50):
                correct = data[f"correct_K{k}"][row_idx]
                variants = {
                    "native": data[f"confidence_native_K{k}"][row_idx],
                    "global_T": data[f"confidence_global_T_corrected_K{k}"][row_idx],
                    "validation_perK_diagnostic": data[f"confidence_oracle_T_K_K{k}"][row_idx],
                }
                for variant, confidence in variants.items():
                    condition = f"K{k}__{variant}"
                    seed_cells[seed][condition] = (confidence, correct)
                    variant_alias = {"native": "native", "global_T": "global_T_corrected", "validation_perK_diagnostic": "oracle_T_K"}[variant]
                    old_row = old[(old.eval_split == legacy_split) & (old.K == k) & (old.variant == variant_alias)] if legacy_split is not None else old.iloc[0:0]
                    if len(old_row) == 1:
                        row = old_row.iloc[0]
                        for metric in CI_METRICS:
                            name = f"{condition}::{metric}"
                            value = row.get(metric)
                            if metric in ("accuracy", "e_aurc", "rer_at_50", "rer_at_80", "rer_at_90", "ece_adaptive", "brier_binary", "nll_binary", "auroc_correct"):
                                legacy_points.setdefault(name, {})[seed] = _float_or_nan(value)
        conditions = tuple(next(iter(seed_cells.values())))
        specs = [
            {"name": f"{condition}::{metric}", "metric": metric, "operation": "condition", "condition": condition}
            for condition in conditions for metric in CI_METRICS
        ]
        for variant in ("native", "global_T", "validation_perK_diagnostic"):
            for metric, a, b in (("auroc_correct", "K5", "K50"), ("e_aurc", "K50", "K5"), ("rer_at_50", "K5", "K50")):
                specs.append({"name": f"crossK_{a}_minus_{b}__{variant}::{metric}", "metric": metric, "operation": "difference", "a": f"{a}__{variant}", "b": f"{b}__{variant}"})
            specs.append({
                "name": f"crossK_relative_eaurc_worsening__{variant}", "metric": "e_aurc",
                "operation": "relative_contrast", "contrast_positive": f"K50__{variant}",
                "contrast_negative": f"K5__{variant}", "denominator_condition": f"K5__{variant}",
            })
        pooled = split == "__pooled_test__"
        all_common = split == "__all_common__"
        cohort_metadata = _cohort_metadata_from_labels(
            np.asarray(ref["eval_split"])[row_idx],
            historical_pool_identity=(
                "new testA+testB-only pool; historical __pooled__ is __all_common__ including validation"
                if pooled else
                "historical B3 all-common pool across val_select/val_calib/testA/testB (no train rows)"
                if all_common else f"historical B3 split {split}"
            ),
        )
        legacy_source_paths = [
            ROOT / "results" / "phase0b_independent" / f"seed_{i}" / table
            for i in (1, 2, 3) for table in ("normalized_selective_metrics.csv", "calibration_metrics.csv", "bootstrap.csv")
        ]
        old_ci: dict[str, dict[str, list[float]]] = {}
        if not pooled:
            for seed_number in (1, 2, 3):
                for name, interval in _read_phase0b_legacy_ci(seed_number, split).items():
                    old_ci.setdefault(name, {})[f"b3_seed{seed_number}"] = interval
        jobs.append({
            "job_id": f"phase0b_{split}", "scope": "Phase0B_temperature", "cohort": str(split),
            "image_ids": image_ids, "seed_cells": seed_cells, "estimates": specs, "metrics": CI_METRICS,
            "legacy_points": legacy_points, "old_ci": old_ci, "source_paths": sources + legacy_source_paths,
            "point_order": point_order,
            "point_estimate_ordering": "sentence_id_ascending_canonical_order",
            "cohort_construction": (
                "testA_plus_testB_new_test_cohort" if pooled else
                "all_available_validation_and_test_rows" if all_common else
                "historical_eval_split"
            ),
            "legacy_pooled_anchor": all_common,
            "legacy_cohort_classification": (
                "NEW_TESTA_TESTB_POOL_NO_HISTORICAL_POOL_ANCHOR" if pooled else
                "ALL_COMMON_INCLUDES_VALIDATION_NO_TRAIN" if all_common else
                "HISTORICAL_SPLIT"
            ),
            "legacy_pool_status": "NEW_SUPPLEMENTAL_NO_MATCHED_LEGACY_TEST_POOL" if pooled else ("EXISTING_LEGACY_ALL_COMMON" if all_common else "NOT_APPLICABLE"),
            "legacy_cohort_note": (
                "Legacy __pooled__ is the all-common 20,799-row validation+test cohort; this new testA+testB pool has 10,286 rows/1,490 images and must not use the legacy point as an anchor or test-only evidence."
                if pooled else
                "Historical __pooled__ is reconstructed over canonical val_select/val_calib/testA/testB rows (no train rows) and retained as an appendix-only all-common analysis; it is not an independent test cohort."
                if all_common else ""
            ),
            "cohort_metadata": cohort_metadata,
            "sentence_id_status": "VERIFIED_NATIVE_ID", "evidence": "three B3 per_sentence_predictions.npz inputs row-aligned on sentence/ref/image/split anchors",
            "recovery_status": "READ_ONLY_PREDICTION_ARTIFACT",
        })
    return jobs


def _phase0b_reference(seed_number: int) -> tuple[dict[str, np.ndarray], Path]:
    path = ROOT / "results" / "phase0b_independent" / f"seed_{seed_number}" / "per_sentence_predictions.npz"
    return load_npz(path), path


def _attach_canonical_sentence_ids(frame, *, seed_number: int, score_column: str) -> tuple[np.ndarray, Path]:
    """Validate each legacy CSV cell against frozen canonical sentence rows."""
    data, canonical_path = _phase0b_reference(seed_number)
    required = ("ref_id", "image_id", "eval_split", "K", "grounding_correct")
    row_ids = np.empty(len(frame), dtype=np.int64)
    seen = np.zeros(len(frame), dtype=np.bool_)
    frame_k = frame["K"].astype(int).to_numpy()
    frame_split = frame["eval_split"].astype(str).to_numpy()
    for k in (5, 10, 20, 50):
        for split in np.unique(frame_split[frame_k == k]):
            candidate_mask = (frame_k == k) & (frame_split == split)
            source_rows = np.flatnonzero(data["eval_split"] == split)
            canonical = {
                "sentence_id": data["sentence_ids"][source_rows],
                "ref_id": data["ref_ids"][source_rows],
                "image_id": data["image_ids"][source_rows],
                "eval_split": data["eval_split"][source_rows],
                "K": np.full(source_rows.size, k, dtype=np.int64),
                "grounding_correct": data[f"correct_K{k}"][source_rows],
            }
            candidate = {
                "ref_id": frame.loc[candidate_mask, "ref_id"].to_numpy(),
                "image_id": frame.loc[candidate_mask, "image_id"].to_numpy(),
                "eval_split": frame.loc[candidate_mask, "eval_split"].astype(str).to_numpy(),
                "K": frame.loc[candidate_mask, "K"].astype(int).to_numpy(),
                "grounding_correct": frame.loc[candidate_mask, "grounding_correct"].to_numpy(),
            }
            recovered = assert_row_alignment(
                canonical,
                candidate,
                required_fields=required,
                reference_name=f"phase0b seed_{seed_number} K{k}/{split}",
                candidate_name=f"legacy prediction CSV ({score_column})",
            )
            row_ids[np.flatnonzero(candidate_mask)] = recovered
            seen[np.flatnonzero(candidate_mask)] = True
    if not bool(np.all(seen)):
        raise ValueError(f"CSV rows not covered by canonical K/split reconstruction: {int((~seen).sum())}")
    return row_ids, canonical_path


def _legacy_metric_value(row: Any, metric: str) -> float:
    column = {
        "accuracy": "b3_accuracy",
        "auroc_correct": "auroc_correct",
        "e_aurc": "e_aurc",
        "rer_at_50": "rer_at_50",
        "rer_at_80": "rer_at_80",
        "rer_at_90": "rer_at_90",
        "ece_adaptive": "ece_adaptive",
        "brier_binary": "brier_binary",
        "nll_binary": "nll_binary",
    }[metric]
    return _float_or_nan(row.get(column))


def _selected_rows(frame, cohort: str) -> np.ndarray:
    split = frame["eval_split"].astype(str).to_numpy()
    if cohort == "__pooled_test__":
        return np.isin(split, ("testA", "testB"))
    return split == cohort


def build_phase05_jobs() -> list[dict[str, Any]]:
    root = ROOT / "results" / "phase05_score_sufficiency"
    prediction_root = root / "predictions"
    old_metrics_path = root / "reliability_metrics.csv"
    old_cross_path = root / "cross_k_degradation.csv"
    gate_path = root / "sufficiency_gate.json"
    old_metrics = read_csv(old_metrics_path)
    old_cross = read_csv(old_cross_path)
    old_gate_json = read_json(gate_path) if gate_path.is_file() else {}
    old_gate = old_gate_json.get("verdict_seed_mean", {}).get("verdict", "NOT_RECORDED")
    thresholds = old_gate_json.get("verdict_seed_mean", {}).get("thresholds", {})
    seeds = (1, 2, 3)
    model_files = sorted(p.stem.removesuffix(".csv") for p in (prediction_root / "b3_seed1").glob("*.csv.gz"))
    if not model_files:
        return []
    frame_by_seed: dict[str, dict[str, Any]] = {}
    ids_by_seed: dict[str, np.ndarray] = {}
    source_paths: list[Path] = [old_metrics_path, old_cross_path, gate_path]
    canonical_paths: set[Path] = set()
    for seed_number in seeds:
        seed = f"b3_seed{seed_number}"
        frame_by_seed[seed] = {}
        for model in model_files:
            path = prediction_root / seed / f"{model}.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen Phase05 prediction: {path}")
            frame = read_csv(path)
            sentence_ids, canonical_path = _attach_canonical_sentence_ids(
                frame, seed_number=seed_number, score_column=model,
            )
            frame_by_seed[seed][model] = (frame, sentence_ids)
            source_paths.append(path)
            canonical_paths.add(canonical_path)
            if model == model_files[0]:
                ids_by_seed[seed] = sentence_ids
            else:
                assert_row_alignment(
                    {"sentence_id": ids_by_seed[seed], "ref_id": frame_by_seed[seed][model_files[0]][0]["ref_id"].to_numpy(),
                     "image_id": frame_by_seed[seed][model_files[0]][0]["image_id"].to_numpy(),
                     "eval_split": frame_by_seed[seed][model_files[0]][0]["eval_split"].astype(str).to_numpy(),
                     "K": frame_by_seed[seed][model_files[0]][0]["K"].astype(int).to_numpy(),
                     "grounding_correct": frame_by_seed[seed][model_files[0]][0]["grounding_correct"].to_numpy()},
                    {"sentence_id": sentence_ids, "ref_id": frame["ref_id"].to_numpy(), "image_id": frame["image_id"].to_numpy(),
                     "eval_split": frame["eval_split"].astype(str).to_numpy(), "K": frame["K"].astype(int).to_numpy(),
                     "grounding_correct": frame["grounding_correct"].to_numpy()},
                    required_fields=("ref_id", "image_id", "eval_split", "K", "grounding_correct"),
                    candidate_name=f"{seed}/{model}",
                )
    source_paths.extend(canonical_paths)
    source_paths = list(dict.fromkeys(source_paths))
    cohorts = ("val_select", "testA", "testB", "__pooled_test__")
    jobs = []
    for cohort in cohorts:
        first_frame = frame_by_seed["b3_seed1"][model_files[0]][0]
        first_k = first_frame["K"].astype(int).to_numpy()
        base_mask = _selected_rows(first_frame, cohort) & (first_k == 5)
        if not np.any(base_mask):
            continue
        canonical_ids = frame_by_seed["b3_seed1"][model_files[0]][1][base_mask]
        image_ids = first_frame.loc[base_mask, "image_id"].to_numpy()
        cohort_metadata = _cohort_metadata_from_labels(
            first_frame.loc[base_mask, "eval_split"].astype(str).to_numpy(),
            historical_pool_identity=(
                "Phase05 K5 prediction-row union testA+testB; historical all-common status is separately represented by __all_common__"
                if cohort == "__pooled_test__" else f"historical Phase05 eval_split={cohort} K5 row mask"
            ),
        )
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        legacy_points: dict[str, dict[str, float]] = {}
        old_ci: dict[str, dict[str, list[Any]]] = {}
        specs: list[dict[str, Any]] = []
        metric_by_name: dict[str, str] = {}
        legacy_anchor_unavailable: list[dict[str, str]] = []
        for seed in (f"b3_seed{v}" for v in seeds):
            seed_cells[seed] = {}
            for model in model_files:
                frame, sentence_ids = frame_by_seed[seed][model]
                for k in (5, 10, 20, 50):
                    cell_mask = _selected_rows(frame, cohort) & (frame["K"].astype(int).to_numpy() == k)
                    if not np.array_equal(sentence_ids[cell_mask], canonical_ids):
                        raise ValueError(f"Sentence order differs for {seed}/{model}/{cohort}/K{k}")
                    condition = f"{model}__K{k}"
                    confidence = frame.loc[cell_mask, "reliability_score"].to_numpy(dtype=np.float64)
                    correct = frame.loc[cell_mask, "grounding_correct"].to_numpy(dtype=np.float64)
                    seed_cells[seed][condition] = (confidence, correct)
                    old_split = cohort
                    old_row = old_metrics[
                        (old_metrics["scorer"] == seed) & (old_metrics["model"] == model)
                        & (old_metrics["eval_split"] == old_split) & (old_metrics["K"] == k)
                    ]
                    for metric in CI_METRICS:
                        name = f"{condition}::{metric}"
                        if len(old_row) == 1:
                            value = _legacy_metric_value(old_row.iloc[0], metric)
                            if np.isfinite(value):
                                legacy_points.setdefault(name, {})[seed] = value
                            else:
                                reason = (
                                    "legacy reliability_metrics.csv has no b3_accuracy column"
                                    if metric == "accuracy" and "b3_accuracy" not in old_metrics.columns else
                                    "legacy metric value is non-finite for this model/scorer/cohort/K"
                                )
                                legacy_anchor_unavailable.append({"estimate": name, "seed": seed, "reason": reason})
                        specs.append({"name": name, "metric": metric, "operation": "condition", "condition": condition})
                        metric_by_name[name] = metric
        for model in model_files:
            for k_hi in (20, 50):
                low = f"{model}__K5"
                high = f"{model}__K{k_hi}"
                entries = [row for row in old_cross.to_dict("records") if row.get("model") == model and int(row.get("K_hi", -1)) == k_hi and row.get("eval_split") == cohort]
                for metric, a, b, legacy_column, ci_low, ci_high, new_tag in (
                    ("auroc_correct", low, high, "auroc_drop", "auroc_drop_ci_low", "auroc_drop_ci_high", "auroc_drop"),
                    ("rer_at_50", low, high, "rer_at_50_drop", "rer50_ci_low", "rer50_ci_high", "rer50_drop"),
                    ("e_aurc", high, low, None, "e_aurc_ci_low", "e_aurc_ci_high", "e_aurc_worsening_absolute"),
                ):
                    name = f"crossK__{model}__K{k_hi}__{new_tag}"
                    specs.append({"name": name, "metric": metric, "operation": "difference", "a": a, "b": b})
                    metric_by_name[name] = metric
                    if entries and legacy_column is not None:
                        for row in entries:
                            scorer = str(row["scorer"])
                            if scorer not in {*[f"b3_seed{i}" for i in seeds], "b3_mean"}:
                                continue
                            legacy_points.setdefault(name, {})[scorer] = _float_or_nan(row.get(legacy_column))
                            if legacy_column and scorer in {f"b3_seed{i}" for i in seeds}:
                                old_ci.setdefault(name, {})[scorer] = [row.get(ci_low), row.get(ci_high)]
                relative_name = f"crossK__{model}__K{k_hi}__e_aurc_worsening_relative"
                specs.append({
                    "name": relative_name,
                    "metric": "e_aurc",
                    "operation": "relative_contrast",
                    "contrast_positive": high,
                    "contrast_negative": low,
                    "denominator_condition": low,
                })
                metric_by_name[relative_name] = "e_aurc_relative"
                if entries:
                    for row in entries:
                        scorer = str(row["scorer"])
                        if scorer not in {*[f"b3_seed{i}" for i in seeds], "b3_mean"}:
                            continue
                        legacy_points.setdefault(relative_name, {})[scorer] = _float_or_nan(row.get("e_aurc_worsening"))
                        if scorer in {f"b3_seed{i}" for i in seeds}:
                            old_ci.setdefault(relative_name, {})[scorer] = [row.get("e_aurc_ci_low"), row.get("e_aurc_ci_high")]
                if any(row.get("e_aurc_abs_hi") is not None for row in entries):
                    for scorer in (f"b3_seed{i}" for i in seeds):
                        legacy_anchor_unavailable.append({
                            "estimate": f"crossK__{model}__K{k_hi}__e_aurc_worsening_absolute",
                            "seed": scorer,
                            "reason": "legacy cross-K table reports high-K E-AURC and relative worsening, not the absolute high-minus-low contrast",
                        })
        jobs.append({
            "job_id": f"phase05_{cohort}", "scope": "Phase05_score_sufficiency", "cohort": cohort,
            "image_ids": image_ids, "seed_cells": seed_cells, "estimates": specs, "metrics": CI_METRICS,
            "legacy_points": legacy_points, "old_ci": old_ci, "old_gate": old_gate, "thresholds": thresholds,
            "metric_by_name": metric_by_name, "source_paths": source_paths,
            "legacy_anchor_unavailable": legacy_anchor_unavailable,
            "cohort_construction": "legacy_testA_plus_testB_pool" if cohort == "__pooled_test__" else "historical_eval_split",
            "legacy_pooled_anchor": cohort == "__pooled_test__",
            "cohort_metadata": cohort_metadata,
            "legacy_cohort_classification": "EXISTING_LEGACY_TESTA_TESTB_POOL" if cohort == "__pooled_test__" else "HISTORICAL_SPLIT",
            "legacy_cohort_note": "The legacy Phase05 metrics and sufficiency gate contain this testA+testB pool; point anchors are checked against the matching scorer/seed rows." if cohort == "__pooled_test__" else "",
            "legacy_pool_status": "EXISTING_LEGACY_TESTA_TESTB_POOL" if cohort == "__pooled_test__" else "NOT_APPLICABLE",
            "sentence_id_status": "RECONSTRUCTED_BY_ORDERED_CANONICAL_ANCHORS",
            "evidence": "all scorer CSV rows validated against seed-specific sentence/ref/image/split/K/correct canonical NPZ; repeated ref_id rows retained",
            "recovery_status": "READ_ONLY_PREDICTION_ARTIFACT",
        })
    return jobs


def _phase1_restore_fixed_predictions(source_paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Restore fixed MSP/Stats/E1b predictions from frozen inputs, never selected CSV columns.

    The Phase1 CSV stores per-seed selected-model outputs at 10 significant digits.
    Its ``semantic_reliability`` column is the selected ``best_semantic`` model and
    is not guaranteed to be E1b. Fixed Stats/E1b values are therefore reconstructed
    from canonical B3 scores, saved E1 features and the checksummed frozen bundle.
    """
    from ccg.reliability import features as rfeat
    from ccg.semantic.frozen_load import load_models

    phase_root = ROOT / "results" / "phase1_semantic_sufficiency"
    bundle_root = ROOT / "results" / "phase1e_refcocog_external" / "frozen_models"
    bundle_json = bundle_root / "models.json"
    checksum_path = bundle_root / "models.sha256"
    verification_path = bundle_root / "bundle_verification.json"
    bundle_record = None
    for path in (bundle_json, checksum_path, verification_path):
        record = frozen_source(path)
        if path == bundle_json:
            bundle_record = record
        source_paths.append(path)
    verification = read_json(verification_path)
    bundle = load_models(bundle_root, verify_checksum=True)
    if set(bundle.scorers) != {"b3_seed1", "b3_seed2", "b3_seed3"}:
        raise ValueError(f"frozen Phase1 reliability bundle has unexpected seeds: {bundle.scorers}")

    fixed: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    canonical: dict[str, dict[str, np.ndarray]] = {}
    source_notes: dict[str, Any] = {
        "bundle_path": bundle_json.relative_to(ROOT).as_posix(),
        "bundle_sha256": (bundle_record or {}).get("sha256", ""),
        "bundle_verification_path": verification_path.relative_to(ROOT).as_posix(),
        "bundle_verification": verification,
        "models": {seed: ["msp", "stats_logistic", "e1b_stats_semantic"] for seed in bundle.scorers},
        "source_precision": "frozen_bundle_float64_and_original_float32_score_inputs",
    }
    for seed_number in (1, 2, 3):
        seed = f"b3_seed{seed_number}"
        seed_root = ROOT / "results" / "phase0b_independent" / f"seed_{seed_number}"
        point_path = seed_root / "per_sentence_predictions.npz"
        metadata_path = seed_root / "eval_metadata.json"
        canonical[seed] = load_npz(point_path)
        if seed_number > 1:
            anchor = canonical["b3_seed1"]
            for field in ("sentence_ids", "ref_ids", "image_ids", "eval_split"):
                if not np.array_equal(canonical[seed][field], anchor[field]):
                    raise ValueError(f"Phase1 canonical row identity differs between b3_seed1 and {seed}: {field}")
        source_paths.extend((point_path, metadata_path))
        temperature = float(read_json(metadata_path)["temperature_corrected"])
        feature_path = phase_root / "semantic_features" / f"e1_stats_{seed}.npz"
        feature_record = frozen_source(feature_path)
        source_paths.append(feature_path)
        fixed[seed] = {}
        for k in (5, 10, 20, 50):
            score_path = seed_root / "raw_scores" / f"K{k}.npz"
            raw = load_npz(score_path)
            source_paths.append(score_path)
            order = np.argsort(np.asarray(raw["sentence_id"], dtype=np.int64), kind="stable")
            sentence_ids = np.asarray(raw["sentence_id"], dtype=np.int64)[order]
            canonical_ids = np.asarray(canonical[seed]["sentence_ids"], dtype=np.int64)
            if not np.array_equal(sentence_ids, canonical_ids) or not np.all(np.diff(sentence_ids) > 0):
                raise ValueError(f"Phase1 frozen score identities do not match canonical rows: {seed}/K{k}")
            score = np.asarray(raw["scores"], dtype=np.float32)[order]
            ref_id = np.asarray(raw["ref_id"], dtype=np.int64)[order]
            image_id = np.asarray(raw["image_id"], dtype=np.int64)[order]
            eval_split = np.asarray(raw["eval_split"])[order].astype(str)
            target_local = np.asarray(raw["target_local"], dtype=np.int64)[order]
            if not np.array_equal(ref_id, canonical[seed]["ref_ids"]) or not np.array_equal(image_id, canonical[seed]["image_ids"]):
                raise ValueError(f"Phase1 frozen score ref/image identity mismatch: {seed}/K{k}")
            if not np.array_equal(eval_split, np.asarray(canonical[seed]["eval_split"]).astype(str)):
                raise ValueError(f"Phase1 frozen score split identity mismatch: {seed}/K{k}")
            embedding_path = ROOT / "cache" / "semantic_phase1" / f"embeddings_K{k}.npz"
            frozen_source(embedding_path)
            source_paths.append(embedding_path)
            with np.load(embedding_path, allow_pickle=False) as embedding_archive:
                for raw_name, embedding_name in (
                    ("sentence_id", "sentence_id"), ("ref_id", "ref_id"),
                    ("image_id", "image_id"), ("eval_split", "eval_split"),
                ):
                    observed = np.asarray(embedding_archive[embedding_name])
                    expected = sentence_ids if raw_name == "sentence_id" else {
                        "ref_id": ref_id, "image_id": image_id, "eval_split": eval_split,
                    }[raw_name]
                    if not np.array_equal(observed.astype(expected.dtype, copy=False), expected):
                        raise ValueError(f"Phase1 frozen semantic embedding identity mismatch: {seed}/K{k}/{raw_name}")
            with np.load(feature_path, allow_pickle=False) as feature_archive:
                feature_names = tuple(str(x) for x in feature_archive["names"].tolist())
                if feature_names != tuple(bundle.semantic_feature_names):
                    raise ValueError(f"Phase1 E1 feature order mismatch for {seed}: {feature_names}")
                sem16 = np.asarray(feature_archive[f"stats_K{k}"], dtype=np.float64)
            if sem16.shape != (sentence_ids.size, len(bundle.semantic_feature_names)):
                raise ValueError(f"Phase1 E1 feature shape mismatch: {seed}/K{k}: {sem16.shape}")
            correct = np.argmax(score, axis=1) == target_local
            if not np.array_equal(correct, np.asarray(canonical[seed][f"correct_K{k}"], dtype=bool)):
                raise ValueError(f"Phase1 fixed-model correctness disagrees with canonical source: {seed}/K{k}")
            stats17 = rfeat.stat_features(score, temperature=temperature)
            conf_stats, conf_e1b = bundle.predict(seed, stats17, sem16)
            conf_msp = rfeat.scalar_confidence(score, temperature=temperature)["msp"]
            fixed[seed][k] = {
                "msp": np.asarray(conf_msp, dtype=np.float64),
                "stats_logistic": np.asarray(conf_stats, dtype=np.float64),
                "e1b_stats_semantic": np.asarray(conf_e1b, dtype=np.float64),
                "correct": np.asarray(correct, dtype=np.float64),
                "sentence_id": sentence_ids,
                "ref_id": ref_id,
                "image_id": image_id,
                "eval_split": eval_split,
            }
            source_notes.setdefault("score_sources", {})[seed] = {}
            source_notes["score_sources"][seed][str(k)] = {
                "path": score_path.relative_to(ROOT).as_posix(),
                "temperature": temperature,
                "e1_features_path": feature_record["path"],
                "identity_source": embedding_path.relative_to(ROOT).as_posix(),
            }
    return fixed, canonical, source_notes


def build_phase1_jobs() -> list[dict[str, Any]]:
    root = ROOT / "results" / "phase1_semantic_sufficiency"
    old_metrics_path = root / "aggregate.csv"
    old_pair_path = root / "pairwise_vs_score_only.csv"
    gate_path = root / "sufficiency_gate.json"
    metadata_path = root / "metadata.json"
    old_metrics = read_csv(old_metrics_path)
    old_pair = read_csv(old_pair_path)
    gate_json = read_json(gate_path)
    metadata = read_json(metadata_path)
    old_gate = gate_json.get("verdict", {}).get("verdict", "NOT_RECORDED")
    thresholds = gate_json.get("thresholds", {})
    seeds = (1, 2, 3)
    source_paths: list[Path] = [old_metrics_path, old_pair_path, gate_path, metadata_path]
    fixed, canonical, restore_notes = _phase1_restore_fixed_predictions(source_paths)
    selected_frames: dict[str, Any] = {}
    selected_rowids: dict[str, np.ndarray] = {}
    selected_models: dict[str, dict[str, str]] = {}
    for seed_number in seeds:
        seed = f"b3_seed{seed_number}"
        prediction_path = root / "predictions" / f"{seed}.csv.gz"
        frame = read_csv(prediction_path)
        ids, canonical_path = _attach_canonical_sentence_ids(
            frame, seed_number=seed_number, score_column="legacy-selected-outputs",
        )
        selected_frames[seed] = frame
        selected_rowids[seed] = ids
        source_paths.extend((prediction_path, canonical_path))
        choice = metadata["selections"][seed]
        selected_models[seed] = {
            "score_only_reliability": str(choice["best_score_only"]),
            "semantic_reliability": str(choice["best_semantic"]),
        }

    fixed_archive_arrays: dict[str, np.ndarray] = {}
    for seed, per_k in fixed.items():
        for k, values in per_k.items():
            prefix = f"{seed}__K{k}__"
            for field in ("sentence_id", "ref_id", "image_id", "eval_split", "correct"):
                fixed_archive_arrays[prefix + ("grounding_correct" if field == "correct" else field)] = values[field]
            for model in ("msp", "stats_logistic", "e1b_stats_semantic"):
                fixed_archive_arrays[prefix + model] = values[model]
    fixed_archive_path = OUT / "predictions" / "phase1_fixed_bundle_predictions.npz"
    fixed_archive_hash = _write_npz_atomic(fixed_archive_path, fixed_archive_arrays)
    selected_archive_arrays: dict[str, np.ndarray] = {}
    for seed_number in seeds:
        seed = f"b3_seed{seed_number}"
        frame = selected_frames[seed]
        row_ids = selected_rowids[seed]
        k_values = frame["K"].astype(int).to_numpy()
        for k in (5, 10, 20, 50):
            mask = k_values == k
            ids = row_ids[mask]
            canonical_ids_for_seed = np.asarray(canonical[seed]["sentence_ids"], dtype=np.int64)
            indexes = np.searchsorted(canonical_ids_for_seed, ids)
            if np.any(indexes >= canonical_ids_for_seed.size) or not np.array_equal(canonical_ids_for_seed[indexes], ids):
                raise ValueError(f"Phase1 selected CSV sentence id not found in canonical universe: {seed}/K{k}")
            expected_correct = np.asarray(canonical[seed][f"correct_K{k}"], dtype=bool)[indexes]
            observed_correct = frame.loc[mask, "grounding_correct"].to_numpy(dtype=bool)
            if not np.array_equal(expected_correct, observed_correct):
                raise ValueError(f"Phase1 selected CSV correctness differs from canonical source: {seed}/K{k}")
            prefix = f"{seed}__K{k}__"
            selected_archive_arrays[prefix + "sentence_id"] = ids
            selected_archive_arrays[prefix + "ref_id"] = frame.loc[mask, "ref_id"].to_numpy(dtype=np.int64)
            selected_archive_arrays[prefix + "image_id"] = frame.loc[mask, "image_id"].to_numpy(dtype=np.int64)
            selected_archive_arrays[prefix + "eval_split"] = frame.loc[mask, "eval_split"].astype(str).to_numpy()
            selected_archive_arrays[prefix + "grounding_correct"] = observed_correct.astype(np.bool_)
            selected_archive_arrays[prefix + "score_only_reliability"] = frame.loc[mask, "score_only_reliability"].to_numpy(dtype=np.float64)
            selected_archive_arrays[prefix + "semantic_reliability"] = frame.loc[mask, "semantic_reliability"].to_numpy(dtype=np.float64)
            selected_archive_arrays[prefix + "selected_score_only_model"] = np.full(ids.size, selected_models[seed]["score_only_reliability"])
            selected_archive_arrays[prefix + "selected_semantic_slot_model"] = np.full(ids.size, selected_models[seed]["semantic_reliability"])
    selected_archive_path = OUT / "predictions" / "phase1_selected_operational_predictions.npz"
    selected_archive_hash = _write_npz_atomic(selected_archive_path, selected_archive_arrays)
    restore_notes["recovery_output_path"] = fixed_archive_path.relative_to(ROOT).as_posix()
    restore_notes["recovery_output_sha256"] = fixed_archive_hash
    operational_identity_payload = {
        "path": selected_archive_path.relative_to(ROOT).as_posix(),
        "sha256": selected_archive_hash,
        "selected_models_by_seed": selected_models,
    }

    fixed_models = ("msp", "stats_logistic", "e1b_stats_semantic")
    jobs: list[dict[str, Any]] = []
    for cohort in ("val_select", "testA", "testB", "__pooled_test__"):
        cohort_mask = np.isin(canonical["b3_seed1"]["eval_split"], ("testA", "testB")) if cohort == "__pooled_test__" else (
            np.asarray(canonical["b3_seed1"]["eval_split"]).astype(str) == cohort
        )
        canonical_ids = np.asarray(canonical["b3_seed1"]["sentence_ids"], dtype=np.int64)[cohort_mask]
        image_ids = np.asarray(canonical["b3_seed1"]["image_ids"], dtype=np.int64)[cohort_mask]
        cohort_metadata = _cohort_metadata_from_labels(
            np.asarray(canonical["b3_seed1"]["eval_split"]).astype(str)[cohort_mask],
            historical_pool_identity=(
                "historical Phase1 pooled gate is exactly testA+testB; no validation rows"
                if cohort == "__pooled_test__" else f"historical Phase1 eval_split={cohort} row mask"
            ),
        )
        if canonical_ids.size == 0:
            continue
        fixed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        fixed_legacy: dict[str, dict[str, float]] = {}
        fixed_old_ci: dict[str, dict[str, list[float]]] = {}
        fixed_specs: list[dict[str, Any]] = []
        fixed_metric_names: dict[str, str] = {}
        for seed_number in seeds:
            seed = f"b3_seed{seed_number}"
            if not np.array_equal(np.asarray(canonical[seed]["sentence_ids"])[cohort_mask], canonical_ids):
                raise ValueError(f"Phase1 canonical sentence identity differs by seed in {cohort}")
            fixed_cells[seed] = {}
            for k in (5, 10, 20, 50):
                per_k = fixed[seed][k]
                if not np.array_equal(per_k["sentence_id"][cohort_mask], canonical_ids):
                    raise ValueError(f"Phase1 restored predictions are not canonical in {seed}/{cohort}/K{k}")
                correct = per_k["correct"][cohort_mask]
                for model in fixed_models:
                    condition = f"{model}__K{k}"
                    fixed_cells[seed][condition] = (per_k[model][cohort_mask], correct)
                    old_row = old_metrics[
                        (old_metrics["scorer"] == seed) & (old_metrics["model"] == model)
                        & (old_metrics["eval_split"] == cohort) & (old_metrics["K"] == k)
                    ]
                    for metric in CI_METRICS:
                        name = f"{condition}::{metric}"
                        if len(old_row) == 1:
                            old_value = _legacy_metric_value(old_row.iloc[0], metric)
                            if np.isfinite(old_value):
                                fixed_legacy.setdefault(name, {})[seed] = old_value
                        fixed_specs.append({"name": name, "metric": metric, "operation": "condition", "condition": condition})
                        fixed_metric_names[name] = metric

            for k in (5, 10, 20, 50):
                for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
                    name = f"e1b_minus_stats__K{k}::{metric}"
                    fixed_specs.append({"name": name, "metric": metric, "operation": "difference",
                                        "a": f"e1b_stats_semantic__K{k}", "b": f"stats_logistic__K{k}"})
                    fixed_metric_names[name] = metric
                    if cohort == "__pooled_test__":
                        rows = old_pair[(old_pair["scorer"] == seed) & (old_pair["K"] == k)
                                        & (old_pair["model_a"] == "e1b_stats_semantic")
                                        & (old_pair["model_b"] == "stats_logistic")
                                        & (old_pair["metric"] == metric) & (old_pair["row_kind"] == "pairwise")]
                        if len(rows) == 1:
                            fixed_legacy.setdefault(name, {})[seed] = _float_or_nan(rows.iloc[0]["diff"])
                            if np.isfinite(_float_or_nan(rows.iloc[0].get("ci_low"))) and np.isfinite(_float_or_nan(rows.iloc[0].get("ci_high"))):
                                fixed_old_ci.setdefault(name, {})[seed] = [float(rows.iloc[0]["ci_low"]), float(rows.iloc[0]["ci_high"])]

            for k in (20, 50):
                for metric, spec in (
                    ("auroc_correct", {"name": f"gate_delta_auroc_e1b_vs_msp__K{k}", "metric": "auroc_correct",
                                       "operation": "difference", "a": f"e1b_stats_semantic__K{k}", "b": f"msp__K{k}"}),
                    ("e_aurc", {"name": f"gate_eaurc_relative_reduction_e1b_vs_msp__K{k}", "metric": "e_aurc",
                                "operation": "relative_contrast", "contrast_positive": f"msp__K{k}",
                                "contrast_negative": f"e1b_stats_semantic__K{k}", "denominator_condition": f"msp__K{k}"}),
                    ("rer_at_50", {"name": f"gate_rer50_gain_e1b_vs_msp_pp__K{k}", "metric": "rer_at_50",
                                   "operation": "difference", "a": f"e1b_stats_semantic__K{k}", "b": f"msp__K{k}", "scale": 100.0}),
                ):
                    fixed_specs.append(spec)
                    fixed_metric_names[spec["name"]] = (
                        "e_aurc_relative_reduction" if metric == "e_aurc" else ("rer_at_50_gain_pp" if metric == "rer_at_50" else metric)
                    )
                    if cohort == "__pooled_test__":
                        row_kind = "pairwise_ratio" if metric == "e_aurc" else "pairwise"
                        rows = old_pair[(old_pair["scorer"] == seed) & (old_pair["K"] == k)
                                        & (old_pair["model_a"] == "e1b_stats_semantic")
                                        & (old_pair["model_b"] == "msp")
                                        & (old_pair["metric"] == metric) & (old_pair["row_kind"] == row_kind)]
                        if len(rows) == 1:
                            if metric == "e_aurc":
                                point_value = _float_or_nan(rows.iloc[0].get("reduction"))
                                low_value = _float_or_nan(rows.iloc[0].get("reduction_ci_low"))
                                high_value = _float_or_nan(rows.iloc[0].get("reduction_ci_high"))
                            elif metric == "rer_at_50":
                                point_value = 100.0 * _float_or_nan(rows.iloc[0].get("diff"))
                                low_value = 100.0 * _float_or_nan(rows.iloc[0].get("ci_low"))
                                high_value = 100.0 * _float_or_nan(rows.iloc[0].get("ci_high"))
                            else:
                                point_value = _float_or_nan(rows.iloc[0].get("diff"))
                                low_value = _float_or_nan(rows.iloc[0].get("ci_low"))
                                high_value = _float_or_nan(rows.iloc[0].get("ci_high"))
                            fixed_legacy.setdefault(spec["name"], {})[seed] = point_value
                            if np.isfinite(low_value) and np.isfinite(high_value):
                                fixed_old_ci.setdefault(spec["name"], {})[seed] = [low_value, high_value]
        fixed_jobs_specs = fixed_specs
        fixed_jobs_legacy = fixed_legacy
        jobs.append({
            "job_id": f"phase1_fixed_{cohort}", "scope": "Phase1_semantic_sufficiency_fixed_models", "cohort": cohort,
            "image_ids": image_ids, "seed_cells": fixed_cells, "estimates": fixed_jobs_specs,
            "metrics": CI_METRICS, "legacy_points": fixed_jobs_legacy, "old_ci": fixed_old_ci,
            "old_gate": old_gate if cohort == "__pooled_test__" else "",
            "thresholds": thresholds if cohort == "__pooled_test__" else {},
            "metric_by_name": fixed_metric_names, "source_paths": list(dict.fromkeys(source_paths)),
            "model_identity_by_seed": {
                f"b3_seed{i}": {str(k): {model: model for model in fixed_models} for k in (5, 10, 20, 50)}
                for i in seeds
            },
            "prediction_source_precision": "frozen_model_bundle_float64_from_original_score_and_semantic_features",
            "recovery_output_path": fixed_archive_path.relative_to(ROOT).as_posix(),
            "recovery_output_sha256": fixed_archive_hash,
            "cohort_construction": "legacy_testA_plus_testB_gate_pool" if cohort == "__pooled_test__" else "historical_eval_split",
            "legacy_pooled_anchor": cohort == "__pooled_test__",
            "cohort_metadata": cohort_metadata,
            "legacy_cohort_classification": "EXISTING_LEGACY_TESTA_TESTB_POOL" if cohort == "__pooled_test__" else "HISTORICAL_SPLIT",
            "legacy_cohort_note": "Historical Phase1 pooled gate is exactly testA+testB (10,286 rows/1,490 images); fixed Stats/E1b predictions are restored from the frozen bundle and anchored per seed/model/K." if cohort == "__pooled_test__" else "",
            "legacy_pool_status": "EXISTING_LEGACY_TESTA_TESTB_POOL" if cohort == "__pooled_test__" else "NOT_APPLICABLE",
            "sentence_id_status": "VERIFIED_NATIVE_ID_AND_EMBEDDING_ROW_ORDER",
            "evidence": json.dumps({"restoration": restore_notes, "bundle_prediction_models": list(fixed_models),
                                    "canonical_ids": "raw score sentence_id sorted ascending; verified against per_sentence_predictions, semantic embedding identity arrays, and correctness",
                                    "historical_gate_models": gate_json.get("comparison", {}),
                                    "metric_anchor_tolerance": ANCHOR_TOLERANCE}, ensure_ascii=False),
            "recovery_status": "FROZEN_BUNDLE_FORWARD_WITH_ORIGINAL_PHASE1_FEATURES",
        })

        # Keep the legacy generic prediction fields only as a named, selected-model
        # operational replay. In this frozen run best_semantic is E2_SCORE, which
        # consumes score statistics but no semantic embedding; it is not the global
        # A7.6 semantic sufficiency comparison and cannot replace fixed E1b.
        operational_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        operational_legacy: dict[str, dict[str, float]] = {}
        operational_specs: list[dict[str, Any]] = []
        operational_identity = {seed: dict(selected_models[seed]) for seed in selected_models}
        for seed_number in seeds:
            seed = f"b3_seed{seed_number}"
            frame = selected_frames[seed]
            row_ids = selected_rowids[seed]
            k_values = frame["K"].astype(int).to_numpy()
            split_values = frame["eval_split"].astype(str).to_numpy()
            operational_cells[seed] = {}
            for k in (5, 10, 20, 50):
                mask = (k_values == k) & (np.isin(split_values, ("testA", "testB")) if cohort == "__pooled_test__" else (split_values == cohort))
                expected_ids = canonical_ids
                observed_ids = row_ids[mask]
                if not np.array_equal(observed_ids, expected_ids):
                    raise ValueError(f"legacy selected Phase1 CSV identity/order mismatch in {seed}/{cohort}/K{k}")
                correctness = frame.loc[mask, "grounding_correct"].to_numpy(dtype=np.float64)
                for slot, column in (("best_score_only_slot", "score_only_reliability"),
                                     ("best_semantic_label_slot", "semantic_reliability")):
                    condition = f"{slot}__K{k}"
                    confidence = frame.loc[mask, column].to_numpy(dtype=np.float64)
                    operational_cells[seed][condition] = (confidence, correctness)
                    old_model = selected_models[seed][column]
                    old_row = old_metrics[(old_metrics["scorer"] == seed) & (old_metrics["model"] == old_model)
                                          & (old_metrics["eval_split"] == cohort) & (old_metrics["K"] == k)]
                    for metric in CI_METRICS:
                        name = f"{condition}::{metric}"
                        if len(old_row) == 1:
                            old_value = _legacy_metric_value(old_row.iloc[0], metric)
                            if np.isfinite(old_value):
                                operational_legacy.setdefault(name, {})[seed] = old_value
                        operational_specs.append({"name": name, "metric": metric, "operation": "condition", "condition": condition})
            # Enforce that the score-only field really serializes the selected model's
            # confidence values as written; this checks the 10g CSV rendering, not losslessness.
            k_values_all = frame["K"].astype(int).to_numpy()
            for k in (5, 10, 20, 50):
                mask = k_values_all == k
                if selected_models[seed]["score_only_reliability"] == "msp":
                    canonical_index = np.searchsorted(fixed[seed][k]["sentence_id"], row_ids[mask])
                    if not np.array_equal(fixed[seed][k]["sentence_id"][canonical_index], row_ids[mask]):
                        raise ValueError(f"selected Phase1 CSV sentence mapping failure: {seed}/K{k}")
                    exact = fixed[seed][k]["msp"][canonical_index]
                    rendered = np.asarray([float(format(float(value), ".10g")) for value in exact], dtype=np.float64)
                    observed = frame.loc[mask, "score_only_reliability"].to_numpy(dtype=np.float64)
                    if not np.array_equal(rendered, observed):
                        raise ValueError(f"Phase1 selected msp CSV rendering mismatch: {seed}/K{k}")
        jobs.append({
            "job_id": f"phase1_selected_operational_{cohort}", "scope": "Phase1_selected_model_operational_replay", "cohort": cohort,
            "image_ids": image_ids, "seed_cells": operational_cells, "estimates": operational_specs,
            "metrics": CI_METRICS, "legacy_points": operational_legacy,
            "old_gate": "OPERATIONAL_SELECTION_FIELDS_ONLY_NOT_THE_GLOBAL_SUFFICIENCY_GATE",
            "thresholds": {}, "source_paths": list(dict.fromkeys(source_paths)),
            "model_identity_by_seed": operational_identity,
            "prediction_source_precision": "legacy_selected_prediction_csv_10_significant_digits; selected outputs reattached to sentence ids, lossy serialization retained and per-model point-anchored",
            "recovery_output_path": selected_archive_path.relative_to(ROOT).as_posix(),
            "recovery_output_sha256": selected_archive_hash,
            "cohort_construction": "legacy_testA_plus_testB_pool" if cohort == "__pooled_test__" else "historical_eval_split",
            "legacy_pooled_anchor": cohort == "__pooled_test__",
            "cohort_metadata": cohort_metadata,
            "legacy_cohort_classification": "EXISTING_LEGACY_TESTA_TESTB_POOL" if cohort == "__pooled_test__" else "HISTORICAL_SPLIT",
            "legacy_cohort_note": "Operational fields are per-seed selected outputs: score_only_reliability selects MSP and semantic_reliability selects E2_SCORE in all three seeds. E2_SCORE is a score-statistics ablation with no candidate semantic input; this replay is not a fixed semantic effect or the A7.6 gate." if cohort == "__pooled_test__" else "",
            "legacy_pool_status": "EXISTING_LEGACY_TESTA_TESTB_POOL" if cohort == "__pooled_test__" else "NOT_APPLICABLE",
            "sentence_id_status": "RECONSTRUCTED_AND_VERIFIED_AGAINST_CANONICAL_ORDER_WITH_DUPLICATE_REF_ROWS_RETAINED",
            "evidence": json.dumps({"selected_model_by_seed": operational_identity,
                                    "selected_prediction_archive": operational_identity_payload,
                                    "global_gate_comparison": gate_json.get("comparison", {}),
                                    "confidence_csv_precision": "10 significant digits; lossy serialization",
                                    "msp_csv_rendering_verified": True,
                                    "selection_source": "metadata.json per-seed best_score_only/best_semantic"}, ensure_ascii=False),
            "recovery_status": "LOSSY_SELECTED_CSV_WITH_PER_MODEL_POINT_ANCHORS",
        })
    return jobs


def build_phase1f_jobs() -> list[dict[str, Any]]:
    root = ROOT / "results" / "phase1f_hard_semantic"
    pred_root = root / "predictions"
    old_metrics_path = root / "reliability_metrics.csv"
    old_increment_path = root / "semantic_increment.csv"
    old_paired_path = root / "paired_random_vs_hard.csv"
    gate_path = root / "gate.json"
    old_metrics = read_csv(old_metrics_path)
    old_increment = read_csv(old_increment_path)
    old_paired = read_csv(old_paired_path)
    gate = read_json(gate_path)
    old_gate = gate.get("verdict", {}).get("verdict", "NOT_RECORDED")
    thresholds = gate.get("verdict", {}).get("thresholds", {})
    seed_names = tuple(f"b3_seed{i}" for i in (1, 2, 3))
    cell_groups = (
        ("k5", ("rand5", "hard5"), "same4"),
        ("k10", ("rand10", "hard10"), "same9"),
        ("dose", ("expb_m0", "expb_m2", "expb_m4", "expb_m8"), "same8"),
    )
    model_fields = {
        "msp": ("conf_msp", "msp"),
        "stats": ("conf_stats", "stats_logistic"),
        "e1b": ("conf_e1b", "e1b_stats_semantic"),
    }
    source_paths: list[Path] = [old_metrics_path, old_increment_path, old_paired_path, gate_path]
    jobs: list[dict[str, Any]] = []

    for group_name, cells, cohort in cell_groups:
        archives: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for cell in cells:
            for seed in seed_names:
                path = pred_root / f"{cell}__{seed}.npz"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing frozen Phase1F prediction: {path}")
                archives[(cell, seed)] = load_npz(path)
                source_paths.append(path)
        ref = archives[(cells[0], seed_names[0])]
        canonical_ids = ref["sentence_id"]
        if np.unique(canonical_ids).size != canonical_ids.size:
            raise ValueError(f"Phase1F {group_name} contains duplicate sentence_id values")
        image_ids = ref["image_id"].copy()
        text_index_path = ROOT / "cache" / "features" / "text_index.csv"
        phase1f_split_counts = _split_metadata_from_text_index(canonical_ids, ref["ref_id"], image_ids)
        source_paths.append(text_index_path)
        cohort_metadata = _cohort_metadata_from_counts(
            phase1f_split_counts,
            historical_pool_identity=(
                f"frozen Phase1F {cohort} sentence/ref/image rows verified against text_index.csv"
            ),
        )
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        legacy_points: dict[str, dict[str, float]] = {}
        old_ci: dict[str, dict[str, list[float]]] = {}
        specs: list[dict[str, Any]] = []
        metric_by_name: dict[str, str] = {}

        for seed in seed_names:
            seed_cells[seed] = {}
            for cell in cells:
                data = archives[(cell, seed)]
                aligned = {
                    "sentence_id": data["sentence_id"],
                    "ref_id": data["ref_id"],
                    "image_id": data["image_id"],
                }
                reference = {
                    "sentence_id": canonical_ids,
                    "ref_id": ref["ref_id"],
                    "image_id": image_ids,
                }
                assert_row_alignment(
                    reference, aligned, required_fields=("ref_id", "image_id"),
                    candidate_name=f"Phase1F/{cell}/{seed}",
                )
                for model, (score_field, old_model) in model_fields.items():
                    condition = f"{cell}__{model}"
                    seed_cells[seed][condition] = (
                        np.asarray(data[score_field], dtype=np.float64),
                        np.asarray(data["correct"], dtype=np.float64),
                    )
                    row = old_metrics[
                        (old_metrics["scorer"] == seed) & (old_metrics["cell"] == cell)
                        & (old_metrics["model"] == old_model)
                    ]
                    if len(row) != 1:
                        raise ValueError(f"Expected one historical Phase1F metric row for {seed}/{cell}/{old_model}")
                    for metric in CI_METRICS:
                        name = f"{condition}::{metric}"
                        legacy_points.setdefault(name, {})[seed] = _legacy_metric_value(row.iloc[0], metric)
                        metric_by_name[name] = metric

            for cell in cells:
                old_row = old_increment[(old_increment["scorer"] == seed) & (old_increment["cell"] == cell)]
                if len(old_row) != 1:
                    raise ValueError(f"Expected one historical Phase1F semantic increment row for {seed}/{cell}")
                old_inc = old_row.iloc[0]
                for metric, old_column, low_column, high_column in (
                    ("auroc_correct", "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high"),
                    ("e_aurc", "delta_e_aurc", "delta_e_aurc_ci_low", "delta_e_aurc_ci_high"),
                    ("rer_at_50", "rer50_gain_pp", "rer50_gain_pp_ci_low", "rer50_gain_pp_ci_high"),
                    ("rer_at_80", "rer80_gain_pp", "rer80_gain_pp_ci_low", "rer80_gain_pp_ci_high"),
                ):
                    effect_name = f"semantic_minus_stats__{cell}::{metric}"
                    old_value = _float_or_nan(old_inc.get(old_column))
                    if np.isfinite(old_value):
                        # Phase1F's RER increment columns are already in percentage points.
                        legacy_points[effect_name] = {seed: old_value}
                        low = _float_or_nan(old_inc.get(low_column))
                        high = _float_or_nan(old_inc.get(high_column))
                        if np.isfinite(low) and np.isfinite(high):
                            old_ci.setdefault(effect_name, {})[seed] = [low, high]
                reduction_name = f"e_aurc_reduction__{cell}"
                reduction = _float_or_nan(old_inc.get("e_aurc_reduction"))
                if np.isfinite(reduction):
                    legacy_points[reduction_name] = {seed: reduction}
                    low = _float_or_nan(old_inc.get("e_aurc_reduction_ci_low"))
                    high = _float_or_nan(old_inc.get("e_aurc_reduction_ci_high"))
                    if np.isfinite(low) and np.isfinite(high):
                        old_ci.setdefault(reduction_name, {})[seed] = [low, high]

            if "rand5" in cells and "hard5" in cells:
                comparisons = (("hard5", "rand5", 5),)
            elif "rand10" in cells and "hard10" in cells:
                comparisons = (("hard10", "rand10", 10),)
            else:
                comparisons = ()
            for hard_cell, rand_cell, k in comparisons:
                hard_e1b, hard_stats = f"{hard_cell}__e1b", f"{hard_cell}__stats"
                rand_e1b, rand_stats = f"{rand_cell}__e1b", f"{rand_cell}__stats"
                for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
                    effect_name = f"semantic_minus_stats__{hard_cell}::{metric}"
                    scale = 100.0 if metric.startswith("rer_at_") else 1.0
                    metric_by_name[effect_name] = "rer_gain_pp" if scale == 100.0 else metric
                    if metric == "auroc_correct":
                        inc_metric, old_point_column, low_column, high_column = (
                            "delta_auroc", "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high",
                        )
                    elif metric == "e_aurc":
                        inc_metric, old_point_column, low_column, high_column = (
                            "delta_e_aurc", "delta_e_aurc", "delta_e_aurc_ci_low", "delta_e_aurc_ci_high",
                        )
                    else:
                        inc_metric, old_point_column, low_column, high_column = (
                            "rer50_gain_pp" if metric == "rer_at_50" else "rer80_gain_pp",
                            "rer50_gain_pp" if metric == "rer_at_50" else "rer80_gain_pp",
                            "rer50_gain_pp_ci_low" if metric == "rer_at_50" else "rer80_gain_pp_ci_low",
                            "rer50_gain_pp_ci_high" if metric == "rer_at_50" else "rer80_gain_pp_ci_high",
                        )
                    old_row = old_increment[(old_increment["scorer"] == seed) & (old_increment["cell"] == hard_cell)]
                    old_row = old_row[old_row["regime"] == "same_category"]
                    if len(old_row) == 1 and metric != "rer_at_80":
                        legacy_points[effect_name] = {seed: _float_or_nan(old_row.iloc[0].get(old_point_column))}
                        if low_column in old_row.columns and high_column in old_row.columns:
                            old_ci.setdefault(effect_name, {})[seed] = [
                                _float_or_nan(old_row.iloc[0].get(low_column)),
                                _float_or_nan(old_row.iloc[0].get(high_column)),
                            ]

                reduction_name = f"e_aurc_reduction__{hard_cell}"
                metric_by_name[reduction_name] = "e_aurc_reduction"
                old_row = old_increment[(old_increment["scorer"] == seed) & (old_increment["cell"] == hard_cell)]
                if len(old_row) == 1:
                    legacy_points[reduction_name] = {seed: _float_or_nan(old_row.iloc[0].get("e_aurc_reduction"))}
                    old_ci.setdefault(reduction_name, {})[seed] = [
                        _float_or_nan(old_row.iloc[0].get("e_aurc_reduction_ci_low")),
                        _float_or_nan(old_row.iloc[0].get("e_aurc_reduction_ci_high")),
                    ]

                if k == 5:
                    hard_dod = f"gate_dod_samecat5::{metric}"
                    metric_dod = "rer_at_50" if metric == "rer_at_50" else metric
                    if metric in ("auroc_correct", "rer_at_50"):
                        pass
                    elif metric == "e_aurc":
                        continue
                    metric_by_name[hard_dod] = "rer_gain_pp" if metric == "rer_at_50" else metric
                    old_dod = old_paired[(old_paired["scorer"] == seed) & (old_paired["row_kind"] == "diff_of_diffs")
                                         & (old_paired["metric"] == ("rer50_gain_pp" if metric == "rer_at_50" else metric))
                                         & (old_paired["pair"].astype(str) == "5")]
                    if len(old_dod) == 1 and metric in ("auroc_correct", "rer_at_50"):
                        legacy_points[hard_dod] = {seed: _float_or_nan(old_dod.iloc[0].get("diff"))}
                        old_ci.setdefault(hard_dod, {})[seed] = [
                            _float_or_nan(old_dod.iloc[0].get("ci_low")),
                            _float_or_nan(old_dod.iloc[0].get("ci_high")),
                        ]

                if k == 5:
                    reduction_dod = "gate_dod_samecat5::e_aurc_reduction"
                    metric_by_name[reduction_dod] = "e_aurc_reduction"
                    old_dod = old_paired[(old_paired["scorer"] == seed) & (old_paired["row_kind"] == "diff_of_diffs")
                                         & (old_paired["metric"] == "e_aurc_reduction")
                                         & (old_paired["pair"].astype(str) == "5")]
                    if len(old_dod) == 1:
                        legacy_points[reduction_dod] = {seed: _float_or_nan(old_dod.iloc[0].get("diff"))}
                        old_ci.setdefault(reduction_dod, {})[seed] = [
                            _float_or_nan(old_dod.iloc[0].get("ci_low")),
                            _float_or_nan(old_dod.iloc[0].get("ci_high")),
                        ]

        for cell in cells:
            for model in model_fields:
                name = f"{cell}__{model}"
                for metric in CI_METRICS:
                    specs.append({"name": f"{name}::{metric}", "metric": metric, "operation": "condition", "condition": name})
                    metric_by_name[f"{name}::{metric}"] = metric
            e1b, stats = f"{cell}__e1b", f"{cell}__stats"
            for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
                effect_name = f"semantic_minus_stats__{cell}::{metric}"
                specs.append({
                    "name": effect_name, "metric": metric, "operation": "difference",
                    "a": e1b, "b": stats, "scale": 100.0 if metric.startswith("rer_at_") else 1.0,
                })
                metric_by_name[effect_name] = "rer_gain_pp" if metric.startswith("rer_at_") else metric
            reduction_name = f"e_aurc_reduction__{cell}"
            specs.append({
                "name": reduction_name, "metric": "e_aurc", "operation": "relative_contrast",
                "contrast_positive": stats, "contrast_negative": e1b, "denominator_condition": stats,
            })
            metric_by_name[reduction_name] = "e_aurc_reduction"
        comparison = (("hard5", "rand5", 5),) if group_name == "k5" else (
            (("hard10", "rand10", 10),) if group_name == "k10" else ()
        )
        for hard_cell, rand_cell, k in comparison:
            hard_e1b, hard_stats = f"{hard_cell}__e1b", f"{hard_cell}__stats"
            rand_e1b, rand_stats = f"{rand_cell}__e1b", f"{rand_cell}__stats"
            for metric in ("auroc_correct", "rer_at_50"):
                name = f"gate_dod_samecat{k}::{metric}"
                specs.append({
                    "name": name, "metric": metric, "operation": "difference_of_differences",
                    "a": hard_e1b, "b": hard_stats, "c": rand_e1b, "d": rand_stats,
                    "scale": 100.0 if metric == "rer_at_50" else 1.0,
                })
                metric_by_name[name] = "rer_gain_pp" if metric == "rer_at_50" else metric
            if k == 5:
                name = "gate_dod_samecat5::e_aurc_reduction"
                specs.append({
                    "name": name, "metric": "e_aurc", "operation": "difference_of_relative_contrasts",
                    "positive_a": hard_stats, "negative_a": hard_e1b, "denominator_a": hard_stats,
                    "positive_b": rand_stats, "negative_b": rand_e1b, "denominator_b": rand_stats,
                })
                metric_by_name[name] = "e_aurc_reduction"

        jobs.append({
            "job_id": f"phase1f_{group_name}_{cohort}", "scope": "Phase1F_hard_semantic",
            "cohort": cohort, "image_ids": image_ids, "seed_cells": seed_cells,
            "estimates": specs, "metrics": CI_METRICS, "legacy_points": legacy_points,
            "old_ci": old_ci, "old_gate": old_gate, "thresholds": thresholds,
            "metric_by_name": metric_by_name, "source_paths": list(dict.fromkeys(source_paths)),
            "cohort_construction": f"frozen_{cohort}_prediction_cells",
            "cohort_metadata": cohort_metadata,
            "legacy_pooled_anchor": True,
            "legacy_cohort_classification": "EXISTING_LEGACY_PHASE1F_COHORT",
            "legacy_cohort_note": "sentence_id, ref_id, and image_id are identical row-by-row across all conditions and fixed B3 seeds; correctness remains condition-specific.",
            "legacy_pool_status": "EXISTING_LEGACY_COHORT",
            "sentence_id_status": "VERIFIED_NATIVE_ID",
            "evidence": f"All {len(cells) * len(seed_names)} phase1f prediction NPZs row-aligned on sentence_id/ref_id/image_id; condition-specific correctness and repeated ref IDs are retained.",
            "recovery_status": "READ_ONLY_PREDICTION_ARTIFACT",
        })
    return jobs


def build_refcocog_jobs() -> list[dict[str, Any]]:
    root = ROOT / "results" / "phase1e_refcocog_external"
    pred_root = root / "predictions"
    result_json = read_json(root / "a11_results.json")
    source_paths = [
        root / "a11_results.json", root / "a11_report.json", root / "anchor_recovery.json",
        root / "main_table.md", root / "cohort" / "cohort.csv",
    ]
    cells = ("rand5", "hard5")
    seeds = tuple(f"b3_seed{i}" for i in (1, 2, 3))
    model_fields = {"msp": "conf_msp", "stats": "conf_stats", "e1b": "conf_e1b"}
    data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for cell in cells:
        for seed in seeds:
            path = pred_root / f"external__{seed}__{cell}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen RefCOCOg external prediction: {path}")
            data[(cell, seed)] = load_npz(path)
            source_paths.append(path)
    ref = data[("rand5", seeds[0])]
    sentence_ids = ref["sentence_id"]
    if np.unique(sentence_ids).size != sentence_ids.size:
        raise ValueError("RefCOCOg prediction archive contains duplicate sentence_id values")
    image_ids = ref["image_id"].copy()
    cohort_metadata = _cohort_metadata_from_counts(
        {"external_strict_test": int(sentence_ids.size)},
        historical_pool_identity="RefCOCOg external strict cohort; rows come from the independently curated external cohort, not internal train/validation splits",
        heldout_test_only=True,
    )
    seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    legacy_points: dict[str, dict[str, float]] = {}
    old_ci: dict[str, dict[str, list[float]]] = {}
    specs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_cells[seed] = {}
        for cell in cells:
            cur = data[(cell, seed)]
            assert_row_alignment(
                {"sentence_id": sentence_ids, "ref_id": ref["ref_id"], "image_id": image_ids},
                {"sentence_id": cur["sentence_id"], "ref_id": cur["ref_id"], "image_id": cur["image_id"]},
                required_fields=("ref_id", "image_id"), candidate_name=f"RefCOCOg/{cell}/{seed}",
            )
            old_regime = result_json.get("per_seed", {}).get(seed, {}).get(
                "random" if cell == "rand5" else "same_category", {}
            )
            for model, field in model_fields.items():
                condition = f"{cell}__{model}"
                seed_cells[seed][condition] = (
                    np.asarray(cur[field], dtype=np.float64), np.asarray(cur["correct"], dtype=np.float64),
                )
                legacy_model = {"stats": "stats", "e1b": "e1b"}.get(model)
                legacy_columns = {
                    "accuracy": "accuracy",
                    "auroc_correct": f"auroc_{legacy_model}" if legacy_model else None,
                    "e_aurc": f"e_aurc_{legacy_model}" if legacy_model else None,
                    "rer_at_50": f"rer50_{legacy_model}" if legacy_model else None,
                }
                for metric, column in legacy_columns.items():
                    if column is not None:
                        value = _float_or_nan(old_regime.get(column))
                        if np.isfinite(value):
                            legacy_points.setdefault(f"{condition}::{metric}", {})[seed] = value
    for cell in cells:
        for model in model_fields:
            condition = f"{cell}__{model}"
            for metric in CI_METRICS:
                specs.append({"name": f"{condition}::{metric}", "metric": metric,
                              "operation": "condition", "condition": condition})
    for cell in cells:
        specs.extend([
            {
                "name": f"semantic_increment_{cell}::auroc_correct", "metric": "auroc_correct",
                "operation": "difference", "a": f"{cell}__e1b", "b": f"{cell}__stats",
            },
            {
                "name": f"semantic_increment_{cell}::rer50_gain_pp", "metric": "rer_at_50",
                "operation": "difference", "a": f"{cell}__e1b", "b": f"{cell}__stats", "scale": 100.0,
            },
            {
                "name": f"e_aurc_reduction__{cell}", "metric": "e_aurc", "operation": "relative_contrast",
                "contrast_positive": f"{cell}__stats", "contrast_negative": f"{cell}__e1b",
                "denominator_condition": f"{cell}__stats",
            },
        ])
    specs.append({
        "name": "hard_minus_random_increment::auroc_correct", "metric": "auroc_correct",
        "operation": "difference_of_differences", "a": "hard5__e1b", "b": "hard5__stats",
        "c": "rand5__e1b", "d": "rand5__stats",
    })
    specs.extend([
        {
            "name": "hard_minus_random_increment::e_aurc_reduction", "metric": "e_aurc",
            "operation": "difference_of_relative_contrasts", "positive_a": "hard5__stats",
            "negative_a": "hard5__e1b", "denominator_a": "hard5__stats",
            "positive_b": "rand5__stats", "negative_b": "rand5__e1b", "denominator_b": "rand5__stats",
        },
        {
            "name": "hard_minus_random_increment::rer50_gain_pp", "metric": "rer_at_50",
            "operation": "difference_of_differences", "a": "hard5__e1b", "b": "hard5__stats",
            "c": "rand5__e1b", "d": "rand5__stats", "scale": 100.0,
        },
    ])
    for seed in seeds:
        seed_rows = result_json.get("per_seed", {}).get(seed, {})
        for cell, old_cell in (("rand5", "random"), ("hard5", "same_category")):
            old_row = seed_rows.get(old_cell, {})
            for name, value_col, low_col, high_col in (
                (f"semantic_increment_{cell}::auroc_correct", "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high"),
                (f"e_aurc_reduction__{cell}", "e_aurc_reduction", "e_aurc_reduction_ci_low", "e_aurc_reduction_ci_high"),
                (f"semantic_increment_{cell}::rer50_gain_pp", "rer50_gain_pp", "rer50_gain_ci_low_pp", "rer50_gain_ci_high_pp"),
            ):
                value = _float_or_nan(old_row.get(value_col))
                if np.isfinite(value):
                    legacy_points.setdefault(name, {})[seed] = value
                    low, high = _float_or_nan(old_row.get(low_col)), _float_or_nan(old_row.get(high_col))
                    if np.isfinite(low) and np.isfinite(high):
                        old_ci.setdefault(name, {})[seed] = [low, high]
        amplification = seed_rows.get("amplification", {})
        for name, value_col, low_col, high_col in (
            ("hard_minus_random_increment::auroc_correct", "ampl_auroc", "ampl_auroc_ci_low", "ampl_auroc_ci_high"),
            ("hard_minus_random_increment::e_aurc_reduction", "ampl_eaurc_reduction", "ampl_eaurc_ci_low", "ampl_eaurc_ci_high"),
            ("hard_minus_random_increment::rer50_gain_pp", "ampl_rer50_gain_pp", "ampl_rer50_ci_low_pp", "ampl_rer50_ci_high_pp"),
        ):
            value = _float_or_nan(amplification.get(value_col))
            if np.isfinite(value):
                legacy_points.setdefault(name, {})[seed] = value
                low, high = _float_or_nan(amplification.get(low_col)), _float_or_nan(amplification.get(high_col))
                if np.isfinite(low) and np.isfinite(high):
                    old_ci.setdefault(name, {})[seed] = [low, high]
    job = {
        "job_id": "refcocog_external_rand5_hard5", "scope": "RefCOCOg_external",
        "cohort": "external_strict", "image_ids": image_ids, "seed_cells": seed_cells,
        "estimates": specs, "metrics": CI_METRICS, "legacy_points": legacy_points, "old_ci": old_ci,
        "cohort_metadata": cohort_metadata,
        "old_gate": result_json.get("full_verdict", {}).get("full_verdict", "NOT_RECORDED"),
        "thresholds": result_json.get("thresholds", {}),
        "source_paths": list(dict.fromkeys(source_paths)),
        "cohort_construction": "strict_refcocog_image_disjoint_external_subset",
        "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_LEGACY_EXTERNAL_COHORT",
        "legacy_cohort_note": "Separate RefCOCOg external cohort; no RefCOCO+ overlap by the frozen cohort manifest.",
        "legacy_pool_status": "EXISTING_LEGACY_EXTERNAL_COHORT",
        "sentence_id_status": "VERIFIED_NATIVE_ID",
        "evidence": "Six frozen external prediction NPZs aligned row-by-row on sentence_id/ref_id/image_id; correctness remains seed/cell-specific.",
        "recovery_status": "READ_ONLY_PREDICTION_ARTIFACT",
    }
    return [job]


def build_v2m_m1_jobs() -> list[dict[str, Any]]:
    """Replay the frozen M1/LCR confidence archives on their original image rows.

    The M1 confidence NPZ omits IDs. Its runner creates Random/K20/K50 cells from
    sentence-sorted embedding stores and creates the matched/hard cells in the
    Phase1F same4 order. We recover those IDs from the immutable Phase0B and
    Phase1F prediction archives and require per-row correctness and point-table
    agreement before a job can be emitted.
    """
    root = ROOT / "results" / "v2_local_competition"
    m1_root = root / "m1_random_only"
    point_path = m1_root / "point_metrics.csv"
    pair_path = root / "bootstrap" / "pairs.csv"
    gate_path = root / "gate.json"
    protocol_path = root / "protocol.json"
    old_points = read_csv(point_path)
    old_pairs = read_csv(pair_path)
    old_gate_doc = read_json(gate_path)
    protocol = read_json(protocol_path)
    seeds = tuple(f"b3_seed{s}" for s in (1, 2, 3))
    cohorts = (
        "Random-K5", "Random-K5-matched", "SameCategory-K5", "K20-random", "K50-random",
    )
    models = ("R1", "E1b", "Aggregate-MLP", "LCR-noGate", "LCR")
    pair_plan = {
        "Random-K5": ("E1b", "Aggregate-MLP", "LCR-noGate", "R1"),
        "Random-K5-matched": ("E1b", "Aggregate-MLP"),
        "SameCategory-K5": ("E1b", "Aggregate-MLP", "LCR-noGate", "R1"),
        "K20-random": ("E1b", "Aggregate-MLP"),
        "K50-random": ("E1b", "Aggregate-MLP"),
    }
    phase0b: dict[str, dict[str, np.ndarray]] = {}
    phase1f: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    source_paths: list[Path] = [point_path, pair_path, gate_path, protocol_path]
    for seed in seeds:
        seed_num = int(seed[-1])
        path = ROOT / "results" / "phase0b_independent" / f"seed_{seed_num}" / "per_sentence_predictions.npz"
        phase0b[seed] = load_npz(path)
        source_paths.append(path)
        for cell in ("rand5", "hard5"):
            p = ROOT / "results" / "phase1f_hard_semantic" / "predictions" / f"{cell}__{seed}.npz"
            phase1f[(cell, seed)] = load_npz(p)
            source_paths.append(p)
    if not all(np.all(np.diff(np.asarray(phase0b[seed]["sentence_ids"], dtype=np.int64)) > 0) for seed in seeds):
        raise ValueError("M1 bridge requires Phase0B scorer rows strictly sorted by sentence_id")
    for seed in seeds:
        rand = phase1f[("rand5", seed)]
        hard = phase1f[("hard5", seed)]
        assert_row_alignment(
            {"sentence_id": rand["sentence_id"], "ref_id": rand["ref_id"], "image_id": rand["image_id"]},
            {"sentence_id": hard["sentence_id"], "ref_id": hard["ref_id"], "image_id": hard["image_id"]},
            required_fields=("ref_id", "image_id"), candidate_name=f"M1 matched/hard/{seed}",
        )
        if not np.all(np.diff(np.asarray(rand["sentence_id"], dtype=np.int64)) > 0):
            raise ValueError(f"M1 matched Phase1F {seed} IDs are not strictly sorted")

    jobs: list[dict[str, Any]] = []
    for cohort in cohorts:
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        legacy_points: dict[str, dict[str, float]] = {}
        legacy_anchor_unavailable: list[dict[str, Any]] = []
        old_ci: dict[str, dict[str, list[float]]] = {}
        specs: list[dict[str, Any]] = []
        ref_ids: np.ndarray | None = None
        image_ids: np.ndarray | None = None
        recovered_ref_ids: np.ndarray | None = None
        recovered_splits: np.ndarray | None = None
        row_n = None
        for seed in seeds:
            data = phase0b[seed]
            m1_path = m1_root / f"seed_{seed[-1]}" / "confidences.npz"
            m1 = load_npz(m1_path)
            source_paths.append(m1_path)
            if cohort in ("Random-K5", "K20-random", "K50-random"):
                k = {"Random-K5": 5, "K20-random": 20, "K50-random": 50}[cohort]
                mask = np.isin(np.asarray(data["eval_split"]).astype(str), ("testA", "testB"))
                idx = np.flatnonzero(mask)
                ids = np.asarray(data["sentence_ids"], dtype=np.int64)[idx]
                image = np.asarray(data["image_ids"], dtype=np.int64)[idx]
                correct = np.asarray(data[f"correct_K{k}"], dtype=np.float64)[idx]
                if not np.all(np.diff(ids) > 0):
                    raise ValueError(f"M1 {cohort}/{seed}: source rows not sentence-sorted")
            else:
                cell = "rand5" if cohort == "Random-K5-matched" else "hard5"
                frozen = phase1f[(cell, seed)]
                ids = np.asarray(frozen["sentence_id"], dtype=np.int64)
                image = np.asarray(frozen["image_id"], dtype=np.int64)
                correct = np.asarray(frozen["correct"], dtype=np.float64)
                if cohort == "Random-K5-matched":
                    base_ids = np.asarray(data["sentence_ids"], dtype=np.int64)
                    pos = np.searchsorted(base_ids, ids)
                    if np.any(pos >= base_ids.size) or not np.array_equal(base_ids[pos], ids):
                        raise ValueError(f"M1 matched IDs are absent from Phase0B scorer rows for {seed}")
                    if not np.array_equal(np.asarray(data["correct_K5"])[pos], correct):
                        raise ValueError(f"M1 matched random correctness does not match Phase0B for {seed}")
                else:
                    # The frozen hard5 archive is the runner's same4 evaluation order.
                    if not np.array_equal(np.asarray(frozen["correct"], dtype=np.float64), correct):
                        raise ValueError(f"M1 hard5 correctness mismatch for {seed}")
            if cohort in ("Random-K5", "K20-random", "K50-random"):
                ref = np.asarray(data["ref_ids"], dtype=np.int64)[idx]
            else:
                ref = np.asarray(frozen["ref_id"], dtype=np.int64)
            phase0b_ids = np.asarray(data["sentence_ids"], dtype=np.int64)
            split_pos = np.searchsorted(phase0b_ids, ids)
            if np.any(split_pos >= phase0b_ids.size) or not np.array_equal(phase0b_ids[split_pos], ids):
                raise ValueError(f"M1 {cohort}/{seed}: sentence IDs cannot be bridged to Phase0B split rows")
            current_splits = np.asarray(data["eval_split"]).astype(str)[split_pos]
            if row_n is None:
                row_n = ids.size
                ref_ids, image_ids = ids.copy(), image.copy()
                recovered_ref_ids = ref.copy()
                recovered_splits = current_splits.copy()
            elif (not np.array_equal(ref_ids, ids) or not np.array_equal(image_ids, image)
                  or not np.array_equal(recovered_ref_ids, ref)
                  or not np.array_equal(recovered_splits, current_splits)):
                raise ValueError(f"M1 {cohort}: seed identity rows do not align")

            archive_cohort = cohort
            archive_correct = np.asarray(m1[f"{archive_cohort}__correct"], dtype=np.float64)
            if archive_correct.shape != correct.shape or not np.array_equal(archive_correct, correct):
                raise ValueError(
                    f"M1 {cohort}/{seed}: archived correctness is not row-aligned to recovered IDs"
                )
            seed_cells[seed] = {}
            for model in models:
                confidence = np.asarray(m1[f"{cohort}__{model}"], dtype=np.float64)
                if confidence.shape != correct.shape:
                    raise ValueError(f"M1 {cohort}/{seed}/{model}: confidence length mismatch")
                seed_cells[seed][f"{cohort}__{model}"] = (confidence, archive_correct)
                old_row = old_points[
                    (old_points["cohort"] == cohort)
                    & (old_points["seed"].astype(int) == int(seed[-1]))
                    & (old_points["model"] == model)
                ]
                if len(old_row) != 1:
                    raise ValueError(f"M1 old point row missing/ambiguous: {cohort}/{seed}/{model}")
                row = old_row.iloc[0]
                if int(row["n"]) != ids.size or int(row["n_images"]) != np.unique(image).size:
                    raise ValueError(f"M1 old count anchor mismatch: {cohort}/{seed}/{model}")
                for metric, col in (
                    ("accuracy", "b3_accuracy"), ("auroc_correct", "auroc_correct"),
                    ("e_aurc", "e_aurc"), ("rer_at_50", "rer_at_50"),
                    ("ece_adaptive", "ece_adaptive"), ("brier_binary", "brier_binary"),
                    ("nll_binary", "nll_binary"),
                ):
                    val = _float_or_nan(row.get(col))
                    if np.isfinite(val):
                        legacy_points.setdefault(f"{cohort}__{model}::{metric}", {})[seed] = val
                    else:
                        legacy_anchor_unavailable.append({
                            "estimate": f"{cohort}__{model}::{metric}", "seed": seed,
                            "reason": "legacy_point_missing_or_nonfinite",
                        })
                for metric in ("rer_at_80", "rer_at_90"):
                    legacy_anchor_unavailable.append({
                        "estimate": f"{cohort}__{model}::{metric}", "seed": seed,
                        "reason": "legacy_point_metric_not_reported_in_M1_point_metrics.csv",
                    })

        assert ref_ids is not None and image_ids is not None
        for model in models:
            for metric in CI_METRICS:
                name = f"{cohort}__{model}::{metric}"
                specs.append({"name": name, "metric": metric, "operation": "condition",
                              "condition": f"{cohort}__{model}"})
        for competitor in pair_plan[cohort]:
            for metric in ("auroc_correct", "e_aurc", "rer_at_50"):
                name = f"LCR_minus_{competitor}::{metric}"
                specs.append({"name": name, "metric": metric, "operation": "difference",
                              "a": f"{cohort}__LCR", "b": f"{cohort}__{competitor}",
                              "scale": 100.0 if metric == "rer_at_50" else 1.0})
                rows = old_pairs[
                    (old_pairs["cohort"] == cohort)
                    & (old_pairs["seed"].astype(int).isin([int(seed[-1]) for seed in seeds]))
                    & (old_pairs["model_a"] == "LCR") & (old_pairs["model_b"] == competitor)
                    & (old_pairs["metric"] == metric)
                ]
                if len(rows) != 3:
                    raise ValueError(f"M1 paired bootstrap rows missing/ambiguous: {cohort}/LCR-{competitor}/{metric}")
                for _, row in rows.iterrows():
                    seed = f"b3_seed{int(row['seed'])}"
                    scale = 100.0 if metric == "rer_at_50" else 1.0
                    legacy_points.setdefault(name, {})[seed] = float(row["diff"]) * scale
                    old_ci.setdefault(name, {})[seed] = [float(row["ci_low"]) * scale, float(row["ci_high"]) * scale]
        rel_name = f"LCR_E1b_relative_eaurc_reduction::{cohort}"
        specs.append({"name": rel_name, "metric": "e_aurc", "operation": "relative_contrast",
                      "contrast_positive": f"{cohort}__E1b", "contrast_negative": f"{cohort}__LCR",
                      "denominator_condition": f"{cohort}__E1b"})
        # This is the exact historical per-seed derived point used by GO/STRONG.
        for seed in seeds:
            sub = old_points[
                (old_points["cohort"] == cohort)
                & (old_points["seed"].astype(int) == int(seed[-1]))
                & (old_points["model"].isin(("LCR", "E1b")))
            ].set_index("model")
            denominator = float(sub.loc["E1b", "e_aurc"])
            legacy_points.setdefault(rel_name, {})[seed] = (
                (denominator - float(sub.loc["LCR", "e_aurc"])) / denominator
                if denominator > 1e-12 else float("nan")
            )

        old_gate = old_gate_doc.get("gates", {})
        assert recovered_splits is not None
        split_labels, split_sizes = np.unique(recovered_splits, return_counts=True)
        split_counts = {str(name): int(count) for name, count in zip(split_labels, split_sizes)}
        split_names = set(split_counts)
        jobs.append({
            "job_id": "m1_" + cohort.replace("-", "_").lower(),
            "scope": "V2M_M1_LCR", "cohort": cohort, "image_ids": image_ids,
            "seed_cells": seed_cells, "estimates": specs, "metrics": CI_METRICS,
            "legacy_points": legacy_points, "old_ci": old_ci,
            "legacy_anchor_unavailable": legacy_anchor_unavailable,
            "old_gate": old_gate.get("M1_GO", {}).get("verdict", "NOT_RECORDED") if cohort == "SameCategory-K5" else "NOT_APPLICABLE",
            "thresholds": protocol.get("M1_gates", {}),
            "source_paths": list(dict.fromkeys(source_paths + [m1_root / f"seed_{s[-1]}" / "model_manifest.json" for s in seeds])),
            "cohort_metadata": {"split_counts": split_counts,
                                "contains_train": any(name.lower().startswith("train") for name in split_names),
                                "contains_validation": any(name.lower().startswith(("val", "valid")) for name in split_names),
                                "heldout_test_only": bool(split_names) and split_names <= {"testA", "testB"},
                                "historical_pool_identity": f"{cohort} frozen runner rows; sentence IDs and ref/image rows bridged to Phase0B eval_split"},
            "cohort_construction": "frozen_M1_runner_cohort_with_IDs_reconstructed_from_sorted_Phase0B_and_Phase1F_same4",
            "legacy_pooled_anchor": False, "legacy_cohort_classification": "EXISTING_LEGACY_M1_COHORT",
            "legacy_cohort_note": "The archived M1 confidence NPZ lacks sentence/image IDs. The frozen runner's sorted-store selection and Phase1F same4 order reconstruct IDs; exact correctness sequences, row/image counts, and finite historical per-model/pair points must agree before use.",
            "legacy_pool_status": "EXISTING_LEGACY_COHORT",
            "sentence_id_status": "RECONSTRUCTED_FROM_FROZEN_RUNNER_ORDER_AND_VERIFIED_CORRECTNESS",
            "evidence": "M1 confidence rows are mapped using run_v2m_m1._load_store_bundle/_cohort_data; random rows use Phase0B sentence-sorted per-row predictions, matched/hard rows use aligned Phase1F rand5/hard5 archives; all archived correctness and historical point rows are checked.",
            "recovery_status": "READ_ONLY_CONFIDENCE_ARCHIVE_WITH_VERIFIED_ID_BRIDGE",
            "model_identity_by_seed": {
                seed: {"scorer": seed, "model_manifest": f"results/v2_local_competition/m1_random_only/seed_{seed[-1]}/model_manifest.json"}
                for seed in seeds
            },
            "prediction_source_precision": "archived_float64_confidences_and_uint8_correctness",
        })
    return jobs


def build_d1_jobs() -> list[dict[str, Any]]:
    """Build D1 clean-annotation C1/C4 jobs from frozen row-level sources.

    The clean manifest intentionally includes validation and held-out rows.
    C1 is split into the single fixed cosine B0 scorer and three trained B3
    seeds so one frozen B0 model is never misreported as three independent
    trained seeds.
    """
    root = ROOT / "results" / "v2_data_robustness" / "d1_reviewed_annotations"
    clean_path = root / "clean_manifest.csv"
    c1_path, c4_path, pairs_path, verdict_path = (
        root / "c1_cardinality.csv", root / "c4_hard_semantic.csv", root / "bootstrap.csv", root / "verdict.json"
    )
    clean = read_csv(clean_path)
    c1 = read_csv(c1_path)
    c4 = read_csv(c4_path)
    pairs = read_csv(pairs_path)
    verdict = read_json(verdict_path)
    clean = clean[clean["in_phase0b_cohort"].astype(bool)].copy()
    clean = clean.sort_values("sent_id", kind="stable").reset_index(drop=True)
    clean_ids = clean["sent_id"].astype(np.int64).to_numpy()
    if np.unique(clean_ids).size != clean_ids.size:
        raise ValueError("D1 clean manifest contains duplicate sentence IDs in the frozen Phase0B cohort")
    clean_set = set(clean_ids.tolist())
    clean_images = clean["image_id"].astype(np.int64).to_numpy()
    clean_refs = clean["ref_id"].astype(np.int64).to_numpy()
    clean_splits = clean["split"].astype(str).to_numpy()
    split_counts = {str(name): int(count) for name, count in zip(*np.unique(clean_splits, return_counts=True))}
    source_paths: list[Path] = [clean_path, c1_path, c4_path, pairs_path, verdict_path]
    phase0a_path = ROOT / "results" / "phase0a_corrected" / "per_sentence_predictions.npz"
    b0 = load_npz(phase0a_path)
    source_paths.append(phase0a_path)

    b0_idx = np.flatnonzero(np.isin(np.asarray(b0["sentence_id"], dtype=np.int64), clean_ids))
    b0_idx = b0_idx[np.argsort(np.asarray(b0["sentence_id"], dtype=np.int64)[b0_idx], kind="stable")]
    b0_ids = np.asarray(b0["sentence_id"], dtype=np.int64)[b0_idx]
    if not np.array_equal(b0_ids, clean_ids):
        raise ValueError("D1 B0 C1 rows are not a complete exact sentence-ID match to clean_manifest")
    if not np.array_equal(np.asarray(b0["image_id"], dtype=np.int64)[b0_idx], clean_images):
        raise ValueError("D1 B0 C1 row image IDs differ from clean_manifest")
    if not np.array_equal(np.asarray(b0["ref_id"], dtype=np.int64)[b0_idx], clean_refs):
        raise ValueError("D1 B0 C1 row ref IDs differ from clean_manifest")

    def make_c1_job(model: str, seed_names: tuple[str, ...], source: dict[str, Any], model_label: str) -> dict[str, Any]:
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        legacy_points: dict[str, dict[str, float]] = {}
        old_ci: dict[str, dict[str, list[float]]] = {}
        unavailable: list[dict[str, Any]] = []
        specs: list[dict[str, Any]] = []
        if model == "cosine_b0":
            seed = seed_names[0]
            seed_cells[seed] = {}
            for k in (5, 10, 20, 50):
                cond = f"K{k}"
                conf = np.asarray(b0[f"confidence_global_T_K{k}"], dtype=np.float64)[b0_idx]
                corr = np.asarray(b0[f"correct_K{k}"], dtype=np.float64)[b0_idx]
                seed_cells[seed][cond] = (conf, corr)
                oldrow = c1[(c1["model"] == model) & (c1["seed"] == "seed1") & (c1["K"].astype(int) == k)]
                if len(oldrow) != 1:
                    raise ValueError(f"D1 C1 old B0 point row missing/ambiguous: K{k}")
                row = oldrow.iloc[0]
                if int(row["n"]) != len(clean_ids) or int(row["n_images"]) != np.unique(clean_images).size:
                    raise ValueError(f"D1 C1 B0 count mismatch at K{k}")
                for metric, col in (("accuracy", "accuracy"), ("auroc_correct", "auroc_correct"),
                                    ("e_aurc", "e_aurc"), ("rer_at_50", "rer_at_50")):
                    legacy_points.setdefault(f"{cond}::{metric}", {})[seed] = float(row[col])
                for metric in set(CI_METRICS) - {"accuracy", "auroc_correct", "e_aurc", "rer_at_50"}:
                    unavailable.append({"estimate": f"{cond}::{metric}", "seed": seed,
                                        "reason": "historical_D1_C1_table_did_not_report_metric"})
        else:
            for seed in seed_names:
                seed_number = int(seed[-1])
                p = ROOT / "results" / "phase0b_independent" / f"seed_{seed_number}" / "per_sentence_predictions.npz"
                data = load_npz(p)
                source_paths.append(p)
                ids = np.asarray(data["sentence_ids"], dtype=np.int64)
                idx = np.flatnonzero(np.isin(ids, clean_ids))
                idx = idx[np.argsort(ids[idx], kind="stable")]
                if not np.array_equal(ids[idx], clean_ids):
                    raise ValueError(f"D1 C1 {seed}: B3 rows are not complete clean manifest IDs")
                if (not np.array_equal(np.asarray(data["image_ids"], dtype=np.int64)[idx], clean_images)
                    or not np.array_equal(np.asarray(data["ref_ids"], dtype=np.int64)[idx], clean_refs)):
                    raise ValueError(f"D1 C1 {seed}: B3 row image/ref IDs differ from clean manifest")
                seed_cells[seed] = {}
                for k in (5, 10, 20, 50):
                    cond = f"K{k}"
                    seed_cells[seed][cond] = (
                        np.asarray(data[f"confidence_global_T_corrected_K{k}"], dtype=np.float64)[idx],
                        np.asarray(data[f"correct_K{k}"], dtype=np.float64)[idx],
                    )
                    oldrow = c1[(c1["model"] == "b3") & (c1["seed"] == f"seed{seed_number}")
                                & (c1["K"].astype(int) == k)]
                    if len(oldrow) != 1:
                        raise ValueError(f"D1 C1 old B3 point row missing/ambiguous: {seed}/K{k}")
                    row = oldrow.iloc[0]
                    if int(row["n"]) != len(clean_ids) or int(row["n_images"]) != np.unique(clean_images).size:
                        raise ValueError(f"D1 C1 B3 count mismatch: {seed}/K{k}")
                    for metric, col in (("accuracy", "accuracy"), ("auroc_correct", "auroc_correct"),
                                        ("e_aurc", "e_aurc"), ("rer_at_50", "rer_at_50")):
                        legacy_points.setdefault(f"{cond}::{metric}", {})[seed] = float(row[col])
                    for metric in set(CI_METRICS) - {"accuracy", "auroc_correct", "e_aurc", "rer_at_50"}:
                        unavailable.append({"estimate": f"{cond}::{metric}", "seed": seed,
                                            "reason": "historical_D1_C1_table_did_not_report_metric"})

        for k in (5, 10, 20, 50):
            cond = f"K{k}"
            for metric in CI_METRICS:
                specs.append({"name": f"{cond}::{metric}", "metric": metric,
                              "operation": "condition", "condition": cond})
        for k_hi in (10, 20, 50):
            for metric, suffix, scale in (("auroc_correct", "auroc_correct", 1.0),
                                          ("e_aurc", "e_aurc", 1.0),
                                          ("rer_at_50", "rer_at_50", 1.0)):
                name = f"crossK_K5_minus_K{k_hi}::{suffix}"
                specs.append({"name": name, "metric": metric, "operation": "difference",
                              "a": "K5", "b": f"K{k_hi}", "scale": scale})
                pair = pairs[(pairs["analysis"] == "C1") & (pairs["model"] == model)
                             & (pairs["seed"].isin(["seed1"] if model == "cosine_b0" else [f"seed{int(s[-1])}" for s in seed_names]))
                             & (pairs["K_a"].astype(int) == 5) & (pairs["K_b"].astype(int) == k_hi)
                             & (pairs["variant"] == ("global_T" if model == "cosine_b0" else "global_T_corrected"))
                             & (pairs["metric"] == metric) & (pairs["diff_kind"] == "absolute")]
                if len(pair) != len(seed_names):
                    raise ValueError(f"D1 C1 paired rows missing: {model}/K5-K{k_hi}/{metric}")
                for _, row in pair.iterrows():
                    seed = seed_names[0] if model == "cosine_b0" else f"b3_seed{int(str(row['seed'])[-1])}"
                    legacy_points.setdefault(name, {})[seed] = float(row["diff"])
                    old_ci.setdefault(name, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]

            rel_name = f"crossK_eaurc_relative_K5_minus_K{k_hi}"
            specs.append({"name": rel_name, "metric": "e_aurc", "operation": "relative_contrast",
                          "contrast_positive": "K5", "contrast_negative": f"K{k_hi}",
                          # D1's legacy table/gate defines r=(K5-Khi)/Khi;
                          # retain that K_hi denominator and negative worsening sign.
                          "denominator_condition": f"K{k_hi}"})
            relative = pairs[(pairs["analysis"] == "C1") & (pairs["model"] == model)
                             & (pairs["seed"].isin(["seed1"] if model == "cosine_b0" else [f"seed{int(s[-1])}" for s in seed_names]))
                             & (pairs["K_a"].astype(int) == 5) & (pairs["K_b"].astype(int) == k_hi)
                             & (pairs["variant"] == ("global_T" if model == "cosine_b0" else "global_T_corrected"))
                             & (pairs["metric"] == "e_aurc") & (pairs["diff_kind"] == "relative")]
            if len(relative) != len(seed_names):
                raise ValueError(f"D1 C1 relative E-AURC rows missing: {model}/K5-K{k_hi}")
            for _, row in relative.iterrows():
                seed = seed_names[0] if model == "cosine_b0" else f"b3_seed{int(str(row['seed'])[-1])}"
                legacy_points.setdefault(rel_name, {})[seed] = float(row["diff"])
                old_ci.setdefault(rel_name, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]

        old_gate = verdict.get("c1_cardinality", {}).get("verdict", {}) if model == "b3" else "NOT_APPLICABLE"
        return {
            "job_id": f"d1_c1_{model}_clean", "scope": "D1_clean_robustness", "cohort": "D1_clean_c1",
            "image_ids": clean_images, "seed_cells": seed_cells, "estimates": specs, "metrics": CI_METRICS,
            "legacy_points": legacy_points, "old_ci": old_ci, "legacy_anchor_unavailable": unavailable,
            "old_gate": old_gate, "thresholds": {"absolute_k5_minus_k50_ci": "preserve old sign/inequality from verdict checks"},
            "source_paths": list(dict.fromkeys(source_paths)), "point_order": np.arange(len(clean_ids), dtype=np.int64),
            "cohort_metadata": {"split_counts": split_counts, "contains_train": False,
                                "contains_validation": "val" in split_counts, "heldout_test_only": False,
                                "historical_pool_identity": "clean_manifest rows with in_phase0b_cohort=true; includes validation and testA/testB"},
            "cohort_construction": "clean_manifest_in_phase0b_cohort_true_exact_sentence_id_intersection",
            "legacy_pooled_anchor": True, "legacy_cohort_classification": "D1_CLEAN_VALIDATION_PLUS_TEST_NO_TRAIN",
            "legacy_pool_status": "EXISTING_D1_CLEAN_COHORT",
            "legacy_cohort_note": "D1 C1 is a reviewed-annotation robustness cohort that includes validation rows; retain as robustness analysis and do not call it held-out-only.",
            "sentence_id_status": "VERIFIED_NATIVE_ID_CLEAN_MANIFEST_FILTER",
            "evidence": f"clean manifest IDs (n={len(clean_ids)}) exactly match filtered frozen source IDs; refs/images and old row/image counts verified; old bootstrap anchors preserve K5-K contrast orientation.",
            "recovery_status": "READ_ONLY_NATIVE_ID_PREDICTIONS",
            "model_identity_by_seed": {seed: {"model": model_label, "scorer": seed} for seed in seed_names},
            "prediction_source_precision": "frozen_float32_confidences_and_binary_correctness",
        }

    b0_seed = ("cosine_b0_fixed",)
    b3_seeds = tuple(f"b3_seed{s}" for s in (1, 2, 3))
    c1_b0 = make_c1_job("cosine_b0", b0_seed, b0, "single frozen cosine B0 (not a trained-seed set)")
    c1_b3 = make_c1_job("b3", b3_seeds, {}, "three frozen B3 trained seeds")

    c4_seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    c4_legacy_points: dict[str, dict[str, float]] = {}
    c4_old_ci: dict[str, dict[str, list[float]]] = {}
    c4_unavailable: list[dict[str, Any]] = []
    c4_specs: list[dict[str, Any]] = []
    f_root = ROOT / "results" / "phase1f_hard_semantic" / "predictions"
    for seed in b3_seeds:
        seed_num = int(seed[-1])
        rand_path = f_root / f"rand5__{seed}.npz"
        hard_path = f_root / f"hard5__{seed}.npz"
        rand, hard = load_npz(rand_path), load_npz(hard_path)
        source_paths.extend((rand_path, hard_path))
        assert_row_alignment(
            {"sentence_id": rand["sentence_id"], "ref_id": rand["ref_id"], "image_id": rand["image_id"]},
            {"sentence_id": hard["sentence_id"], "ref_id": hard["ref_id"], "image_id": hard["image_id"]},
            required_fields=("ref_id", "image_id"), candidate_name=f"D1 C4 rand/hard/{seed}",
        )
        ids = np.asarray(rand["sentence_id"], dtype=np.int64)
        keep = np.isin(ids, clean_ids)
        idx = np.flatnonzero(keep)
        order = np.argsort(ids[idx], kind="stable")
        idx = idx[order]
        ids_clean = ids[idx]
        clean_pos = np.searchsorted(clean_ids, ids_clean)
        if np.any(clean_pos >= clean_ids.size) or not np.array_equal(clean_ids[clean_pos], ids_clean):
            raise ValueError(f"D1 C4 clean IDs not exact: {seed}")
        images = np.asarray(rand["image_id"], dtype=np.int64)[idx]
        refs = np.asarray(rand["ref_id"], dtype=np.int64)[idx]
        if not np.array_equal(images, clean_images[clean_pos]) or not np.array_equal(refs, clean_refs[clean_pos]):
            raise ValueError(f"D1 C4 row image/ref mismatch: {seed}")
        c4_seed_cells[seed] = {}
        for cell, data in (("rand", rand), ("hard", hard)):
            correct = np.asarray(data["correct"], dtype=np.float64)[idx]
            for model, field in (("stats", "conf_stats"), ("e1b", "conf_e1b")):
                condition = f"{cell}_{model}"
                c4_seed_cells[seed][condition] = (np.asarray(data[field], dtype=np.float64)[idx], correct)
                oldrow = c4[(c4["scorer"] == seed) & (c4["cell"] == f"{cell}5")
                            & (c4["model"] == ("stats_logistic" if model == "stats" else "e1b_stats_semantic"))]
                if len(oldrow) != 1:
                    raise ValueError(f"D1 C4 old point missing/ambiguous: {seed}/{condition}")
                row = oldrow.iloc[0]
                if int(row["n"]) != ids_clean.size or int(row["n_images"]) != np.unique(images).size:
                    raise ValueError(f"D1 C4 old row/image count mismatch: {seed}/{condition}")
                for metric, col in (("accuracy", "b3_accuracy"), ("auroc_correct", "auroc_correct"),
                                    ("e_aurc", "e_aurc"), ("rer_at_50", "rer_at_50")):
                    c4_legacy_points.setdefault(f"{condition}::{metric}", {})[seed] = float(row[col])
                for metric in set(CI_METRICS) - {"accuracy", "auroc_correct", "e_aurc", "rer_at_50"}:
                    c4_unavailable.append({"estimate": f"{condition}::{metric}", "seed": seed,
                                           "reason": "historical_D1_C4_table_did_not_report_metric"})

    for condition in ("rand_stats", "rand_e1b", "hard_stats", "hard_e1b"):
        for metric in CI_METRICS:
            c4_specs.append({"name": f"{condition}::{metric}", "metric": metric,
                             "operation": "condition", "condition": condition})
    c4_effects = (
        ("d1_c4_hard_semantic_delta_auroc", "auroc_correct", "hard_e1b", "hard_stats", 1.0, "delta_auroc"),
        ("d1_c4_hard_rer50_gain_pp", "rer_at_50", "hard_e1b", "hard_stats", 100.0, "rer50_gain_pp"),
    )
    for name, metric, a, b, scale, old_metric in c4_effects:
        c4_specs.append({"name": name, "metric": metric, "operation": "difference", "a": a, "b": b, "scale": scale})
        rows = pairs[(pairs["analysis"] == "C4") & (pairs["model"] == "e1b_stats_semantic vs stats_logistic")
                     & (pairs["seed"].isin([f"b3_seed{s}" for s in (1, 2, 3)]))
                     & (pairs["variant"] == "hard5") & (pairs["metric"] == old_metric)]
        if len(rows) != 3:
            raise ValueError(f"D1 C4 paired anchor missing for {name}")
        for _, row in rows.iterrows():
            seed = str(row["seed"])
            c4_legacy_points.setdefault(name, {})[seed] = float(row["diff"])
            c4_old_ci.setdefault(name, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]
    c4_hard_relative = "d1_c4_hard_eaurc_reduction"
    c4_specs.append({"name": c4_hard_relative, "metric": "e_aurc", "operation": "relative_contrast",
                     "contrast_positive": "hard_stats", "contrast_negative": "hard_e1b",
                     "denominator_condition": "hard_stats"})
    rows = pairs[(pairs["analysis"] == "C4") & (pairs["model"] == "e1b_stats_semantic vs stats_logistic")
                 & (pairs["seed"].isin([f"b3_seed{s}" for s in (1, 2, 3)]))
                 & (pairs["variant"] == "hard5") & (pairs["metric"] == "e_aurc_reduction")]
    if len(rows) != 3:
        raise ValueError("D1 C4 relative E-AURC reduction point anchors missing")
    for _, row in rows.iterrows():
        seed = str(row["seed"])
        c4_legacy_points.setdefault(c4_hard_relative, {})[seed] = float(row["diff"])
        c4_old_ci.setdefault(c4_hard_relative, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]
    dod_specs = (
        ("d1_c4_hard_minus_rand_dod_auroc", "auroc_correct", "hard_e1b", "hard_stats", "rand_e1b", "rand_stats", 1.0, "auroc_correct"),
        ("d1_c4_hard_minus_rand_dod_rer50_gain_pp", "rer_at_50", "hard_e1b", "hard_stats", "rand_e1b", "rand_stats", 100.0, "rer_at_50_gain_pp"),
    )
    for name, metric, a, b, c, d, scale, old_metric in dod_specs:
        c4_specs.append({"name": name, "metric": metric, "operation": "difference_of_differences",
                         "a": a, "b": b, "c": c, "d": d, "scale": scale})
        rows = pairs[(pairs["analysis"] == "C4") & (pairs["model"] == "diff_of_diffs")
                     & (pairs["seed"].isin([f"b3_seed{s}" for s in (1, 2, 3)]))
                     & (pairs["variant"] == "hard5_vs_rand5") & (pairs["metric"] == old_metric)]
        if len(rows) != 3:
            raise ValueError(f"D1 C4 DoD bootstrap anchor missing for {name}")
        for _, row in rows.iterrows():
            seed = str(row["seed"])
            c4_legacy_points.setdefault(name, {})[seed] = float(row["diff"])
            c4_old_ci.setdefault(name, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]

    dod_relative = "d1_c4_hard_minus_rand_dod_eaurc_reduction"
    c4_specs.append({"name": dod_relative, "metric": "e_aurc", "operation": "difference_of_relative_contrasts",
                     "positive_a": "hard_stats", "negative_a": "hard_e1b", "denominator_a": "hard_stats",
                     "positive_b": "rand_stats", "negative_b": "rand_e1b", "denominator_b": "rand_stats"})
    rows = pairs[(pairs["analysis"] == "C4") & (pairs["model"] == "diff_of_diffs")
                 & (pairs["seed"].isin([f"b3_seed{s}" for s in (1, 2, 3)]))
                 & (pairs["variant"] == "hard5_vs_rand5") & (pairs["metric"] == "e_aurc_reduction")]
    if len(rows) != 3:
        raise ValueError("D1 C4 relative E-AURC difference-of-differences anchors missing")
    for _, row in rows.iterrows():
        seed = str(row["seed"])
        c4_legacy_points.setdefault(dod_relative, {})[seed] = float(row["diff"])
        c4_old_ci.setdefault(dod_relative, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]

    c4_manifest = clean[clean["sent_id"].isin(np.asarray(rand["sentence_id"], dtype=np.int64))]
    c4_manifest = c4_manifest[c4_manifest["sent_id"].isin(np.asarray(hard["sentence_id"], dtype=np.int64))]
    c4_manifest = c4_manifest.sort_values("sent_id", kind="stable")
    c4_splits = c4_manifest["split"].astype(str).to_numpy()
    c4_split_counts = {str(name): int(count) for name, count in zip(*np.unique(c4_splits, return_counts=True))}
    d1_c4 = {
        "job_id": "d1_c4_clean", "scope": "D1_clean_robustness", "cohort": "D1_clean_same4_c4",
        "image_ids": c4_manifest["image_id"].astype(np.int64).to_numpy(), "seed_cells": c4_seed_cells,
        "estimates": c4_specs, "metrics": CI_METRICS, "legacy_points": c4_legacy_points,
        "old_ci": c4_old_ci, "legacy_anchor_unavailable": c4_unavailable,
        "old_gate": verdict.get("c4_hard_semantic", {}).get("verdict", {}),
        "thresholds": {"delta_auroc_ci_low_gt0": True, "dod_ci_low_gt0": True,
                       "dod_same_direction_as_v1": True, "a8_6_gate_confirmed_or_strong": True},
        "source_paths": list(dict.fromkeys(source_paths)), "point_order": np.arange(c4_manifest.shape[0], dtype=np.int64),
        "cohort_metadata": {"split_counts": c4_split_counts, "contains_train": False,
                            "contains_validation": any(name.lower().startswith(("val", "valid")) for name in c4_split_counts),
                            "heldout_test_only": bool(c4_split_counts) and set(c4_split_counts) <= {"testA", "testB"},
                            "historical_pool_identity": "Phase1F same4 rand5/hard5 intersection filtered by D1 clean sentence IDs"},
        "cohort_construction": "Phase1F_same4_rand_hard_exact_sentence_id_intersection_filtered_by_clean_manifest",
        "legacy_pooled_anchor": True, "legacy_cohort_classification": "D1_CLEAN_SAME4_TESTA_TESTB_ONLY",
        "legacy_pool_status": "EXISTING_D1_CLEAN_C4_COHORT",
        "legacy_cohort_note": "C4 retains the Phase1F same4 mask and filters both random/hard conditions by reviewed clean sent_id; the verified intersection contains only testA/testB rows.",
        "sentence_id_status": "VERIFIED_NATIVE_ID_FILTERED_FROM_PHASE1F",
        "evidence": f"Per-seed rand5/hard5 archives are aligned by sentence/ref/image; clean manifest filtering yields n={c4_manifest.shape[0]} rows, old point counts match, all available per-seed effects and DoD anchors match table values.",
        "recovery_status": "READ_ONLY_NATIVE_ID_PREDICTIONS",
        "model_identity_by_seed": {seed: {"scorer": seed, "models": ["stats_logistic", "e1b_stats_semantic"]} for seed in b3_seeds},
        "prediction_source_precision": "frozen_float64_model_confidence_and_binary_correctness",
    }
    return [c1_b0, c1_b3, d1_c4]


def build_v2g_g3_jobs() -> list[dict[str, Any]]:
    """Replay V2-G G3 from the saved per-sentence outputs for B1/B2.

    Historical ``__pooled__`` rows cover all 20,799 common rows (validation plus
    testA/testB). The testA+testB-only pool is emitted as a separate, unanchored
    cohort so it cannot inherit the all-common gate's point estimate.
    """
    root = ROOT / "results" / "v2_backbone_generalization"
    gate_doc = read_json(root / "g3_cardinality_gate.json")
    seeds = tuple(f"b3_seed{s}" for s in (1, 2, 3))
    variants = {
        "native": ("native", "native"),
        "global_T": ("global_T_corrected", "global_T_corrected"),
        "validation_perK_diagnostic": ("oracle_T_K", "oracle_T_K"),
    }
    jobs: list[dict[str, Any]] = []
    for backbone, tag in (("b1_phase0b", "b1"), ("b2_phase0b", "b2")):
        arrays = {
            f"b3_seed{seed}": load_npz(root / backbone / f"seed_{seed}" / "per_sentence_predictions.npz")
            for seed in (1, 2, 3)
        }
        ref = arrays["b3_seed1"]
        id_ref = {"sentence_id": ref["sentence_ids"], "ref_id": ref["ref_ids"],
                  "image_id": ref["image_ids"], "eval_split": ref["eval_split"]}
        for seed, data in arrays.items():
            assert_row_alignment(
                id_ref,
                {"sentence_id": data["sentence_ids"], "ref_id": data["ref_ids"],
                 "image_id": data["image_ids"], "eval_split": data["eval_split"]},
                required_fields=("ref_id", "image_id", "eval_split"), candidate_name=f"V2G/{backbone}/{seed}",
            )
        for cohort, mask in _split_masks(np.asarray(ref["eval_split"]).astype(str), pooled_alias="__pooled__"):
            row_idx = np.flatnonzero(mask)
            point_order = np.argsort(np.asarray(ref["sentence_ids"])[row_idx], kind="stable")
            image_ids = np.asarray(ref["image_ids"], dtype=np.int64)[row_idx]
            split_values = np.asarray(ref["eval_split"]).astype(str)[row_idx]
            split_labels, split_sizes = np.unique(split_values, return_counts=True)
            split_counts = {str(name): int(count) for name, count in zip(split_labels, split_sizes)}
            split_names = set(split_counts)
            legacy_split = "__pooled__" if cohort == "__all_common__" else (None if cohort == "__pooled_test__" else cohort)
            seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
            legacy_points: dict[str, dict[str, float]] = {}
            old_ci: dict[str, dict[str, list[float]]] = {}
            specs: list[dict[str, Any]] = []
            source_paths: list[Path] = [root / "g3_cardinality_gate.json", root / "final_summary.csv"]
            for seed, data in arrays.items():
                seed_dir = root / backbone / f"seed_{seed[-1]}"
                point_path = seed_dir / "normalized_selective_metrics.csv"
                calibration_path = seed_dir / "calibration_metrics.csv"
                bootstrap_path = seed_dir / "bootstrap.csv"
                point_frame = read_csv(point_path)
                calibration_frame = read_csv(calibration_path)
                old = point_frame.merge(
                    calibration_frame, on=["eval_split", "K", "variant", "n"],
                    suffixes=("", "_cal"),
                )
                boot = read_csv(bootstrap_path)
                source_paths.extend((
                    root / backbone / f"seed_{seed[-1]}" / "per_sentence_predictions.npz",
                    seed_dir / "normalized_selective_metrics.csv", calibration_path, bootstrap_path,
                    seed_dir / "eval_metadata.json", root / backbone / "metadata.json",
                ))
                seed_cells[seed] = {}
                for k in (5, 10, 20, 50):
                    correct = np.asarray(data[f"correct_K{k}"], dtype=np.float64)[row_idx]
                    confidence_fields = {
                        "native": data[f"confidence_native_K{k}"],
                        "global_T": data[f"confidence_global_T_corrected_K{k}"],
                        "validation_perK_diagnostic": data[f"confidence_oracle_T_K_K{k}"],
                    }
                    for variant, confidence_all in confidence_fields.items():
                        condition = f"K{k}__{variant}"
                        seed_cells[seed][condition] = (
                            np.asarray(confidence_all, dtype=np.float64)[row_idx], correct,
                        )
                        if legacy_split is None:
                            continue
                        old_variant = variants[variant][1]
                        old_row = old[(old["eval_split"] == legacy_split) & (old["K"].astype(int) == k)
                                      & (old["variant"] == old_variant)]
                        if len(old_row) != 1:
                            raise ValueError(f"V2G historical metric row missing/ambiguous: {backbone}/{seed}/{legacy_split}/K{k}/{old_variant}")
                        row = old_row.iloc[0]
                        for metric in CI_METRICS:
                            col = "accuracy" if metric == "accuracy" else metric
                            value = _float_or_nan(row.get(col))
                            name = f"{condition}::{metric}"
                            if np.isfinite(value):
                                legacy_points.setdefault(name, {})[seed] = value
                            else:
                                # Some legacy tables intentionally omit a metric or record it as non-finite.
                                legacy_points.setdefault(name, {})[seed] = float("nan")

                if legacy_split is not None:
                    for variant in variants:
                        old_variant = variants[variant][1]
                        for kb in (10, 20, 50):
                            rows = boot[(boot["eval_split"] == legacy_split)
                                        & (boot["variant"] == old_variant)
                                        & (boot["K_a"].astype(int) == 5)
                                        & (boot["K_b"].astype(int) == kb)]
                            by_key = {
                                f"{r.metric}|{r.diff_kind}": r
                                for r in rows.itertuples(index=False)
                            }
                            for metric, kind, estimate_name, sign in (
                                ("auroc_correct", "absolute", f"crossK_K5_minus_K{kb}__{variant}::auroc_correct", 1.0),
                                ("e_aurc", "absolute", f"crossK_K{kb}_minus_K5__{variant}::e_aurc", -1.0),
                                ("rer_at_50", "absolute", f"crossK_K5_minus_K{kb}__{variant}::rer_at_50", 1.0),
                            ):
                                old_row = by_key.get(f"{metric}|{kind}")
                                if old_row is None:
                                    raise ValueError(f"V2G bootstrap anchor missing {backbone}/{seed}/{legacy_split}/K{kb}/{metric}")
                                legacy_points.setdefault(estimate_name, {})[seed] = sign * float(old_row.diff)
                                old_ci.setdefault(estimate_name, {})[seed] = sorted((
                                    sign * float(old_row.ci_low), sign * float(old_row.ci_high),
                                ))
                            # Historical relative E-AURC is (K5-Kb)/Kb. Convert its
                            # estimate and limits to (Kb-K5)/K5 before comparison.
                            rel_row = by_key.get("e_aurc|relative")
                            if rel_row is None:
                                raise ValueError(f"V2G relative E-AURC anchor missing {backbone}/{seed}/{legacy_split}/K{kb}")
                            rel_name = f"crossK_relative_eaurc_worsening_K{kb}__{variant}"
                            transform = lambda x: -float(x) / (1.0 + float(x))
                            legacy_points.setdefault(rel_name, {})[seed] = transform(rel_row.diff)
                            limits = sorted((transform(rel_row.ci_low), transform(rel_row.ci_high)))
                            old_ci.setdefault(rel_name, {})[seed] = limits

            for k in (5, 10, 20, 50):
                for variant in variants:
                    condition = f"K{k}__{variant}"
                    for metric in CI_METRICS:
                        specs.append({"name": f"{condition}::{metric}", "metric": metric,
                                      "operation": "condition", "condition": condition})
            for variant in variants:
                for kb in (10, 20, 50):
                    specs.extend((
                        {"name": f"crossK_K5_minus_K{kb}__{variant}::auroc_correct", "metric": "auroc_correct",
                         "operation": "difference", "a": f"K5__{variant}", "b": f"K{kb}__{variant}"},
                        {"name": f"crossK_K{kb}_minus_K5__{variant}::e_aurc", "metric": "e_aurc",
                         "operation": "difference", "a": f"K{kb}__{variant}", "b": f"K5__{variant}"},
                        {"name": f"crossK_K5_minus_K{kb}__{variant}::rer_at_50", "metric": "rer_at_50",
                         "operation": "difference", "a": f"K5__{variant}", "b": f"K{kb}__{variant}"},
                        {"name": f"crossK_relative_eaurc_worsening_K{kb}__{variant}", "metric": "e_aurc",
                         "operation": "relative_contrast", "contrast_positive": f"K{kb}__{variant}",
                         "contrast_negative": f"K5__{variant}", "denominator_condition": f"K5__{variant}"},
                    ))
            g3_entry = gate_doc.get("backbones", {}).get("B1_openclip_b16" if tag == "b1" else "B2_siglip_b16", {})
            jobs.append({
                "job_id": f"v2g_g3_{tag}_{cohort}", "scope": "V2G_G3_cardinality",
                "cohort": str(cohort), "image_ids": image_ids, "seed_cells": seed_cells,
                "estimates": specs, "metrics": CI_METRICS, "legacy_points": legacy_points,
                "old_ci": old_ci, "old_gate": g3_entry.get("CARDINALITY_REPLICATED", "NOT_APPLICABLE")
                if cohort == "__all_common__" else "NOT_APPLICABLE",
                "thresholds": gate_doc.get("thresholds", {}),
                "source_paths": list(dict.fromkeys(source_paths)), "point_order": point_order,
                "point_estimate_ordering": "sentence_id_ascending_canonical_order",
                "cohort_metadata": {"split_counts": split_counts,
                                    "contains_train": any(name.lower().startswith("train") for name in split_names),
                                    "contains_validation": any(name.lower().startswith(("val", "valid")) for name in split_names),
                                    "heldout_test_only": bool(split_names) and split_names <= {"testA", "testB"},
                                    "historical_pool_identity": "V2G G3 split mask from frozen per-sentence eval_split"},
                "cohort_construction": "testA_plus_testB_new_test_cohort" if cohort == "__pooled_test__" else
                    "historical_all_common_validation_plus_test" if cohort == "__all_common__" else "historical_eval_split",
                "legacy_pooled_anchor": cohort == "__all_common__",
                "legacy_cohort_classification": "NEW_TESTA_TESTB_POOL_NO_MATCHED_LEGACY_TEST_POOL" if cohort == "__pooled_test__" else
                    "ALL_COMMON_INCLUDES_VALIDATION_NO_TRAIN" if cohort == "__all_common__" else "HISTORICAL_SPLIT",
                "legacy_cohort_note": (
                    "Historical G3 __pooled__ means all common 20,799 rows, including val_select and val_calib; this testA+testB pool is a new 10,286-row cohort and has no legacy pool anchor."
                    if cohort == "__pooled_test__" else
                    "Historical __pooled__ includes val_select, val_calib, testA, and testB (no train rows); retain as an appendix and do not call it independent test evidence."
                    if cohort == "__all_common__" else ""
                ),
                "legacy_pool_status": "NEW_SUPPLEMENTAL_NO_MATCHED_LEGACY_TEST_POOL" if cohort == "__pooled_test__" else
                    "EXISTING_LEGACY_ALL_COMMON" if cohort == "__all_common__" else "NOT_APPLICABLE",
                "sentence_id_status": "VERIFIED_NATIVE_ID",
                "evidence": "B1/B2 per_sentence_predictions.npz carry sentence/ref/image/eval_split IDs, aligned across all three frozen seeds; historical normalized/calibration tables and per-seed bootstrap rows anchor each available point/effect.",
                "recovery_status": "READ_ONLY_PREDICTION_ARTIFACT",
                "model_identity_by_seed": {seed: {"backbone": backbone, "seed": seed} for seed in seeds},
                "prediction_source_precision": "frozen_float32_confidences_and_binary_correctness",
            })
    return jobs


def build_d2_c1_jobs() -> list[dict[str, Any]]:
    """Replay D2-C1 on the exact native-ID intersection of all random K cells."""
    root = ROOT / "results" / "v2_d2_refcoco_lang"
    pred_root = root / "predictions"
    point_path = root / "c1_point.csv"
    bootstrap_path = root / "c1_bootstrap.csv"
    verdict_path = root / "c1_c4_verdict.json"
    source_paths: list[Path] = [point_path, bootstrap_path, verdict_path,
                                root / "inference_report.json", root / "phase1" / "cohort.csv",
                                root / "phase1" / "cohort_audit.json"]
    old_point = read_csv(point_path)
    old_boot = read_csv(bootstrap_path)
    verdict = read_json(verdict_path)
    seeds = tuple(f"b3_seed{s}" for s in (1, 2, 3))
    arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    id_sets: list[set[int]] = []
    for k in (5, 10, 20, 50):
        for seed in seeds:
            path = pred_root / f"d2__random_k{k}__{seed}.npz"
            arrays[(k, seed)] = load_npz(path)
            source_paths.append(path)
            ids = np.asarray(arrays[(k, seed)]["sentence_id"], dtype=np.int64)
            if np.unique(ids).size != ids.size:
                raise ValueError(f"D2 C1 random K{k}/{seed}: duplicate sentence_id")
            if seed == seeds[0]:
                id_sets.append(set(ids.tolist()))
            else:
                ref = arrays[(k, seeds[0])]
                cur = arrays[(k, seed)]
                assert_row_alignment(
                    {"sentence_id": ref["sentence_id"], "ref_id": ref["ref_id"], "image_id": ref["image_id"]},
                    {"sentence_id": cur["sentence_id"], "ref_id": cur["ref_id"], "image_id": cur["image_id"]},
                    required_fields=("ref_id", "image_id"), candidate_name=f"D2/C1/K{k}/{seed}",
                )
    common_ids = set.intersection(*id_sets)
    if not common_ids:
        raise ValueError("D2 C1 common K50 intersection is empty")

    seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    legacy_points: dict[str, dict[str, float]] = {}
    legacy_anchor_unavailable: list[dict[str, Any]] = []
    old_ci: dict[str, dict[str, list[float]]] = {}
    specs: list[dict[str, Any]] = []
    image_ids: np.ndarray | None = None
    sid_ref: np.ndarray | None = None
    ref_ref: np.ndarray | None = None
    old_metric_columns = {"accuracy": "accuracy", "auroc_correct": "auroc_correct",
                          "e_aurc": "e_aurc", "rer_at_50": "rer_at_50"}
    for seed in seeds:
        seed_cells[seed] = {}
        for k in (5, 10, 20, 50):
            data = arrays[(k, seed)]
            raw_ids = np.asarray(data["sentence_id"], dtype=np.int64)
            if not np.all(np.diff(raw_ids) > 0):
                raise ValueError(f"P1/P2 {family}/K{k}/{seed}: canonical sentence IDs must be ascending")
            mask = np.isin(raw_ids, np.fromiter(sorted(common_ids), dtype=np.int64))
            idx = np.flatnonzero(mask)
            order = np.argsort(raw_ids[idx], kind="stable")
            idx = idx[order]
            ids = raw_ids[idx]
            ref_ids = np.asarray(data["ref_id"], dtype=np.int64)[idx]
            images = np.asarray(data["image_id"], dtype=np.int64)[idx]
            correct = np.asarray(data["correct"], dtype=np.float64)[idx]
            if sid_ref is None:
                sid_ref, ref_ref, image_ids = ids.copy(), ref_ids.copy(), images.copy()
            elif not np.array_equal(sid_ref, ids) or not np.array_equal(ref_ref, ref_ids) or not np.array_equal(image_ids, images):
                raise ValueError(f"D2 C1 common K cohort differs across K/seed at K{k}/{seed}")
            condition = f"K{k}__global_T"
            confidence = np.asarray(data["conf_msp"], dtype=np.float64)[idx]
            seed_cells[seed][condition] = (confidence, correct)
            old_row = old_point[(old_point["seed"].astype(str) == f"seed{seed[-1]}")
                                & (old_point["K"].astype(int) == k)
                                & (old_point["model"] == "global_T_corrected")]
            if len(old_row) != 1:
                raise ValueError(f"D2 C1 old point row missing/ambiguous: {seed}/K{k}")
            row = old_row.iloc[0]
            if int(row["n"]) != len(common_ids) or int(row["n_images"]) != np.unique(images).size:
                raise ValueError(f"D2 C1 old point count mismatch: {seed}/K{k}")
            for metric, col in old_metric_columns.items():
                value = _float_or_nan(row.get(col))
                if np.isfinite(value):
                    legacy_points.setdefault(f"{condition}::{metric}", {})[seed] = value
            for metric in set(CI_METRICS) - set(old_metric_columns):
                legacy_anchor_unavailable.append({
                    "estimate": f"{condition}::{metric}", "seed": seed,
                    "reason": "legacy_D2_C1_point_table_did_not_report_this_metric",
                })
        # Every condition must carry exactly the same canonical identities.
        for k in (5, 10, 20, 50):
            data = arrays[(k, seed)]
            raw_ids = np.asarray(data["sentence_id"], dtype=np.int64)
            mask = np.isin(raw_ids, np.fromiter(sorted(common_ids), dtype=np.int64))
            idx = np.flatnonzero(mask)
            idx = idx[np.argsort(raw_ids[idx], kind="stable")]
            if not np.array_equal(np.asarray(data["sentence_id"])[idx], sid_ref):
                raise ValueError(f"D2 C1 seed {seed} K{k} sentence order differs")

    for k in (5, 10, 20, 50):
        condition = f"K{k}__global_T"
        for metric in CI_METRICS:
            specs.append({"name": f"{condition}::{metric}", "metric": metric,
                          "operation": "condition", "condition": condition})
    for kb in (10, 20, 50):
        names = (
            (f"d2_c1_auroc_drop_K5_K{kb}", "auroc_correct", "K5__global_T", f"K{kb}__global_T", 1.0, "auroc_correct", "absolute"),
            (f"d2_c1_eaurc_abs_worsening_K{kb}_minus_K5", "e_aurc", f"K{kb}__global_T", "K5__global_T", 1.0, "e_aurc", "absolute"),
            (f"d2_c1_rer50_drop_K5_K{kb}", "rer_at_50", "K5__global_T", f"K{kb}__global_T", 100.0, "rer_at_50", "absolute"),
        )
        for name, metric, a, b, scale, old_metric, old_kind in names:
            specs.append({"name": name, "metric": metric, "operation": "difference",
                          "a": a, "b": b, "scale": scale})
            rows = old_boot[(old_boot["seed"].astype(str).isin([f"seed{s[-1]}" for s in seeds]))
                            & (old_boot["K_a"].astype(int) == 5)
                            & (old_boot["K_b"].astype(int) == kb)
                            & (old_boot["variant"] == "global_T_corrected")
                            & (old_boot["metric"] == old_metric)
                            & (old_boot["diff_kind"] == old_kind)]
            if len(rows) != 3:
                raise ValueError(f"D2 C1 old contrast rows missing: K5/K{kb}/{old_metric}")
            for _, row in rows.iterrows():
                seed = f"b3_seed{int(str(row['seed'])[-1])}"
                unit_scale = 100.0 if metric == "rer_at_50" else 1.0
                sign = -1.0 if metric == "e_aurc" else 1.0
                legacy_points.setdefault(name, {})[seed] = sign * unit_scale * float(row["diff"])
                limits = sorted((sign * unit_scale * float(row["ci_low"]),
                                 sign * unit_scale * float(row["ci_high"])))
                old_ci.setdefault(name, {})[seed] = limits

        rel_name = f"d2_c1_eaurc_relative_worsening_K5_K{kb}"
        specs.append({"name": rel_name, "metric": "e_aurc", "operation": "relative_contrast",
                      "contrast_positive": f"K{kb}__global_T", "contrast_negative": "K5__global_T",
                      "denominator_condition": "K5__global_T"})
        rows = old_boot[(old_boot["seed"].astype(str).isin([f"seed{s[-1]}" for s in seeds]))
                        & (old_boot["K_a"].astype(int) == 5)
                        & (old_boot["K_b"].astype(int) == kb)
                        & (old_boot["variant"] == "global_T_corrected")
                        & (old_boot["metric"] == "e_aurc")
                        & (old_boot["diff_kind"] == "relative")]
        if len(rows) != 3:
            raise ValueError(f"D2 C1 old relative E-AURC rows missing: K5/K{kb}")
        for _, row in rows.iterrows():
            seed = f"b3_seed{int(str(row['seed'])[-1])}"
            trans = lambda x: -float(x) / (1.0 + float(x))
            legacy_points.setdefault(rel_name, {})[seed] = trans(float(row["diff"]))
            old_ci.setdefault(rel_name, {})[seed] = sorted((trans(float(row["ci_low"])), trans(float(row["ci_high"]))))

    g1 = verdict.get("c1", {})
    gate = g1.get("gate", {}) if isinstance(g1, dict) else {}
    cohort_frame = read_csv(root / "phase1" / "cohort.csv")
    split_counts = _d2_verified_split_counts(cohort_frame, sid_ref, ref_ref, image_ids)
    job = {
        "job_id": "d2_c1_common_k50", "scope": "D2_C1_cardinality",
        "cohort": "common_k50", "image_ids": image_ids, "seed_cells": seed_cells,
        "estimates": specs, "metrics": CI_METRICS, "legacy_points": legacy_points,
        "old_ci": old_ci, "legacy_anchor_unavailable": legacy_anchor_unavailable,
        "old_gate": gate.get("replicated", "NOT_RECORDED"),
        "thresholds": gate.get("thresholds", {}), "source_paths": list(dict.fromkeys(source_paths)),
        "point_estimate_ordering": "sentence_id_ascending_canonical_order",
        "cohort_metadata": {"split_counts": split_counts, "contains_train": False,
                            "contains_validation": False, "heldout_test_only": set(split_counts) <= {"testA", "testB"},
                            "historical_pool_identity": "D2 RefCOCO external C1 common-K50; sentence_id=cohort expr_id rowwise verified"},
        "cohort_construction": "intersection_of_native_sentence_ids_across_random_K5_K10_K20_K50",
        "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_LEGACY_D2_COMMON_K50",
        "legacy_pool_status": "EXISTING_LEGACY_COMMON_K50",
        "sentence_id_status": "VERIFIED_NATIVE_ID",
        "evidence": f"All four K prediction arrays carry sentence/ref/image IDs; intersection n={len(common_ids)} is identical across all three seeds and every K; points and original paired-bootstrap contrasts are anchored from frozen D2 tables.",
        "recovery_status": "READ_ONLY_PREDICTION_ARTIFACT",
        "model_identity_by_seed": {seed: {"scorer": seed, "model": "frozen B3 MSP/global-T"} for seed in seeds},
        "prediction_source_precision": "archived_float64_conf_msp_and_binary_correctness",
    }
    return [job]


def _d2_verified_split_counts(cohort_frame: Any, sentence_ids: np.ndarray,
                              ref_ids: np.ndarray, image_ids: np.ndarray) -> dict[str, int]:
    """Bridge D2 sentence IDs to the frozen cohort's expr_id rows, with anchors."""
    frame = cohort_frame
    if "expr_id" not in frame or "split" not in frame or frame["expr_id"].duplicated().any():
        raise ValueError("D2 cohort CSV needs unique expr_id and split columns for metadata mapping")
    indexed = frame.set_index("expr_id", drop=False)
    ids = np.asarray(sentence_ids, dtype=np.int64)
    missing = [int(value) for value in ids if value not in indexed.index]
    if missing:
        raise ValueError(f"D2 prediction sentence IDs absent from cohort expr_id rows: {missing[:5]}")
    rows = indexed.loc[ids]
    if not np.array_equal(rows["ref_id"].to_numpy(dtype=np.int64), np.asarray(ref_ids, dtype=np.int64)):
        raise ValueError("D2 cohort expr_id bridge ref_id mismatch")
    if not np.array_equal(rows["image_id"].to_numpy(dtype=np.int64), np.asarray(image_ids, dtype=np.int64)):
        raise ValueError("D2 cohort expr_id bridge image_id mismatch")
    labels, counts = np.unique(rows["split"].astype(str).to_numpy(), return_counts=True)
    return {str(name): int(count) for name, count in zip(labels, counts)}


def build_d2_c4_jobs() -> list[dict[str, Any]]:
    """Replay D2 same-category C4 on native-ID hard/random common rows."""
    root = ROOT / "results" / "v2_d2_refcoco_lang"
    pred_root = root / "predictions"
    amp_path, manip_path = root / "c4_amplification.csv", root / "c4_manipulation.csv"
    verdict_path = root / "c1_c4_verdict.json"
    cohort_path = root / "phase1" / "cohort.csv"
    amp, manip, verdict = read_csv(amp_path), read_csv(manip_path), read_json(verdict_path)
    cohort_frame = read_csv(cohort_path)
    seeds = tuple(f"b3_seed{s}" for s in (1, 2, 3))
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    source_paths: list[Path] = [amp_path, manip_path, verdict_path, cohort_path,
                                root / "phase1" / "cohort_audit.json", root / "inference_report.json"]
    common_ids: np.ndarray | None = None
    for seed in seeds:
        rand_path = pred_root / f"d2__random_k5__{seed}.npz"
        hard_path = pred_root / f"d2__hard_k5__{seed}.npz"
        rand, hard = load_npz(rand_path), load_npz(hard_path)
        arrays[("rand", seed)], arrays[("hard", seed)] = rand, hard
        source_paths.extend((rand_path, hard_path))
        rand_ids = np.asarray(rand["sentence_id"], dtype=np.int64)
        hard_ids = np.asarray(hard["sentence_id"], dtype=np.int64)
        if np.unique(rand_ids).size != rand_ids.size or np.unique(hard_ids).size != hard_ids.size:
            raise ValueError(f"D2 C4 {seed}: duplicate expression sentence IDs")
        ids = np.intersect1d(rand_ids, hard_ids, assume_unique=True)
        common_ids = ids if common_ids is None else np.intersect1d(common_ids, ids, assume_unique=True)
    assert common_ids is not None and common_ids.size > 0
    cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {seed: {} for seed in seeds}
    aux_cells: dict[str, dict[str, np.ndarray]] = {seed: {} for seed in seeds}
    legacy_points: dict[str, dict[str, float]] = {}
    old_ci: dict[str, dict[str, list[float]]] = {}
    image_ref: np.ndarray | None = None
    ref_ref: np.ndarray | None = None
    for seed in seeds:
        expected_ids: np.ndarray | None = None
        for regime in ("rand", "hard"):
            data = arrays[(regime, seed)]
            idx = _canonical_indices(np.asarray(data["sentence_id"], dtype=np.int64), common_ids,
                                     label=f"D2 C4 {seed}/{regime}")
            ids = np.asarray(data["sentence_id"], dtype=np.int64)[idx]
            refs = np.asarray(data["ref_id"], dtype=np.int64)[idx]
            images = np.asarray(data["image_id"], dtype=np.int64)[idx]
            correct = np.asarray(data["correct"], dtype=np.float64)[idx]
            if expected_ids is None:
                expected_ids = ids.copy()
            elif not np.array_equal(ids, expected_ids):
                raise ValueError(f"D2 C4 {seed}: hard/random sentence IDs differ after intersection")
            if regime == "rand":
                if image_ref is None:
                    image_ref, ref_ref = images.copy(), refs.copy()
                elif not np.array_equal(image_ref, images) or not np.array_equal(ref_ref, refs):
                    raise ValueError(f"D2 C4 {seed}: identity rows differ across seeds")
            else:
                random = arrays[("rand", seed)]
                random_idx = _canonical_indices(np.asarray(random["sentence_id"], dtype=np.int64), common_ids,
                                                label=f"D2 C4 {seed}/rand anchors")
                if not np.array_equal(refs, np.asarray(random["ref_id"], dtype=np.int64)[random_idx]) or not np.array_equal(images, np.asarray(random["image_id"], dtype=np.int64)[random_idx]):
                    raise ValueError(f"D2 C4 {seed}: hard/random ref/image identity mismatch")
            condition_regime = "random" if regime == "rand" else "hard"
            cells[seed][f"{regime}__R1"] = (np.asarray(data["conf_stats"], dtype=np.float64)[idx], correct)
            cells[seed][f"{regime}__E1b"] = (np.asarray(data["conf_e1b"], dtype=np.float64)[idx], correct)
            semantic = np.asarray(data["sem14"], dtype=np.float64)[idx]
            for feature, _direction in MANIPULATION_METRICS:
                aux_cells[seed][f"{regime}__feature__{feature}"] = semantic[:, V2_PRIMARY_SEMANTIC_NAMES.index(feature)]
    split_counts = _d2_verified_split_counts(cohort_frame, common_ids, ref_ref, image_ref)
    specs = [
        {"name": f"{regime}__{model}::auroc_correct", "metric": "auroc_correct",
         "operation": "condition", "condition": f"{regime}__{model}"}
        for regime in ("rand", "hard") for model in ("R1", "E1b")
    ]
    specs.extend([
        {"name": "delta_hard_auroc", "metric": "auroc_correct", "operation": "difference", "a": "hard__E1b", "b": "hard__R1"},
        {"name": "delta_rand_auroc", "metric": "auroc_correct", "operation": "difference", "a": "rand__E1b", "b": "rand__R1"},
        {"name": "amplification_auroc", "metric": "auroc_correct", "operation": "difference_of_differences", "a": "hard__E1b", "b": "hard__R1", "c": "rand__E1b", "d": "rand__R1"},
    ])
    aux_specs: list[dict[str, Any]] = []
    for feature, direction in MANIPULATION_METRICS:
        left, right = f"hard__feature__{feature}", f"rand__feature__{feature}"
        name = f"manipulation__{feature}__hard_minus_rand"
        aux_specs.append({"name": name, "metric": "mean", "operation": "difference", "a": left, "b": right})
        aux_specs.append({"name": f"manipulation__{feature}__directional", "metric": "mean", "operation": "difference",
                          "a": left, "b": right, "scale": 1.0 if direction == "up" else -1.0})
    if amp.shape[0] != 3 or not np.array_equal(amp["scorer"].astype(str).to_numpy(), np.asarray(seeds)):
        amp = amp.set_index("scorer").loc[list(seeds)].reset_index()
    for _, row in amp.iterrows():
        seed = str(row["scorer"])
        if int(row["n"]) != common_ids.size:
            raise ValueError(f"D2 C4 {seed}: historical amplification row count mismatch")
        for regime in ("hard", "rand"):
            for model, column in (("R1", f"auroc_r1_{regime}"), ("E1b", f"auroc_e1b_{regime}")):
                legacy_points.setdefault(f"{('rand' if regime == 'rand' else regime)}__{model}::auroc_correct", {})[seed] = float(row[column])
        for name, col in (("delta_hard_auroc", "delta_hard"), ("delta_rand_auroc", "delta_rand"),
                          ("amplification_auroc", "amplification")):
            legacy_points.setdefault(name, {})[seed] = float(row[col])
        for name, lo, hi in (("delta_hard_auroc", "delta_hard_ci_low", "delta_hard_ci_high"),
                             ("delta_rand_auroc", "delta_rand_ci_low", "delta_rand_ci_high"),
                             ("amplification_auroc", "amplification_ci_low", "amplification_ci_high")):
            old_ci.setdefault(name, {})[seed] = [float(row[lo]), float(row[hi])]
    for _, row in manip.iterrows():
        seed, feature = str(row["scorer"]), str(row["feature"])
        if int(row["n"]) != common_ids.size:
            raise ValueError(f"D2 C4 {seed}/{feature}: manipulation row count mismatch")
        name = f"manipulation__{feature}__hard_minus_rand"
        legacy_points.setdefault(name, {})[seed] = float(row["diff"])
        old_ci.setdefault(name, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]
    old_c4 = verdict.get("c4", {}).get("gate", {})
    return [{
        "job_id": "d2_c4_random_hard5", "scope": "D2_C4_semantic", "cohort": "matched_random_hard5",
        "image_ids": image_ref, "seed_cells": cells, "estimates": specs, "metrics": ("auroc_correct",),
        "auxiliary_cells": aux_cells, "auxiliary_estimates": aux_specs,
        "legacy_points": legacy_points, "old_ci": old_ci, "old_gate": old_c4.get("verdict", "NOT_RECORDED"),
        "thresholds": old_c4.get("thresholds", {}), "source_paths": list(dict.fromkeys(source_paths)),
        "point_ordering": "ascending_native_expression_sentence_id",
        "cohort_construction": "intersection_of_D2_hard_and_random_native_expression_ids_across_fixed_seeds",
        "cohort_metadata": {"split_counts": split_counts, "contains_train": False, "contains_validation": False,
                            "heldout_test_only": set(split_counts) <= {"testA", "testB"},
                            "historical_pool_identity": "D2 RefCOCO external matched C4; prediction sentence_id=cohort expr_id verified rowwise"},
        "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_D2_SAMECAT_TESTA_TESTB",
        "legacy_pool_status": "EXISTING_LEGACY_MATCHED_COHORT",
        "sentence_id_status": "VERIFIED_NATIVE_ID_AND_EXPR_ID_BRIDGE",
        "evidence": f"Common native sentence IDs n={common_ids.size}; cohort expr_id/ref/image joins verified; seedwise hard/random ref/image rows match; AUROC and manipulation anchors read from frozen D2 tables.",
        "recovery_status": "READ_ONLY_NATIVE_ID_PREDICTIONS",
        "model_identity_by_seed": {seed: {"scorer": seed, "R1": "stats_logistic", "E1b": "e1b_stats_semantic"} for seed in seeds},
        "prediction_source_precision": "archived_float64_confidence_and_sem14",
    }]


def build_v2m_m2_jobs() -> list[dict[str, Any]]:
    """Replay M2 curriculum confidence on its frozen same8 rows and severity cells.

    The legacy M2 confidence NPZ omits sentence IDs. This adapter bridges rows
    to Phase1F's frozen expb archives in their original ascending sentence-ID
    order, verifies image/correctness arrays per severity and seed, then checks
    the complete source row set against Phase0B to recover eval-split counts.
    """
    root = ROOT / "results" / "v2_local_competition" / "m2_curriculum"
    point_path = root / "point_metrics.csv"
    pair_path = root / "bootstrap_pairs.csv"
    gate_path = root / "gate.json"
    protocol_path = ROOT / "results" / "v2_local_competition" / "protocol.json"
    old_points, old_pairs = read_csv(point_path), read_csv(pair_path)
    gate_doc, protocol = read_json(gate_path), read_json(protocol_path)
    seeds = tuple(f"b3_seed{s}" for s in (1, 2, 3))
    levels = (0, 2, 4, 8)
    models = ("R1", "E1b", "Aggregate-MLP", "LCR-noGate", "LCR")
    old_columns = {
        "accuracy": "b3_accuracy", "auroc_correct": "auroc_correct", "e_aurc": "e_aurc",
        "rer_at_50": "rer_at_50", "rer_at_80": "rer_at_80", "ece_adaptive": "ece_adaptive",
        "brier_binary": "brier_binary", "nll_binary": "nll_binary",
    }
    source_paths: list[Path] = [point_path, pair_path, gate_path, protocol_path]
    cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {seed: {} for seed in seeds}
    ids_ref: np.ndarray | None = None
    images_ref: np.ndarray | None = None
    split_ref: np.ndarray | None = None
    for seed in seeds:
        seed_number = int(seed[-1])
        archive_path = root / f"seed_{seed_number}" / "confidences.npz"
        archive = load_npz(archive_path)
        source_paths.extend((archive_path, root / f"seed_{seed_number}" / f"model_manifest_seed{seed_number}.json"))
        phase0b_path = ROOT / "results" / "phase0b_independent" / f"seed_{seed_number}" / "per_sentence_predictions.npz"
        phase0b = load_npz(phase0b_path)
        source_paths.append(phase0b_path)
        base_ids = np.asarray(phase0b["sentence_ids"], dtype=np.int64)
        if not np.all(np.diff(base_ids) > 0):
            raise ValueError(f"M2 Phase0B row bridge must be strictly sentence-sorted: {seed}")

        seed_ids: np.ndarray | None = None
        seed_images: np.ndarray | None = None
        for level in levels:
            expb_path = ROOT / "results" / "phase1f_hard_semantic" / "predictions" / f"expb_m{level}__{seed}.npz"
            expb = load_npz(expb_path)
            source_paths.append(expb_path)
            ids = np.asarray(expb["sentence_id"], dtype=np.int64)
            refs = np.asarray(expb["ref_id"], dtype=np.int64)
            images = np.asarray(expb["image_id"], dtype=np.int64)
            correctness = np.asarray(expb["correct"], dtype=np.float64)
            if not (np.all(np.diff(ids) > 0) and np.unique(ids).size == ids.size):
                raise ValueError(f"M2 expb m{level}/{seed}: sentence IDs must be unique and ascending")
            if seed_ids is None:
                seed_ids, seed_images = ids.copy(), images.copy()
            elif not np.array_equal(seed_ids, ids) or not np.array_equal(seed_images, images):
                raise ValueError(f"M2 severity rows differ within seed {seed} at m{level}")
            if ids_ref is None:
                ids_ref, images_ref = ids.copy(), images.copy()
            elif not np.array_equal(ids_ref, ids) or not np.array_equal(images_ref, images):
                raise ValueError(f"M2 frozen row IDs differ across seed/severity at m{level}/{seed}")
            pos = np.searchsorted(base_ids, ids)
            if np.any(pos >= base_ids.size) or not np.array_equal(base_ids[pos], ids):
                raise ValueError(f"M2 expb sentence IDs missing from Phase0B: m{level}/{seed}")
            base_images = np.asarray(phase0b["image_ids"], dtype=np.int64)[pos]
            base_refs = np.asarray(phase0b["ref_ids"], dtype=np.int64)[pos]
            if not np.array_equal(base_images, images) or not np.array_equal(base_refs, refs):
                raise ValueError(f"M2 frozen rows do not match Phase0B image/ref IDs: m{level}/{seed}")
            splits = np.asarray(phase0b["eval_split"]).astype(str)[pos]
            if split_ref is None:
                split_ref = splits.copy()
            elif not np.array_equal(split_ref, splits):
                raise ValueError(f"M2 split identity differs across seed/severity at m{level}/{seed}")

            archive_images = np.asarray(archive[f"m{level}__image_id"], dtype=np.int64)
            archive_correct = np.asarray(archive[f"m{level}__correct"], dtype=np.float64)
            if not np.array_equal(archive_images, images) or not np.array_equal(archive_correct, correctness):
                raise ValueError(f"M2 confidence archive cannot be bridged rowwise at m{level}/{seed}")
            for model in models:
                condition = f"m{level}__{model}"
                confidence = np.asarray(archive[condition], dtype=np.float64)
                if confidence.shape != ids.shape:
                    raise ValueError(f"M2 confidence row count mismatch: {condition}/{seed}")
                cells[seed][condition] = (confidence, correctness)

    assert ids_ref is not None and images_ref is not None and split_ref is not None
    split_counts = {str(name): int(count) for name, count in zip(*np.unique(split_ref, return_counts=True))}
    if set(split_counts) - {"testA", "testB"}:
        raise ValueError(f"M2 same8 contains a non-test split: {split_counts}")

    specs: list[dict[str, Any]] = []
    legacy_points: dict[str, dict[str, float]] = {}
    legacy_ci: dict[str, dict[str, list[float]]] = {}
    legacy_anchor_unavailable: list[dict[str, Any]] = []
    for level in levels:
        for model in models:
            condition = f"m{level}__{model}"
            for metric in CI_METRICS:
                name = f"{condition}::{metric}"
                specs.append({"name": name, "metric": metric, "operation": "condition", "condition": condition})
                for seed in seeds:
                    row = old_points[(old_points["cohort"] == f"m{level}")
                                     & (old_points["seed"].astype(int) == int(seed[-1]))
                                     & (old_points["model"] == model)]
                    if len(row) != 1:
                        raise ValueError(f"M2 old point missing/ambiguous: m{level}/{seed}/{model}")
                    old_row = row.iloc[0]
                    if int(old_row["n"]) != ids_ref.size or int(old_row["n_images"]) != np.unique(images_ref).size:
                        raise ValueError(f"M2 row/image count mismatch: m{level}/{seed}/{model}")
                    column = old_columns.get(metric)
                    if column is not None:
                        value = _float_or_nan(old_row.get(column))
                        if np.isfinite(value):
                            legacy_points.setdefault(name, {})[seed] = value
                        else:
                            legacy_anchor_unavailable.append({"estimate": name, "seed": seed,
                                                              "reason": "historical_point_missing_or_nonfinite"})
                    else:
                        legacy_anchor_unavailable.append({"estimate": name, "seed": seed,
                                                          "reason": "RER90_not_reported_by_historical_M2_point_table"})

    # Historical point tables report all four severity cells independently.
    # Add supplementary severity macros as explicit within-replicate means;
    # derive their legacy point anchor from the four anchored per-severity rows.
    for model in models:
        for metric in CI_METRICS:
            name = f"severity_macro__{model}::{metric}"
            specs.append({"name": name, "metric": metric, "operation": "macro_mean",
                          "conditions": [f"m{level}__{model}" for level in levels]})
            for seed in seeds:
                components = [legacy_points.get(f"m{level}__{model}::{metric}", {}).get(seed) for level in levels]
                if all(value is not None and np.isfinite(value) for value in components):
                    legacy_points.setdefault(name, {})[seed] = float(np.mean(components))
                else:
                    legacy_anchor_unavailable.append({"estimate": name, "seed": seed,
                                                      "reason": "historical_M2_reports_per_severity_not_macro"})

    competitors = ("E1b", "Aggregate-MLP", "LCR-noGate", "R1")
    for competitor in competitors:
        for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
            name = f"severity_macro__LCR_minus_{competitor}::{metric}"
            specs.append({"name": name, "metric": metric, "operation": "macro_difference",
                          "conditions_a": [f"m{level}__LCR" for level in levels],
                          "conditions_b": [f"m{level}__{competitor}" for level in levels]})
            for seed in seeds:
                deltas = [legacy_points.get(f"m{level}__LCR::{metric}", {}).get(seed)
                          for level in levels]
                refs = [legacy_points.get(f"m{level}__{competitor}::{metric}", {}).get(seed)
                        for level in levels]
                if all(v is not None and np.isfinite(v) for v in deltas + refs):
                    legacy_points.setdefault(name, {})[seed] = float(np.mean(np.asarray(deltas) - np.asarray(refs)))
                else:
                    legacy_anchor_unavailable.append({"estimate": name, "seed": seed,
                                                      "reason": "historical_M2_per_severity_point_component_unavailable"})
            # M2 published per-level percentile intervals but no joint macro raw
            # draws, so there is no honest legacy macro CI to anchor.

    for level in levels:
        available_pairs = ["E1b", "Aggregate-MLP"] + (["LCR-noGate", "R1"] if level == 8 else [])
        for competitor in available_pairs:
            for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
                name = f"m{level}__LCR_minus_{competitor}::{metric}"
                specs.append({"name": name, "metric": metric, "operation": "difference",
                              "a": f"m{level}__LCR", "b": f"m{level}__{competitor}",
                              "scale": 100.0 if metric in ("rer_at_50", "rer_at_80") else 1.0})
                pair_rows = old_pairs[(old_pairs["cohort"] == f"m{level}")
                                      & (old_pairs["seed"].astype(int).isin([1, 2, 3]))
                                      & (old_pairs["model_a"] == "LCR")
                                      & (old_pairs["model_b"] == competitor)
                                      & (old_pairs["metric"] == metric)]
                if len(pair_rows) != 3:
                    raise ValueError(f"M2 old paired bootstrap missing: m{level}/LCR-{competitor}/{metric}")
                for _, row in pair_rows.iterrows():
                    seed = f"b3_seed{int(row['seed'])}"
                    scale = 100.0 if metric in ("rer_at_50", "rer_at_80") else 1.0
                    legacy_points.setdefault(name, {})[seed] = float(row["diff"]) * scale
                    legacy_ci.setdefault(name, {})[seed] = sorted((float(row["ci_low"]) * scale,
                                                                    float(row["ci_high"]) * scale))

    old_m2_gate = gate_doc.get("gates", {}).get("M2_GO", {})
    return [{
        "job_id": "m2_curriculum_m0_m2_m4_m8", "scope": "V2M_M2_curriculum",
        "cohort": "same8_m0_m2_m4_m8", "image_ids": images_ref, "seed_cells": cells,
        "estimates": specs, "metrics": CI_METRICS, "legacy_points": legacy_points,
        "old_ci": legacy_ci, "legacy_anchor_unavailable": legacy_anchor_unavailable,
        "old_gate": old_m2_gate.get("verdict", "NOT_RECORDED"),
        "old_gate_detail": old_m2_gate,
        "thresholds": protocol.get("M2_preregistration", {}).get("gate", {}),
        "source_paths": list(dict.fromkeys(source_paths)), "point_order": np.arange(ids_ref.size, dtype=np.int64),
        "point_estimate_ordering": "ascending_native_sentence_id_same_as_historical_runner",
        "cohort_construction": "frozen_Phase1F_expB_same8_testA_testB_intersection_all_seeds_and_m0_m2_m4_m8",
        "cohort_metadata": {
            "split_counts": split_counts, "contains_train": False, "contains_validation": False,
            "heldout_test_only": True,
            "historical_pool_identity": "same8 testA+testB row universe; confidence archive bridged to native expb sentence_id rows",
        },
        "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_M2_SAME8_TESTA_TESTB",
        "legacy_pool_status": "EXISTING_LEGACY_M2_SAME8",
        "legacy_cohort_note": "M2 confidence NPZ lacks sentence IDs. Row identity is bridged to expb_m{0,2,4,8} frozen per-sentence predictions, then mapped to sorted Phase0B native IDs; image/correct arrays are checked exactly. Historical table reports severity-specific estimates; macro effects are supplementary and macro CIs are unavailable in the old results.",
        "sentence_id_status": "VERIFIED_BY_ROWWISE_EXPb_AND_PHASE0B_BRIDGE",
        "evidence": f"All 12 expb archives have ascending sentence IDs and identical same8 rows (n={ids_ref.size}); M2 confidence image/correct arrays equal expb arrays per cell; Phase0B sentence/ref/image/split anchors match rowwise; split counts={split_counts}.",
        "recovery_status": "READ_ONLY_CONFIDENCE_ARCHIVE_WITH_VERIFIED_NATIVE_ID_BRIDGE",
        "model_identity_by_seed": {seed: {"scorer": seed, "models": list(models)} for seed in seeds},
        "prediction_source_precision": "archived_float64_confidence_and_binary_correctness",
    }]


def _proposal_cardinality_job(family: str, point_path: Path, boot_path: Path,
                              verdict_path: Path, prediction_root: Path,
                              *, scope: str) -> dict[str, Any]:
    """Build one native-ID, common-K50 P1/P2 cardinality replay."""
    old_points, old_boot = read_csv(point_path), read_csv(boot_path)
    verdict = read_json(verdict_path)
    seeds = tuple(f"b3_seed{s}" for s in (1, 2, 3))
    ks = (5, 10, 20, 50)
    arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    row_sets: list[set[int]] = []
    source_paths = [point_path, boot_path, verdict_path]
    for k in ks:
        first: dict[str, np.ndarray] | None = None
        for seed in seeds:
            path = prediction_root / f"p1__{family}__random_k{k}__{seed}.npz"
            data = load_npz(path)
            arrays[(k, seed)] = data
            source_paths.append(path)
            ids = np.asarray(data["sentence_id"], dtype=np.int64)
            if ids.size != np.unique(ids).size:
                raise ValueError(f"P1/P2 {family}/K{k}/{seed}: duplicate sentence IDs")
            if first is None:
                first = data
                row_sets.append(set(ids.tolist()))
            else:
                assert_row_alignment(
                    {key: first[key] for key in ("sentence_id", "ref_id", "image_id")},
                    {key: data[key] for key in ("sentence_id", "ref_id", "image_id")},
                    required_fields=("ref_id", "image_id"), candidate_name=f"{family}/K{k}/{seed}",
                )
    common_ids = np.asarray(sorted(set.intersection(*row_sets)), dtype=np.int64)
    if common_ids.size == 0:
        raise ValueError(f"P1/P2 {family}: common-K50 intersection is empty")
    seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {seed: {} for seed in seeds}
    point_orders: dict[str, dict[str, np.ndarray]] = {seed: {} for seed in seeds}
    legacy_points: dict[str, dict[str, float]] = {}
    old_ci: dict[str, dict[str, list[float]]] = {}
    unavailable: list[dict[str, Any]] = []
    image_ref: np.ndarray | None = None
    ref_ref: np.ndarray | None = None
    old_columns = {"accuracy": "accuracy", "auroc_correct": "auroc_correct", "e_aurc": "e_aurc",
                   "rer_at_50": "rer_at_50", "rer_at_80": "rer_at_80"}
    for seed in seeds:
        for k in ks:
            data = arrays[(k, seed)]
            raw_ids = np.asarray(data["sentence_id"], dtype=np.int64)
            pos = _canonical_indices(raw_ids, common_ids, label=f"P1/P2 {family}/K{k}/{seed}")
            refs = np.asarray(data["ref_id"], dtype=np.int64)[pos]
            images = np.asarray(data["image_id"], dtype=np.int64)[pos]
            corr = np.asarray(data["correct"], dtype=np.float64)[pos]
            if image_ref is None:
                image_ref, ref_ref = images.copy(), refs.copy()
            elif not np.array_equal(image_ref, images) or not np.array_equal(ref_ref, refs):
                raise ValueError(f"P1/P2 {family}/K{k}/{seed}: canonical common-K50 IDs differ")
            condition = f"K{k}__global_T"
            seed_cells[seed][condition] = (np.asarray(data["conf_msp"], dtype=np.float64)[pos], corr)
            # The original loader sorts by sentence_id; canonical_ids has that
            # order even though the archive's physical rows are unsorted.
            point_orders[seed][condition] = np.arange(common_ids.size, dtype=np.int64)
            old_row = old_points[(old_points["family"] == family)
                                 & (old_points["model"] == "global_T_corrected")
                                 & (old_points["seed"].astype(str) == f"seed{seed[-1]}")
                                 & (old_points["K"].astype(int) == k)]
            if len(old_row) != 1:
                raise ValueError(f"P1/P2 {family} old point missing/ambiguous: {seed}/K{k}")
            row = old_row.iloc[0]
            if int(row["n"]) != common_ids.size or int(row["n_images"]) != np.unique(images).size:
                raise ValueError(f"P1/P2 {family} common-K50 row/image count anchor failed: {seed}/K{k}")
            for metric in CI_METRICS:
                estimate = f"{condition}::{metric}"
                col = old_columns.get(metric)
                value = _float_or_nan(row.get(col)) if col else float("nan")
                if np.isfinite(value):
                    legacy_points.setdefault(estimate, {})[seed] = value
                else:
                    unavailable.append({"estimate": estimate, "seed": seed,
                                        "reason": "historical_C1_point_table_did_not_report_metric"})

    specs: list[dict[str, Any]] = []
    for k in ks:
        condition = f"K{k}__global_T"
        for metric in CI_METRICS:
            specs.append({"name": f"{condition}::{metric}", "metric": metric,
                          "operation": "condition", "condition": condition})
    # The frozen C1 tables define changes as K5 minus Kb. E-AURC worsening is
    # reported in the gate as Kb minus K5 and normalized by K5 E-AURC.
    for kb in (10, 20, 50):
        low, high = f"K5__global_T", f"K{kb}__global_T"
        for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
            name = f"K5_minus_K{kb}::{metric}"
            specs.append({"name": name, "metric": metric, "operation": "difference",
                          "a": low, "b": high, "scale": 100.0 if metric.startswith("rer_") else 1.0})
            old_rows = old_boot[(old_boot["family"] == family)
                                & (old_boot["seed"].astype(str).isin([f"seed{s[-1]}" for s in seeds]))
                                & (old_boot["K_a"].astype(int) == 5)
                                & (old_boot["K_b"].astype(int) == kb)
                                & (old_boot["metric"] == metric)
                                & (old_boot["diff_kind"] == "absolute")]
            if len(old_rows) != 3:
                raise ValueError(f"P1/P2 {family} C1 absolute anchor missing: K5/K{kb}/{metric}")
            for _, row in old_rows.iterrows():
                seed = f"b3_seed{int(str(row['seed'])[-1])}"
                scale = 100.0 if metric.startswith("rer_") else 1.0
                legacy_points.setdefault(name, {})[seed] = scale * float(row["diff"])
                old_ci.setdefault(name, {})[seed] = sorted((scale * float(row["ci_low"]), scale * float(row["ci_high"])))

        rel = f"K{kb}_relative_eaurc_worsening_over_K5"
        specs.append({"name": rel, "metric": "e_aurc", "operation": "relative_contrast",
                      "contrast_positive": high, "contrast_negative": low,
                      "denominator_condition": low})
        old_rows = old_boot[(old_boot["family"] == family)
                            & (old_boot["seed"].astype(str).isin([f"seed{s[-1]}" for s in seeds]))
                            & (old_boot["K_a"].astype(int) == 5)
                            & (old_boot["K_b"].astype(int) == kb)
                            & (old_boot["metric"] == "e_aurc")
                            & (old_boot["diff_kind"] == "relative")]
        if len(old_rows) != 3:
            raise ValueError(f"P1/P2 {family} C1 relative anchor missing: K5/K{kb}/e_aurc")
        for _, row in old_rows.iterrows():
            seed = f"b3_seed{int(str(row['seed'])[-1])}"
            # Legacy stores r=(K5-Kb)/Kb. The frozen gate consumes the
            # K5-baselined worsening w=(Kb-K5)/K5=-r/(1+r).
            transform = lambda value: -float(value) / (1.0 + float(value))
            legacy_points.setdefault(rel, {})[seed] = transform(row["diff"])
            old_ci.setdefault(rel, {})[seed] = sorted((transform(row["ci_low"]), transform(row["ci_high"])))

    family_gates = verdict.get("c1_by_family", {}).get(family, {}).get("gate", {})
    text_index_path = ROOT / "cache" / "features" / "text_index.csv"
    split_counts = _split_metadata_from_text_index(common_ids, ref_ref, image_ref)
    split_names = set(split_counts)
    splits_path = prediction_root.parent / "inference_report.json"
    if splits_path.exists():
        source_paths.append(splits_path)
    source_paths.append(text_index_path)
    return {
        "job_id": f"p1_c1_{family}", "scope": scope, "cohort": "common_k50",
        "image_ids": image_ref, "seed_cells": seed_cells, "estimates": specs,
        "point_order_by_seed_condition": point_orders,
        "metrics": CI_METRICS, "legacy_points": legacy_points, "old_ci": old_ci,
        "legacy_anchor_unavailable": unavailable,
        "old_gate": family_gates.get("replicated", "NOT_RECORDED"),
        "thresholds": family_gates.get("thresholds", {}),
        "source_paths": list(dict.fromkeys(source_paths)),
        "point_estimate_ordering": "ascending_sentence_id_as_in_original_p1_loader; bootstrap uses the same canonical row order",
        "cohort_construction": "intersection_of_native_sentence_ids_across_K5_K10_K20_K50_and_all_three_seeds",
        "cohort_metadata": {"split_counts": split_counts,
                            "contains_train": any(name.lower().startswith("train") for name in split_names),
                            "contains_validation": any(name.lower().startswith(("val", "valid")) for name in split_names),
                            "heldout_test_only": bool(split_names) and split_names <= {"testA", "testB"},
                            "historical_pool_identity": "historical C1 common-K50 cohort, bridged rowwise by sentence_id/ref_id/image_id to frozen text_index.csv"},
        "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_P1_P2_COMMON_K50",
        "legacy_pool_status": "EXISTING_LEGACY_COMMON_K50",
        "sentence_id_status": "VERIFIED_NATIVE_ID",
        "evidence": f"Native sentence/ref/image arrays intersect exactly across all K and fixed seeds; row/image counts checked against historical family point table (n={common_ids.size}).",
        "recovery_status": "READ_ONLY_NATIVE_ID_PREDICTIONS",
        "model_identity_by_seed": {seed: {"scorer": seed, "model": "frozen global_T_corrected"} for seed in seeds},
        "prediction_source_precision": "archived_float64_conf_msp_and_binary_correctness",
        "source_paths": list(dict.fromkeys(source_paths)),
        "family_gate_record": family_gates,
    }


def build_p_jobs() -> list[dict[str, Any]]:
    root = ROOT / "results" / "v2_proposal_robustness"
    pred_root = root / "predictions"
    p1 = root / "p1_f4_c1"
    p2 = root / "p2_c1_gdino"
    p1_verdict = read_json(p1 / "f4_verdict.json")
    p2_verdict = read_json(p2 / "p2_c1_verdict.json")
    jobs = [
        _proposal_cardinality_job("RPN", p1 / "c1_point.csv", p1 / "c1_bootstrap.csv", p1 / "f4_verdict.json", pred_root, scope="P1_F4_C1"),
        _proposal_cardinality_job("DETR", p1 / "c1_point.csv", p1 / "c1_bootstrap.csv", p1 / "f4_verdict.json", pred_root, scope="P1_F4_C1"),
        _proposal_cardinality_job("GDINO", p2 / "c1_point.csv", p2 / "c1_bootstrap.csv", p2 / "p2_c1_verdict.json", pred_root, scope="P2_C1_GDINO"),
    ]
    # P1-F5 C4 matched random/hard rows share the native IDs; continuous
    # manipulation features are added to the same image-cluster draws.
    c4 = root / "p1_f5_c4"
    selective = read_csv(c4 / "c4_selective.csv")
    amplification = read_csv(c4 / "c4_amplification.csv")
    manipulation = read_csv(c4 / "c4_manipulation.csv")
    f5 = read_json(c4 / "f5_verdict.json")
    for family in ("RPN", "DETR"):
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        point_orders: dict[str, dict[str, np.ndarray]] = {}
        aux_cells: dict[str, dict[str, np.ndarray]] = {}
        legacy_points: dict[str, dict[str, float]] = {}
        old_ci: dict[str, dict[str, list[float]]] = {}
        unavailable: list[dict[str, Any]] = []
        source_paths: list[Path] = [c4 / "c4_selective.csv", c4 / "c4_amplification.csv", c4 / "c4_manipulation.csv", c4 / "f5_verdict.json"]
        canonical_ids: np.ndarray | None = None
        image_ref: np.ndarray | None = None
        for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
            rand_path = pred_root / f"p1__{family}__random_k5__{seed}.npz"
            hard_path = pred_root / f"p1__{family}__hard_k5__{seed}.npz"
            rand, hard = load_npz(rand_path), load_npz(hard_path)
            source_paths.extend((rand_path, hard_path))
            rand_ids = np.asarray(rand["sentence_id"], dtype=np.int64)
            hard_ids = np.asarray(hard["sentence_id"], dtype=np.int64)
            if rand_ids.size != np.unique(rand_ids).size or hard_ids.size != np.unique(hard_ids).size:
                raise ValueError(f"P1 C4 {family}/{seed}: duplicate sentence IDs in random/hard archive")
            common = np.intersect1d(rand_ids, hard_ids, assume_unique=True)
            ids = common if canonical_ids is None else np.intersect1d(canonical_ids, common, assume_unique=True)
            if ids.size == 0:
                raise ValueError(f"P1 C4 {family}: empty hard/random identity intersection")
            canonical_ids = ids
        assert canonical_ids is not None
        for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
            rand = load_npz(pred_root / f"p1__{family}__random_k5__{seed}.npz")
            hard = load_npz(pred_root / f"p1__{family}__hard_k5__{seed}.npz")
            seed_cells[seed], aux_cells[seed], point_orders[seed] = {}, {}, {}
            for regime, data in (("rand", rand), ("hard", hard)):
                raw_ids = np.asarray(data["sentence_id"], dtype=np.int64)
                idx = _canonical_indices(raw_ids, canonical_ids, label=f"P1 C4 {family}/{seed}/{regime}")
                refs = np.asarray(data["ref_id"], dtype=np.int64)[idx]
                images = np.asarray(data["image_id"], dtype=np.int64)[idx]
                correct = np.asarray(data["correct"], dtype=np.float64)[idx]
                if regime == "rand":
                    if image_ref is None:
                        image_ref = images.copy()
                        canonical_ref = refs.copy()
                    elif not np.array_equal(image_ref, images) or not np.array_equal(canonical_ref, refs):
                        raise ValueError(f"P1 C4 {family}/{seed}: common row identity differs across seeds")
                    seed_cells[seed][f"{regime}__R1"] = (np.asarray(data["conf_stats"], dtype=np.float64)[idx], correct)
                    seed_cells[seed][f"{regime}__E1b"] = (np.asarray(data["conf_e1b"], dtype=np.float64)[idx], correct)
                    point_orders[seed][f"{regime}__R1"] = np.arange(canonical_ids.size, dtype=np.int64)
                    point_orders[seed][f"{regime}__E1b"] = point_orders[seed][f"{regime}__R1"].copy()
                    expected_refs, expected_images = refs.copy(), images.copy()
                elif not np.array_equal(refs, expected_refs) or not np.array_equal(images, expected_images):
                    raise ValueError(f"P1 C4 {family}/{seed}: random/hard common row anchors differ")
                seed_cells[seed][f"{regime}__R1"] = (np.asarray(data["conf_stats"], dtype=np.float64)[idx], correct)
                seed_cells[seed][f"{regime}__E1b"] = (np.asarray(data["conf_e1b"], dtype=np.float64)[idx], correct)
                if regime == "hard":
                    point_orders[seed][f"{regime}__R1"] = np.arange(canonical_ids.size, dtype=np.int64)
                    point_orders[seed][f"{regime}__E1b"] = point_orders[seed][f"{regime}__R1"].copy()
                sem = np.asarray(data["sem14"], dtype=np.float64)[idx]
                for feature, _direction in MANIPULATION_METRICS:
                    col = V2_PRIMARY_SEMANTIC_NAMES.index(feature)
                    aux_cells[seed][f"{regime}__feature__{feature}"] = sem[:, col]
            if seed_cells[seed]["rand__R1"][0].size != canonical_ids.size:
                raise ValueError(f"P1 C4 {family}/{seed}: confidence-row count mismatch")

        specs: list[dict[str, Any]] = []
        for regime in ("rand", "hard"):
            for model in ("R1", "E1b"):
                cond = f"{regime}__{model}"
                for metric in CI_METRICS:
                    specs.append({"name": f"{cond}::{metric}", "metric": metric,
                                  "operation": "condition", "condition": cond})
        for regime in ("rand", "hard"):
            for metric in CI_METRICS:
                if metric not in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
                    continue
                name = f"{regime}__E1b_minus_R1::{metric}"
                scale = 100.0 if metric.startswith("rer_") else 1.0
                specs.append({"name": name, "metric": metric, "operation": "difference",
                              "a": f"{regime}__E1b", "b": f"{regime}__R1", "scale": scale})
        specs.extend([
            {"name": "delta_hard_auroc", "metric": "auroc_correct", "operation": "difference", "a": "hard__E1b", "b": "hard__R1"},
            {"name": "delta_rand_auroc", "metric": "auroc_correct", "operation": "difference", "a": "rand__E1b", "b": "rand__R1"},
            {"name": "amplification_auroc", "metric": "auroc_correct", "operation": "difference_of_differences",
             "a": "hard__E1b", "b": "hard__R1", "c": "rand__E1b", "d": "rand__R1"},
        ])
        aux_specs: list[dict[str, Any]] = []
        for feature, direction in MANIPULATION_METRICS:
            left, right = f"hard__feature__{feature}", f"rand__feature__{feature}"
            raw_name = f"manipulation__{feature}__hard_minus_rand"
            aux_specs.append({"name": raw_name, "metric": "mean", "operation": "difference", "a": left, "b": right})
            directional = 1.0 if direction == "up" else -1.0
            aux_specs.append({"name": f"manipulation__{feature}__directional", "metric": "mean",
                              "operation": "difference", "a": left, "b": right, "scale": directional})
        for _, row in amplification[amplification["family"] == family].iterrows():
            seed = str(row["scorer"])
            if int(row["n"]) != canonical_ids.size:
                raise ValueError(f"P1 C4 {family}/{seed}: amplification row count anchor mismatch")
            for regime in ("hard", "rand"):
                for model, old_column in (("R1", f"auroc_r1_{regime}"), ("E1b", f"auroc_e1b_{regime}")):
                    legacy_points.setdefault(f"{regime}__{model}::auroc_correct", {})[seed] = float(row[old_column])
            for name, col in (("delta_hard_auroc", "delta_hard"), ("delta_rand_auroc", "delta_rand"),
                              ("amplification_auroc", "amplification")):
                legacy_points.setdefault(name, {})[seed] = float(row[col])
            old_ci.setdefault("delta_hard_auroc", {})[seed] = [float(row["delta_hard_ci_low"]), float(row["delta_hard_ci_high"])]
            old_ci.setdefault("delta_rand_auroc", {})[seed] = [float(row["delta_rand_ci_low"]), float(row["delta_rand_ci_high"])]
            old_ci.setdefault("amplification_auroc", {})[seed] = [float(row["amplification_ci_low"]), float(row["amplification_ci_high"])]
        for _, row in selective[selective["family"] == family].iterrows():
            seed, regime = str(row["scorer"]), str(row["regime"])
            condition_regime = "rand" if regime == "random" else regime
            for model, suffix in (("R1", "r1"), ("E1b", "e1b")):
                legacy_points.setdefault(f"{condition_regime}__{model}::e_aurc", {})[seed] = float(row[f"e_aurc_{suffix}"])
                legacy_points.setdefault(f"{condition_regime}__{model}::rer_at_50", {})[seed] = float(row[f"rer50_{suffix}"])
                legacy_points.setdefault(f"{condition_regime}__{model}::rer_at_80", {})[seed] = float(row[f"rer80_{suffix}"])
            for metric, column, scale in (("e_aurc", "e1b_minus_r1_e_aurc_reduction", 1.0),
                                          ("rer_at_50", "e1b_minus_r1_rer50_gain", 100.0)):
                name = f"{condition_regime}__R1_minus_E1b::{metric}"
                # Both stored contrasts are R1-E1b. The legacy RER column is
                # mislabeled "e1b_minus_r1_rer50_gain" but its source computes
                # R1 - E1b; preserve the arithmetic and name the sign honestly.
                specs.append({"name": name, "metric": metric, "operation": "difference",
                              "a": f"{condition_regime}__R1", "b": f"{condition_regime}__E1b", "scale": scale})
                legacy_points.setdefault(name, {})[seed] = float(row[column]) * scale
        for _, row in manipulation[manipulation["family"] == family].iterrows():
            seed, feature = str(row["scorer"]), str(row["feature"])
            name = f"manipulation__{feature}__hard_minus_rand"
            legacy_points.setdefault(name, {})[seed] = float(row["diff"])
            old_ci.setdefault(name, {})[seed] = [float(row["ci_low"]), float(row["ci_high"])]
        family_doc = f5.get("c4_by_family", {}).get(family, {})
        text_index_path = ROOT / "cache" / "features" / "text_index.csv"
        c4_split_counts = _split_metadata_from_text_index(canonical_ids, canonical_ref, image_ref)
        c4_split_names = set(c4_split_counts)
        source_paths.append(text_index_path)
        cohort_meta = {"split_counts": c4_split_counts,
                       "contains_train": any(name.lower().startswith("train") for name in c4_split_names),
                       "contains_validation": any(name.lower().startswith(("val", "valid")) for name in c4_split_names),
                       "heldout_test_only": bool(c4_split_names) and c4_split_names <= {"testA", "testB"},
                       "historical_pool_identity": "historical same-category hard5 ∩ random5 cohort, bridged rowwise by sentence_id/ref_id/image_id to frozen text_index.csv"}
        jobs.append({
            "job_id": f"p1_c4_{family}", "scope": "P1_F5_C4", "cohort": f"samecategory_k5_{family}",
            "image_ids": image_ref, "seed_cells": seed_cells, "estimates": specs, "metrics": CI_METRICS,
            "point_order_by_seed_condition": point_orders,
            "auxiliary_cells": aux_cells, "auxiliary_estimates": aux_specs,
            "legacy_points": legacy_points, "old_ci": old_ci,
            "old_gate": family_doc.get("gate", {}).get("verdict", "NOT_RECORDED"),
            "thresholds": family_doc.get("gate", {}).get("thresholds", {}),
            "legacy_anchor_unavailable": unavailable,
            "source_paths": list(dict.fromkeys(source_paths)),
            "point_estimate_ordering": "ascending_sentence_id_as_in_original_p1_loader; bootstrap uses the same canonical row order",
            "cohort_construction": "intersection_of_native_sentence_ids_across_three_seeds_and_matched_random_hard_cells",
            "cohort_metadata": cohort_meta,
            "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_P1_F5_SAMECATEGORY_C4",
            "legacy_pool_status": "EXISTING_LEGACY_MATCHED_COHORT",
            "sentence_id_status": "VERIFIED_NATIVE_ID",
            "evidence": f"Per-family random/hard arrays intersect on identical sentence/ref/image rows across all three seeds; n={canonical_ids.size}; continuous sem14 checks and frozen c4 table anchors retained.",
            "recovery_status": "READ_ONLY_NATIVE_ID_PREDICTIONS",
            "model_identity_by_seed": {seed: {"scorer": seed, "R1": "stats_logistic", "E1b": "e1b_stats_semantic"} for seed in seed_cells},
            "prediction_source_precision": "archived_float64_confidence_and_sem14",
            "family_gate_record": family_doc.get("gate", {}),
        })
    return jobs


def build_v2g_g4_jobs() -> list[dict[str, Any]]:
    """Use only the strictly anchored B1/B2 recovery archives for G4."""
    jobs: list[dict[str, Any]] = []
    metrics = ("auroc_correct", "e_aurc", "rer_at_50")
    for backbone in ("b1", "b2"):
        job_id = f"v2g_g4_{backbone}"
        records = _completed_recovery_job(job_id)
        gate_path = ROOT / "results" / "v2_backbone_generalization" / "g4_phaseB" / backbone / "gate.json"
        old_gate = read_json(gate_path)
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        point_orders: dict[str, dict[str, np.ndarray]] = {}
        image_ids: np.ndarray | None = None
        reference_identity: dict[str, np.ndarray] | None = None
        legacy_points: dict[str, dict[str, float]] = {
            "delta_hard_auroc": {}, "amplification_auroc": {},
        }
        source_paths: list[Path] = []
        model_identity: dict[str, Any] = {}
        for record in records:
            seed = f"b3_seed{record['seed']}"
            seed_cells[seed], point_orders[seed] = {}, {}
            source_paths.extend(
                ROOT / str(item["path"])
                for item in record["sidecar"].get("source_files", [])
                if item.get("status") != "SUPPLEMENTAL_SOURCE_SNAPSHOT"
            )
            source_paths.extend((record["archive_path"], record["sidecar_path"]))
            model_identity[seed] = record["sidecar"].get("model_fit_provenance", {})
            for regime in ("rand5", "hard5"):
                raw = _recovery_cell(record, regime)
                cell, point_order = _canonicalize_recovery_cell(raw, label=f"{job_id}/{seed}/{regime}")
                # Random and hard K=5 rows share query/ref/image identities,
                # while candidate correctness is allowed to differ by regime.
                identity = {field: cell[field] for field in ("sentence_id", "ref_id", "image_id", "eval_split")}
                if reference_identity is None:
                    reference_identity = identity
                    image_ids = np.asarray(cell["image_id"], dtype=np.int64)
                else:
                    assert_row_alignment(reference_identity, identity,
                                         required_fields=("ref_id", "image_id", "eval_split"),
                                         reference_name=f"{job_id}/canonical", candidate_name=f"{job_id}/{seed}/{regime}")
                for source_condition, short_condition in (("R1_stats", "R1"), ("R2_stats_sem", "R2")):
                    if source_condition not in cell["predictions"]:
                        raise ValueError(f"{job_id}/{regime}: missing archived condition {source_condition}")
                    condition = f"{regime}__{short_condition}"
                    seed_cells[seed][condition] = (
                        np.asarray(cell["predictions"][source_condition], dtype=np.float64),
                        np.asarray(cell["correct"], dtype=np.float64),
                    )
                    point_orders[seed][condition] = point_order
        assert image_ids is not None
        for seed in seed_cells:
            legacy_points["delta_hard_auroc"][seed] = float(old_gate["per_seed_delta_hard"][seed])
            legacy_points["amplification_auroc"][seed] = float(old_gate["per_seed_amplification"][seed])
        specs: list[dict[str, Any]] = []
        for regime in ("rand5", "hard5"):
            for model in ("R1", "R2"):
                condition = f"{regime}__{model}"
                for metric in metrics:
                    specs.append({"name": f"{condition}::{metric}", "metric": metric,
                                  "operation": "condition", "condition": condition})
        specs.extend([
            {"name": "delta_hard_auroc", "metric": "auroc_correct", "operation": "difference",
             "a": "hard5__R2", "b": "hard5__R1"},
            {"name": "delta_rand_auroc", "metric": "auroc_correct", "operation": "difference",
             "a": "rand5__R2", "b": "rand5__R1"},
            {"name": "amplification_auroc", "metric": "auroc_correct", "operation": "difference_of_differences",
             "a": "hard5__R2", "b": "hard5__R1", "c": "rand5__R2", "d": "rand5__R1"},
        ])
        source_paths.extend((gate_path, ROOT / "results" / "v2_backbone_generalization" / "g4_phaseB" / "g5_overall_gate.json"))
        source_paths = list(dict.fromkeys(path.resolve() for path in source_paths))
        all_splits = np.asarray(reference_identity["eval_split"]).astype(str)
        split_names, split_counts = np.unique(all_splits, return_counts=True)
        jobs.append({
            "job_id": job_id, "scope": "V2G_G4", "cohort": "matched_rand5_hard5_testA_testB",
            "image_ids": image_ids, "seed_cells": seed_cells, "estimates": specs, "metrics": metrics,
            "point_order_by_seed_condition": point_orders, "legacy_points": legacy_points,
            "old_gate": old_gate.get("verdict", "NOT_RECORDED"), "thresholds": old_gate.get("thresholds", {}),
            "source_paths": source_paths,
            "point_estimate_ordering": "native sentence_id ascending for bootstrap; point metrics replay archived original row order",
            "cohort_construction": "matched_samecategory_K5_random_hard_shared_native_sentence_ids",
            "cohort_metadata": _cohort_metadata_from_counts(
                {str(name): int(count) for name, count in zip(split_names, split_counts)},
                historical_pool_identity=f"EXISTING_V2G_G4_{backbone.upper()}_MATCHED_RANDOM_HARD_K5",
                heldout_test_only=True,
            ),
            "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_V2G_G4_SAMECATEGORY_TESTA_TESTB",
            "legacy_pool_status": "EXISTING_LEGACY_MATCHED_COHORT",
            "sentence_id_status": "VERIFIED_RECOVERY_ARCHIVE_NATIVE_IDS",
            "evidence": "Recovery manifest, archive and sidecar SHA match; all stored G4 anchors pass; row identity is checked across random/hard cells and fixed seeds.",
            "recovery_status": "STRICT_FROZEN_PHASE_A_CHECKPOINT_FORWARD",
            "model_identity_by_seed": model_identity,
            "prediction_source_precision": "recovery_archive_float64_probabilities",
            "recovery_output_path": records[0]["manifest_path"].resolve().relative_to(ROOT.resolve()).as_posix(),
            "recovery_output_sha256": sha256(records[0]["manifest_path"]),
            "manipulation_validity_from_frozen_gate": bool(old_gate.get("manipulation_ok", False)),
        })
    return jobs


def build_v2m_m25_jobs() -> list[dict[str, Any]]:
    """Read-only diagnostic effects for the four frozen M25 severity cells."""
    records = _completed_recovery_job("v2m_m25")
    verdict_path = ROOT / "results" / "v2_local_competition" / "m25_specialist_audit" / "verdict.json"
    verdict = read_json(verdict_path)
    levels = (0, 2, 4, 8)
    metrics = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
    seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    point_orders: dict[str, dict[str, np.ndarray]] = {}
    image_ids: np.ndarray | None = None
    reference_identity: dict[str, np.ndarray] | None = None
    legacy_points: dict[str, dict[str, float]] = {}
    old_ci: dict[str, dict[str, list[float]]] = {}
    model_identity: dict[str, Any] = {}
    for record in records:
        seed = f"b3_seed{record['seed']}"
        seed_cells[seed], point_orders[seed] = {}, {}
        model_identity[seed] = record["sidecar"].get("model_fit_provenance", {})
        for level in levels:
            raw = _recovery_cell(record, f"m{level}")
            cell, point_order = _canonicalize_recovery_cell(raw, label=f"v2m_m25/{seed}/m{level}")
            # Candidate correctness may legitimately change with severity;
            # the matched row universe is established by native IDs and
            # sentence/ref/image/split identity.
            identity = {field: cell[field] for field in ("sentence_id", "ref_id", "image_id", "eval_split")}
            if reference_identity is None:
                reference_identity = identity
                image_ids = np.asarray(cell["image_id"], dtype=np.int64)
            else:
                assert_row_alignment(reference_identity, identity,
                                     required_fields=("ref_id", "image_id", "eval_split"),
                                     reference_name="v2m_m25/canonical", candidate_name=f"{seed}/m{level}")
            for model in ("E1b_random", "E1b_curriculum"):
                if model not in cell["predictions"]:
                    raise ValueError(f"v2m_m25/{seed}/m{level}: missing {model}")
                condition = f"m{level}__{model}"
                seed_cells[seed][condition] = (
                    np.asarray(cell["predictions"][model], dtype=np.float64),
                    np.asarray(cell["correct"], dtype=np.float64),
                )
                point_orders[seed][condition] = point_order
    assert image_ids is not None
    specs: list[dict[str, Any]] = []
    for level in levels:
        for model in ("E1b_random", "E1b_curriculum"):
            condition = f"m{level}__{model}"
            for metric in metrics:
                specs.append({"name": f"{condition}::{metric}", "metric": metric,
                              "operation": "condition", "condition": condition})
        for metric in metrics:
            name = f"m{level}__E1b_curriculum_minus_random::{metric}"
            specs.append({"name": name, "metric": metric, "operation": "difference",
                          "a": f"m{level}__E1b_curriculum", "b": f"m{level}__E1b_random"})
            legacy = verdict["severity"][f"m{level}"][metric]
            for seed_num, value in legacy["per_seed"].items():
                legacy_points.setdefault(name, {})[f"b3_seed{seed_num}"] = float(value)
            for seed_num, limits in legacy.get("per_seed_ci", {}).items():
                old_ci.setdefault(name, {})[f"b3_seed{seed_num}"] = [float(limits[0]), float(limits[1])]
    source_paths = _recovery_source_paths(
        records,
        (verdict_path, ROOT / "results" / "v2_local_competition" / "protocol.json",
         ROOT / "results" / "v2_local_competition" / "m25_specialist_audit" / "metadata.json"),
    )
    split_names, split_counts = np.unique(np.asarray(reference_identity["eval_split"]).astype(str), return_counts=True)
    return [{
        "job_id": "m25_specialist_m0_m2_m4_m8", "scope": "V2M_M25_SPECIALIST_AUDIT",
        "cohort": "same8_m0_m2_m4_m8", "image_ids": image_ids,
        "seed_cells": seed_cells, "estimates": specs, "metrics": metrics,
        "point_order_by_seed_condition": point_orders,
        "legacy_points": legacy_points, "old_ci": old_ci,
        "old_gate": "DIAGNOSTIC_NO_CONFIRMATORY_GATE", "thresholds": {},
        "source_paths": source_paths,
        "point_estimate_ordering": "native sentence_id ascending for bootstrap; point metrics replay archived original row order",
        "cohort_construction": "existing_same8_frozen_M1_M2_specialist_m0_m2_m4_m8_rows",
        "cohort_metadata": _cohort_metadata_from_counts(
            {str(name): int(count) for name, count in zip(split_names, split_counts)},
            historical_pool_identity="EXISTING_M25_M1_M2_SAME8_TESTA_TESTB",
            heldout_test_only=True,
        ),
        "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_M25_SAME8_TESTA_TESTB",
        "legacy_pool_status": "EXISTING_LEGACY_MATCHED_COHORT",
        "sentence_id_status": "VERIFIED_RECOVERY_ARCHIVE_NATIVE_IDS",
        "evidence": "All three seed archives and sidecars hash-match their recovery manifest; M1/M2 confidence and M25 point-table anchors pass at the frozen 1e-9 tolerance.",
        "recovery_status": "FROZEN_M1_M2_COEFFICIENTS_AND_TRAIN_ONLY_NORMALIZATION",
        "model_identity_by_seed": model_identity,
        "prediction_source_precision": "recovery_archive_float64_probabilities",
        "recovery_output_path": records[0]["manifest_path"].resolve().relative_to(ROOT.resolve()).as_posix(),
        "recovery_output_sha256": sha256(records[0]["manifest_path"]),
        "diagnostic_interpretation": verdict.get("interpretation", {}),
    }]


def build_v2m_m3_conf_jobs() -> list[dict[str, Any]]:
    """Build each confirmatory M3.1 backbone only when all three seeds pass recovery."""
    levels = (0, 2, 4, 8)
    models = ("E_random", "E_curriculum", "EqualMix", "StaticMix", "AdaptiveMix")
    metrics = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80", "ece_adaptive", "brier_binary", "nll_binary")
    jobs: list[dict[str, Any]] = []
    for backbone in ("b1", "b2"):
        job_id = f"v2m_m3_conf_{backbone}"
        try:
            records = _completed_recovery_job(job_id)
        except FileNotFoundError:
            # An incomplete or failed recovery is tracked by the recovery ledger;
            # it must never be made into a partial-seed statistics job.
            continue
        gate_path = ROOT / "results" / "v2_local_competition" / "m3_mixture" / "conf" / backbone / "gate.json"
        old_gate = read_json(gate_path)
        point_path = ROOT / "results" / "v2_local_competition" / "m3_mixture" / "conf" / backbone / "point_metrics.csv"
        point_table = read_csv(point_path)
        seed_cells: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        point_orders: dict[str, dict[str, np.ndarray]] = {}
        image_ids: np.ndarray | None = None
        reference_identity: dict[str, np.ndarray] | None = None
        legacy_points: dict[str, dict[str, float]] = {}
        old_ci: dict[str, dict[str, list[float]]] = {}
        model_identity: dict[str, Any] = {}
        for record in records:
            seed_num = int(record["seed"])
            seed = f"b3_seed{seed_num}"
            seed_cells[seed], point_orders[seed] = {}, {}
            model_identity[seed] = record["sidecar"].get("model_fit_provenance", {})
            for level in levels:
                raw = _recovery_cell(record, f"m{level}")
                cell, point_order = _canonicalize_recovery_cell(raw, label=f"{job_id}/{seed}/m{level}")
                # The fixed query rows are shared across severity levels, but
                # correctness can change as the candidate pool changes.
                identity = {field: cell[field] for field in ("sentence_id", "ref_id", "image_id", "eval_split")}
                if reference_identity is None:
                    reference_identity = identity
                    image_ids = np.asarray(cell["image_id"], dtype=np.int64)
                else:
                    assert_row_alignment(reference_identity, identity,
                                         required_fields=("ref_id", "image_id", "eval_split"),
                                         reference_name=f"{job_id}/canonical", candidate_name=f"{seed}/m{level}")
                for model in models:
                    if model not in cell["predictions"]:
                        raise ValueError(f"{job_id}/{seed}/m{level}: missing archived condition {model}")
                    condition = f"m{level}__{model}"
                    seed_cells[seed][condition] = (
                        np.asarray(cell["predictions"][model], dtype=np.float64),
                        np.asarray(cell["correct"], dtype=np.float64),
                    )
                    point_orders[seed][condition] = point_order
        assert image_ids is not None
        point_index = point_table.set_index(["seed", "level", "model"], drop=False)

        def point_value(seed_num: int, level: int, model: str, metric: str) -> float:
            key = (seed_num, f"m{level}", model)
            if key not in point_index.index:
                raise ValueError(f"{job_id}: historical point row missing {key}")
            raw_value = point_index.loc[key][metric]
            if hasattr(raw_value, "__len__") and not isinstance(raw_value, str):
                raise ValueError(f"{job_id}: duplicate historical point row {key}")
            return float(raw_value)

        specs: list[dict[str, Any]] = []
        for level in levels:
            for model in models:
                condition = f"m{level}__{model}"
                for metric in metrics:
                    name = f"{condition}::{metric}"
                    specs.append({"name": name, "metric": metric, "operation": "condition", "condition": condition})
                    for seed in seed_cells:
                        legacy_points.setdefault(name, {})[seed] = point_value(int(seed[-1]), level, model, metric)
        for model in models:
            conditions = [f"m{level}__{model}" for level in levels]
            for metric in metrics:
                name = f"severity_macro__{model}::{metric}"
                specs.append({"name": name, "metric": metric, "operation": "macro_mean", "conditions": conditions})
                for seed in seed_cells:
                    legacy_points.setdefault(name, {})[seed] = float(np.mean([
                        point_value(int(seed[-1]), level, model, metric) for level in levels
                    ]))

        primary_name = "m3_primary_delta_macro_auroc"
        specs.append({"name": primary_name, "metric": "auroc_correct", "operation": "macro_difference",
                      "conditions_a": [f"m{level}__AdaptiveMix" for level in levels],
                      "conditions_b": [f"m{level}__StaticMix" for level in levels]})
        old_primary = old_gate["delta_macro"]["vs_static"]
        for seed_num, value in old_primary["per_seed"].items():
            seed = f"b3_seed{int(seed_num)}"
            legacy_points.setdefault(primary_name, {})[seed] = float(value)
            limits = old_primary.get("per_seed_ci", {}).get(str(seed_num))
            if limits is not None:
                old_ci.setdefault(primary_name, {})[seed] = [float(limits[0]), float(limits[1])]
        specs.extend([
            {"name": "m3_extreme_safety_m0_adaptive_minus_random", "metric": "auroc_correct",
             "operation": "difference", "a": "m0__AdaptiveMix", "b": "m0__E_random"},
            {"name": "m3_extreme_safety_m8_adaptive_minus_curriculum", "metric": "auroc_correct",
             "operation": "difference", "a": "m8__AdaptiveMix", "b": "m8__E_curriculum"},
            {"name": "m3_selective_macro_eaurc_relative_improvement", "metric": "e_aurc",
             "operation": "relative_contrast_of_macros",
             "contrast_positive_conditions": [f"m{level}__StaticMix" for level in levels],
             "contrast_negative_conditions": [f"m{level}__AdaptiveMix" for level in levels]},
            {"name": "m3_selective_macro_rer50_gain_pp", "metric": "rer_at_50",
             "operation": "macro_difference", "scale": 100.0,
             "conditions_a": [f"m{level}__AdaptiveMix" for level in levels],
             "conditions_b": [f"m{level}__StaticMix" for level in levels]},
        ])
        for seed in seed_cells:
            seed_num = int(seed[-1])
            for name, model_a, model_b, level in (
                ("m3_extreme_safety_m0_adaptive_minus_random", "AdaptiveMix", "E_random", 0),
                ("m3_extreme_safety_m8_adaptive_minus_curriculum", "AdaptiveMix", "E_curriculum", 8),
            ):
                legacy_points.setdefault(name, {})[seed] = (
                    point_value(seed_num, level, model_a, "auroc_correct")
                    - point_value(seed_num, level, model_b, "auroc_correct")
                )
            legacy_points.setdefault("m3_selective_macro_rer50_gain_pp", {})[seed] = 100.0 * float(np.mean([
                point_value(seed_num, level, "AdaptiveMix", "rer_at_50")
                - point_value(seed_num, level, "StaticMix", "rer_at_50") for level in levels
            ]))

        source_paths = _recovery_source_paths(
            records,
            (point_path, gate_path, ROOT / "results" / "v2_local_competition" / "m3_mixture" / "amendment_v2m31.json",
             ROOT / "results" / "v2_local_competition" / "m3_mixture" / "conf" / "cross_backbone_verdict.json"),
        )
        split_names, split_counts = np.unique(np.asarray(reference_identity["eval_split"]).astype(str), return_counts=True)
        jobs.append({
            "job_id": job_id, "scope": "V2M_M3_1_CONFIRMATORY", "cohort": "same8_m0_m2_m4_m8",
            "image_ids": image_ids, "seed_cells": seed_cells, "estimates": specs, "metrics": metrics,
            "point_order_by_seed_condition": point_orders,
            "legacy_points": legacy_points, "old_ci": old_ci,
            "old_gate": old_gate.get("pass", "NOT_RECORDED"),
            "thresholds": old_gate.get("frozen_thresholds", {}), "source_paths": source_paths,
            "point_estimate_ordering": "native sentence_id ascending for shared draws; condition points replay archived original row order",
            "cohort_construction": "frozen_same8_K10_shared_m0_m2_m4_m8_rows",
            "cohort_metadata": _cohort_metadata_from_counts(
                {str(name): int(count) for name, count in zip(split_names, split_counts)},
                historical_pool_identity=f"EXISTING_M3_1_{backbone.upper()}_SAME8_TESTA_TESTB",
                heldout_test_only=True,
            ),
            "legacy_pooled_anchor": True, "legacy_cohort_classification": "EXISTING_M3_1_SAME8_TESTA_TESTB",
            "legacy_pool_status": "EXISTING_LEGACY_MATCHED_COHORT",
            "sentence_id_status": "VERIFIED_RECOVERY_ARCHIVE_NATIVE_IDS",
            "evidence": "Recovery archives and sidecars hash-match the B manifest; all model/severity rows are shared and old point tables/primary per-seed effects are checked at original tolerances.",
            "recovery_status": "FROZEN_E_RANDOM_PLUS_ORIGINAL_M2_CURRICULUM_FIT_AND_TUNE_ONLY_MIXERS",
            "model_identity_by_seed": model_identity,
            "prediction_source_precision": "recovery_archive_float64_probabilities",
            "recovery_output_path": records[0]["manifest_path"].resolve().relative_to(ROOT.resolve()).as_posix(),
            "recovery_output_sha256": sha256(records[0]["manifest_path"]),
            "legacy_anchor_unavailable": [{
                "estimate": "m3_selective_macro_eaurc_relative_improvement",
                "reason": "legacy M3 reports relative improvement from the seed-mean macro E-AURC endpoints; repair forms the paired per-seed relative effect inside each shared image draw before seed averaging, so no point-equivalent legacy scalar is substituted",
            }],
        })
    return jobs


def _normalise_builder_specs(builder):
    """Collapse identical per-seed copies of job-level estimate declarations.

    A few historical adapters declare the same estimator while iterating each
    seed, even though the bootstrap API correctly requires one unique spec per
    estimate name. Conflicting definitions remain a hard error.
    """
    def wrapped():
        jobs = builder()
        for job in jobs:
            unique: dict[str, dict[str, Any]] = {}
            ordered: list[dict[str, Any]] = []
            removed = 0
            for spec in job["estimates"]:
                name = str(spec["name"])
                prior = unique.get(name)
                if prior is None:
                    unique[name] = spec
                    ordered.append(spec)
                elif prior == spec:
                    removed += 1
                else:
                    raise ValueError(f"conflicting estimate specs share name {name!r} in job {job['job_id']}")
            job["estimates"] = ordered
            if removed:
                job["duplicate_estimate_specs_collapsed"] = removed
        return jobs
    return wrapped


BUILDERS = {
    "phase0a": build_phase0a_jobs,
    "phase0b": build_phase0b_jobs,
    "phase05": build_phase05_jobs,
    "phase1": build_phase1_jobs,
    "phase1f": build_phase1f_jobs,
    "refcocog": build_refcocog_jobs,
    "d1": build_d1_jobs,
    "m1": build_v2m_m1_jobs,
    "m2": build_v2m_m2_jobs,
    "v2g_g3": build_v2g_g3_jobs,
    "v2g_g4": build_v2g_g4_jobs,
    "d2_c1": build_d2_c1_jobs,
    "d2_c4": build_d2_c4_jobs,
    "p": build_p_jobs,
    "m25": build_v2m_m25_jobs,
    "m3": build_v2m_m3_conf_jobs,
}
BUILDERS = {name: _normalise_builder_specs(builder) for name, builder in BUILDERS.items()}


def _serialized_estimate_summary(item: dict[str, Any]) -> dict[str, Any]:
    return item


def run_job(job: dict[str, Any], n_replicates: int, seed: int, ci: float) -> dict[str, Any]:
    _derive_legacy_points(job)
    source_records = [frozen_source(Path(path)) for path in job["source_paths"]]
    result = joint_prediction_bootstrap(
        job["image_ids"], job["seed_cells"], estimates=job["estimates"], metrics=job["metrics"],
        auxiliary_cells=job.get("auxiliary_cells"),
        auxiliary_estimates=job.get("auxiliary_estimates", ()),
        n_replicates=n_replicates, seed=seed, ci=ci, point_order=job.get("point_order"),
        point_order_by_seed_condition=job.get("point_order_by_seed_condition"),
    )
    raw_path = OUT / "raw_replicates" / f"{job['job_id']}.npz"
    _write_raw(raw_path, result)
    raw_rel = raw_path.relative_to(ROOT).as_posix()
    summary = _job_summary(job, result, source_records, raw_rel)
    summary["created_utc"] = utc_now()
    summary["config"] = {"n_replicates": n_replicates, "seed": seed, "ci_level": ci, "ci_method": "percentile"}
    return summary


def _load_existing_summary() -> dict[str, Any]:
    validity_rule = {
        "minimum_valid_fraction_for_exploratory_ci": 0.95,
        "minimum_valid_replicates": "ceil(0.95 * planned_replicates)",
        "gate_rule": "all planned replicates must be valid for gate_eligible=true; any CI with invalid draws is conditional on finite draws and is not gate-eligible",
        "rationale": "invalid draws can remove probability mass from the 2.5%/97.5% tails; below 95% valid draws the interval is explicitly uncertainty-insufficient",
    }
    path = OUT / "summary.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SUMMARY_SCHEMA:
            raise RuntimeError(f"Unexpected statistics summary schema: {data.get('schema')}")
        data["inference_scope"] = (
            "conditional test-sample uncertainty for each job's fixed model/seed set; "
            "report n_seeds from each estimate and show seed SD separately, which does not estimate training randomness"
        )
        data.setdefault("bootstrap", {})["validity_sufficiency_rule"] = validity_rule
        return data
    return {
        "schema": SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "classification": "RESULT_DRIVEN_SUPPLEMENTARY_REPAIR",
        "baseline_manifest": "results/research_repair_v1/input_manifest.json",
        "baseline_manifest_sha256": sha256(BASELINE_MANIFEST),
        "bootstrap": {
            "resample_unit": "image_cluster", "replicates": 5000, "seed": 0, "ci_level": 0.95,
            "method": "percentile", "validity_sufficiency_rule": validity_rule,
        },
        "inference_scope": "conditional test-sample uncertainty for each job's fixed model/seed set; report n_seeds from each estimate and show seed SD separately, which does not estimate training randomness",
        "jobs": [],
        "estimates": [],
    }


def _backfill_uncertainty_metadata(estimate: dict[str, Any], planned_replicates: int) -> None:
    """Add the current validity rule to summaries created before it was recorded."""
    n_replicates = int(estimate.get("n_replicates", planned_replicates))
    valid = int(estimate.get("valid_replicates", 0))
    invalid = int(estimate.get("invalid_replicates", max(0, n_replicates - valid)))
    n_replicates = int(estimate.get("n_replicates", valid + invalid))
    minimum_valid = int(np.ceil(0.95 * n_replicates)) if n_replicates else 0
    if n_replicates and valid == n_replicates:
        status, gate_eligible = "SUFFICIENT_ALL_PLANNED_REPLICATES_VALID", True
    elif valid >= minimum_valid and valid >= 2:
        status, gate_eligible = "CONDITIONAL_ON_VALID_DRAWS_NOT_GATE_ELIGIBLE", False
    else:
        status, gate_eligible = "UNCERTAINTY_INSUFFICIENT", False
    estimate.update({
        "n_replicates": n_replicates,
        "valid_replicates": valid,
        "invalid_replicates": invalid,
        "valid_fraction": float(valid / n_replicates) if n_replicates else 0.0,
        "uncertainty_status": status,
        "gate_eligible": gate_eligible,
        "ci_conditioning": "all_planned_draws_valid" if valid == n_replicates else "conditional_on_finite_draws",
        "minimum_valid_fraction_for_exploratory_ci": 0.95,
        "minimum_valid_replicates_for_exploratory_ci": minimum_valid,
    })


def _persist(summary: dict[str, Any], *, stage_status: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    point_rows = []
    coverage_rows = []
    for job_summary in summary["jobs"]:
        for estimate in job_summary["estimates"]:
            planned = int(estimate.get("n_replicates", job_summary.get("result", {}).get("n_replicates", 0)))
            _backfill_uncertainty_metadata(estimate, planned)
            for seed_stats in estimate.get("per_seed", {}).values():
                seed_n = int(seed_stats.get("valid_replicates", 0)) + int(seed_stats.get("invalid_replicates", 0))
                _backfill_uncertainty_metadata(seed_stats, seed_n)
            point_rows.append({
                "scope": estimate["scope"], "cohort": estimate["cohort"], "estimate": estimate["name"],
                "metric": estimate["metric"], "operation": estimate.get("operation", ""),
                "scale": estimate.get("scale", 1.0), "formula": estimate.get("formula", ""),
                "unit": estimate.get("unit", ""), "point": estimate["point"], "ci_low": estimate["ci_low"],
                "ci_high": estimate["ci_high"], "bootstrap_std": estimate["bootstrap_std"],
                "seed_standard_deviation": estimate["seed_standard_deviation"],
                "seed_estimates": estimate["seed_estimates"], "old_point": estimate["old_point"],
                "old_seed_points": estimate["old_seed_points"], "old_ci_low": estimate["old_ci_low"],
                "old_ci_high": estimate["old_ci_high"], "valid_replicates": estimate["valid_replicates"],
                "invalid_replicates": estimate["invalid_replicates"], "raw_replicates": estimate["raw_replicates"],
                "valid_fraction": estimate["valid_fraction"], "uncertainty_status": estimate["uncertainty_status"],
                "gate_eligible": estimate["gate_eligible"], "ci_conditioning": estimate["ci_conditioning"],
                "minimum_valid_replicates_for_exploratory_ci": estimate["minimum_valid_replicates_for_exploratory_ci"],
                "anchor_status": estimate["anchor_status"], "n_images": estimate["n_images"],
                "nonfinite_old_seed_points": ";".join(estimate.get("nonfinite_old_seed_points", [])),
                "nonfinite_anchor_reason": estimate.get("nonfinite_anchor_reason", ""),
                "legacy_anchor_unavailable": json.dumps(estimate.get("legacy_anchor_unavailable", []), ensure_ascii=False),
            })
            coverage_rows.append(_default_coverage_row(job_summary, estimate))
    summary["estimates"] = [estimate for job_summary in summary["jobs"] for estimate in job_summary["estimates"]]
    atomic_json(OUT / "summary.json", summary)
    point_fields = ["scope", "cohort", "estimate", "metric", "operation", "scale", "formula", "unit",
                    "point", "ci_low", "ci_high", "bootstrap_std",
                    "seed_standard_deviation", "seed_estimates", "old_point", "old_seed_points", "old_ci_low",
                    "old_ci_high", "valid_replicates", "invalid_replicates", "valid_fraction", "uncertainty_status",
                    "gate_eligible", "ci_conditioning", "minimum_valid_replicates_for_exploratory_ci",
                    "raw_replicates", "anchor_status", "nonfinite_old_seed_points", "nonfinite_anchor_reason",
                    "legacy_anchor_unavailable", "n_images"]
    coverage_fields = ["scope", "endpoint_or_gate", "source_artifacts", "source_hashes", "old_point", "old_ci_low", "old_ci_high",
                       "new_point", "new_ci_low", "new_ci_high", "operation", "scale", "formula", "unit",
                       "old_gate", "new_gate", "thresholds", "n_rows", "n_images",
                       "n_replicates", "n_seeds", "valid_replicates", "invalid_replicates", "valid_fraction",
                       "uncertainty_status", "gate_eligible", "ci_conditioning",
                       "minimum_valid_replicates_for_exploratory_ci", "recovery_status",
                       "cohort_construction", "point_estimate_ordering", "legacy_pooled_anchor", "legacy_cohort_classification",
                       "legacy_cohort_note", "legacy_pool_status", "anchor_status", "nonfinite_old_seed_points", "nonfinite_anchor_reason",
                       "legacy_anchor_unavailable",
                       "anchor_tolerance", "sentence_id_status", "model_identity_by_seed", "prediction_source_precision",
                       "recovery_output_path", "recovery_output_sha256",
                       "evidence", "status", "reason", "output_path"]
    _csv_write(OUT / "point_metrics.csv", point_fields, point_rows)
    _csv_write(OUT / "coverage.csv", coverage_fields, coverage_rows)
    atomic_json(OUT / "status.json", {
        "schema": "ccg.research_repair_v1.statistics.status.v1",
        "status": stage_status,
        "updated_utc": utc_now(),
        "completed_jobs": [job["job_id"] for job in summary["jobs"]],
        "n_estimates": len(summary["estimates"]),
        "n_replicates": summary["bootstrap"]["replicates"],
        "blocking_resource_slot": "root grant required before formal 5000-replicate run" if stage_status == "WAITING_RESOURCE_SLOT" else None,
    })


def _refresh_required_scope_coverage(
    jobs_by_scope: dict[str, list[dict[str, Any]]], summary: dict[str, Any],
) -> int:
    """Persist an exact planned-estimator ledger and observed cohort metadata."""
    path = OUT / "required_scope_coverage.json"
    # This is a repair-owned output.  read_json() deliberately enforces the
    # frozen-input manifest, so using it here would reject our own output file.
    document = json.loads(path.read_text(encoding="utf-8"))
    scopes = document.get("scopes", [])
    if isinstance(scopes, dict):
        scopes = [dict(value, scope_id=key) for key, value in scopes.items()]
        document["scopes"] = scopes
    scope_by_id = {str(item["scope_id"]): item for item in scopes}
    completed = {job["job_id"]: job for job in summary.get("jobs", [])}
    updated = 0
    for scope_id, jobs in jobs_by_scope.items():
        record = scope_by_id.get(scope_id)
        if record is None:
            continue
        unverifiable_jobs = record.get("unverifiable_jobs", [])

        def strict_evidence_ok(item: dict[str, Any]) -> bool:
            relative = item.get("evidence_path")
            expected_sha = item.get("evidence_sha256", item.get("sha256"))
            if item.get("status") != "UNVERIFIABLE" or not relative or not expected_sha:
                return False
            evidence_path = (ROOT / str(relative)).resolve()
            if not evidence_path.is_relative_to(ROOT.resolve()) or not evidence_path.is_file():
                return False
            return sha256(evidence_path) == str(expected_sha)

        verified_unverifiable = bool(unverifiable_jobs) and all(
            strict_evidence_ok(item) for item in unverifiable_jobs
        )
        ledger = []
        for job in jobs:
            specs = list(job.get("estimates", ())) + list(job.get("auxiliary_estimates", ()))
            by_name = {str(spec["name"]): spec for spec in specs}
            current = completed.get(str(job["job_id"]), {})
            current_estimates = {str(item["name"]): item for item in current.get("estimates", [])}
            ledger.append({
                "job_id": str(job["job_id"]),
                "cohort": str(job.get("cohort", "")),
                "n_rows": int(np.asarray(job.get("image_ids", [])).size),
                "n_images": int(np.unique(np.asarray(job.get("image_ids", []))).size),
                "seed_names": sorted(str(seed) for seed in job.get("seed_cells", {})),
                "n_seeds": len(job.get("seed_cells", {})),
                "n_estimates": len(by_name),
                "estimate_names": list(by_name),
                "estimate_specs": [
                    {"estimate_name": name, **_estimate_spec_metadata(spec)}
                    for name, spec in by_name.items()
                ],
                "cohort_metadata": job.get("cohort_metadata", {}),
                "source_artifacts": [Path(value).resolve().relative_to(ROOT.resolve()).as_posix()
                                     for value in job.get("source_paths", [])],
                "anchor_status": (
                    "PASS" if current and current.get("status") == "RECOMPUTED_5000"
                    and current_estimates and all(item.get("anchor_status") == "PASS" for item in current_estimates.values())
                    else record.get("anchor_status", "PENDING")
                ),
                "formal_status": current.get("status", "PENDING_FORMAL_BOOTSTRAP"),
                "n_replicates": current.get("result", {}).get("n_replicates"),
                "completed": current.get("status") == "RECOMPUTED_5000",
            })
        if not ledger:
            if scope_id == "m3" and not record.get("job_ids") and unverifiable_jobs:
                required_m3_jobs = {"m3_b0_severity_macro", "m3_b1_severity_macro", "m3_b2_severity_macro"}
                observed_m3_jobs = {str(item.get("job_id")) for item in unverifiable_jobs}
                record["formal_jobs_complete"] = 0
                record["formal_jobs_required"] = 0
                if observed_m3_jobs == required_m3_jobs and verified_unverifiable:
                    record["status"] = "UNVERIFIABLE"
                    record["anchor_status"] = "STRICT_FAILURE_EVIDENCE_FOR_ALL_REQUIRED_JOBS"
                else:
                    record["status"] = "UNVERIFIABLE_EVIDENCE_PENDING"
                    record["anchor_status"] = "PENDING_COMPLETE_REQUIRED_JOB_FAILURE_EVIDENCE"
                record["updated_utc"] = utc_now()
                updated += 1
            continue
        expected_ids = [str(job["job_id"]) for job in jobs]
        if record.get("job_ids") and set(record["job_ids"]) != set(expected_ids):
            raise ValueError(
                f"required scope job inventory mismatch for {scope_id}: "
                f"coverage={record['job_ids']} builders={expected_ids}"
            )
        record["job_ids"] = expected_ids
        record["completed_job_ids"] = [job_id for job_id in expected_ids if completed.get(job_id, {}).get("status") == "RECOMPUTED_5000"]
        record["formal_jobs_complete"] = len(record["completed_job_ids"])
        record["formal_jobs_required"] = len(expected_ids)
        record["job_estimator_ledger"] = ledger
        record["cohort_metadata_by_job"] = {item["job_id"]: item["cohort_metadata"] for item in ledger}
        record["estimator_ledger_status"] = "EXACT_FROM_REGISTERED_ADAPTER"
        if record["formal_jobs_complete"] == len(expected_ids):
            if verified_unverifiable:
                record["status"] = "COMPLETE_WITH_UNVERIFIABLE_RECOVERY"
                record["anchor_status"] = "PASS_FOR_FORMAL_JOBS; STRICT_FAILURE_EVIDENCE_FOR_UNVERIFIABLE_JOBS"
            elif unverifiable_jobs:
                record["status"] = "UNVERIFIABLE_EVIDENCE_PENDING"
                record["anchor_status"] = "PENDING_STRICT_FAILURE_EVIDENCE"
            else:
                record["status"] = "FORMAL_5000_COMPLETE"
                record["anchor_status"] = "PASS_FOR_COMPLETED_JOBS"
        elif unverifiable_jobs and verified_unverifiable and record["formal_jobs_complete"] > 0:
            record["status"] = "PARTIAL_WITH_UNVERIFIABLE_RECOVERY"
            record["anchor_status"] = "SOME_FORMAL_JOBS_COMPLETE; OTHER_REQUIRED_JOBS_HAVE_STRICT_FAILURE_EVIDENCE"
        updated += 1
    document["updated_utc"] = utc_now()
    atomic_json(path, document)
    return updated


def _verify_observed_point_anchors(job: dict[str, Any]) -> dict[str, float]:
    """Check all available observed points before paying for any bootstrap draws."""
    legacy = job.get("legacy_points", {})
    if not legacy:
        return {}
    order = job.get("point_order")
    default_point_rows = np.arange(len(job["image_ids"]), dtype=np.int64) if order is None else np.asarray(order, dtype=np.int64)
    observed_by_seed: dict[str, dict[str, float]] = {}
    nonfinite_old: dict[str, list[str]] = {}
    for seed_key, conditions in job["seed_cells"].items():
        values: dict[str, float] = {}
        for condition, (confidence, correctness) in conditions.items():
            point_rows = default_point_rows
            condition_orders = job.get("point_order_by_seed_condition")
            if condition_orders is not None:
                point_rows = np.asarray(condition_orders[seed_key][condition], dtype=np.int64)
            point = metric_bundle(
                np.asarray(confidence)[point_rows], np.asarray(correctness)[point_rows],
                metrics=job["metrics"], zero_error_rer="legacy_zero",
            )
            values.update({f"{condition}::{metric}": float(value) for metric, value in point.items()})
        for condition, raw_values in job.get("auxiliary_cells", {}).get(seed_key, {}).items():
            auxiliary = np.asarray(raw_values, dtype=np.float64)
            values[f"{condition}::mean"] = float(np.mean(auxiliary))
        all_specs = list(job["estimates"]) + list(job.get("auxiliary_estimates", ()))
        estimates = {str(spec["name"]): _estimate_value(values, spec) for spec in all_specs}
        observed_by_seed[str(seed_key)] = estimates
    checked: dict[str, float] = {}
    for name, seed_points in legacy.items():
        if not isinstance(seed_points, dict):
            seed_points = {"legacy": seed_points}
        for seed_key, old_value in seed_points.items():
            expected = _float_or_nan(old_value)
            if not np.isfinite(expected):
                nonfinite_old.setdefault(str(name), []).append(str(seed_key))
                continue
            if seed_key in {"b3_mean", "__aggregate__", "seed_mean"}:
                if name not in next(iter(observed_by_seed.values()), {}):
                    raise ValueError(f"observed point anchor estimate {name!r} absent from {job['job_id']}")
                observed = float(np.mean([per_seed[name] for per_seed in observed_by_seed.values()]))
            else:
                if seed_key not in observed_by_seed:
                    raise ValueError(f"observed point anchor seed {seed_key!r} absent from {job['job_id']}/{name}")
                if name not in observed_by_seed[seed_key]:
                    raise ValueError(f"observed point anchor estimate {name!r} absent from {job['job_id']}")
                observed = observed_by_seed[seed_key][name]
            try:
                _check_anchor({"point": observed}, expected)
            except ValueError as exc:
                raise ValueError(f"observed anchor failed job={job['job_id']} estimate={name} seed={seed_key}: {exc}") from exc
            checked[f"{seed_key}::{name}"] = observed
    job["nonfinite_legacy_anchors"] = [
        {"estimate": name, "seeds": sorted(seed_keys),
         "reason": "legacy point is non-finite or absent; no finite-status anchor exists"}
        for name, seed_keys in sorted(nonfinite_old.items())
    ]
    return checked


def preflight(scopes: list[str]) -> None:
    selected = list(scopes)
    missing = sorted(set(selected) - set(BUILDERS))
    if missing:
        raise ValueError(f"No prediction-only adapter is registered yet for scopes: {missing}")
    all_jobs = [job for name in selected for job in BUILDERS[name]()]
    plan = []
    for job in all_jobs:
        observed_anchors = _verify_observed_point_anchors(job)
        sources = [frozen_source(Path(path)) for path in job["source_paths"]]
        plan.append({
            "job_id": job["job_id"], "scope": job["scope"], "cohort": job["cohort"],
            "n_rows": int(np.asarray(job["image_ids"]).size),
            "n_images": int(np.unique(job["image_ids"]).size),
            "n_seeds": len(job["seed_cells"]), "n_conditions": len(next(iter(job["seed_cells"].values()))),
            "n_estimates": len(job["estimates"]), "source_artifacts": sources,
            "sentence_id_status": job.get("sentence_id_status", "VERIFIED_NATIVE_ID"),
            "evidence": job.get("evidence", ""),
            "model_identity_by_seed": job.get("model_identity_by_seed", {}),
            "prediction_source_precision": job.get("prediction_source_precision", "unspecified"),
            "recovery_output_path": job.get("recovery_output_path", ""),
            "recovery_output_sha256": job.get("recovery_output_sha256", ""),
            "observed_anchor_status": (
                "PASS" if observed_anchors and not job.get("nonfinite_legacy_anchors") and not job.get("legacy_anchor_unavailable") else
                "PARTIAL_PASS_OLD_NONFINITE_UNANCHORED" if observed_anchors else
                "LEGACY_POINT_UNAVAILABLE" if job.get("legacy_anchor_unavailable") else
                "OLD_POINT_NONFINITE_UNANCHORED" if job.get("nonfinite_legacy_anchors") else
                "NO_LEGACY_POINT_ANCHOR"
            ),
            "observed_anchor_count": len(observed_anchors),
            "observed_anchors": observed_anchors,
            "nonfinite_legacy_anchors": job.get("nonfinite_legacy_anchors", []),
            "legacy_anchor_unavailable": job.get("legacy_anchor_unavailable", []),
        })
    atomic_json(OUT / "preflight.json", {
        "schema": "ccg.research_repair_v1.statistics.preflight.v1",
        "created_utc": utc_now(), "selected_scopes": selected,
        "formal_bootstrap_started": False,
        "resource_slot": "not granted",
        "jobs": plan,
    })
    print(f"Preflighted {len(plan)} jobs; formal 5000-replicate statistics not started.")


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=sorted(BUILDERS), action="append", help="select one or more registered prediction-only cohorts")
    parser.add_argument("--preflight", action="store_true", help="verify inputs and emit a resource plan only")
    parser.add_argument("--n-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--resource-slot-granted", help="required root handoff token for formal 5000-draw work")
    parser.add_argument("--pilot", action="store_true", help="allow at most 100 non-formal draws, isolated under pilot output")
    parser.add_argument("--refresh-existing-metadata", action="store_true",
                        help="refresh adapter/spec/identity metadata for existing jobs without drawing bootstrap replicates")
    parser.add_argument("--refresh-coverage-ledger", action="store_true",
                        help="write exact job estimator lists and cohort split metadata to required_scope_coverage.json")
    args = parser.parse_args()
    if args.n_replicates != 5000 and not (args.pilot and 1 <= args.n_replicates <= 100):
        parser.error("formal runs require exactly 5000 draws; pilot requires --pilot and 1..100 draws")
    if args.n_replicates == 5000 and not args.preflight and not args.refresh_existing_metadata and not args.resource_slot_granted:
        parser.error("formal 5000-replicate statistics require --resource-slot-granted from the root coordinator")
    if args.seed != 0 or args.ci != 0.95:
        parser.error("the frozen repair protocol requires seed=0 and percentile 95% CI")
    scopes = args.scope or list(BUILDERS)
    if args.refresh_existing_metadata or args.refresh_coverage_ledger:
        if args.preflight or args.pilot:
            parser.error("metadata refresh cannot be combined with --preflight or --pilot")
        summary = _load_existing_summary() if args.refresh_existing_metadata else json.loads(
            (OUT / "summary.json").read_text(encoding="utf-8")
        )
        status_path = OUT / "status.json"
        current_status = (
            json.loads(status_path.read_text(encoding="utf-8")).get("status", "PARTIAL_SCOPE_COMPLETE")
            if status_path.is_file() else "PARTIAL_SCOPE_COMPLETE"
        )
        completed = {job["job_id"] for job in summary["jobs"]}
        refreshed = []
        jobs_by_scope: dict[str, list[dict[str, Any]]] = {}
        for scope in scopes:
            if scope not in BUILDERS:
                raise ValueError(f"No prediction-only adapter is registered yet for scope: {scope}")
            jobs_by_scope[scope] = BUILDERS[scope]()
            for job in jobs_by_scope[scope]:
                if args.refresh_existing_metadata and job["job_id"] in completed:
                    _refresh_persisted_job(summary, job)
                    refreshed.append(job["job_id"])
        if args.refresh_existing_metadata:
            _persist(summary, stage_status=current_status)
        if args.refresh_coverage_ledger:
            updated_scopes = _refresh_required_scope_coverage(jobs_by_scope, summary)
        else:
            updated_scopes = 0
        print(
            f"Refreshed metadata for {len(refreshed)} existing jobs and exact ledger for "
            f"{updated_scopes} scopes; no bootstrap replicates drawn."
        )
        return 0
    if args.preflight:
        preflight(scopes)
        return 0
    if args.pilot:
        OUT = OUT / "pilot"
    summary = _load_existing_summary()
    completed = {job["job_id"] for job in summary["jobs"]}
    _persist(summary, stage_status="RUNNING_FORMAL" if args.n_replicates == 5000 else "RUNNING_PILOT")
    for scope in scopes:
        if scope not in BUILDERS:
            raise ValueError(f"No prediction-only adapter is registered yet for scope: {scope}")
        for job in BUILDERS[scope]():
            if job["job_id"] in completed and args.n_replicates == 5000:
                _refresh_persisted_job(summary, job)
                _persist(summary, stage_status="RUNNING_FORMAL")
                print(f"Skipping completed job {job['job_id']}", flush=True)
                continue
            job_summary = run_job(job, args.n_replicates, args.seed, args.ci)
            summary["jobs"] = [item for item in summary["jobs"] if item["job_id"] != job["job_id"]] + [job_summary]
            summary["estimates"] = [estimate for item in summary["jobs"] for estimate in item["estimates"]]
            _persist(summary, stage_status="RUNNING_FORMAL" if args.n_replicates == 5000 else "RUNNING_PILOT")
            print(f"Completed {job['job_id']}: {len(job_summary['estimates'])} estimates", flush=True)
    stage_status = "PILOT_COMPLETE_NOT_FORMAL" if args.pilot else "PARTIAL_SCOPE_COMPLETE"
    _persist(summary, stage_status=stage_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
