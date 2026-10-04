"""Derive the missing M2 relative E-AURC gate estimate from saved draws.

This utility consumes existing joint bootstrap endpoint arrays. It performs no
sampling and writes only beneath results/research_repair_v1/statistics/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccg.repairs.atomic import replace_with_retry
from ccg.repairs.m2_auxiliary import (
    M2_ANCHOR_ATOL,
    M2_RELATIVE_EAURC_NAME,
    M2_RELATIVE_EAURC_SPEC,
    M2_SEEDS,
    derive_m2_relative_eaurc_reduction,
)

STAT_DIR = ROOT / "results" / "research_repair_v1" / "statistics"
JOB_ID = "m2_curriculum_m0_m2_m4_m8"
GATE_PATH = "results/v2_local_competition/m2_curriculum/gate.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".m2aux-{uuid.uuid4().hex[:8]}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def _find_job(summary: dict[str, Any]) -> dict[str, Any]:
    jobs = [job for job in summary.get("jobs", []) if job.get("job_id") == JOB_ID]
    if len(jobs) != 1:
        raise ValueError(f"expected exactly one {JOB_ID} in statistics summary; found {len(jobs)}")
    return jobs[0]


def _estimate_map(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    estimates = {str(item["name"]): item for item in job.get("estimates", [])}
    if len(estimates) != len(job.get("estimates", [])):
        raise ValueError(f"duplicate estimate names in {JOB_ID}")
    return estimates


def _shared_endpoint_archive(estimates: dict[str, dict[str, Any]], endpoint_names: tuple[str, ...]) -> tuple[str, str | None]:
    paths = {str(estimates[name].get("raw_replicates", "")) for name in endpoint_names}
    if len(paths) != 1 or not next(iter(paths)):
        raise ValueError("M2 E1b and LCR endpoints must declare the same raw replicate archive")
    declared_hashes = {
        str(estimates[name]["raw_replicates_sha256"])
        for name in endpoint_names
        if estimates[name].get("raw_replicates_sha256")
    }
    if len(declared_hashes) > 1:
        raise ValueError("M2 E1b and LCR endpoint raw archive SHA declarations differ")
    return next(iter(paths)), next(iter(declared_hashes)) if declared_hashes else None


def _verify_endpoint_aggregates(raw: Any, component_names: tuple[str, ...]) -> None:
    for name in component_names:
        arrays = [np.asarray(raw[f"seed_{seed}__{name}"], dtype=np.float64) for seed in M2_SEEDS]
        aggregate = np.asarray(raw[f"aggregate__{name}"], dtype=np.float64)
        stack = np.stack(arrays)
        valid = np.all(np.isfinite(stack), axis=0)
        expected = np.full(aggregate.shape, np.nan, dtype=np.float64)
        expected[valid] = np.mean(stack[:, valid], axis=0)
        if not np.array_equal(expected, aggregate, equal_nan=True):
            if not np.allclose(expected, aggregate, rtol=0.0, atol=1e-12, equal_nan=True):
                raise ValueError(f"saved aggregate draws do not match same-index seed endpoints: {name}")


def derive_and_write(*, apply: bool = False) -> dict[str, Any]:
    summary_path = STAT_DIR / "summary.json"
    status_path = STAT_DIR / "status.json"
    coverage_path = STAT_DIR / "required_scope_coverage.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status == "RUNNING_FORMAL":
            raise RuntimeError("statistics writer is still RUNNING_FORMAL; wait before deriving the auxiliary artifact")
    bootstrap = summary.get("bootstrap", {})
    if (bootstrap.get("replicates") != 5000 or bootstrap.get("seed") != 0
            or float(bootstrap.get("ci_level", -1)) != 0.95
            or bootstrap.get("method") != "percentile"
            or bootstrap.get("resample_unit") != "image_cluster"):
        raise ValueError("summary does not carry the frozen 5000/seed0/percentile95/image-cluster method")

    job = _find_job(summary)
    if job.get("status") != "RECOMPUTED_5000":
        raise RuntimeError(f"M2 formal job is not complete: status={job.get('status')!r}")
    result = job.get("result", {})
    config = job.get("config", {})
    if (config.get("n_replicates") != 5000 or config.get("seed") != 0
            or float(config.get("ci_level", -1)) != 0.95
            or config.get("ci_method") != "percentile"
            or result.get("n_replicates") != 5000
            or result.get("seed") != 0
            or float(result.get("ci_level", -1)) != 0.95
            or result.get("method") != "percentile"
            or result.get("resample_unit") != "image_cluster"):
        raise ValueError("M2 job metadata does not match the frozen formal bootstrap method")

    estimates = _estimate_map(job)
    if M2_RELATIVE_EAURC_NAME in estimates:
        existing = estimates[M2_RELATIVE_EAURC_NAME]
        expected_source = existing.get("derived_from", {})
        source_path = ROOT / str(existing.get("derived_from", {}).get("raw_replicates", ""))
        if source_path.is_file() and _sha256(source_path) == expected_source.get("raw_replicates_sha256"):
            return {"status": "ALREADY_DERIVED", "estimate": M2_RELATIVE_EAURC_NAME,
                    "raw_replicates": existing.get("raw_replicates")}
        raise FileExistsError("M2 derived estimate already exists with a different or unverifiable input fingerprint")

    endpoint_names = ("m8__E1b::e_aurc", "m8__LCR::e_aurc")
    for name in endpoint_names:
        if name not in estimates:
            raise KeyError(f"M2 endpoint estimate missing from completed job: {name}")
    source_raw_rel, declared_raw_sha = _shared_endpoint_archive(estimates, endpoint_names)
    source_raw_path = (ROOT / source_raw_rel).resolve()
    if not source_raw_path.is_relative_to(STAT_DIR.resolve()) or not source_raw_path.is_file():
        raise ValueError("M2 raw replicate archive is missing or outside repair statistics output")
    source_raw_sha = _sha256(source_raw_path)
    if declared_raw_sha is not None and declared_raw_sha != source_raw_sha:
        raise ValueError("M2 endpoint raw archive SHA differs from its estimate declarations")
    gate_path = ROOT / GATE_PATH
    frozen_sources = {str(item.get("path")): str(item.get("sha256")) for item in job.get("source_artifacts", [])}
    if frozen_sources.get(GATE_PATH) != _sha256(gate_path):
        raise ValueError("M2 historical gate artifact hash differs from the job's frozen source manifest")
    old_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    selective = old_gate.get("gates", {}).get("selective_metrics", {})
    old_seed_points = {
        seed: float(selective.get("per_seed", {}).get(seed[-1], {}).get("e_aurc_reduction"))
        for seed in M2_SEEDS
    }
    old_mean_point = float(selective.get("e_aurc_reduction"))

    points: dict[str, dict[str, float]] = {}
    for name in endpoint_names:
        endpoint = estimates[name]
        if set(endpoint.get("seed_estimates", {})) != set(M2_SEEDS):
            raise ValueError(f"endpoint points lack exactly the fixed seeds: {name}")
        points[name] = {seed: float(endpoint["seed_estimates"][seed]) for seed in M2_SEEDS}

    with np.load(source_raw_path, allow_pickle=False) as source_raw:
        _verify_endpoint_aggregates(source_raw, endpoint_names)
        e1b_draws = {seed: np.asarray(source_raw[f"seed_{seed}__{endpoint_names[0]}"], dtype=np.float64)
                     for seed in M2_SEEDS}
        lcr_draws = {seed: np.asarray(source_raw[f"seed_{seed}__{endpoint_names[1]}"], dtype=np.float64)
                     for seed in M2_SEEDS}
    if _sha256(source_raw_path) != source_raw_sha:
        raise RuntimeError("M2 source raw archive changed while being read")

    estimate, raw_arrays = derive_m2_relative_eaurc_reduction(
        e1b_draws=e1b_draws,
        lcr_draws=lcr_draws,
        e1b_points=points[endpoint_names[0]],
        lcr_points=points[endpoint_names[1]],
        old_seed_points=old_seed_points,
        old_mean_point=old_mean_point,
        ci_level=0.95,
        minimum_valid_fraction=float(summary.get("bootstrap", {}).get("validity_sufficiency_rule", {}).get(
            "minimum_valid_fraction_for_exploratory_ci", 0.95)),
        expected_replicates=5000,
        anchor_atol=M2_ANCHOR_ATOL,
    )

    output_raw = STAT_DIR / "raw_replicates" / f"{JOB_ID}__m2_relative_auxiliary.npz"
    output_rel = output_raw.relative_to(ROOT).as_posix()

    if not apply:
        return {
            "status": "PREFLIGHT_ONLY",
            "job_id": JOB_ID,
            "estimate": M2_RELATIVE_EAURC_NAME,
            "source_raw_replicates": source_raw_rel,
            "source_raw_replicates_sha256": source_raw_sha,
            "derived_point": estimate["point"],
            "old_point": estimate["old_point"],
            "valid_replicates": estimate["valid_replicates"],
            "invalid_replicates": estimate["invalid_replicates"],
            "gate_eligible": estimate["gate_eligible"],
            "would_write": output_rel,
        }

    output_raw.parent.mkdir(parents=True, exist_ok=True)
    if output_raw.exists():
        with np.load(output_raw, allow_pickle=False) as prior:
            if set(prior.files) != set(raw_arrays) or any(
                not np.array_equal(np.asarray(prior[key]), value, equal_nan=True)
                for key, value in raw_arrays.items()
            ):
                raise FileExistsError("existing M2 auxiliary raw archive conflicts with deterministic derivation")
    else:
        temporary = output_raw.parent / f".m2aux-{uuid.uuid4().hex[:8]}.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, **raw_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, output_raw)
    output_sha = _sha256(output_raw)

    spec = dict(M2_RELATIVE_EAURC_SPEC)
    estimate.update({
        "scope": job["scope"],
        "cohort": job["cohort"],
        "metric": "e_aurc",
        "estimate_spec": spec,
        "operation": "relative_contrast",
        "scale": 1.0,
        "formula": "(e_aurc(m8__E1b) - e_aurc(m8__LCR)) / e_aurc(m8__E1b), computed per seed and shared draw, then mean over fixed seeds",
        "unit": "relative_fraction",
        "n_rows": result.get("n_rows"),
        "n_images": result.get("n_images"),
        "n_replicates": 5000,
        "resample_unit": "image_cluster",
        "ci_level": 0.95,
        "ci_method": "percentile",
        "raw_replicates": output_rel,
        "raw_replicates_sha256": output_sha,
        "source_artifacts": job.get("source_artifacts", []),
        "derived_from": {
            "raw_replicates": source_raw_rel,
            "raw_replicates_sha256": source_raw_sha,
            "raw_seed_keys": {
                seed: {
                    "e1b": f"seed_{seed}__{endpoint_names[0]}",
                    "lcr": f"seed_{seed}__{endpoint_names[1]}",
                }
                for seed in M2_SEEDS
            },
            "shared_draw_alignment": "same NPZ replicate index across all conditions and seeds; no random numbers drawn",
            "historical_anchor": {
                "path": GATE_PATH,
                "sha256": frozen_sources[GATE_PATH],
                "seed_pointer_template": "/gates/selective_metrics/per_seed/{seed}/e_aurc_reduction",
                "mean_pointer": "/gates/selective_metrics/e_aurc_reduction",
            },
            "derivation_module": "src/ccg/repairs/m2_auxiliary.py",
            "derivation_module_sha256": _sha256(ROOT / "src" / "ccg" / "repairs" / "m2_auxiliary.py"),
            "derivation_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "recovery_status": "DETERMINISTIC_TRANSFORM_OF_SAVED_JOINT_DRAWS_NO_RESAMPLING",
        "sentence_id_status": "INHERITED_FROM_M2_SHARED_CANONICAL_COHORT",
        "evidence": "Derived from same-draw per-seed m8 E1b/LCR E-AURC arrays; historical standalone M2 gate relative points anchored at abs tol 1e-9.",
    })

    estimates[ M2_RELATIVE_EAURC_NAME ] = estimate
    job["estimates"].append(estimate)
    job["derived_auxiliary_estimates"] = [M2_RELATIVE_EAURC_NAME]
    summary["estimates"] = [item for stored_job in summary["jobs"] for item in stored_job["estimates"]]
    summary.setdefault("derived_estimate_runs", []).append({
        "estimate": M2_RELATIVE_EAURC_NAME,
        "job_id": JOB_ID,
        "status": "DERIVED_FROM_EXISTING_DRAWS",
        "source_raw_replicates": source_raw_rel,
        "source_raw_replicates_sha256": source_raw_sha,
        "output_raw_replicates": output_rel,
        "output_raw_replicates_sha256": output_sha,
        "n_replicates": 5000,
        "seed": 0,
        "ci_level": 0.95,
        "ci_method": "percentile",
        "resample_unit": "image_cluster",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json(summary_path, summary)
    return {
        "status": "DERIVED_AND_WRITTEN",
        "job_id": JOB_ID,
        "estimate": M2_RELATIVE_EAURC_NAME,
        "source_raw_replicates": source_raw_rel,
        "source_raw_replicates_sha256": source_raw_sha,
        "output_raw_replicates": output_rel,
        "output_raw_replicates_sha256": output_sha,
        "point": estimate["point"],
        "ci_low": estimate["ci_low"],
        "ci_high": estimate["ci_high"],
        "valid_replicates": estimate["valid_replicates"],
        "invalid_replicates": estimate["invalid_replicates"],
        "gate_eligible": estimate["gate_eligible"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the repair-owned derived archive and summary entry")
    args = parser.parse_args()
    result = derive_and_write(apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
