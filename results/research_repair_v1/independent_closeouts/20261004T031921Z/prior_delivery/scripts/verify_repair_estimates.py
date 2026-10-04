"""Recheck saved percentile intervals and seed means without drawing samples."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_repair_v1"


def records(axis: str):
    """Yield archive, mean key, seed keys, mean summary, per-seed summaries."""
    if axis == "statistics":
        summary = json.loads((OUT / "statistics/summary.json").read_text(encoding="utf-8-sig"))
        for job in summary.get("jobs", []):
            if job.get("status") != "RECOMPUTED_5000":
                continue
            for estimate in job["estimates"]:
                name = estimate["name"]
                seed_names = list(estimate["per_seed"])
                yield (ROOT / estimate["raw_replicates"], f"aggregate__{name}",
                       [f"seed_{seed}__{name}" for seed in seed_names], estimate,
                       [estimate["per_seed"][seed] for seed in seed_names])
    elif axis == "information":
        with (OUT / "information/information_bootstrap.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                archive, keys_json = row["per_seed_evidence_path"].split("::", 1)
                seed_keys = json.loads(keys_json)
                per_seed = json.loads(row["per_seed_ci_json"])
                yield (OUT / "information" / archive, row["raw_replicates_key"], list(seed_keys.values()),
                       row, [per_seed[seed] for seed in seed_keys])
    elif axis == "mechanism":
        with (OUT / "mechanism/bootstrap_ci.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                prefix = f"{row['family']}__{row['estimate']}__"
                mean = dict(row, point=row["point_mean3"], ci_low=row["ci_low_mean3"], ci_high=row["ci_high_mean3"])
                per_seed = [{key: row[f"seed{seed}_{key}"]
                             for key in ("point", "ci_low", "ci_high", "valid_replicates", "invalid_replicates")}
                            for seed in (1, 2, 3)]
                yield (OUT / "mechanism/bootstrap_raw_replicates.npz", prefix + "mean3",
                       [prefix + f"seed{seed}" for seed in (1, 2, 3)], mean, per_seed)
    elif axis == "candidates":
        pointer = json.loads((OUT / "candidates/current_audit.json").read_text(encoding="utf-8-sig"))
        directory = (ROOT / pointer["summary_path"]).parent / "forward/sensitivity_bootstrap"
        with (directory / "estimates.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                per_seed = json.loads(row["per_seed_json"])
                keys = json.loads(row["per_seed_raw_replicates_keys_json"])
                keys = list(keys.values()) if isinstance(keys, dict) else keys
                path = Path(row["raw_replicates_path"])
                archive = path if path.is_absolute() else directory / path
                yield archive, row["raw_replicates_key"], keys, row, list(per_seed.values())
    else:
        raise ValueError(axis)


def same_number(actual: float, expected: object) -> bool:
    try:
        number = float(expected)
    except (TypeError, ValueError):
        return bool(np.isnan(actual) and expected in (None, "", "None"))
    return bool((np.isnan(actual) and np.isnan(number))
                or np.isclose(actual, number, rtol=0, atol=1e-12))


def check(axes: list[str]) -> dict:
    failures, archive_records, counts = [], [], {}
    source_hashes = {}
    for axis in axes:
        source = OUT / {"statistics": "statistics/summary.json",
                        "information": "information/information_bootstrap.csv",
                        "mechanism": "mechanism/bootstrap_ci.csv"}.get(axis, "candidates/current_audit.json")
        if axis == "candidates":
            pointer = json.loads(source.read_text(encoding="utf-8-sig"))
            source = (ROOT / pointer["summary_path"]).parent / "forward/sensitivity_bootstrap/estimates.csv"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        source_hashes[source.relative_to(OUT).as_posix()] = before
        grouped = defaultdict(list)
        for archive, mean_key, seed_keys, mean, seeds in records(axis):
            grouped[archive].append((mean_key, seed_keys, mean, seeds))
        counts[axis] = sum(len(rows) for rows in grouped.values())
        for archive, rows in grouped.items():
            digest = hashlib.sha256()
            with archive.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            archive_records.append({"axis": axis, "path": archive.relative_to(ROOT).as_posix(),
                                    "size_bytes": archive.stat().st_size, "sha256": digest.hexdigest()})
            with np.load(archive, allow_pickle=False) as raw:
                for mean_key, seed_keys, mean, seeds in rows:
                    try:
                        aggregate = raw[mean_key]
                        seed_arrays = [raw[key] for key in seed_keys]
                    except KeyError as error:
                        failures.append(f"{axis}/{mean_key}: missing raw array {error}")
                        continue
                    if any(values.dtype != np.float64 or values.shape != (5000,) or np.any(np.isinf(values))
                           for values in [aggregate, *seed_arrays]):
                        failures.append(f"{axis}/{mean_key}: wrong raw shape/dtype or infinite values")
                        continue
                    for values, summary in [(aggregate, mean), *zip(seed_arrays, seeds)]:
                        valid = int(np.count_nonzero(np.isfinite(values)))
                        if valid != int(summary["valid_replicates"]) or 5000 - valid != int(summary["invalid_replicates"]):
                            failures.append(f"{axis}/{mean_key}: validity counts differ from raw arrays")
                        # Reproduce the saved finite-draw percentile calculation.
                        # Its sufficiency label and gate eligibility are checked separately.
                        low, high = (np.percentile(values[np.isfinite(values)], [2.5, 97.5])
                                     if valid >= 2 else (np.nan, np.nan))
                        if not same_number(low, summary.get("ci_low")) or not same_number(high, summary.get("ci_high")):
                            failures.append(f"{axis}/{mean_key}: stored interval differs from raw percentile interval")
                    seed_matrix = np.vstack(seed_arrays)
                    eligible = np.all(np.isfinite(seed_matrix), axis=0)
                    expected_mean = np.full(5000, np.nan)
                    expected_mean[eligible] = np.mean(seed_matrix[:, eligible], axis=0)
                    if not np.allclose(aggregate, expected_mean, rtol=0, atol=1e-12, equal_nan=True):
                        failures.append(f"{axis}/{mean_key}: aggregate draws are not per-draw fixed-seed means")
                    points = np.asarray([float(row["point"]) if row.get("point") not in (None, "", "None")
                                         else np.nan for row in seeds])
                    expected_point = float(np.mean(points)) if np.all(np.isfinite(points)) else np.nan
                    if not same_number(expected_point, mean.get("point")):
                        failures.append(f"{axis}/{mean_key}: aggregate point is not the mean per-seed effect")
                    expected_sd = (float(np.std(points, ddof=1))
                                   if points.size > 1 and np.all(np.isfinite(points)) else np.nan)
                    if not same_number(expected_sd, mean.get("seed_standard_deviation")):
                        failures.append(f"{axis}/{mean_key}: stored seed SD differs from per-seed points")
            print(f"Checked {axis}: {archive.name} ({len(rows)} estimates)", flush=True)
        if hashlib.sha256(source.read_bytes()).hexdigest() != before:
            failures.append(f"{axis}: estimate source changed during numerical verification")
    return {"schema": "research-repair-v1-stored-estimate-verification-v1",
            "status": "PASS" if not failures else "FAIL", "axes": axes, "estimate_counts": counts,
            "checked_utc": datetime.now(timezone.utc).isoformat(), "failures": failures,
            "raw_archives": archive_records, "comparison_atol": 1e-12,
            "estimate_source_sha256": source_hashes,
            "comparison_scope": "serialized arithmetic/percentile consistency; original recovery tolerances are unchanged",
            "new_resampling": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("statistics", "information", "candidates", "mechanism"), action="append")
    args = parser.parse_args()
    result = check(args.axis or ["statistics", "information", "candidates", "mechanism"])
    suffix = "" if args.axis is None else "_" + "_".join(args.axis)
    target = OUT / "logs" / f"stored_estimate_verification{suffix}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{result['status']}: {result['estimate_counts']}; {len(result['failures'])} inconsistencies")
    raise SystemExit(0 if result["status"] == "PASS" else 1)
