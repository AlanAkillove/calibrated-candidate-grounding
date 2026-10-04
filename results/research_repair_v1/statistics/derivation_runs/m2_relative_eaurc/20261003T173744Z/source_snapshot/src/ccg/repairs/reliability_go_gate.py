"""Read-only evaluator for the historical Amendment A5.4 Reliability GO rule.

This module deliberately does not decide whether a reliability map has a
"stable shift" from numeric bins.  That clause had no frozen numerical
threshold; its historical human adjudication is preserved as provenance.
The repaired GO can be established mechanically from Route A when both
historical scorer families pass it.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any


_SPECS: tuple[dict[str, Any], ...] = (
    {
        "scope_id": "phase0a",
        "gate_id": "A5_4_RELIABILITY_GO_COSINE",
        "model": "Phase0A B1 frozen cosine",
        "job_id": "phase0a___all_common__",
        "experiment_id": "p0a1-corrected-metrics-20260928-01",
        "metadata_path": "results/phase0a_corrected/metadata.json",
        "map_paths": ["results/phase0a_corrected/reliability_bins_globalT.csv"],
        "expected_seeds": ["B0"],
        "map_status": "NOT_STABLE_SHIFT_BY_HISTORICAL_ADJUDICATION",
        "map_evidence": "Phase0A log records that the corrected reliability shift did not meet the historical qualitative stability criterion.",
    },
    {
        "scope_id": "phase0b",
        "gate_id": "A5_4_RELIABILITY_GO_B3",
        "model": "Phase0B B3 Independent MLP",
        "job_id": "phase0b___all_common__",
        "experiment_id": "p0b-b3-independent-20260928-01",
        "metadata_path": "results/phase0b_independent/metadata.json",
        "map_paths": [
            "results/phase0b_independent/seed_1/reliability_bins.csv",
            "results/phase0b_independent/seed_2/reliability_bins.csv",
            "results/phase0b_independent/seed_3/reliability_bins.csv",
        ],
        "expected_seeds": ["b3_seed1", "b3_seed2", "b3_seed3"],
        "map_status": "STABLE_DIRECTIONAL_SHIFT_BY_HISTORICAL_ADJUDICATION",
        "map_evidence": "Phase0B log records a conservative shift in supported mid/high bins, with the same direction across three seeds.",
    },
)

_ESTIMATE_NAMES = {
    "eaurc": "crossK_relative_eaurc_worsening__global_T",
    "rer50": "crossK_K5_minus_K50__global_T::rer_at_50",
    "auroc": "crossK_K5_minus_K50__global_T::auroc_correct",
    "rer80_k5": "K5__global_T::rer_at_80",
    "rer80_k50": "K50__global_T::rer_at_80",
    "rer80_delta": "crossK_K5_minus_K50__global_T::rer_at_80",
}
_PROTOCOL = "docs/research_protocol.md"
_EXPERIMENT_LOG = "docs/experiment_log.md"
_PHASE0A_INTERPRETATION = "docs/phase0a_interpretation.md"
_N_ROWS = 20_799
_N_IMAGES = 2_981


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_text(root: Path, relative: str) -> str | None:
    try:
        return (root / relative).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _formal_log_block(log: str, experiment_id: str) -> str | None:
    marker = f"experiment_id: {experiment_id}"
    position = log.find(marker)
    if position < 0:
        return None
    start = log.rfind("# ===== FORMAL ENTRY", 0, position)
    if start < 0:
        start = position
    end = log.find("\n```", position)
    if end < 0:
        end = len(log)
    return log[start:end]


def _parse_bool_field(block: str | None, name: str) -> tuple[bool | None, str | None]:
    if block is None:
        return None, None
    match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*(true|false)\b([^\n]*)", block, re.I)
    if match is None:
        return None, None
    value = match.group(1).lower() == "true"
    return value, match.group(2).strip()


def _old_log_evidence(root: Path, spec: dict[str, Any], protocol: str | None, log: str | None) -> dict[str, Any]:
    block = _formal_log_block(log or "", spec["experiment_id"])
    route_a, _ = _parse_bool_field(block, "route_A_selective")
    route_b, route_b_comment = _parse_bool_field(block, "route_B_correctness")
    errors: list[str] = []
    if protocol is None or "### A5.4 Reliability GO" not in protocol:
        errors.append("The frozen protocol's Amendment A5.4 section is missing or unreadable.")
    else:
        section_start = protocol.find("### A5.4 Reliability GO")
        section_end = protocol.find("### A5.5", section_start)
        section = protocol[section_start:section_end if section_end >= 0 else None]
        for token in ("20%", "10 个百分点", "0.03", "稳定 shift"):
            if token not in section:
                errors.append(f"The Amendment A5.4 protocol section is missing frozen predicate text {token!r}.")
    if block is None:
        errors.append(f"The historical experiment-log block {spec['experiment_id']} is missing.")
    if route_a is None or route_b is None:
        errors.append("The historical experiment-log amendment_gate route flags are missing or not boolean.")

    if spec["scope_id"] == "phase0a":
        map_status = "NOT_STABLE_SHIFT_BY_HISTORICAL_ADJUDICATION" if "未达稳定阈值" in (route_b_comment or "") else "HISTORICAL_MAP_ADJUDICATION_UNVERIFIED"
        map_note = route_b_comment or spec["map_evidence"]
    else:
        map_status = "STABLE_DIRECTIONAL_SHIFT_BY_HISTORICAL_ADJUDICATION" if "稳定 shift" in (route_b_comment or "") else "HISTORICAL_MAP_ADJUDICATION_UNVERIFIED"
        map_note = route_b_comment or spec["map_evidence"]
    if not any((root / path).is_file() for path in spec["map_paths"]):
        map_status = "HISTORICAL_MAP_FILES_UNAVAILABLE"

    old_decision = "UNVERIFIABLE_OLD_DECISION" if errors else ("GO" if route_a or route_b else "NO_GO")
    return {
        "route_a": route_a,
        "route_b": route_b,
        "errors": errors,
        "old_decision": old_decision,
        "map_status": map_status,
        "map_note": map_note,
        "log_block_found": block is not None,
    }


def _source_hashes(root: Path, paths: list[str]) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            hashes[relative] = _sha256(path)
        except OSError:
            missing.append(relative)
    return hashes, missing


def _job_index(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(job.get("job_id")): job for job in summary.get("jobs", []) if isinstance(job, dict)}


def _find_estimate(job: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not isinstance(job, dict):
        return None
    for item in job.get("estimates", []):
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return None


def _finite_number(item: dict[str, Any], field: str) -> float | None:
    try:
        value = float(item[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _estimate_method_errors(
    estimate: dict[str, Any] | None,
    job: dict[str, Any] | None,
    *,
    expected_seeds: list[str],
    model: str,
    estimate_name: str,
) -> list[str]:
    errors: list[str] = []
    label = f"{model} / {estimate_name}"
    if estimate is None:
        return [f"Missing required estimate: {label}."]
    if job is None:
        return [f"Missing required bootstrap job for {label}."]

    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    if config.get("n_replicates") != 5000:
        errors.append(f"{label}: job config n_replicates must be 5000.")
    if config.get("seed") != 0:
        errors.append(f"{label}: job config seed must be 0.")
    if not _is_close(config.get("ci_level"), 0.95):
        errors.append(f"{label}: job config ci_level must be 0.95.")
    if config.get("ci_method") != "percentile":
        errors.append(f"{label}: job config ci_method must be percentile.")
    if job.get("cohort") != "__all_common__":
        errors.append(f"{label}: job must use the historical __all_common__ cohort.")
    if job.get("legacy_cohort_classification") != "ALL_COMMON_INCLUDES_VALIDATION_NO_TRAIN":
        errors.append(f"{label}: job cohort is not certified as all-common validation+test with no training rows.")
    if job.get("status") != "RECOMPUTED_5000":
        errors.append(f"{label}: job status is not RECOMPUTED_5000.")

    for field, expected in (("n_replicates", 5000), ("valid_replicates", 5000), ("invalid_replicates", 0)):
        if estimate.get(field) != expected:
            errors.append(f"{label}: estimate {field} must be {expected}.")
    if estimate.get("gate_eligible") is not True:
        errors.append(f"{label}: estimate gate_eligible must be true.")
    if estimate.get("resample_unit") != "image_cluster":
        errors.append(f"{label}: estimate resample_unit must be image_cluster.")
    if not _is_close(estimate.get("ci_level"), 0.95):
        errors.append(f"{label}: estimate ci_level must be 0.95.")
    if estimate.get("n_rows") != _N_ROWS or estimate.get("n_images") != _N_IMAGES:
        errors.append(f"{label}: estimate must match the historical 20,799-row / 2,981-image cohort.")
    if estimate.get("cohort") != "__all_common__":
        errors.append(f"{label}: estimate cohort must be __all_common__.")

    per_seed = estimate.get("per_seed")
    if not isinstance(per_seed, dict) or set(per_seed) != set(expected_seeds):
        errors.append(f"{label}: per_seed keys must exactly match {expected_seeds!r}.")
    else:
        seed_points: list[float] = []
        for seed in expected_seeds:
            values = per_seed.get(seed)
            if not isinstance(values, dict):
                errors.append(f"{label}: per_seed[{seed!r}] is missing.")
                continue
            for field, expected in (("n_replicates", 5000), ("valid_replicates", 5000), ("invalid_replicates", 0)):
                if values.get(field) != expected:
                    errors.append(f"{label}: per_seed[{seed!r}].{field} must be {expected}.")
            if values.get("gate_eligible") is not True:
                errors.append(f"{label}: per_seed[{seed!r}].gate_eligible must be true.")
            seed_point = _finite_number(values, "point")
            seed_low = _finite_number(values, "ci_low")
            seed_high = _finite_number(values, "ci_high")
            if seed_point is None or seed_low is None or seed_high is None or seed_low > seed_high:
                errors.append(f"{label}: per_seed[{seed!r}] requires finite point and ordered CI endpoints.")
            else:
                seed_points.append(seed_point)
        aggregate_point = _finite_number(estimate, "point")
        if aggregate_point is None or len(seed_points) != len(expected_seeds) or not math.isclose(
            aggregate_point, sum(seed_points) / len(seed_points), rel_tol=0.0, abs_tol=1e-9,
        ):
            errors.append(f"{label}: aggregate point must equal the mean of the expected per-seed points.")
    return errors


def _is_close(value: Any, expected: float) -> bool:
    try:
        return math.isfinite(float(value)) and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _effect_payload(estimate: dict[str, Any] | None, *, units: str, sign: str, scale: float = 1.0) -> dict[str, Any]:
    if estimate is None:
        return {"point": None, "ci95": [None, None], "units": units, "positive_effect": sign}
    point = _finite_number(estimate, "point")
    low = _finite_number(estimate, "ci_low")
    high = _finite_number(estimate, "ci_high")
    return {
        "point": None if point is None else point * scale,
        "ci95": [None if low is None else low * scale, None if high is None else high * scale],
        "units": units,
        "positive_effect": sign,
    }


def _route_a(
    estimate_e: dict[str, Any] | None,
    estimate_r50: dict[str, Any] | None,
    estimate_r80_k5: dict[str, Any] | None,
    estimate_r80_k50: dict[str, Any] | None,
    estimate_r80_delta: dict[str, Any] | None,
    *,
    e_errors: list[str],
    r50_errors: list[str],
    r80_endpoint_errors: list[str],
    r80_delta_errors: list[str],
) -> dict[str, Any]:
    e = _effect_payload(
        estimate_e, units="fraction relative worsening (display percent = x100)",
        sign="E-AURC(K50)-E-AURC(K5), relative to E-AURC(K5)",
    )
    r50 = _effect_payload(
        estimate_r50, units="percentage points (source estimate is fraction; displayed x100)",
        sign="RER@50(K5)-RER@50(K50)", scale=100.0,
    )
    epoint, elow = e["point"], e["ci95"][0]
    if e_errors or epoint is None or elow is None:
        reasons = list(e_errors)
        if epoint is None or elow is None:
            reasons.append("Route A E-AURC point or CI lower endpoint is missing/non-finite.")
        return {"status": "UNCERTAINTY_INSUFFICIENT", "reasons": reasons,
                "eaurc_relative_worsening": e, "rer50_drop": r50}
    e_pass = epoint >= 0.20 and elow > 0.0
    if not e_pass:
        return {
            "status": "FAIL",
            "witness": "none; E-AURC prerequisite did not pass",
            "rer_witness_selection": "Route A requires the E-AURC prerequisite and either RER@50 or RER@80.",
            "eaurc_relative_worsening": {**e, "point_threshold": 0.20, "ci_excludes_zero_in_positive_direction": elow > 0.0, "threshold_pass": False},
            "rer50_drop": r50,
            "rer80_alternative": _rer80_alternative(estimate_r80_k5, estimate_r80_k50, estimate_r80_delta, r80_endpoint_errors, r80_delta_errors),
        }

    r50_numeric = (r50["point"] is not None and r50["ci95"][0] is not None)
    r50_pass = not r50_errors and r50_numeric and r50["point"] >= 10.0 and r50["ci95"][0] > 0.0
    r80 = _rer80_alternative(estimate_r80_k5, estimate_r80_k50, estimate_r80_delta, r80_endpoint_errors, r80_delta_errors)
    if r50_pass:
        status = "PASS"
    elif r50_errors or not r50_numeric:
        status = "UNCERTAINTY_INSUFFICIENT"
    elif r80["status"] == "PASS":
        status = "PASS"
    elif r80["status"] in ("POINT_PASSES_BUT_PAIRED_CI_UNAVAILABLE", "UNCERTAINTY_INSUFFICIENT"):
        status = "UNCERTAINTY_INSUFFICIENT"
    else:
        status = "FAIL"
    return {
        "status": status,
        "witness": "K5_to_K50_RER50" if r50_pass else ("K5_to_K50_RER80" if r80["status"] == "PASS" else None),
        "rer_witness_selection": "RER@50 is the tested witness. The frozen protocol permits RER@50 OR RER@80; the observed RER@80 K5/K50 point contrast is shown as an auxiliary branch check, but no paired CI is reconstructed from separate endpoint CIs.",
        "eaurc_relative_worsening": {**e, "point_threshold": 0.20, "ci_excludes_zero_in_positive_direction": elow > 0.0, "threshold_pass": True},
        "rer50_drop": {**r50, "point_threshold": 10.0,
                       "ci_excludes_zero_in_positive_direction": r50["ci95"][0] is not None and r50["ci95"][0] > 0.0,
                       "threshold_pass": bool(r50_pass), "eligibility_reasons": r50_errors},
        "rer80_alternative": r80,
        "reasons": (r50_errors if status == "UNCERTAINTY_INSUFFICIENT" and r50_errors else
                    ["RER@80 observed point reaches the 10pp threshold, but the paired K5-minus-K50 RER@80 CI is not named in the summary."]
                    if r80["status"] == "POINT_PASSES_BUT_PAIRED_CI_UNAVAILABLE" else []),
    }


def _rer80_alternative(
    estimate_k5: dict[str, Any] | None,
    estimate_k50: dict[str, Any] | None,
    estimate_delta: dict[str, Any] | None,
    endpoint_errors: list[str],
    delta_errors: list[str],
) -> dict[str, Any]:
    if estimate_delta is not None:
        effect = _effect_payload(
            estimate_delta, units="percentage points (source estimate is fraction; displayed x100)",
            sign="RER@80(K5)-RER@80(K50)", scale=100.0,
        )
        if delta_errors:
            return {"status": "UNCERTAINTY_INSUFFICIENT", "reasons": delta_errors, "paired_drop": effect}
        point, low = effect["point"], effect["ci95"][0]
        if point is None or low is None:
            return {"status": "UNCERTAINTY_INSUFFICIENT", "reasons": ["Named paired RER@80 point/CI is incomplete."], "paired_drop": effect}
        passed = point >= 10.0 and low > 0.0
        return {"status": "PASS" if passed else "FAIL", "paired_drop": {**effect, "point_threshold": 10.0, "ci_excludes_zero_in_positive_direction": low > 0.0, "threshold_pass": passed}}
    if endpoint_errors or estimate_k5 is None or estimate_k50 is None:
        reasons = list(endpoint_errors)
        if estimate_k5 is None or estimate_k50 is None:
            reasons.append("RER@80 K5/K50 endpoint estimates are missing.")
        return {"status": "UNCERTAINTY_INSUFFICIENT", "reasons": reasons, "paired_drop": None}
    k5_point = _finite_number(estimate_k5, "point")
    k50_point = _finite_number(estimate_k50, "point")
    if k5_point is None or k50_point is None:
        return {"status": "UNCERTAINTY_INSUFFICIENT", "reasons": ["RER@80 endpoint point estimate is missing/non-finite."], "paired_drop": None}
    drop_pp = 100.0 * (k5_point - k50_point)
    if drop_pp < 10.0:
        status = "POINT_BELOW_THRESHOLD"
    else:
        status = "POINT_PASSES_BUT_PAIRED_CI_UNAVAILABLE"
    return {
        "status": status,
        "k5_point_fraction": k5_point,
        "k50_point_fraction": k50_point,
        "observed_drop_pp": drop_pp,
        "point_threshold_pp": 10.0,
        "paired_ci": None,
        "endpoint_cis_used_for_difference": False,
        "note": "The drop is a point-only difference of same-job endpoints. Do not subtract their marginal CI endpoints to create a paired CI.",
    }


def _route_b(estimate: dict[str, Any] | None, errors: list[str], map_status: str) -> dict[str, Any]:
    auc = _effect_payload(
        estimate, units="AUROC", sign="AUROC_correct(K5)-AUROC_correct(K50)",
    )
    if errors:
        return {
            "status": "UNCERTAINTY_INSUFFICIENT",
            "numeric_status": "UNCERTAINTY_INSUFFICIENT",
            "reasons": errors,
            "auroc_drop": auc,
            "map_component": {
                "status": map_status,
                "new_manual_review": "NOT_PERFORMED",
                "numeric_threshold": None,
                "note": "Historical manual map adjudication is preserved; no new numeric stability threshold is defined.",
            },
        }
    point, low = auc["point"], auc["ci95"][0]
    if point is None or low is None:
        numeric_status = "UNCERTAINTY_INSUFFICIENT"
    else:
        numeric_status = "PASS" if point >= 0.03 and low > 0.0 else "FAIL"
    # A historical qualitative label is provenance, not a fresh manual review
    # of the repaired comparison.  Therefore a passing numeric Route B alone
    # remains unverifiable under the current run.
    complete_route = "UNVERIFIABLE_MANUAL_MAP_REVIEW_REQUIRED" if numeric_status == "PASS" else "FAIL"
    return {
        "status": complete_route,
        "numeric_status": numeric_status,
        "auroc_drop": {**auc, "point_threshold": 0.03, "ci_excludes_zero_in_positive_direction": low is not None and low > 0.0},
        "map_component": {
            "status": map_status,
            "new_manual_review": "NOT_PERFORMED",
            "numeric_threshold": None,
            "note": "The source protocol says 'stable shift' qualitatively and freezes no numerical bin-gap, support, or stability cutoff.",
            "use_in_new_go": False,
        },
    }


def _estimate_ref(root: Path, job_id: str, name: str, estimate: dict[str, Any] | None, units: str, sign: str) -> dict[str, Any]:
    ref = {
        "job_id": job_id,
        "estimate_name": name,
        "units": units,
        "positive_effect": sign,
    }
    if estimate is not None:
        raw_path = estimate.get("raw_replicates")
        ref["raw_replicates"] = raw_path
        ref["raw_replicates_key"] = estimate.get("raw_replicates_key")
        if isinstance(raw_path, str):
            path = root / raw_path
            if path.is_file():
                try:
                    ref["raw_replicates_sha256"] = _sha256(path)
                except OSError:
                    ref["raw_replicates_hash_status"] = "UNREADABLE"
            else:
                ref["raw_replicates_hash_status"] = "MISSING"
        ref["source_artifacts"] = estimate.get("source_artifacts", [])
    return ref


def _evaluate_source(root: Path, summary: dict[str, Any], spec: dict[str, Any], protocol: str | None, log: str | None) -> dict[str, Any]:
    jobs = _job_index(summary)
    job = jobs.get(spec["job_id"])
    e_est = _find_estimate(job, _ESTIMATE_NAMES["eaurc"])
    r_est = _find_estimate(job, _ESTIMATE_NAMES["rer50"])
    a_est = _find_estimate(job, _ESTIMATE_NAMES["auroc"])
    r80_k5_est = _find_estimate(job, _ESTIMATE_NAMES["rer80_k5"])
    r80_k50_est = _find_estimate(job, _ESTIMATE_NAMES["rer80_k50"])
    r80_delta_est = _find_estimate(job, _ESTIMATE_NAMES["rer80_delta"])

    e_errors = _estimate_method_errors(e_est, job, expected_seeds=spec["expected_seeds"], model=spec["model"], estimate_name=_ESTIMATE_NAMES["eaurc"])
    r_errors = _estimate_method_errors(r_est, job, expected_seeds=spec["expected_seeds"], model=spec["model"], estimate_name=_ESTIMATE_NAMES["rer50"])
    a_errors = _estimate_method_errors(a_est, job, expected_seeds=spec["expected_seeds"], model=spec["model"], estimate_name=_ESTIMATE_NAMES["auroc"])
    r80_k5_errors = _estimate_method_errors(r80_k5_est, job, expected_seeds=spec["expected_seeds"], model=spec["model"], estimate_name=_ESTIMATE_NAMES["rer80_k5"])
    r80_k50_errors = _estimate_method_errors(r80_k50_est, job, expected_seeds=spec["expected_seeds"], model=spec["model"], estimate_name=_ESTIMATE_NAMES["rer80_k50"])
    r80_delta_errors = _estimate_method_errors(r80_delta_est, job, expected_seeds=spec["expected_seeds"], model=spec["model"], estimate_name=_ESTIMATE_NAMES["rer80_delta"])
    old = _old_log_evidence(root, spec, protocol, log)
    map_hashes, missing_maps = _source_hashes(root, spec["map_paths"])
    map_status = old["map_status"]
    if missing_maps and map_status != "HISTORICAL_MAP_ADJUDICATION_UNVERIFIED":
        map_status = "HISTORICAL_MAP_FILES_UNAVAILABLE"
    route_b = _route_b(a_est, a_errors, map_status)
    route_a = _route_a(
        e_est, r_est, r80_k5_est, r80_k50_est, r80_delta_est,
        e_errors=e_errors, r50_errors=r_errors,
        r80_endpoint_errors=r80_k5_errors + r80_k50_errors,
        r80_delta_errors=r80_delta_errors,
    )
    if route_a["status"] == "PASS":
        new_decision = "GO_ROUTE_A"
    elif route_a["status"] == "UNCERTAINTY_INSUFFICIENT":
        new_decision = "UNCERTAINTY_INSUFFICIENT"
    elif route_b["numeric_status"] == "PASS":
        new_decision = "UNVERIFIABLE_ROUTE_B_MANUAL_MAP_REVIEW_REQUIRED"
    elif route_b["numeric_status"] == "UNCERTAINTY_INSUFFICIENT":
        new_decision = "UNCERTAINTY_INSUFFICIENT"
    else:
        new_decision = "NO_ROUTE_MET"

    source_paths = [
        _PROTOCOL, _EXPERIMENT_LOG, _PHASE0A_INTERPRETATION,
        spec["metadata_path"], *spec["map_paths"],
    ]
    source_hashes, missing_sources = _source_hashes(root, source_paths)
    old_decision = old["old_decision"]
    method = {
        "n_replicates": 5000,
        "seed": 0,
        "ci_level": 0.95,
        "ci_method": "percentile",
        "resample_unit": "image_cluster",
        "cohort": "__all_common__ (validation plus test, no train)",
        "n_rows": _N_ROWS,
        "n_images": _N_IMAGES,
        "same_image_draws_for_all_K_and_seeds": True,
    }
    refs = [
        _estimate_ref(root, spec["job_id"], _ESTIMATE_NAMES["eaurc"], e_est, "relative fraction", "K50 E-AURC minus K5 E-AURC, divided by K5 E-AURC"),
        _estimate_ref(root, spec["job_id"], _ESTIMATE_NAMES["rer50"], r_est, "fraction; report as percentage points x100", "RER@50(K5) minus RER@50(K50)"),
        _estimate_ref(root, spec["job_id"], _ESTIMATE_NAMES["rer80_k5"], r80_k5_est, "condition endpoint fraction; point-only fallback", "RER@80(K5) endpoint"),
        _estimate_ref(root, spec["job_id"], _ESTIMATE_NAMES["rer80_k50"], r80_k50_est, "condition endpoint fraction; point-only fallback", "RER@80(K50) endpoint"),
        _estimate_ref(root, spec["job_id"], _ESTIMATE_NAMES["auroc"], a_est, "AUROC", "AUROC_correct(K5) minus AUROC_correct(K50)"),
    ]
    return {
        "scope_id": spec["scope_id"],
        "gate_id": spec["gate_id"],
        "criterion": "Amendment A5.4 post-hoc Reliability GO (replication); not Gate Q1/Q2/Q3",
        "model": spec["model"],
        "historical_experiment_id": spec["experiment_id"],
        "source": [
            f"{_PROTOCOL}#Amendment-A5.4",
            f"{_EXPERIMENT_LOG}#{spec['experiment_id']}",
            spec["metadata_path"],
            *spec["map_paths"],
        ],
        "predicate_source": f"{_PROTOCOL}#A5.4; {_EXPERIMENT_LOG}#{spec['experiment_id']}::amendment_gate",
        "old_thresholds": {
            "route_A": {
                "eaurc_relative_worsening_gte_fraction": 0.20,
                "eaurc_95pct_ci_excludes_zero": True,
                "rer50_or_rer80_drop_gte_percentage_points": 10.0,
                "rer_95pct_ci_excludes_zero": True,
            },
            "route_B": {
                "auroc_correct_drop_gte": 0.03,
                "auroc_95pct_ci_excludes_zero": True,
                "additional_manual_condition": "stable corrected-global-T reliability-map shift; qualitative with no frozen numeric cutoff",
            },
            "threshold_change": "NONE",
        },
        "old_decision": old_decision,
        "old_route_decisions": {"route_A": old["route_a"], "route_B": old["route_b"]},
        "old_map_adjudication": {
            "status": old["map_status"],
            "historical_log_note": old["map_note"],
            "new_manual_review": "NOT_PERFORMED",
            "numeric_threshold": None,
            "interpretation": "Historical qualitative adjudication is retained as reported; the evaluator does not manufacture a quantitative stability rule.",
        },
        "new_estimates": refs,
        "bootstrap": method,
        "new_result": {
            "route_A": route_a,
            "route_B": route_b,
            "decision_basis": "Route A only" if route_a["status"] == "PASS" else "Route A did not establish GO; Route B cannot establish new GO without a fresh manual map adjudication.",
        },
        "new_decision": new_decision,
        "status": (
            "UNCERTAINTY_INSUFFICIENT" if new_decision == "UNCERTAINTY_INSUFFICIENT" else
            "UNVERIFIABLE" if new_decision == "UNVERIFIABLE_ROUTE_B_MANUAL_MAP_REVIEW_REQUIRED" else
            "EVALUATED"
        ),
        "source_sha256": source_hashes,
        "missing_source_files": missing_sources,
        "source_validation_errors": old["errors"],
        "cohort_identity_warning": "Legacy Repair scope is named Phase0A_B0_temperature/Phase0B_temperature, but the A5.4 scorer identities are B1 cosine and B3 Independent; corrected global-T only.",
        "interpretation_limit": "A Route A GO supports the selective-metric finding from E-AURC/RER; it does not establish an AUROC_correct decline. Phase0A's AUROC Route B numeric estimate is reported separately and its CI crosses zero.",
    }


def build_reliability_go_decisions(root: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-scorer and composite A5.4 decision rows without mutation.

    The function reads frozen protocol/log/map files under ``root`` and only
    reads the supplied repair summary.  It never recomputes bootstrap draws.
    """
    root = Path(root)
    protocol = _read_text(root, _PROTOCOL)
    log = _read_text(root, _EXPERIMENT_LOG)
    source_rows = [_evaluate_source(root, summary, spec, protocol, log) for spec in _SPECS]
    both_route_a_pass = all(row["new_result"]["route_A"]["status"] == "PASS" for row in source_rows)
    any_bootstrap_unknown = any(row["new_decision"] == "UNCERTAINTY_INSUFFICIENT" for row in source_rows)
    route_b_needed_but_manual_unverified = any(
        row["new_decision"] == "UNVERIFIABLE_ROUTE_B_MANUAL_MAP_REVIEW_REQUIRED" for row in source_rows
    )
    if both_route_a_pass:
        composite_new = "GO_ROUTE_A_REPLICATED"
        composite_status = "EVALUATED"
        basis = "Both the frozen B1 cosine and independent B3 source pass the unchanged Route A K5-to-K50 thresholds. Route B map adjudication is not used to establish this repaired GO."
    elif any_bootstrap_unknown:
        composite_new = "UNCERTAINTY_INSUFFICIENT"
        composite_status = "UNCERTAINTY_INSUFFICIENT"
        basis = "At least one source lacks eligible Route A bootstrap evidence; Route B cannot replace this until its numeric condition and manual map review are independently established."
    elif route_b_needed_but_manual_unverified:
        composite_new = "UNVERIFIABLE_ROUTE_B_MANUAL_MAP_REVIEW_REQUIRED"
        composite_status = "UNVERIFIABLE"
        basis = "Route A does not establish replication and at least one Route B numeric result passes, but the qualitative map clause has no new manual adjudication or numeric cutoff."
    elif all(row["new_decision"] == "NO_ROUTE_MET" for row in source_rows):
        composite_new = "NO_GO"
        composite_status = "EVALUATED"
        basis = "Neither source meets Route A or the numeric Route B condition."
    else:
        composite_new = "NO_GO_REPLICATION_NOT_ESTABLISHED"
        composite_status = "EVALUATED"
        basis = "The unchanged criterion requires both scorer families to show the same-direction phenomenon; at least one source fails all mechanically established routes."

    log_block = _formal_log_block(log or "", _SPECS[1]["experiment_id"])
    old_composite_recorded = bool(log_block and re.search(r"Case A\s*[：:]\s*GO", log_block, re.I))
    old_composite = "GO_CASE_A" if old_composite_recorded and all(row["old_route_decisions"]["route_A"] is not None for row in source_rows) else "UNVERIFIABLE_OLD_COMPOSITE"
    shared_sources = sorted({path for row in source_rows for path in row["source_sha256"]})
    shared_hashes = {path: source_rows[0]["source_sha256"].get(path) or source_rows[1]["source_sha256"].get(path) for path in shared_sources}
    composite = {
        "scope_id": "reliability_go",
        "gate_id": "A5_4_RELIABILITY_GO_REPLICATION",
        "criterion": "Amendment A5.4 post-hoc Reliability GO (replication); not Gate Q1/Q2/Q3",
        "source": [f"{_PROTOCOL}#Amendment-A5.4", f"{_EXPERIMENT_LOG}#Phase0A-and-Phase0B-amendment_gate"],
        "predicate_source": f"{_PROTOCOL}#A5.4; {_EXPERIMENT_LOG}#Phase0B-Case-A",
        "old_thresholds": source_rows[0]["old_thresholds"],
        "old_decision": old_composite,
        "old_route_decisions": {row["scope_id"]: row["old_route_decisions"] for row in source_rows},
        "new_estimates": [ref for row in source_rows for ref in row["new_estimates"]],
        "bootstrap": {
            "n_replicates": 5000,
            "seed": 0,
            "ci_level": 0.95,
            "ci_method": "percentile",
            "resample_unit": "image_cluster",
            "cohort": "__all_common__",
            "n_rows": _N_ROWS,
            "n_images": _N_IMAGES,
            "same_image_draws_within_each_source_across_K_and_seeds": True,
        },
        "source_rows": [
            {"scope_id": row["scope_id"], "gate_id": row["gate_id"], "new_decision": row["new_decision"],
             "route_A_status": row["new_result"]["route_A"]["status"], "route_B_status": row["new_result"]["route_B"]["status"]}
            for row in source_rows
        ],
        "new_decision": composite_new,
        "status": composite_status,
        "decision_basis": basis,
        "map_clause_handling": "Preserve the historical manual outcome and map files. The map condition remains qualitative, with no numerical threshold; repaired GO here is supported by Route A only.",
        "interpretation_limit": "The replicated A5.4 GO is supported by selective-metric Route A only; it does not imply that cosine AUROC_correct declined. Its K5−K50 AUROC CI crosses zero.",
        "threshold_change": "NONE",
        "source_sha256": shared_hashes,
        "source_validation_errors": sorted({error for row in source_rows for error in row["source_validation_errors"]}),
    }
    return [*source_rows, composite]
