"""Evaluate the frozen V2-M legacy gates on eligible repair estimates.

This module is deliberately read-only with respect to legacy artifacts.  It
returns registry rows; callers own any output file updates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAL_REPLICATES = 5000
FORMAL_SEED = 0
FORMAL_CI = 0.95
FORMAL_CI_METHOD = "percentile"
RESAMPLE_UNIT = "image_cluster"
FIXED_SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")

M1_JOB = {
    "primary": "m1_samecategory_k5",
    "random_guard": "m1_random_k5",
}
M2_JOB = "m2_curriculum_m0_m2_m4_m8"
M25_JOB = "m25_specialist_m0_m2_m4_m8"
M2_RELATIVE_EAURC = "m8__LCR_E1b_relative_eaurc_reduction"


def _relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"legacy source escapes repository root: {relative}")
    return path


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    path = _relative(root, relative)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_meta(
    root: Path,
    json_sources: Sequence[tuple[str, str]],
    predicate_sources: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Hash every JSON/code source and attach its exact symbol or pointer."""
    artifact_hashes: dict[str, str] = {}
    source_pointers: list[str] = []
    for relative, pointer in json_sources:
        path = _relative(root, relative)
        artifact_hashes[relative] = _sha256(path)
        source_pointers.append(f"{relative}#{pointer}")
    predicate_hashes: dict[str, str] = {}
    predicate_labels: list[str] = []
    for relative, symbol in predicate_sources:
        path = _relative(root, relative)
        predicate_hashes[relative] = _sha256(path)
        predicate_labels.append(f"{relative}::{symbol}")
    return {
        "source": "; ".join(source_pointers),
        "source_sha256": artifact_hashes,
        "predicate_source": "; ".join(predicate_labels),
        "predicate_source_sha256": predicate_hashes,
    }


def _ref(job_id: str, estimate_name: str) -> dict[str, str]:
    return {"job_id": job_id, "estimate_name": estimate_name}


def _estimate_index(summary: Mapping[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs: dict[str, dict[str, Any]] = {}
    estimates: dict[tuple[str, str], dict[str, Any]] = {}
    for job in summary.get("jobs", []):
        if not isinstance(job, Mapping) or not job.get("job_id"):
            continue
        job_id = str(job["job_id"])
        jobs[job_id] = dict(job)
        for estimate in job.get("estimates", []):
            if isinstance(estimate, Mapping) and estimate.get("name"):
                estimates[(job_id, str(estimate["name"]))] = dict(estimate)
    # A's post-processing adapter also publishes derived M2 effects at the
    # summary root.  Keep the ordinary job estimate authoritative, but accept
    # the root view when callers pass a summary before nesting is normalized.
    for estimate in summary.get("estimates", []):
        if not isinstance(estimate, Mapping) or not estimate.get("name"):
            continue
        job_id = str(estimate.get("job_id", ""))
        if job_id:
            estimates.setdefault((job_id, str(estimate["name"])), dict(estimate))
    return estimates, jobs


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _equals_int(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def _coverage_issues(
    coverage: Mapping[str, Any], scope_id: str, refs: Sequence[Mapping[str, str]],
) -> tuple[list[str], list[str]]:
    """Return (pending, invalid) issues for the exact jobs behind a gate."""
    scope = _scope_record(coverage, scope_id)
    if scope is None:
        return [f"coverage scope {scope_id} is missing"], []
    ledger = scope.get("job_estimator_ledger", [])
    by_job = {
        str(item.get("job_id")): item
        for item in ledger
        if isinstance(item, Mapping) and item.get("job_id")
    }
    pending: list[str] = []
    invalid: list[str] = []
    for ref in refs:
        job_id = str(ref["job_id"])
        estimate_name = str(ref["estimate_name"])
        item = by_job.get(job_id)
        label = f"coverage {scope_id}/{job_id}/{estimate_name}"
        if item is None:
            pending.append(f"{label}: job estimator ledger is missing")
            continue
        status = item.get("formal_status")
        if status in (None, "PENDING_FORMAL_BOOTSTRAP", "WAITING_RESOURCE_SLOT", "RUNNING"):
            pending.append(f"{label}: formal job is still pending")
        elif status != "RECOMPUTED_5000":
            invalid.append(f"{label}: formal_status is {status!r}, expected RECOMPUTED_5000")
        if item.get("completed") is not True:
            if status == "RECOMPUTED_5000":
                invalid.append(f"{label}: completed flag conflicts with formal_status")
            elif not any(issue.startswith(label) and "still pending" in issue for issue in pending):
                pending.append(f"{label}: completed flag is false")
        if not _equals_int(item.get("n_replicates"), FORMAL_REPLICATES):
            if status in (None, "PENDING_FORMAL_BOOTSTRAP", "WAITING_RESOURCE_SLOT", "RUNNING"):
                pass
            else:
                invalid.append(f"{label}: coverage n_replicates != 5000")
        registered = set(str(x) for x in item.get("estimate_names", []))
        registered.update(str(x) for x in item.get("auxiliary_estimate_names", []))
        if estimate_name not in registered:
            invalid.append(f"{label}: estimate is absent from the exact coverage ledger")
    return pending, invalid


def _global_method_issues(summary: Mapping[str, Any]) -> list[str]:
    bootstrap = summary.get("bootstrap", {})
    if not isinstance(bootstrap, Mapping):
        bootstrap = {}
    issues: list[str] = []
    if not _equals_int(bootstrap.get("replicates"), FORMAL_REPLICATES):
        issues.append("summary bootstrap replicates != 5000")
    if not _equals_int(bootstrap.get("seed"), FORMAL_SEED):
        issues.append("summary bootstrap seed != 0")
    try:
        if not math.isclose(float(bootstrap.get("ci_level", -1)), FORMAL_CI, rel_tol=0.0, abs_tol=1e-12):
            issues.append("summary bootstrap CI != 0.95")
    except (TypeError, ValueError):
        issues.append("summary bootstrap CI missing")
    if bootstrap.get("resample_unit") != RESAMPLE_UNIT:
        issues.append("summary resample unit != image_cluster")
    if bootstrap.get("method", FORMAL_CI_METHOD) != FORMAL_CI_METHOD:
        issues.append("summary CI method != percentile")
    return issues


def _estimate_eligibility(
    summary: Mapping[str, Any],
    refs: Sequence[Mapping[str, str]],
    coverage: Mapping[str, Any],
    coverage_scope_id: str,
    *,
    fixed_seeds: Sequence[str] = FIXED_SEEDS,
) -> tuple[str, list[str], dict[tuple[str, str], dict[str, Any]]]:
    """Require the frozen method and every planned replicate to be gate-usable."""
    estimates, jobs = _estimate_index(summary)
    found: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    bad: list[str] = _global_method_issues(summary)
    pending, coverage_bad = _coverage_issues(coverage, coverage_scope_id, refs)
    bad.extend(coverage_bad)
    missing.extend(pending)
    for ref in refs:
        job_id = str(ref["job_id"])
        name = str(ref["estimate_name"])
        key = (job_id, name)
        estimate = estimates.get(key)
        job = jobs.get(job_id)
        label = f"{job_id}/{name}"
        if estimate is None or job is None:
            missing.append(label)
            continue
        found[key] = estimate
        if job.get("status") != "RECOMPUTED_5000":
            bad.append(f"{label}: job status is not RECOMPUTED_5000")
        config = job.get("config", {})
        result = job.get("result", {})
        if not isinstance(config, Mapping):
            config = {}
        if not isinstance(result, Mapping):
            result = {}
        if not _equals_int(config.get("n_replicates"), FORMAL_REPLICATES):
            bad.append(f"{label}: job config replicate count != 5000")
        if not _equals_int(config.get("seed"), FORMAL_SEED):
            bad.append(f"{label}: job config seed != 0")
        try:
            if not math.isclose(float(config.get("ci_level", -1)), FORMAL_CI, rel_tol=0.0, abs_tol=1e-12):
                bad.append(f"{label}: job config CI != 0.95")
        except (TypeError, ValueError):
            bad.append(f"{label}: job config CI missing")
        if config.get("ci_method", FORMAL_CI_METHOD) != FORMAL_CI_METHOD:
            bad.append(f"{label}: job config CI method != percentile")
        if result.get("resample_unit") != RESAMPLE_UNIT:
            bad.append(f"{label}: result resample unit != image_cluster")
        if not _equals_int(result.get("n_replicates"), FORMAL_REPLICATES):
            bad.append(f"{label}: result replicate count != 5000")
        if not _equals_int(result.get("seed"), FORMAL_SEED):
            bad.append(f"{label}: result seed != 0")
        try:
            if not math.isclose(float(result.get("ci_level", -1)), FORMAL_CI, rel_tol=0.0, abs_tol=1e-12):
                bad.append(f"{label}: result CI != 0.95")
        except (TypeError, ValueError):
            bad.append(f"{label}: result CI missing")
        if result.get("method", FORMAL_CI_METHOD) != FORMAL_CI_METHOD:
            bad.append(f"{label}: result CI method != percentile")

        valid_ok = _equals_int(estimate.get("valid_replicates"), FORMAL_REPLICATES)
        invalid_ok = _equals_int(estimate.get("invalid_replicates"), 0)
        if estimate.get("gate_eligible") is not True or not valid_ok or not invalid_ok:
            bad.append(f"{label}: estimate is not 5000/5000 valid and gate eligible")
        if not _equals_int(estimate.get("n_replicates"), FORMAL_REPLICATES):
            bad.append(f"{label}: estimate n_replicates != 5000")
        if estimate.get("resample_unit") != RESAMPLE_UNIT:
            bad.append(f"{label}: estimate resample_unit != image_cluster")
        try:
            if not math.isclose(float(estimate.get("ci_level", -1)), FORMAL_CI, rel_tol=0.0, abs_tol=1e-12):
                bad.append(f"{label}: estimate CI != 0.95")
        except (TypeError, ValueError):
            bad.append(f"{label}: estimate CI missing")
        if not all(_is_finite_number(estimate.get(field)) for field in ("point", "ci_low", "ci_high")):
            bad.append(f"{label}: point or CI is missing/non-finite")
        per_seed = estimate.get("per_seed")
        if not isinstance(per_seed, Mapping) or set(str(k) for k in per_seed) != set(fixed_seeds):
            bad.append(f"{label}: per_seed does not contain exactly {list(fixed_seeds)}")
            continue
        for seed in fixed_seeds:
            item = per_seed[seed]
            if not isinstance(item, Mapping):
                bad.append(f"{label}/{seed}: seed estimate is not a mapping")
                continue
            if (item.get("gate_eligible") is not True
                    or not _equals_int(item.get("valid_replicates"), FORMAL_REPLICATES)
                    or not _equals_int(item.get("invalid_replicates"), 0)):
                bad.append(f"{label}/{seed}: seed estimate is not 5000/5000 valid and gate eligible")
            if not all(_is_finite_number(item.get(field)) for field in ("point", "ci_low", "ci_high")):
                bad.append(f"{label}/{seed}: point or CI is missing/non-finite")

    if missing:
        return "PENDING_FORMAL_BOOTSTRAP", [f"formal evidence pending: {item}" for item in missing] + bad, found
    if bad:
        return "UNCERTAINTY_INSUFFICIENT", bad, found
    return "ELIGIBLE", [], found


def _metric(index: Mapping[tuple[str, str], Mapping[str, Any]], ref: Mapping[str, str], field: str) -> float:
    return float(index[(str(ref["job_id"]), str(ref["estimate_name"]))][field])


def _decision_base(
    *,
    scope_id: str,
    gate_id: str,
    source_meta: Mapping[str, Any],
    old_thresholds: Any,
    old_decision: Any,
    refs: Sequence[Mapping[str, str]],
    predicate_note: str,
) -> dict[str, Any]:
    return {
        "scope_id": scope_id,
        "gate_id": gate_id,
        **dict(source_meta),
        "old_thresholds": old_thresholds,
        "old_decision": old_decision,
        "new_estimates": [dict(ref) for ref in refs],
        "threshold_change": "NONE",
        "predicate_note": predicate_note,
    }


def _finish(
    record: dict[str, Any],
    summary: Mapping[str, Any],
    refs: Sequence[Mapping[str, str]],
    coverage: Mapping[str, Any],
    coverage_scope_id: str,
    evaluator: Any,
) -> dict[str, Any]:
    state, issues, index = _estimate_eligibility(summary, refs, coverage, coverage_scope_id)
    if state != "ELIGIBLE":
        record.update(status=state, new_decision=state, eligibility_issues=issues)
        return record
    result = evaluator(index)
    record.update(status="EVALUATED", new_decision=result.pop("decision"), new_result=result)
    return record


def _scope_record(coverage: Mapping[str, Any], scope_id: str) -> dict[str, Any] | None:
    scopes = coverage.get("scopes", [])
    if isinstance(scopes, Mapping):
        value = scopes.get(scope_id)
        if isinstance(value, Mapping):
            return dict(value, scope_id=value.get("scope_id", scope_id))
        return None
    for value in scopes:
        if isinstance(value, Mapping) and str(value.get("scope_id")) == scope_id:
            return dict(value)
    return None


def _verified_m3_failures(root: Path, coverage: Mapping[str, Any]) -> list[dict[str, Any]]:
    scope = _scope_record(coverage, "m3")
    if scope is None:
        raise ValueError("M3 coverage scope is missing; cannot bind strict recovery evidence")
    entries = scope.get("unverifiable_jobs", [])
    expected = {
        "m3_b0_severity_macro",
        "m3_b1_severity_macro",
        "m3_b2_severity_macro",
    }
    by_job = {str(item.get("job_id")): item for item in entries if isinstance(item, Mapping)}
    if set(by_job) != expected:
        raise ValueError(f"M3 requires strict failure evidence for exactly {sorted(expected)}")
    verified: list[dict[str, Any]] = []
    for job_id in sorted(expected):
        item = by_job[job_id]
        relative = str(item.get("evidence_path", ""))
        expected_hash = str(item.get("evidence_sha256", item.get("sha256", "")))
        if item.get("status") != "UNVERIFIABLE" or item.get("required", True) is not True:
            raise ValueError(f"M3 {job_id}: recovery evidence is not required UNVERIFIABLE evidence")
        if not relative or len(expected_hash) != 64:
            raise ValueError(f"M3 {job_id}: failure path or SHA-256 is missing")
        failure_path = _relative(root, relative)
        if not failure_path.is_file() or _sha256(failure_path) != expected_hash:
            raise ValueError(f"M3 {job_id}: failure record hash does not match coverage")
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if (failure.get("schema") != "research-repair-v1-recovery-failure-v1"
                or failure.get("scope_id") != "m3"
                or failure.get("job_id") != job_id
                or failure.get("status") != "UNVERIFIABLE"
                or not str(failure.get("failure_reason", "")).strip()
                or not failure.get("original_tolerance")
                or not isinstance(failure.get("attempted_route"), list)
                or not failure.get("attempted_route")):
            raise ValueError(f"M3 {job_id}: failure record schema/provenance is incomplete")
        source_hashes = failure.get("source_sha256")
        if (not isinstance(source_hashes, Mapping) or not source_hashes
                or any(not isinstance(value, str) or len(value) != 64
                       or any(ch not in "0123456789abcdef" for ch in value.lower())
                       for value in source_hashes.values())):
            raise ValueError(f"M3 {job_id}: source SHA-256 inventory is incomplete")
        inventory = failure.get("no_checkpoint_inventory")
        if not isinstance(inventory, Mapping):
            raise ValueError(f"M3 {job_id}: checkpoint inventory evidence is missing")
        inventory_relative = str(inventory.get("path", ""))
        inventory_hash = str(inventory.get("sha256", ""))
        if not inventory_relative or len(inventory_hash) != 64:
            raise ValueError(f"M3 {job_id}: checkpoint inventory path/hash is missing")
        inventory_path = _relative(root, inventory_relative)
        if not inventory_path.is_file() or _sha256(inventory_path) != inventory_hash:
            raise ValueError(f"M3 {job_id}: checkpoint inventory hash does not match")
        log_relative = str(failure.get("log_path", ""))
        log_hash = str(failure.get("log_sha256", ""))
        if not log_relative or len(log_hash) != 64:
            raise ValueError(f"M3 {job_id}: final log path/hash is missing")
        log_path = _relative(root, log_relative)
        if not log_path.is_file() or _sha256(log_path) != log_hash:
            raise ValueError(f"M3 {job_id}: final log hash does not match failure record")
        verified.append({
            "job_id": job_id,
            "status": "UNVERIFIABLE",
            "evidence_path": relative,
            "evidence_sha256": expected_hash,
            "failure_reason": failure["failure_reason"],
            "original_tolerance": failure["original_tolerance"],
            "observed_error": failure.get("observed_error"),
            "source_sha256": dict(source_hashes),
            "checkpoint_inventory_path": inventory_relative,
            "checkpoint_inventory_sha256": inventory_hash,
            "log_path": log_relative,
            "log_sha256": log_hash,
        })
    return verified


def build_legacy_model_gate_decisions(
    root: Path,
    summary: dict[str, Any],
    coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return M1/M2/M2.5/M3 decision rows from immutable legacy sources.

    Missing repair estimates stay PENDING_FORMAL_BOOTSTRAP.  A present but
    incompatible or invalid estimate set becomes UNCERTAINTY_INSUFFICIENT.
    M3 is a special strict recovery case: it is marked UNVERIFIABLE only after
    all three required parent failure records and their final logs hash-check.
    """
    root = Path(root).resolve()
    protocol_rel = "results/v2_local_competition/protocol.json"
    m1_gate_rel = "results/v2_local_competition/gate.json"
    m2_gate_rel = "results/v2_local_competition/m2_curriculum/gate.json"
    m25_verdict_rel = "results/v2_local_competition/m25_specialist_audit/verdict.json"
    m3_amendment_rel = "results/v2_local_competition/m3_mixture/amendment_v2m31.json"

    protocol = _read_json(root, protocol_rel)
    m1_gate = _read_json(root, m1_gate_rel)
    m2_gate = _read_json(root, m2_gate_rel)
    m25_verdict = _read_json(root, m25_verdict_rel)
    out: list[dict[str, Any]] = []

    # M1 primary GO. K-generalization remains a separate diagnostic in the
    # source artifact and is not silently promoted into this GO predicate.
    m1_cfg = protocol["M1_gates"]
    m1_thresholds = m1_cfg["GO_gate"]
    m1_cell = str(m1_cfg.get("primary_cell", "SameCategory-K5"))
    m1_auroc = _ref(M1_JOB["primary"], "LCR_minus_E1b::auroc_correct")
    m1_eaurc = _ref(M1_JOB["primary"], f"LCR_E1b_relative_eaurc_reduction::{m1_cell}")
    m1_rer50 = _ref(M1_JOB["primary"], "LCR_minus_E1b::rer_at_50")
    m1_guard = _ref(M1_JOB["random_guard"], "LCR_minus_E1b::auroc_correct")
    m1_refs = (m1_auroc, m1_eaurc, m1_rer50, m1_guard)
    m1_source = _source_meta(
        root,
        ((m1_gate_rel, "/gates/M1_GO"), (protocol_rel, "/M1_gates/GO_gate")),
        (("scripts/run_v2m_m1.py", "_build_gate"),),
    )
    m1 = _decision_base(
        scope_id="m1", gate_id="M1_GO", source_meta=m1_source,
        old_thresholds={"primary_cell": m1_cell, **m1_thresholds},
        old_decision=m1_gate["gates"]["M1_GO"]["verdict"], refs=m1_refs,
        predicate_note=(
            "Frozen conjunction: primary AUROC delta >= configured minimum and CI low > 0; "
            "E-AURC relative reduction >= configured minimum OR RER50 gain (in pp) >= configured minimum; "
            "Random-K5 AUROC delta >= its guard floor. Repair estimates are accepted only when all planned "
            "5000 image-cluster draws are valid under seed 0 / percentile 95% and all three fixed B3 seeds qualify."
        ),
    )

    def eval_m1(index: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
        delta = _metric(index, m1_auroc, "point")
        ci_low = _metric(index, m1_auroc, "ci_low")
        eaurc_reduction = _metric(index, m1_eaurc, "point")
        rer50_gain_pp = _metric(index, m1_rer50, "point")
        random_delta = _metric(index, m1_guard, "point")
        primary_ok = delta >= float(m1_thresholds["delta_auroc_LCR_minus_E1b_ge"]) and ci_low > float(m1_thresholds["delta_auroc_ci_low_gt"])
        selective_ok = (
            eaurc_reduction >= float(m1_thresholds["at_least_one_of"]["e_aurc_reduction_ge"])
            or rer50_gain_pp >= float(m1_thresholds["at_least_one_of"]["rer50_gain_pp_ge"])
        )
        guard_ok = random_delta >= float(m1_thresholds["random_k5_guard"]["delta_auroc_LCR_minus_E1b_ge"])
        return {
            "decision": bool(primary_ok and selective_ok and guard_ok),
            "primary_cell": m1_cell,
            "delta_auroc": delta, "ci_low": ci_low,
            "primary_condition_pass": bool(primary_ok),
            "e_aurc_relative_reduction": eaurc_reduction,
            "rer50_gain_pp": rer50_gain_pp,
            "selective_condition_pass": bool(selective_ok),
            "random_k5_guard_delta": random_delta,
            "random_k5_guard_pass": bool(guard_ok),
        }

    out.append(_finish(m1, summary, m1_refs, coverage, "m1", eval_m1))

    # M2 at its pre-registered unseen m=8 severity cell. Selective and gate
    # contribution are separate registry decisions, never folded into M2_GO.
    m2_thresholds = protocol["M2_preregistration"]["gate"]
    m2_primary_ref = _ref(M2_JOB, "m8__LCR_minus_E1b::auroc_correct")
    m2_structure_ref = _ref(M2_JOB, "m8__LCR_minus_Aggregate-MLP::auroc_correct")
    m2_go_refs = (m2_primary_ref, m2_structure_ref)
    m2_source = _source_meta(
        root,
        ((m2_gate_rel, "/gates/M2_GO"), (protocol_rel, "/M2_preregistration/gate")),
        (("scripts/run_v2m_m2.py", "_build_gate"),),
    )
    m2_go = _decision_base(
        scope_id="m2", gate_id="M2_GO", source_meta=m2_source,
        old_thresholds={
            "vs_curriculum_E1b": dict(m2_thresholds["vs_curriculum_E1b"]),
            "vs_Aggregate_MLP": {**dict(m2_thresholds["vs_Aggregate_MLP"]), "delta_gt": 0.0},
            "primary_cell": "m8", "unseen_test_level": 8,
        },
        old_decision=m2_gate["gates"]["M2_GO"]["verdict"], refs=m2_go_refs,
        predicate_note=(
            "Frozen M2 conjunction on unseen m=8: LCR-E1b AUROC delta >= protocol minimum and CI low > 0; "
            "LCR-Aggregate-MLP AUROC delta > 0 and CI low > 0. Selective-risk diagnostics are separate."
        ),
    )

    def eval_m2_go(index: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
        d1 = _metric(index, m2_primary_ref, "point")
        d1_low = _metric(index, m2_primary_ref, "ci_low")
        d2 = _metric(index, m2_structure_ref, "point")
        d2_low = _metric(index, m2_structure_ref, "ci_low")
        c1 = d1 >= float(m2_thresholds["vs_curriculum_E1b"]["delta_auroc_ge"]) and d1_low > float(m2_thresholds["vs_curriculum_E1b"]["ci_low_gt"])
        c2 = d2 > 0.0 and d2_low > float(m2_thresholds["vs_Aggregate_MLP"]["ci_low_gt"])
        go = bool(c1 and c2)
        return {
            "decision": go,
            "delta_lcr_minus_e1b": d1, "delta_lcr_minus_e1b_ci_low": d1_low,
            "condition_lcr_minus_e1b": bool(c1),
            "delta_lcr_minus_aggregate_mlp": d2,
            "delta_lcr_minus_aggregate_mlp_ci_low": d2_low,
            "condition_lcr_minus_aggregate_mlp": bool(c2),
            "capacity_case": "A" if go else ("B" if c1 else "C"),
        }

    out.append(_finish(m2_go, summary, m2_go_refs, coverage, "m2", eval_m2_go))

    # The exact M2.2 diagnostic is the sign of relative E-AURC reduction, not
    # the sign of an absolute difference. A supplies this as a derived
    # auxiliary estimate from existing shared raw draws; absence remains pending.
    m2_sel_auroc = m2_primary_ref
    m2_sel_eaurc = _ref(M2_JOB, M2_RELATIVE_EAURC)
    m2_sel_rer50 = _ref(M2_JOB, "m8__LCR_minus_E1b::rer_at_50")
    m2_sel_rer80 = _ref(M2_JOB, "m8__LCR_minus_E1b::rer_at_80")
    m2_selective_refs = (m2_sel_auroc, m2_sel_eaurc, m2_sel_rer50, m2_sel_rer80)
    m2_selective_source = _source_meta(
        root,
        ((m2_gate_rel, "/gates/selective_metrics"), (protocol_rel, "/M2_preregistration/gate")),
        (("scripts/run_v2m_m2.py", "_derived; _build_gate"),),
    )
    m2_selective_thresholds = {
        "numeric_thresholds": "NONE; diagnostic sign predicate only",
        "auroc_condition": {
            "delta_auroc_ge": float(m2_thresholds["vs_curriculum_E1b"]["delta_auroc_ge"]),
            "ci_low_gt": float(m2_thresholds["vs_curriculum_E1b"]["ci_low_gt"]),
        },
        "e_aurc_reduction_lt": 0.0,
        "rer50_gain_pp_lt": 0.0,
        "rer80_gain_pp_lt": 0.0,
    }
    m2_selective = _decision_base(
        scope_id="m2", gate_id="M2_SELECTIVE_SIGNAL", source_meta=m2_selective_source,
        old_thresholds=m2_selective_thresholds,
        old_decision=m2_gate["gates"]["selective_metrics"]["method_signal_mixed"],
        refs=m2_selective_refs,
        predicate_note=(
            "Diagnostic only: mixed signal is true iff the LCR-E1b AUROC condition passes and all of "
            "relative E-AURC reduction, RER50 gain, and RER80 gain are negative. The relative E-AURC effect "
            "is recomputed from shared stored raw draws; E1b E-AURC <=1e-12 or nonfinite endpoints "
            "invalidate that draw."
        ),
    )

    def eval_m2_selective(index: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
        auroc_delta = _metric(index, m2_sel_auroc, "point")
        auroc_ci_low = _metric(index, m2_sel_auroc, "ci_low")
        relative_reduction = _metric(index, m2_sel_eaurc, "point")
        rer50 = _metric(index, m2_sel_rer50, "point")
        rer80 = _metric(index, m2_sel_rer80, "point")
        auroc_pass = (
            auroc_delta >= float(m2_selective_thresholds["auroc_condition"]["delta_auroc_ge"])
            and auroc_ci_low > float(m2_selective_thresholds["auroc_condition"]["ci_low_gt"])
        )
        mixed = bool(auroc_pass and relative_reduction < 0.0 and rer50 < 0.0 and rer80 < 0.0)
        return {
            "decision": mixed,
            "auroc_condition_pass": bool(auroc_pass),
            "e_aurc_relative_reduction": relative_reduction,
            "rer50_gain_pp": rer50,
            "rer80_gain_pp": rer80,
            "method_signal_mixed": mixed,
        }

    out.append(_finish(m2_selective, summary, m2_selective_refs, coverage, "m2", eval_m2_selective))

    m2_nogate = _ref(M2_JOB, "m8__LCR_minus_LCR-noGate::auroc_correct")
    m2_contrib_refs = (m2_nogate,)
    m2_contrib_source = _source_meta(
        root,
        ((m2_gate_rel, "/gates/gate_contribution"), (protocol_rel, "/M2_preregistration/gate")),
        (("scripts/run_v2m_m2.py", "_build_gate"),),
    )
    contrib_thresholds = {
        "supported": {"delta_gt": 0.0, "ci_low_gt": 0.0},
        "unsupported": {"delta_lt": 0.0, "ci_high_lt": 0.0},
        "ci_includes_zero": True,
    }
    old_contrib = m2_gate["gates"]["gate_contribution"]
    old_contrib_decision = (
        "SUPPORTED" if old_contrib["supported"] else
        "UNSUPPORTED" if old_contrib["unsupported"] else
        "CI_INCLUDES_ZERO" if old_contrib["effectively_equal"] else "INCONCLUSIVE"
    )
    m2_contrib = _decision_base(
        scope_id="m2", gate_id="M2_GATE_CONTRIBUTION", source_meta=m2_contrib_source,
        old_thresholds=contrib_thresholds, old_decision=old_contrib_decision,
        refs=m2_contrib_refs,
        predicate_note=(
            "At m=8, LCR gate contribution is supported only when LCR-LCR-noGate delta > 0 and CI low > 0; "
            "it is unsupported when delta < 0 and CI high < 0; a CI spanning zero is reported as CI_INCLUDES_ZERO, "
            "not as proof of equivalence."
        ),
    )

    def eval_m2_contrib(index: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
        delta = _metric(index, m2_nogate, "point")
        low, high = _metric(index, m2_nogate, "ci_low"), _metric(index, m2_nogate, "ci_high")
        supported = delta > 0.0 and low > 0.0
        unsupported = delta < 0.0 and high < 0.0
        includes_zero = low <= 0.0 <= high
        decision = "SUPPORTED" if supported else "UNSUPPORTED" if unsupported else "CI_INCLUDES_ZERO" if includes_zero else "INCONCLUSIVE"
        return {"decision": decision,
            "delta_lcr_minus_nogate": delta, "ci_low": low, "ci_high": high,
            "supported": bool(supported), "unsupported": bool(unsupported),
            "ci_includes_zero": bool(includes_zero),
        }

    out.append(_finish(m2_contrib, summary, m2_contrib_refs, coverage, "m2", eval_m2_contrib))

    # M2.5 is a frozen diagnostic classifier; all four levels are required to
    # distinguish the two uniform patterns from the specialist trade-off.
    m25_refs = tuple(
        _ref(M25_JOB, f"m{level}__E1b_curriculum_minus_random::auroc_correct")
        for level in (0, 2, 4, 8)
    )
    m25_source = _source_meta(
        root,
        ((m25_verdict_rel, "/delta_definition"), (m25_verdict_rel, "/pattern"),
         (m25_verdict_rel, "/conditions")),
        (("scripts/run_v2m_m25.py", "_pattern"),),
    )
    m25_thresholds = {
        "low_levels": [0, 2], "low_ci_high_lt": 0.0,
        "high_level": 8, "high_ci_low_gt": 0.0,
        "all_levels": [0, 2, 4, 8],
    }
    m25 = _decision_base(
        scope_id="m25", gate_id="M25_DIAGNOSTIC_VERDICT", source_meta=m25_source,
        old_thresholds=m25_thresholds, old_decision=m25_verdict["pattern"], refs=m25_refs,
        predicate_note=(
            "Diagnostic classifier, not a success gate. delta(m)=AUROC(E1b_curriculum)-AUROC(E1b_random); "
            "low specialization is any m0/m2 CI high < 0, high specialization is m8 CI low > 0. "
            "The uniform curriculum/random alternatives require all four m={0,2,4,8} confidence intervals "
            "to lie strictly on the corresponding side of zero."
        ),
    )

    def eval_m25(index: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
        limits = {
            level: (_metric(index, ref, "ci_low"), _metric(index, ref, "ci_high"))
            for level, ref in zip((0, 2, 4, 8), m25_refs)
        }
        low_sig = any(limits[level][1] < 0.0 for level in (0, 2))
        high_sig = limits[8][0] > 0.0
        all_curr = all(limits[level][0] > 0.0 for level in (0, 2, 4, 8))
        all_rand = all(limits[level][1] < 0.0 for level in (0, 2, 4, 8))
        pattern = (
            "SPECIALIST TRADEOFF PRESENT" if low_sig and high_sig else
            "CURRICULUM DOMINATES" if all_curr else
            "CURRICULUM NOT USEFUL" if all_rand else "INCONCLUSIVE"
        )
        return {"decision": pattern,
            "pattern": pattern,
            "conditions": {
                "low_m_random_better_significant": bool(low_sig),
                "m8_curriculum_better_significant": bool(high_sig),
                "all_levels_curriculum_better": bool(all_curr),
                "all_levels_random_better": bool(all_rand),
            },
            "ci_by_level": {f"m{level}": [low, high] for level, (low, high) in limits.items()},
            "confirmatory_gate": False,
        }

    out.append(_finish(m25, summary, m25_refs, coverage, "m25", eval_m25))

    # M3 B0 has no gate. The actual confirmatory B1/B2 cross-backbone decision
    # cannot be recomputed because all three legacy recovery jobs, including
    # B0 provenance needed by the frozen M3 bundle, have strict evidence-backed
    # failures. Preserve the old decision and bind all parent records/logs.
    m3_b0_rel = "results/v2_local_competition/m3_mixture/b0/summary.json"
    m3_amend_rel = "results/v2_local_competition/m3_mixture/amendment_v2m31.json"
    m3_b1_rel = "results/v2_local_competition/m3_mixture/conf/b1/gate.json"
    m3_b2_rel = "results/v2_local_competition/m3_mixture/conf/b2/gate.json"
    m3_cross_rel = "results/v2_local_competition/m3_mixture/conf/cross_backbone_verdict.json"
    m3_protocol_rel = "results/v2_local_competition/m3_mixture/protocol_m3.json"
    m3_b0 = _read_json(root, m3_b0_rel)
    amendment = _read_json(root, m3_amend_rel)
    cross = _read_json(root, m3_cross_rel)
    # Read these explicit original per-backbone decisions before freezing their
    # hashes into the provenance record.
    _read_json(root, m3_b1_rel)
    _read_json(root, m3_b2_rel)
    failure_evidence = _verified_m3_failures(root, coverage)
    m3_source = _source_meta(
        root,
        ((m3_b0_rel, "/no_gate"), (m3_amend_rel, "/frozen_gates"),
         (m3_amend_rel, "/frozen_verdict_rule"), (m3_b1_rel, "/pass"),
         (m3_b2_rel, "/pass"), (m3_cross_rel, "/verdict"),
         (m3_protocol_rel, "/confirmatory_gates")),
        (("scripts/run_v2m_m3_b0.py", "_summary (B0 no-gate record)"),
         ("scripts/run_v2m_m3_conf.py", "_backbone_gate; _cross_backbone_verdict")),
    )
    m3 = _decision_base(
        scope_id="m3", gate_id="M3_B1_B2_FROZEN_GATE", source_meta=m3_source,
        old_thresholds={
            "B0": {"no_gate": bool(m3_b0["no_gate"]), "label": m3_b0["label"]},
            "B1_B2_frozen_gates": amendment["frozen_gates"],
            "cross_backbone_verdict_rule": amendment["frozen_verdict_rule"],
        },
        old_decision=cross["verdict"], refs=(),
        predicate_note=(
            "B0 is POST-HOC DEVELOPMENTAL and no_gate=true. The registered gate is the frozen B1/B2 confirmatory "
            "cross-backbone predicate. It is UNVERIFIABLE only with strict, hash-verified parent recovery failures "
            "for m3_b0_severity_macro, m3_b1_severity_macro, and m3_b2_severity_macro, including final log hashes; "
            "the historical cross-backbone verdict and thresholds are retained."
        ),
    )
    m3.update(
        status="UNVERIFIABLE",
        new_decision="UNVERIFIABLE",
        unverifiable_evidence=failure_evidence,
        new_result={
            "historical_cross_backbone_verdict": cross["verdict"],
            "historical_n_pass": cross["n_pass"],
            "b0_no_gate": bool(m3_b0["no_gate"]),
            "recovery_jobs": [item["job_id"] for item in failure_evidence],
        },
    )
    out.append(m3)
    return out

