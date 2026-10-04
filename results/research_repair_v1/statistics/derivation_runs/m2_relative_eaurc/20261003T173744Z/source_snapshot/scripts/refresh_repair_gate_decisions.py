"""Re-evaluate frozen gate predicates over Research Repair v1 estimates.

The legacy threshold and decision remain alongside the repair decision.  The
repair decision consumes only the named estimates listed in ``new_estimates``;
ineligible/invalid bootstrap estimates cannot enter a threshold predicate.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "research_repair_v1" / "statistics"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _read(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _estimate_index(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for job in summary.get("jobs", []):
        for item in job.get("estimates", []):
            indexed = dict(item)
            indexed["_job_config"] = dict(job.get("config", {}))
            indexed["_job_result"] = dict(job.get("result", {}))
            indexed["_job_n_seeds"] = int(job.get("n_seeds", 0))
            seed_names = list((job.get("model_identity_by_seed") or {}).keys())
            if not seed_names and int(job.get("n_seeds", 0)) == 3:
                seed_names = ["b3_seed1", "b3_seed2", "b3_seed3"]
            elif not seed_names and int(job.get("n_seeds", 0)) == 1 and str(job.get("scope", "")).startswith("Phase0"):
                seed_names = ["B0"]
            indexed["_job_model_seed_names"] = seed_names
            indexed["_job_id"] = str(job.get("job_id", ""))
            indexed["_job_status"] = str(job.get("status", ""))
            index[(str(job["job_id"]), str(item["name"]))] = indexed
    return index


def _refs(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"job_id": job, "estimate_name": name} for job, name in pairs]


def _get(index: dict[tuple[str, str], dict[str, Any]], ref: dict[str, str]) -> dict[str, Any] | None:
    return index.get((ref["job_id"], ref["estimate_name"]))


def _eligibility(index: dict[tuple[str, str], dict[str, Any]], refs: list[dict[str, str]]) -> str:
    missing = [ref for ref in refs if _get(index, ref) is None]
    if missing:
        return "PENDING_FORMAL_BOOTSTRAP"
    reasons = _eligibility_reasons(index, refs)
    return "UNCERTAINTY_INSUFFICIENT" if reasons else "ELIGIBLE"


def _eligibility_reasons(index: dict[tuple[str, str], dict[str, Any]], refs: list[dict[str, str]]) -> list[str]:
    reasons: list[str] = []
    expected_config = {"n_replicates": 5000, "seed": 0, "ci_level": 0.95, "ci_method": "percentile"}
    for ref in refs:
        item = _get(index, ref)
        if item is None:
            reasons.append(f"missing estimate {ref['job_id']}::{ref['estimate_name']}")
            continue
        config, result = item.get("_job_config", {}), item.get("_job_result", {})
        if item.get("_job_status") != "RECOMPUTED_5000":
            reasons.append(f"{ref['job_id']} status={item.get('_job_status')!r}, expected 'RECOMPUTED_5000'")
        for field, expected in expected_config.items():
            actual = config.get(field)
            if actual != expected:
                reasons.append(f"{ref['job_id']} config.{field}={actual!r}, expected {expected!r}")
        if result.get("resample_unit") != "image_cluster":
            reasons.append(f"{ref['job_id']} result.resample_unit={result.get('resample_unit')!r}, expected 'image_cluster'")
        if result.get("method") != "percentile":
            reasons.append(f"{ref['job_id']} result.method={result.get('method')!r}, expected 'percentile'")
        if int(item.get("n_replicates", -1)) != 5000 or int(item.get("valid_replicates", -1)) != 5000:
            reasons.append(f"{ref['job_id']}::{ref['estimate_name']} does not have 5000/5000 valid draws")
        if int(item.get("invalid_replicates", -1)) != 0:
            reasons.append(f"{ref['job_id']}::{ref['estimate_name']} has invalid draws")
        if item.get("resample_unit") != "image_cluster" or float(item.get("ci_level", -1.0)) != 0.95:
            reasons.append(f"{ref['job_id']}::{ref['estimate_name']} CI method metadata is incompatible")
        if item.get("gate_eligible") is not True:
            reasons.append(f"{ref['job_id']}::{ref['estimate_name']} is not gate_eligible")
        per_seed = item.get("per_seed", {})
        expected_seeds = int(item.get("_job_n_seeds", 0))
        expected_seed_names = set(item.get("_job_model_seed_names", []))
        if (not per_seed or (expected_seeds and len(per_seed) != expected_seeds)
                or (expected_seed_names and set(per_seed) != expected_seed_names)):
            reasons.append(f"{ref['job_id']}::{ref['estimate_name']} fixed seed set is incomplete")
        if any(int(v.get("valid_replicates", -1)) != 5000 or int(v.get("invalid_replicates", -1)) != 0
               or int(v.get("n_replicates", -1)) != 5000
               or v.get("gate_eligible") is not True for v in per_seed.values()):
            reasons.append(f"{ref['job_id']}::{ref['estimate_name']} has an ineligible per-seed interval")
    return reasons


def _number(index: dict[tuple[str, str], dict[str, Any]], job: str, name: str, field: str = "point") -> float:
    item = index[(job, name)]
    return float(item[field])


def _ref(index: dict[tuple[str, str], dict[str, Any]], job: str, name: str) -> dict[str, Any]:
    return index[(job, name)]


def _phase05(index: dict[tuple[str, str], dict[str, Any]], old: dict[str, Any]) -> dict[str, Any]:
    job_ids = {"testA": "phase05_testA", "testB": "phase05_testB"}
    refs = []
    for job in job_ids.values():
        for k in (20, 50):
            refs.extend(_refs(
                (job, f"crossK__msp__K{k}__auroc_drop"),
                (job, f"crossK__msp__K{k}__e_aurc_worsening_relative"),
                (job, f"crossK__msp__K{k}__rer50_drop"),
            ))
    pooled_job = "phase05___pooled_test__"
    refs.extend(_refs(
        (pooled_job, "msp__K50::auroc_correct"),
        (pooled_job, "msp__K50::e_aurc"),
        (pooled_job, "msp__K50::rer_at_50"),
        (pooled_job, "crossK__msp__K50__auroc_drop"),
        (pooled_job, "crossK__msp__K50__e_aurc_worsening_relative"),
        (pooled_job, "crossK__msp__K50__rer50_drop"),
    ))
    meta = {
        "scope_id": "phase05", "gate_id": "PHASE05_SUFFICIENCY_GATE",
        "source": "results/phase05_score_sufficiency/sufficiency_gate.json#/verdict_seed_mean",
        "predicate_source": "src/ccg/reliability/evaluate.py::sufficiency_verdict (Amendment A6.6); scripts/run_phase05.py::_gate",
        "old_thresholds": old["verdict_seed_mean"]["thresholds"],
        "old_decision": old["verdict_seed_mean"]["verdict"], "new_estimates": refs,
        "threshold_change": "NONE",
        "predicate_note": "Replay the frozen seed-mean predicate over testA/testB K20/K50 cells, then apply the original K50 pooled NO_GO and override point rules. Although Amendment A6.6 and the note discuss >=2/3 seed consistency, scripts/run_phase05.py::_gate does not use its seed_consistency field in the verdict; that historical implementation detail is preserved here.",
        "document_vs_runner": "The protocol/note calls for >=2/3 seed consistency; the frozen runner records that diagnostic but its headline verdict consumes only verdict_seed_mean. Repair reproduces the runner and exposes the discrepancy without adding a new seed-consistency condition.",
        "old_new_aggregation": "old gate uses the frozen MSP family and seed-mean table; repair uses the same MSP points with per-seed effects computed inside shared image draws then seed mean.",
    }
    state = _eligibility(index, refs)
    if state != "ELIGIBLE":
        meta.update(status=state, new_decision=state, eligibility_reasons=_eligibility_reasons(index, refs))
        return meta
    thresholds = meta["old_thresholds"]
    route_a, route_b = [], []
    for split, job in job_ids.items():
        for k in (20, 50):
            au = _ref(index, job, f"crossK__msp__K{k}__auroc_drop")
            ea = _ref(index, job, f"crossK__msp__K{k}__e_aurc_worsening_relative")
            rr = _ref(index, job, f"crossK__msp__K{k}__rer50_drop")
            if (float(ea["point"]) >= thresholds["e_aurc_worsen"]
                    and 100.0 * float(rr["point"]) >= thresholds["rer_drop_pp"]
                    and float(ea["ci_low"]) > 0.0 and float(rr["ci_low"]) > 0.0):
                route_a.append(f"{split}/K{k}")
            if (float(au["point"]) >= thresholds["auroc_drop"] and float(au["ci_low"]) > 0.0):
                route_b.append(f"{split}/K{k}")
    n_failing = len(set(route_a) | set(route_b))
    go = n_failing >= 2
    pooled_auc_drop = _number(index, pooled_job, "crossK__msp__K50__auroc_drop")
    pooled_eaurc_rel = _number(index, pooled_job, "crossK__msp__K50__e_aurc_worsening_relative")
    pooled_eaurc_abs = _number(index, pooled_job, "msp__K50::e_aurc")
    pooled_rer_drop_pp = 100.0 * _number(index, pooled_job, "crossK__msp__K50__rer50_drop")
    no_go = (
        abs(pooled_auc_drop) < thresholds["auroc_stable"]
        and (pooled_eaurc_rel < thresholds["e_aurc_worsen"] or pooled_eaurc_abs < thresholds["e_aurc_abs"])
        and pooled_rer_drop_pp < thresholds["rer_drop_pp"]
    )
    override = (
        _number(index, pooled_job, "msp__K50::auroc_correct") >= thresholds["override_auroc"]
        and pooled_eaurc_abs <= thresholds["override_e_aurc"]
        and _number(index, pooled_job, "msp__K50::rer_at_50") >= thresholds["override_rer"]
    )
    decision = (
        "practically_solved_by_score_only" if override else
        "GO_candidate_embeddings" if go else
        "NO_GO_candidate_embeddings" if no_go else "inconclusive"
    )
    meta.update(
        status="EVALUATED", new_decision=decision,
        new_result={"route_A_cells": route_a, "route_B_cells": route_b,
                    "n_failing_ood_cells": n_failing, "override": override, "no_go_sufficient": no_go},
    )
    return meta


def _phase1(index: dict[tuple[str, str], dict[str, Any]], old: dict[str, Any]) -> dict[str, Any]:
    from ccg.semantic.evaluate import semantic_gate

    job = "phase1_fixed___pooled_test__"
    refs = []
    for k in (20, 50):
        refs.extend(_refs(
            (job, f"gate_delta_auroc_e1b_vs_msp__K{k}"),
            (job, f"gate_eaurc_relative_reduction_e1b_vs_msp__K{k}"),
            (job, f"gate_rer50_gain_e1b_vs_msp_pp__K{k}"),
        ))
    meta = {
        "scope_id": "phase1", "gate_id": "PHASE1_GLOBAL_SUFFICIENCY_GATE",
        "source": "results/phase1_semantic_sufficiency/sufficiency_gate.json#/verdict",
        "predicate_source": "src/ccg/semantic/evaluate.py::semantic_gate; scripts/run_phase1.py::_gate_stage",
        "old_thresholds": old["thresholds"], "old_decision": old["verdict"]["verdict"],
        "new_estimates": refs, "threshold_change": "NONE",
        "predicate_note": "K20 and K50 must each pass the A7.6 AUROC floor and positive lower bound plus at least one selective branch; at least 2/3 per-seed K50 AUROC CI lower bounds must be positive and mean K50 delta positive.",
        "old_new_aggregation": "old A7.6 selection is fixed MSP versus fixed E1b; selected generic CSV semantic output is operational E2_SCORE and is excluded from this gate.",
    }
    state = _eligibility(index, refs)
    if state != "ELIGIBLE":
        meta.update(status=state, new_decision=state, eligibility_reasons=_eligibility_reasons(index, refs))
        return meta
    cells = {}
    for k in (20, 50):
        delta = _ref(index, job, f"gate_delta_auroc_e1b_vs_msp__K{k}")
        red = _ref(index, job, f"gate_eaurc_relative_reduction_e1b_vs_msp__K{k}")
        rer = _ref(index, job, f"gate_rer50_gain_e1b_vs_msp_pp__K{k}")
        cells[k] = {
            "delta_auroc": float(delta["point"]), "delta_auroc_ci_low": float(delta["ci_low"]),
            "e_aurc_reduction": float(red["point"]), "e_aurc_reduction_ci_low": float(red["ci_low"]),
            "rer50_gain_pp": float(rer["point"]), "rer50_gain_pp_ci_low": float(rer["ci_low"]),
        }
    k50 = _ref(index, job, "gate_delta_auroc_e1b_vs_msp__K50")
    seed_lows = [float(k50["per_seed"][seed]["ci_low"]) for seed in sorted(k50["per_seed"])]
    result = semantic_gate(cells, seed_lows, seed_mean_delta=float(k50["point"]), thresholds=old["thresholds"])
    meta.update(status="EVALUATED", new_decision=result["verdict"], new_result=result)
    return meta


def _phase1f(index: dict[tuple[str, str], dict[str, Any]], old: dict[str, Any]) -> dict[str, Any]:
    from ccg.semantic.hard_eval import hard_gate

    job = "phase1f_k5_same4"
    delta_name = "semantic_minus_stats__hard5::auroc_correct"
    refs = _refs(
        (job, delta_name), (job, "e_aurc_reduction__hard5"),
        (job, "semantic_minus_stats__hard5::rer_at_50"),
        (job, "gate_dod_samecat5::auroc_correct"),
    )
    meta = {
        "scope_id": "phase1f", "gate_id": "PHASE1F_SEMANTIC_GATE",
        "source": "results/phase1f_hard_semantic/gate.json#/verdict",
        "predicate_source": "src/ccg/semantic/hard_eval.py::hard_gate; scripts/run_phase1f.py::_gate_payload",
        "old_thresholds": old["verdict"]["thresholds"], "old_decision": old["verdict"]["verdict"],
        "new_estimates": refs, "threshold_change": "NONE",
        "predicate_note": "A8.6 SameCategory-K5 dAUROC/CI, E-AURC reduction or RER50 gain branch, at least 2/3 positive per-seed K5 AUROC CI lows, positive mean delta, and hard-minus-random dAUROC >= frozen minimum; strong rule adds frozen strong thresholds.",
        "old_new_aggregation": "repair computes per-seed hard effects and hard-minus-random difference-of-differences within each shared image draw before seed averaging.",
    }
    state = _eligibility(index, refs)
    if state != "ELIGIBLE":
        meta.update(status=state, new_decision=state, eligibility_reasons=_eligibility_reasons(index, refs))
        return meta
    delta = _ref(index, job, delta_name)
    reduction = _ref(index, job, "e_aurc_reduction__hard5")
    rer = _ref(index, job, "semantic_minus_stats__hard5::rer_at_50")
    dod = _ref(index, job, "gate_dod_samecat5::auroc_correct")
    seed_lows = [float(delta["per_seed"][seed]["ci_low"]) for seed in sorted(delta["per_seed"])]
    result = hard_gate(
        {"delta_auroc": delta["point"], "delta_auroc_ci_low": delta["ci_low"],
         "e_aurc_reduction": reduction["point"], "e_aurc_reduction_ci_low": reduction["ci_low"],
         "rer50_gain_pp": rer["point"], "rer50_gain_pp_ci_low": rer["ci_low"]},
        {"diff": dod["point"]}, seed_delta_ci_lows=seed_lows,
        seed_mean_delta=delta["point"], thresholds=old["verdict"]["thresholds"],
    )
    meta.update(status="EVALUATED", new_decision=result["verdict"], new_result=result)
    return meta


def _refcocog(index: dict[tuple[str, str], dict[str, Any]], old: dict[str, Any]) -> dict[str, Any]:
    result = _read("results/phase1e_refcocog_external/a11_results.json")
    job = "refcocog_external_rand5_hard5"
    name_map = {
        "delta": "semantic_increment_hard5::auroc_correct",
        "reduction": "e_aurc_reduction__hard5",
        "rer": "semantic_increment_hard5::rer50_gain_pp",
        "amp_delta": "hard_minus_random_increment::auroc_correct",
        "amp_reduction": "hard_minus_random_increment::e_aurc_reduction",
        "amp_rer": "hard_minus_random_increment::rer50_gain_pp",
    }
    refs = _refs(*[(job, name) for name in name_map.values()])
    meta = {
        "scope_id": "refcocog", "gate_id": "REFCOCOG_EXTERNAL_GATE",
        "source": "results/phase1e_refcocog_external/a11_results.json#/full_verdict",
        "predicate_source": "scripts/a11_external_analysis.py::_gate(transfer=True/False), _full_verdict",
        "old_thresholds": result["thresholds"], "old_decision": result["full_verdict"]["full_verdict"],
        "new_estimates": refs, "threshold_change": "NONE",
        "predicate_note": "A11 Q1/Q2 thresholds and full-verdict case logic are retained; the frozen hard-manipulation validity check is also required and remains false in the historical artifact.",
        "old_new_aggregation": "Q1/Q2 seed-specific paired increments are computed within shared image draws and then averaged; the existing manipulation-validity finding is held fixed.",
    }
    state = _eligibility(index, refs)
    if state != "ELIGIBLE":
        meta.update(status=state, new_decision=state, eligibility_reasons=_eligibility_reasons(index, refs))
        return meta
    points = {key: _ref(index, job, name) for key, name in name_map.items()}
    manipulation_valid = bool(result.get("manipulation", {}).get("valid", False))
    seed_names = sorted(points["delta"].get("per_seed", {}))
    q1_mean = float(points["delta"]["point"])
    q1_ci_positive = sum(float(points["delta"]["per_seed"][s]["ci_low"]) > 0 for s in seed_names)
    q1_delta_positive = sum(float(points["delta"]["per_seed"][s]["point"]) > 0 for s in seed_names)
    q1_branch_e = (
        float(points["reduction"]["point"]) >= result["thresholds"]["q1_e_aurc_reduction"]
        and sum(float(points["reduction"]["per_seed"][s]["ci_low"]) > 0 for s in seed_names)
        >= result["thresholds"]["seed_sig_min"]
    )
    q1_branch_r = (
        float(points["rer"]["point"]) >= result["thresholds"]["q1_rer50_gain_pp"]
        and sum(float(points["rer"]["per_seed"][s]["ci_low"]) > 0 for s in seed_names)
        >= result["thresholds"]["seed_sig_min"]
    )
    q1_core = (
        q1_mean >= result["thresholds"]["q1_delta_auroc"]
        and q1_ci_positive >= result["thresholds"]["seed_sig_min"]
        and q1_delta_positive >= result["thresholds"]["seed_sig_min"]
        and (q1_branch_e or q1_branch_r)
    )
    if q1_core:
        q1_decision = "YES"
    elif q1_mean <= 0:
        q1_decision = "NO"
    else:
        q1_decision = "GRAY"
    # A11's external hard manipulation did not pass.  As in the original
    # analysis, this blocks Q2 and therefore prevents a full replication claim.
    if manipulation_valid:
        amp_mean = float(points["amp_delta"]["point"])
        amp_ci_positive = sum(float(points["amp_delta"]["per_seed"][s]["ci_low"]) > 0 for s in seed_names)
        amp_selective = (
            float(points["amp_reduction"]["point"]) > 0
            and sum(float(points["amp_reduction"]["per_seed"][s]["ci_low"]) > 0 for s in seed_names)
            >= result["thresholds"]["seed_sig_min"]
        ) or (
            float(points["amp_rer"]["point"]) > 0
            and sum(float(points["amp_rer"]["per_seed"][s]["ci_low"]) > 0 for s in seed_names)
            >= result["thresholds"]["seed_sig_min"]
        )
        q2_decision = (
            "YES" if amp_mean >= result["thresholds"]["q2_amplification_auroc"]
            and amp_ci_positive >= result["thresholds"]["seed_sig_min"] and amp_selective
            else ("NO" if amp_mean <= 0 else "GRAY")
        )
    else:
        q2_decision = "INVALID_MANIPULATION"
    severe = bool(result.get("severe_external_failure", False))
    if severe:
        decision = "EXTERNAL INCONCLUSIVE"
    elif q1_decision == "YES" and q2_decision == "YES":
        decision = "FULL EXTERNAL CONFIRMATION"
    elif q1_decision == "YES" and q2_decision == "NO":
        decision = "PARTIAL EXTERNAL CONFIRMATION"
    elif q1_decision == "NO" and q2_decision == "NO":
        decision = "EXTERNAL NOT CONFIRMED"
    else:
        decision = "EXTERNAL INCONCLUSIVE"
    meta.update(status="EVALUATED", new_decision=decision,
                new_result={"q1_decision": q1_decision, "q2_decision": q2_decision,
                            "manipulation_valid": manipulation_valid, "case_blocked_by_manipulation": not manipulation_valid})
    return meta


def _cardinality_gate_result(
    index: dict[tuple[str, str], dict[str, Any]],
    job_ids: list[str],
    *,
    auc_name: str,
    eaurc_name: str,
    rer_name: str,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    refs = [ref for job in job_ids for ref in _refs((job, auc_name), (job, eaurc_name), (job, rer_name))]
    state = _eligibility(index, refs)
    if state != "ELIGIBLE":
        return ({"status": state, "reason": _eligibility_reasons(index, refs)}, refs, state)
    rows: dict[str, Any] = {}
    all_replicated = True
    for job in job_ids:
        auc = _ref(index, job, auc_name)
        eaurc = _ref(index, job, eaurc_name)
        rer = _ref(index, job, rer_name)
        auc_seeds = list(auc["per_seed"].values())
        eaurc_seeds = list(eaurc["per_seed"].values())
        rer_seeds = list(rer["per_seed"].values())
        route_a_ci = all(float(value["ci_low"]) > 0.0 for value in auc_seeds)
        route_b_eaurc_ci = all(float(value["ci_low"]) > 0.0 or float(value["ci_high"]) < 0.0
                               for value in eaurc_seeds)
        route_b_rer_ci = all(float(value["ci_low"]) > 0.0 for value in rer_seeds)
        rer_point = float(rer["point"])
        # The frozen thresholds are proportions (e.g. .10 means a 10pp RER
        # change). Proposal-table repair estimates are stored in pp, so convert
        # only that explicit scale before comparing with the unchanged cutoff.
        rer_fraction = rer_point / 100.0 if float(rer.get("scale", 1.0)) == 100.0 else rer_point
        route_a = float(auc["point"]) >= float(thresholds["route_a_auc_drop_min"]) and route_a_ci
        route_b = (
            float(eaurc["point"]) >= float(thresholds["route_b_eaurc_worsen_min"])
            and rer_fraction >= float(thresholds["route_b_rer50_drop_min"])
            and route_b_eaurc_ci and route_b_rer_ci
        )
        replicated = bool(route_a or route_b)
        all_replicated = all_replicated and replicated
        rows[job] = {
            "route_a": route_a, "route_b": route_b, "replicated": replicated,
            "auc_drop_mean": float(auc["point"]),
            "auc_per_seed_ci_low": {seed: float(value["ci_low"]) for seed, value in auc["per_seed"].items()},
            "relative_eaurc_worsening_mean": float(eaurc["point"]),
            "eaurc_ci_excludes_zero_by_seed": {seed: bool(float(value["ci_low"]) > 0.0 or float(value["ci_high"]) < 0.0)
                                                 for seed, value in eaurc["per_seed"].items()},
            "rer50_drop_fraction_mean": rer_fraction,
            "rer50_ci_low_by_seed": {seed: float(value["ci_low"]) for seed, value in rer["per_seed"].items()},
        }
    return ({"backbones_or_families": rows, "replicated_all": all_replicated}, refs, "EVALUATED")


def _hard_gate_result(
    index: dict[tuple[str, str], dict[str, Any]],
    job_ids: list[str],
    *,
    manipulation_by_job: dict[str, bool],
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    names = ("delta_hard_auroc", "amplification_auroc")
    refs = [ref for job in job_ids for ref in _refs(*[(job, name) for name in names])]
    state = _eligibility(index, refs)
    if state != "ELIGIBLE":
        return ({"status": state, "reason": _eligibility_reasons(index, refs)}, refs, state)
    output: dict[str, Any] = {}
    all_passed = True
    for job in job_ids:
        delta = _ref(index, job, names[0])
        amp = _ref(index, job, names[1])
        delta_low = sum(float(v["ci_low"]) for v in delta["per_seed"].values()) / len(delta["per_seed"])
        amp_low = sum(float(v["ci_low"]) for v in amp["per_seed"].values()) / len(amp["per_seed"])
        pass_delta = float(delta["point"]) >= float(thresholds["delta_hard_min"]) and delta_low > 0.0
        pass_amp = float(amp["point"]) >= float(thresholds["amplification_min"]) and amp_low > 0.0
        manipulation_ok = bool(manipulation_by_job[job])
        replicated = manipulation_ok and pass_delta and pass_amp
        all_passed = all_passed and replicated
        output[job] = {
            "delta_hard_mean": float(delta["point"]), "delta_hard_ci_low_mean": delta_low,
            "amplification_mean": float(amp["point"]), "amplification_ci_low_mean": amp_low,
            "pass_delta_hard": pass_delta, "pass_amplification": pass_amp,
            "manipulation_ok_from_frozen_gate": manipulation_ok,
            "verdict": "HARD_SEMANTIC_REPLICATED" if replicated else
                ("NOT_ASSESSABLE" if not manipulation_ok else "NOT_REPLICATED"),
        }
    return ({"jobs": output, "replicated_all": all_passed}, refs, "EVALUATED")


def _g3_gate(index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    source = _read("results/v2_backbone_generalization/g3_cardinality_gate.json")
    thresholds = source["thresholds"]
    job_ids = ["v2g_g3_b1___all_common__", "v2g_g3_b2___all_common__"]
    result, refs, state = _cardinality_gate_result(
        index, job_ids,
        auc_name="crossK_K5_minus_K50__global_T::auroc_correct",
        eaurc_name="crossK_relative_eaurc_worsening_K50__global_T",
        rer_name="crossK_K5_minus_K50__global_T::rer_at_50",
        thresholds=thresholds,
    )
    old_backbones = source["backbones"]
    old_decisions = {"B1_openclip_b16": old_backbones["B1_openclip_b16"]["CARDINALITY_REPLICATED"],
                     "B2_siglip_b16": old_backbones["B2_siglip_b16"]["CARDINALITY_REPLICATED"]}
    row = {
        "scope_id": "v2g_g3", "gate_id": "V2G_G3_CARDINALITY_GATE",
        "source": "results/v2_backbone_generalization/g3_cardinality_gate.json#/backbones/*",
        "predicate_source": "scripts/p1_replication_core.py::c1_gate; scripts/analyze_v2g_cardinality.py::evaluate_backbone",
        "old_thresholds": thresholds, "old_decision": old_decisions, "new_estimates": refs,
        "threshold_change": "NONE", "status": state,
        "new_decision": state if state != "EVALUATED" else result["replicated_all"],
        "new_result": result,
        "predicate_note": "Route A uses mean K5−K50 AUROC drop >= 0.03 and positive per-seed CI lower bounds. Route B uses mean K50-baselined relative E-AURC worsening >= 0.20 and mean RER50 drop >= 0.10 as a fraction, E-AURC CIs merely exclude zero (either sign), and positive RER50 CI lower bounds.",
        "cohort_note": "Historical gate cohort is __all_common__ (validation plus testA/testB, no train); the new testA+testB pool is not substituted.",
        "old_new_aggregation": "Same frozen B1/B2 seed set; each seed effect is computed within each shared image draw before the seed mean.",
    }
    if state != "EVALUATED":
        row["eligibility_reasons"] = result.get("reason", [])
    return row


def _g4_gate(index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    root = "results/v2_backbone_generalization/g4_phaseB"
    source_paths = [f"{root}/b1/gate.json", f"{root}/b2/gate.json"]
    source_docs = [_read(path) for path in source_paths]
    thresholds = source_docs[0]["thresholds"]
    jobs = ["v2g_g4_b1", "v2g_g4_b2"]
    manipulation = {job: bool(doc.get("manipulation_ok", False)) for job, doc in zip(jobs, source_docs)}
    result, refs, state = _hard_gate_result(index, jobs, manipulation_by_job=manipulation, thresholds=thresholds)
    old_decision = {"B1": source_docs[0].get("verdict"), "B2": source_docs[1].get("verdict")}
    row = {
        "scope_id": "v2g_g4", "gate_id": "V2G_G4_HARD_SEMANTIC_GATE",
        "source": "; ".join(f"{path}#/verdict" for path in source_paths),
        "predicate_source": "scripts/run_v2g_hard.py::_hard_gate (V2-G G4 item 9)",
        "old_thresholds": {"delta_hard_min": thresholds["delta_hard_min"],
                           "amplification_min": thresholds["amplification_min"],
                           "requires_manipulation_valid": thresholds.get("requires_manipulation_valid", True)},
        "old_decision": old_decision, "new_estimates": refs, "threshold_change": "NONE",
        "status": state, "new_decision": state if state != "EVALUATED" else result["replicated_all"],
        "new_result": result,
        "predicate_note": "For each backbone, mean hard AUROC increment >= 0.015 and mean per-seed CI lower bound > 0; mean hard-minus-random amplification >= 0.01 and mean per-seed CI lower bound > 0; the frozen manipulation-validity check is required.",
        "old_new_aggregation": "Per-seed hard and hard-minus-random effects are computed inside the shared image draw, then seed-mean CIs are evaluated.",
    }
    if state != "EVALUATED":
        row["eligibility_reasons"] = result.get("reason", [])
    return row


def _d1_gate(index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    old = _read("results/v2_data_robustness/d1_reviewed_annotations/verdict.json")
    job_c1, job_c4 = "d1_c1_b3_clean", "d1_c4_clean"
    c1_names = [
        "crossK_K5_minus_K10::auroc_correct", "crossK_K5_minus_K20::auroc_correct", "crossK_K5_minus_K50::auroc_correct",
        "crossK_K5_minus_K10::e_aurc", "crossK_K5_minus_K20::e_aurc", "crossK_K5_minus_K50::e_aurc",
        "crossK_eaurc_relative_K5_minus_K10", "crossK_eaurc_relative_K5_minus_K20", "crossK_eaurc_relative_K5_minus_K50",
        "crossK_K5_minus_K10::rer_at_50", "crossK_K5_minus_K20::rer_at_50", "crossK_K5_minus_K50::rer_at_50",
    ]
    c4_names = ["d1_c4_hard_semantic_delta_auroc", "d1_c4_hard_rer50_gain_pp", "d1_c4_hard_eaurc_reduction",
                "d1_c4_hard_minus_rand_dod_auroc"]
    refs = _refs(*[(job_c1, name) for name in c1_names], *[(job_c4, name) for name in c4_names])
    state = _eligibility(index, refs)
    row: dict[str, Any] = {
        "scope_id": "d1", "gate_id": "D1_C1_C4_GATE",
        "source": "results/v2_data_robustness/d1_reviewed_annotations/verdict.json#/c1_cardinality;/c4_hard_semantic",
        "predicate_source": "scripts/run_d1_c1_c4.py::c1_verdict, c4_verdict; src/ccg/semantic/hard_eval.py::hard_gate",
        "old_thresholds": {"c1": "No numeric cutoff; exact CI-sign and monotonic-point checks are recorded in the source verdict.",
                           "c4": _read("results/phase1f_hard_semantic/gate.json").get("verdict", {}).get("thresholds", {})},
        "old_decision": {"c1_robust": old["c1_cardinality"]["verdict"]["robust"],
                         "c4_robust": old["c4_hard_semantic"]["verdict"]["robust"],
                         "combined": old["combined_verdict"]},
        "new_estimates": refs, "threshold_change": "NONE", "status": state,
        "predicate_note": "C1 uses only the historical mean-seed row: K50 AUROC lower bound > 0, E-AURC absolute upper bound < 0, relative E-AURC lower bound < 0, RER50 lower bound > 0, plus monotone point pattern across K10/K20/K50. C4 retains A8.6, hard-minus-random AUROC DoD lower bound > 0, and its direction must match V1. The historical C1 predicate is not a per-seed majority gate.",
        "old_new_aggregation": "Both predicates use three fixed B3 seeds; effects are formed per seed per shared image draw, then averaged. C1 gate conditions apply to the mean-seed repair interval.",
    }
    if state != "ELIGIBLE":
        row.update(status=state, new_decision=state, eligibility_reasons=_eligibility_reasons(index, refs))
        return row
    def e(name: str) -> dict[str, Any]:
        return _ref(index, job_c1, name)
    mean_row = {name: e(name) for name in c1_names}
    c1_checks = {
        "auroc_gain_k50_ci_low_gt0": float(mean_row["crossK_K5_minus_K50::auroc_correct"]["ci_low"]) > 0.0,
        "e_aurc_worse_k50_ci_high_lt0": float(mean_row["crossK_K5_minus_K50::e_aurc"]["ci_high"]) < 0.0,
        "e_aurc_relative_worse_k50_ci_low_lt0": float(mean_row["crossK_eaurc_relative_K5_minus_K50"]["ci_low"]) < 0.0,
        "rer50_gain_k50_ci_low_gt0": float(mean_row["crossK_K5_minus_K50::rer_at_50"]["ci_low"]) > 0.0,
    }
    auc_values = [float(mean_row[f"crossK_K5_minus_K{k}::auroc_correct"]["point"]) for k in (10, 20, 50)]
    rel_values = [-float(mean_row[f"crossK_eaurc_relative_K5_minus_K{k}"]["point"]) for k in (10, 20, 50)]
    rer_values = [float(mean_row[f"crossK_K5_minus_K{k}::rer_at_50"]["point"]) for k in (10, 20, 50)]
    c1_checks["monotone_in_k"] = all(a <= b + 1e-12 for values in (auc_values, rel_values, rer_values)
                                      for a, b in zip(values, values[1:]))

    from ccg.semantic.hard_eval import hard_gate
    delta = _ref(index, job_c4, c4_names[0])
    rer = _ref(index, job_c4, c4_names[1])
    reduction = _ref(index, job_c4, c4_names[2])
    dod = _ref(index, job_c4, c4_names[3])
    gate = hard_gate(
        {"delta_auroc": delta["point"], "delta_auroc_ci_low": delta["ci_low"],
         "e_aurc_reduction": reduction["point"], "e_aurc_reduction_ci_low": reduction["ci_low"],
         "rer50_gain_pp": rer["point"], "rer50_gain_pp_ci_low": rer["ci_low"]},
        {"diff": dod["point"]},
        seed_delta_ci_lows=[float(v["ci_low"]) for v in delta["per_seed"].values()],
        seed_mean_delta=float(delta["point"]), thresholds=row["old_thresholds"]["c4"],
    )
    v1_dod = old["c4_hard_semantic"]["vs_v1"]["diff_of_diffs_auroc"]["v1_full_cohort"]
    c4_checks = {
        "delta_auroc_ci_low_gt0": float(delta["ci_low"]) > 0.0,
        "dod_ci_low_gt0": float(dod["ci_low"]) > 0.0,
        "dod_same_direction_as_v1": float(dod["point"]) * float(v1_dod) > 0.0,
        "a8_6_gate_confirmed_or_strong": gate["verdict"] in ("CONFIRMED", "STRONG"),
    }
    c1_robust, c4_robust = all(c1_checks.values()), all(c4_checks.values())
    result = {"c1": {"checks": c1_checks, "robust": c1_robust},
              "c4": {"checks": c4_checks, "robust": c4_robust, "a8_6_gate": gate},
              "combined_robust": c1_robust and c4_robust}
    row.update(status="EVALUATED", new_decision=result["combined_robust"], new_result=result)
    return row


def _d2_gate_rows(index: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    source = _read("results/v2_d2_refcoco_lang/c1_c4_verdict.json")
    c1 = source["c1"]["gate"]
    thresholds = c1["thresholds"]
    c1_job = "d2_c1_common_k50"
    c1_result, c1_refs, c1_state = _cardinality_gate_result(
        index, [c1_job],
        auc_name="crossK_K5_minus_K50__global_T::auroc_correct",
        eaurc_name="crossK_relative_eaurc_worsening_K50__global_T",
        rer_name="crossK_K5_minus_K50__global_T::rer_at_50",
        thresholds=thresholds,
    )
    c1_row = {
        "scope_id": "d2_c1", "gate_id": "D2_C1_GATE",
        "source": "results/v2_d2_refcoco_lang/c1_c4_verdict.json#/c1/gate",
        "predicate_source": "scripts/run_d2_c1_c4.py::c1_gate (V2-G G3 predicate)",
        "old_thresholds": thresholds, "old_decision": c1["replicated"], "new_estimates": c1_refs,
        "threshold_change": "NONE", "status": c1_state,
        "new_decision": c1_state if c1_state != "EVALUATED" else c1_result["replicated_all"],
        "new_result": c1_result,
        "predicate_note": "Route A: mean AUROC K5−K50 >= .03 and every fixed-seed CI lower bound >0. Route B: mean relative E-AURC worsening >=.20, mean RER50 drop >=.10 fraction, E-AURC CI excludes zero without a directional sign requirement, and RER50 lower bounds >0.",
        "cohort_note": "Native-ID D2 C1 common-K50 test cohort, split metadata verified by expr_id/ref/image bridge.",
    }
    if c1_state != "EVALUATED":
        c1_row["eligibility_reasons"] = c1_result.get("reason", [])

    c4 = source["c4"]["gate"]
    c4_job = "d2_c4_random_hard5"
    c4_result, c4_refs, c4_state = _hard_gate_result(
        index, [c4_job], manipulation_by_job={c4_job: bool(c4["manipulation_ok"])},
        thresholds=c4["thresholds"],
    )
    c4_row = {
        "scope_id": "d2_c4", "gate_id": "D2_C4_GATE",
        "source": "results/v2_d2_refcoco_lang/c1_c4_verdict.json#/c4/gate",
        "predicate_source": "scripts/run_d2_c1_c4.py::c4_gate; V2-G G4 item 9",
        "old_thresholds": c4["thresholds"], "old_decision": c4["verdict"],
        "new_estimates": c4_refs, "threshold_change": "NONE", "status": c4_state,
        "new_decision": c4_state if c4_state != "EVALUATED" else c4_result["replicated_all"],
        "new_result": c4_result,
        "predicate_note": "Mean hard AUROC increment >=.015 and mean per-seed lower CI >0; mean hard-minus-random amplification >=.01 and mean per-seed lower CI >0; frozen D2 manipulation_ok is required.",
    }
    if c4_state != "EVALUATED":
        c4_row["eligibility_reasons"] = c4_result.get("reason", [])
    return [c1_row, c4_row]


def _proposal_gate_rows(index: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    f4_path = "results/v2_proposal_robustness/p1_f4_c1/f4_verdict.json"
    p2_path = "results/v2_proposal_robustness/p2_c1_gdino/p2_c1_verdict.json"
    f5_path = "results/v2_proposal_robustness/p1_f5_c4/f5_verdict.json"
    f4, p2, f5 = _read(f4_path), _read(p2_path), _read(f5_path)
    rows: list[dict[str, Any]] = []
    c1_names = {"auc": "K5_minus_K50::auroc_correct",
                "eaurc": "K50_relative_eaurc_worsening_over_K5",
                "rer": "K5_minus_K50::rer_at_50"}
    f4_jobs = ["p1_c1_RPN", "p1_c1_DETR"]
    f4_thresholds = f4["c1_by_family"]["RPN"]["gate"]["thresholds"]
    result, refs, state = _cardinality_gate_result(
        index, f4_jobs, auc_name=c1_names["auc"], eaurc_name=c1_names["eaurc"],
        rer_name=c1_names["rer"], thresholds=f4_thresholds,
    )
    rows.append({
        "scope_id": "p", "gate_id": "P1_F4_ROUTE_GATE", "source": f"{f4_path}#/c1_by_family",
        "predicate_source": "scripts/p1_f4_cardinality.py; scripts/p1_replication_core.py::c1_gate",
        "old_thresholds": f4_thresholds,
        "old_decision": {family: f4["c1_by_family"][family]["gate"]["replicated"] for family in ("RPN", "DETR")},
        "new_estimates": refs, "threshold_change": "NONE", "status": state,
        "new_decision": state if state != "EVALUATED" else result["replicated_all"], "new_result": result,
        "predicate_note": "The frozen V2-G G3 route predicates are evaluated separately for RPN and DETR. AUROC and RER50 CI lower bounds must be positive; E-AURC CIs only need to exclude zero, matching c1_gate. RER50 threshold .10 is a proportion (10pp); explicit estimate scale is converted from pp when required.",
    })
    if state != "EVALUATED": rows[-1]["eligibility_reasons"] = result.get("reason", [])

    p2_job = ["p1_c1_GDINO"]
    p2_thresholds = p2["c1_by_family"]["GDINO"]["gate"]["thresholds"]
    result, refs, state = _cardinality_gate_result(
        index, p2_job, auc_name=c1_names["auc"], eaurc_name=c1_names["eaurc"],
        rer_name=c1_names["rer"], thresholds=p2_thresholds,
    )
    rows.append({
        "scope_id": "p", "gate_id": "P2_C1_GATE", "source": f"{p2_path}#/c1_by_family/GDINO",
        "predicate_source": "scripts/p2_c1_cardinality.py; scripts/p1_replication_core.py::c1_gate",
        "old_thresholds": p2_thresholds, "old_decision": p2["c1_by_family"]["GDINO"]["gate"]["replicated"],
        "new_estimates": refs, "threshold_change": "NONE", "status": state,
        "new_decision": state if state != "EVALUATED" else result["replicated_all"], "new_result": result,
        "predicate_note": "The frozen V2-G G3 routes and thresholds are replayed for GDINO; E-AURC CIs require zero exclusion but not a particular direction, while AUROC and RER50 CIs must be positive.",
    })
    if state != "EVALUATED": rows[-1]["eligibility_reasons"] = result.get("reason", [])

    f5_jobs = ["p1_c4_RPN", "p1_c4_DETR"]
    f5_thresholds = f5["c4_by_family"]["RPN"]["gate"]["thresholds"]
    manipulation = {job: bool(f5["c4_by_family"][family]["manipulation"]["manipulation_valid"])
                    for job, family in zip(f5_jobs, ("RPN", "DETR"))}
    result, refs, state = _hard_gate_result(index, f5_jobs, manipulation_by_job=manipulation,
                                            thresholds=f5_thresholds)
    rows.append({
        "scope_id": "p", "gate_id": "P1_F5_ROUTE_GATE", "source": f"{f5_path}#/c4_by_family",
        "predicate_source": "scripts/p1_f5_hard_semantic.py; scripts/p1_replication_core.py::c4_hard_gate",
        "old_thresholds": f5_thresholds,
        "old_decision": {family: f5["c4_by_family"][family]["gate"]["verdict"] for family in ("RPN", "DETR")},
        "new_estimates": refs, "threshold_change": "NONE", "status": state,
        "new_decision": state if state != "EVALUATED" else result["replicated_all"], "new_result": result,
        "predicate_note": "For each proposal family, retain G4 delta-hard >=.015 and hard-minus-random AUROC amplification >=.01 with positive mean CI lows; frozen per-family manipulation validity is required.",
    })
    if state != "EVALUATED": rows[-1]["eligibility_reasons"] = result.get("reason", [])
    return rows


def refresh() -> dict[str, Any]:
    summary = _read("results/research_repair_v1/statistics/summary.json")
    old_doc = _read("results/research_repair_v1/statistics/gate_decisions.json")
    coverage = _read("results/research_repair_v1/statistics/required_scope_coverage.json")
    index = _estimate_index(summary)
    phase05 = _read("results/phase05_score_sufficiency/sufficiency_gate.json")
    phase1 = _read("results/phase1_semantic_sufficiency/sufficiency_gate.json")
    phase1f = _read("results/phase1f_hard_semantic/gate.json")
    ref = _read("results/phase1e_refcocog_external/a11_results.json")
    refreshed = [
        {
            "scope_id": "phase0a", "gate_id": "B0_temperature_gate",
            "source": "results/phase0a_corrected/metadata.json; results/phase0a_corrected/temperature_fit.json",
            "predicate_source": "NOT_APPLICABLE: original B0 artifacts define native/global-T/oracle-T diagnostics and no threshold-based go/no-go predicate",
            "old_thresholds": {}, "old_decision": "NO_GATE_DECLARED", "new_estimates": [],
            "new_decision": "NOT_APPLICABLE", "status": "NOT_APPLICABLE", "threshold_change": "NONE",
            "predicate_note": "Report old/new point estimates and image-cluster CIs as diagnostics; no original gate threshold exists to preserve.",
        },
        {
            "scope_id": "phase0b", "gate_id": "B0_temperature_gate",
            "source": "results/phase0b_independent/metadata.json; results/phase0b_independent/aggregate.csv",
            "predicate_source": "NOT_APPLICABLE: original B0 artifacts define native/global-T/oracle-T diagnostics and no threshold-based go/no-go predicate",
            "old_thresholds": {}, "old_decision": "NO_GATE_DECLARED", "new_estimates": [],
            "new_decision": "NOT_APPLICABLE", "status": "NOT_APPLICABLE", "threshold_change": "NONE",
            "predicate_note": "Report native/global-T and validation-fitted per-K diagnostic endpoints separately; oracle_T_K is diagnostic, not an OOD method.",
        },
        _phase05(index, phase05), _phase1(index, phase1), _phase1f(index, phase1f), _refcocog(index, ref),
    ]
    # Add the registered, executable predicates. Each row includes concrete
    # source decisions and named estimate references rather than scaffold data.
    from ccg.repairs.reliability_go_gate import build_reliability_go_decisions
    from ccg.repairs.legacy_gate_predicates import build_legacy_model_gate_decisions

    reliability_rows = build_reliability_go_decisions(ROOT, summary)
    legacy_model_rows = build_legacy_model_gate_decisions(ROOT, summary, coverage)
    refreshed.extend(reliability_rows)
    refreshed.extend([
        _g3_gate(index), _g4_gate(index), _d1_gate(index), *_d2_gate_rows(index), *_proposal_gate_rows(index),
    ])
    refreshed.extend(legacy_model_rows)
    output = {
        **old_doc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "decisions": refreshed,
        "method": {
            "decision_rule": "Evaluate each original predicate with its unchanged thresholds over named repair estimates. Estimates are eligible only at 5000/5000 valid image-cluster draws, seed 0, percentile 95%; otherwise report UNCERTAINTY_INSUFFICIENT.",
            "seed_aggregation": "Compute each fixed seed's effect inside the shared image draw, then take the mean across seeds; per-seed CIs and seed SD remain separate.",
        },
    }
    return output


def main() -> int:
    output = refresh()
    target = OUT / "gate_decisions.json"
    _atomic_write_json(target, output)
    coverage_path = OUT / "required_scope_coverage.json"
    coverage = _read("results/research_repair_v1/statistics/required_scope_coverage.json")
    scopes = coverage.get("scopes", [])
    gate_ids_by_scope: dict[str, set[str]] = {}
    for item in output.get("decisions", []):
        gate_ids_by_scope.setdefault(str(item["scope_id"]), set()).add(str(item["gate_id"]))
    for row in scopes:
        scope_id = str(row.get("scope_id", ""))
        present = set(row.get("gate_ids", []))
        row["gate_ids"] = sorted(present | gate_ids_by_scope.get(scope_id, set()))
    coverage["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(coverage_path, coverage)
    print(f"Wrote {len(output['decisions'])} gate rows to {target.relative_to(ROOT)}")
    return 0


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    from ccg.repairs.atomic import replace_with_retry

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix="gate_", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temp_path, path)


if __name__ == "__main__":
    raise SystemExit(main())
