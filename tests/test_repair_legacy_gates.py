from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ccg.repairs.legacy_gate_predicates import (
    FIXED_SEEDS,
    M2_RELATIVE_EAURC,
    build_legacy_model_gate_decisions,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _put_estimate(
    estimates: dict[tuple[str, str], dict[str, Any]],
    job_id: str,
    name: str,
    *,
    point: float = 0.01,
    low: float = 0.001,
    high: float = 0.02,
) -> None:
    estimates[(job_id, name)] = {
        "name": name,
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "n_replicates": 5000,
        "resample_unit": "image_cluster",
        "ci_level": 0.95,
        "valid_replicates": 5000,
        "invalid_replicates": 0,
        "gate_eligible": True,
        "per_seed": {
            seed: {
                "point": point,
                "ci_low": low,
                "ci_high": high,
                "valid_replicates": 5000,
                "invalid_replicates": 0,
                "gate_eligible": True,
            }
            for seed in FIXED_SEEDS
        },
    }


def _make_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = {
        "M1_gates": {
            "primary_cell": "SameCategory-K5",
            "GO_gate": {
                "delta_auroc_LCR_minus_E1b_ge": 0.01,
                "delta_auroc_ci_low_gt": 0.0,
                "at_least_one_of": {
                    "e_aurc_reduction_ge": 0.05,
                    "rer50_gain_pp_ge": 3.0,
                },
                "random_k5_guard": {"delta_auroc_LCR_minus_E1b_ge": -0.005},
            },
        },
        "M2_preregistration": {
            "gate": {
                "vs_curriculum_E1b": {"delta_auroc_ge": 0.01, "ci_low_gt": 0.0},
                "vs_Aggregate_MLP": {"ci_low_gt": 0.0},
            },
        },
    }
    _write_json(root / "results/v2_local_competition/protocol.json", protocol)
    _write_json(root / "results/v2_local_competition/gate.json", {
        "gates": {"M1_GO": {"verdict": False}},
    })
    _write_json(root / "results/v2_local_competition/m2_curriculum/gate.json", {
        "gates": {
            "M2_GO": {"verdict": False},
            "selective_metrics": {"method_signal_mixed": False},
            "gate_contribution": {
                "supported": False, "unsupported": True, "effectively_equal": False,
            },
        },
    })
    _write_json(root / "results/v2_local_competition/m25_specialist_audit/verdict.json", {
        "delta_definition": "curriculum minus random",
        "pattern": "SPECIALIST TRADEOFF PRESENT",
        "conditions": {},
    })
    _write_json(root / "results/v2_local_competition/m3_mixture/b0/summary.json", {
        "no_gate": True, "label": "POST-HOC DEVELOPMENTAL",
    })
    _write_json(root / "results/v2_local_competition/m3_mixture/amendment_v2m31.json", {
        "frozen_gates": {"primary": "delta >= .003 and CI low > 0"},
        "frozen_verdict_rule": {"2/2 pass": "strong", "otherwise": "stop"},
    })
    _write_json(root / "results/v2_local_competition/m3_mixture/protocol_m3.json", {
        "confirmatory_gates": {"B1": True, "B2": True},
    })
    _write_json(root / "results/v2_local_competition/m3_mixture/conf/b1/gate.json", {"pass": False})
    _write_json(root / "results/v2_local_competition/m3_mixture/conf/b2/gate.json", {"pass": False})
    _write_json(root / "results/v2_local_competition/m3_mixture/conf/cross_backbone_verdict.json", {
        "verdict": "ADAPTIVE MIXTURE NOT SUPPORTED", "n_pass": 0,
    })
    for filename in (
        "run_v2m_m1.py", "run_v2m_m2.py", "run_v2m_m25.py",
        "run_v2m_m3_b0.py", "run_v2m_m3_conf.py",
    ):
        path = root / "scripts" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture source for {filename}\n", encoding="utf-8")

    estimates: dict[tuple[str, str], dict[str, Any]] = {}
    _put_estimate(estimates, "m1_samecategory_k5", "LCR_minus_E1b::auroc_correct", point=0.01, low=0.001)
    _put_estimate(estimates, "m1_samecategory_k5", "LCR_E1b_relative_eaurc_reduction::SameCategory-K5", point=0.05)
    _put_estimate(estimates, "m1_samecategory_k5", "LCR_minus_E1b::rer_at_50", point=0.0)
    _put_estimate(estimates, "m1_random_k5", "LCR_minus_E1b::auroc_correct", point=-0.005)

    _put_estimate(estimates, "m2_curriculum_m0_m2_m4_m8", "m8__LCR_minus_E1b::auroc_correct", point=-0.001, low=-0.005, high=0.002)
    _put_estimate(estimates, "m2_curriculum_m0_m2_m4_m8", "m8__LCR_minus_Aggregate-MLP::auroc_correct", point=0.03, low=0.02, high=0.04)
    _put_estimate(estimates, "m2_curriculum_m0_m2_m4_m8", M2_RELATIVE_EAURC, point=-0.01, low=-0.02, high=-0.001)
    _put_estimate(estimates, "m2_curriculum_m0_m2_m4_m8", "m8__LCR_minus_E1b::rer_at_50", point=-0.5, low=-0.9, high=-0.1)
    _put_estimate(estimates, "m2_curriculum_m0_m2_m4_m8", "m8__LCR_minus_E1b::rer_at_80", point=0.1, low=-0.1, high=0.3)
    _put_estimate(estimates, "m2_curriculum_m0_m2_m4_m8", "m8__LCR_minus_LCR-noGate::auroc_correct", point=-0.01, low=-0.02, high=-0.001)

    for level, (low, high) in {
        0: (-0.02, -0.01),
        2: (-0.01, 0.01),
        4: (0.0, 0.02),
        8: (0.01, 0.02),
    }.items():
        _put_estimate(
            estimates, "m25_specialist_m0_m2_m4_m8",
            f"m{level}__E1b_curriculum_minus_random::auroc_correct",
            point=(low + high) / 2, low=low, high=high,
        )

    jobs_by_id: dict[str, dict[str, Any]] = {}
    for (job_id, _), estimate in estimates.items():
        job = jobs_by_id.setdefault(job_id, {
            "job_id": job_id,
            "status": "RECOMPUTED_5000",
            "config": {"n_replicates": 5000, "seed": 0, "ci_level": 0.95, "ci_method": "percentile"},
            "result": {"resample_unit": "image_cluster", "n_replicates": 5000, "seed": 0, "ci_level": 0.95, "method": "percentile"},
            "estimates": [],
        })
        job["estimates"].append(estimate)

    summary = {
        "bootstrap": {
            "replicates": 5000, "seed": 0, "ci_level": 0.95,
            "resample_unit": "image_cluster", "method": "percentile",
        },
        "jobs": list(jobs_by_id.values()),
    }

    # Build the same narrow ledger consumed by the production refresh path.
    scopes: list[dict[str, Any]] = []
    for scope_id, job_ids in {
        "m1": ["m1_samecategory_k5", "m1_random_k5"],
        "m2": ["m2_curriculum_m0_m2_m4_m8"],
        "m25": ["m25_specialist_m0_m2_m4_m8"],
    }.items():
        ledger = []
        for job_id in job_ids:
            job = jobs_by_id[job_id]
            names = [entry["name"] for entry in job["estimates"]]
            ledger.append({
                "job_id": job_id,
                "estimate_names": names,
                "auxiliary_estimate_names": [],
                "formal_status": "RECOMPUTED_5000",
                "n_replicates": 5000,
                "completed": True,
            })
        scopes.append({"scope_id": scope_id, "job_estimator_ledger": ledger})

    m3_failures = []
    for job_id in ("m3_b0_severity_macro", "m3_b1_severity_macro", "m3_b2_severity_macro"):
        base = Path("results/research_repair_v1/statistics/recovery/fixture")
        inventory_rel = (base / f"{job_id}_inventory.json").as_posix()
        log_rel = (base / f"{job_id}.log").as_posix()
        inventory_path = root / inventory_rel
        log_path = root / log_rel
        _write_json(inventory_path, {"job_id": job_id, "checkpoint_candidates": 0})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"strict historical recovery failed: {job_id}\n", encoding="utf-8")
        failure = {
            "schema": "research-repair-v1-recovery-failure-v1",
            "scope_id": "m3", "job_id": job_id, "status": "UNVERIFIABLE",
            "failure_reason": "strict point anchor mismatch",
            "original_tolerance": {"committed_point_metrics": 1e-9},
            "attempted_route": [{"step": "load/forward original route", "applicable": True}],
            "source_sha256": {"scripts/run_v2m_m3_conf.py": "a" * 64},
            "no_checkpoint_inventory": {"path": inventory_rel, "sha256": _sha(inventory_path)},
            "observed_error": 2e-7,
            "log_path": log_rel, "log_sha256": _sha(log_path),
        }
        failure_rel = (base / f"{job_id}.json").as_posix()
        failure_path = root / failure_rel
        _write_json(failure_path, failure)
        m3_failures.append({
            "job_id": job_id, "required": True, "status": "UNVERIFIABLE",
            "evidence_path": failure_rel, "evidence_sha256": _sha(failure_path),
        })
    scopes.append({"scope_id": "m3", "unverifiable_jobs": m3_failures})
    return summary, {"scopes": scopes}


def _decision(rows: list[dict[str, Any]], gate_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["gate_id"] == gate_id)


def test_frozen_m1_threshold_boundaries_and_m2_m8_signs(tmp_path: Path) -> None:
    summary, coverage = _make_fixture(tmp_path)
    rows = build_legacy_model_gate_decisions(tmp_path, summary, coverage)

    # Inclusive numeric minima, strict positive CI low, and the random guard
    # retain the legacy boundary behavior exactly.
    assert _decision(rows, "M1_GO")["new_decision"] is True
    assert _decision(rows, "M1_GO")["new_result"]["random_k5_guard_pass"] is True

    # The original M2 m=8 pattern has positive structural-capacity evidence,
    # but fails LCR-vs-E1b; case C and no M2_GO follow from that conjunction.
    m2 = _decision(rows, "M2_GO")
    assert m2["new_decision"] is False
    assert m2["new_result"]["capacity_case"] == "C"
    assert m2["new_result"]["condition_lcr_minus_aggregate_mlp"] is True
    selective = _decision(rows, "M2_SELECTIVE_SIGNAL")
    assert selective["new_decision"] is False
    assert selective["new_result"]["e_aurc_relative_reduction"] < 0
    assert selective["new_result"]["rer50_gain_pp"] < 0
    assert selective["new_result"]["rer80_gain_pp"] > 0
    assert _decision(rows, "M2_GATE_CONTRIBUTION")["new_decision"] == "UNSUPPORTED"


def test_m2_selective_requires_auroc_and_all_three_negative_effects(tmp_path: Path) -> None:
    summary, coverage = _make_fixture(tmp_path)
    job = next(j for j in summary["jobs"] if j["job_id"] == "m2_curriculum_m0_m2_m4_m8")
    by_name = {item["name"]: item for item in job["estimates"]}
    by_name["m8__LCR_minus_E1b::auroc_correct"].update(point=0.01, ci_low=0.001)
    by_name["m8__LCR_minus_E1b::rer_at_80"].update(point=-0.1, ci_low=-0.2, ci_high=-0.01)
    rows = build_legacy_model_gate_decisions(tmp_path, summary, coverage)
    assert _decision(rows, "M2_SELECTIVE_SIGNAL")["new_decision"] is True


def test_m25_uses_strict_ci_signs_and_keeps_zero_boundary_inconclusive(tmp_path: Path) -> None:
    summary, coverage = _make_fixture(tmp_path)
    rows = build_legacy_model_gate_decisions(tmp_path, summary, coverage)
    m25 = _decision(rows, "M25_DIAGNOSTIC_VERDICT")
    assert m25["new_decision"] == "SPECIALIST TRADEOFF PRESENT"
    assert m25["new_result"]["confirmatory_gate"] is False

    job = next(j for j in summary["jobs"] if j["job_id"] == "m25_specialist_m0_m2_m4_m8")
    by_name = {item["name"]: item for item in job["estimates"]}
    by_name["m0__E1b_curriculum_minus_random::auroc_correct"].update(ci_low=-0.02, ci_high=0.0)
    by_name["m8__E1b_curriculum_minus_random::auroc_correct"].update(ci_low=0.0, ci_high=0.02)
    rows = build_legacy_model_gate_decisions(tmp_path, summary, coverage)
    assert _decision(rows, "M25_DIAGNOSTIC_VERDICT")["new_decision"] == "INCONCLUSIVE"


def test_missing_formal_estimates_stay_pending_and_wrong_seed_is_uncertain(tmp_path: Path) -> None:
    summary, coverage = _make_fixture(tmp_path)
    summary["jobs"] = []
    rows = build_legacy_model_gate_decisions(tmp_path, summary, coverage)
    assert _decision(rows, "M1_GO")["status"] == "PENDING_FORMAL_BOOTSTRAP"
    assert _decision(rows, "M2_GO")["new_decision"] == "PENDING_FORMAL_BOOTSTRAP"

    summary, coverage = _make_fixture(tmp_path / "wrong-seed")
    hard_job = next(j for j in summary["jobs"] if j["job_id"] == "m1_samecategory_k5")
    hard_job["config"]["seed"] = 17
    rows = build_legacy_model_gate_decisions(tmp_path / "wrong-seed", summary, coverage)
    m1 = _decision(rows, "M1_GO")
    assert m1["status"] == "UNCERTAINTY_INSUFFICIENT"
    assert any("job config seed != 0" in issue for issue in m1["eligibility_issues"])


def test_m3_unverifiable_is_bound_to_all_three_hash_verified_failures(tmp_path: Path) -> None:
    summary, coverage = _make_fixture(tmp_path)
    rows = build_legacy_model_gate_decisions(tmp_path, summary, coverage)
    m3 = _decision(rows, "M3_B1_B2_FROZEN_GATE")
    assert m3["old_decision"] == "ADAPTIVE MIXTURE NOT SUPPORTED"
    assert m3["new_decision"] == "UNVERIFIABLE"
    assert len(m3["unverifiable_evidence"]) == 3
    assert all(item["checkpoint_inventory_sha256"] for item in m3["unverifiable_evidence"])

    failure = tmp_path / coverage["scopes"][-1]["unverifiable_jobs"][0]["evidence_path"]
    failure.write_text(failure.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="failure record hash"):
        build_legacy_model_gate_decisions(tmp_path, summary, coverage)
