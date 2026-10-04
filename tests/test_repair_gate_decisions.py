from __future__ import annotations

from scripts.refresh_repair_gate_decisions import _d2_gate_rows, _phase05, _phase1, _phase1f


def _estimate(point: float, ci_low: float = 0.01, *, seed_points: dict | None = None) -> dict:
    item = {
        "point": point, "ci_low": ci_low, "ci_high": point + 0.02,
        "gate_eligible": True, "valid_replicates": 5000, "invalid_replicates": 0,
    }
    if seed_points is not None:
        item["per_seed"] = {
            key: {"point": value, "ci_low": ci_low, "ci_high": value + 0.02}
            for key, value in seed_points.items()
        }
    return item


def _index(rows: dict[tuple[str, str], dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for (job_id, name), raw in rows.items():
        item = dict(raw)
        item.update(n_replicates=5000, ci_level=0.95, resample_unit="image_cluster")
        item.setdefault("per_seed", {
            f"b3_seed{seed}": {
                "point": (raw.get("per_seed", {}).get(f"b3_seed{seed}", {}).get("point", raw["point"])),
                "ci_low": raw.get("per_seed", {}).get(f"b3_seed{seed}", {}).get("ci_low", raw["ci_low"]),
                "ci_high": raw.get("per_seed", {}).get(f"b3_seed{seed}", {}).get("ci_high", raw["ci_high"]),
                "n_replicates": 5000, "valid_replicates": 5000, "invalid_replicates": 0,
                "ci_level": 0.95, "resample_unit": "image_cluster", "gate_eligible": True,
            } for seed in (1, 2, 3)
        })
        for seed_item in item["per_seed"].values():
            seed_item.update(n_replicates=5000, valid_replicates=5000, invalid_replicates=0,
                             ci_level=0.95, resample_unit="image_cluster", gate_eligible=True)
        item["_job_config"] = {"n_replicates": 5000, "seed": 0, "ci_level": 0.95, "ci_method": "percentile"}
        item["_job_result"] = {"resample_unit": "image_cluster", "method": "percentile"}
        item["_job_n_seeds"] = 3
        item["_job_model_seed_names"] = ["b3_seed1", "b3_seed2", "b3_seed3"]
        item["_job_status"] = "RECOMPUTED_5000"
        indexed[(job_id, name)] = item
    return indexed


def test_phase05_repair_gate_preserves_a6_threshold_routes() -> None:
    thresholds = {
        "auroc_drop": 0.03, "auroc_stable": 0.02, "e_aurc_worsen": 0.20,
        "e_aurc_abs": 0.03, "rer_drop_pp": 10.0, "override_auroc": 0.85,
        "override_e_aurc": 0.03, "override_rer": 0.70,
    }
    rows = {}
    for split in ("testA", "testB"):
        job = f"phase05_{split}"
        for k in (20, 50):
            rows[(job, f"crossK__msp__K{k}__auroc_drop")] = _estimate(0.04)
            rows[(job, f"crossK__msp__K{k}__e_aurc_worsening_relative")] = _estimate(0.30)
            rows[(job, f"crossK__msp__K{k}__rer50_drop")] = _estimate(0.15)
    pooled = "phase05___pooled_test__"
    rows.update({
        (pooled, "msp__K50::auroc_correct"): _estimate(0.80),
        (pooled, "msp__K50::e_aurc"): _estimate(0.10),
        (pooled, "msp__K50::rer_at_50"): _estimate(0.60),
        (pooled, "crossK__msp__K50__auroc_drop"): _estimate(0.05),
        (pooled, "crossK__msp__K50__e_aurc_worsening_relative"): _estimate(0.30),
        (pooled, "crossK__msp__K50__rer50_drop"): _estimate(0.15),
    })
    result = _phase05(_index(rows), {"verdict_seed_mean": {"thresholds": thresholds, "verdict": "OLD"}})
    assert result["new_decision"] == "GO_candidate_embeddings"
    assert len(result["new_result"]["route_A_cells"]) == 4
    assert len(result["new_result"]["route_B_cells"]) == 4
    assert result["threshold_change"] == "NONE"


def test_gate_eligibility_rejects_wrong_replicate_method_metadata() -> None:
    from scripts.refresh_repair_gate_decisions import _eligibility

    rows = _index({("job", "estimate"): _estimate(0.1)})
    rows[("job", "estimate")]["_job_config"]["seed"] = 1
    assert _eligibility(rows, [{"job_id": "job", "estimate_name": "estimate"}]) == "UNCERTAINTY_INSUFFICIENT"

    rows[("job", "estimate")]["_job_config"]["seed"] = 0
    rows[("job", "estimate")]["_job_result"]["resample_unit"] = "sentence"
    assert _eligibility(rows, [{"job_id": "job", "estimate_name": "estimate"}]) == "UNCERTAINTY_INSUFFICIENT"


def test_phase1_and_phase1f_reuse_original_predicates_on_repair_estimates() -> None:
    thresholds1 = {
        "auroc_gain": 0.02, "auroc_gain_strong": 0.03, "e_aurc_reduction": 0.10,
        "e_aurc_reduction_strong": 0.20, "rer50_gain_pp": 5.0,
        "rer50_gain_pp_strong": 10.0, "no_go_auroc": 0.01,
        "no_go_e_aurc": 0.05, "no_go_rer50_pp": 3.0, "seed_sig_min": 2, "seed_total": 3,
    }
    job = "phase1_fixed___pooled_test__"
    rows = {}
    for k in (20, 50):
        rows[(job, f"gate_delta_auroc_e1b_vs_msp__K{k}")] = _estimate(
            0.035, 0.02, seed_points={"b3_seed1": 0.034, "b3_seed2": 0.035, "b3_seed3": 0.036},
        )
        rows[(job, f"gate_eaurc_relative_reduction_e1b_vs_msp__K{k}")] = _estimate(0.22, 0.10)
        rows[(job, f"gate_rer50_gain_e1b_vs_msp_pp__K{k}")] = _estimate(6.0, 2.0)
    phase1 = _phase1(_index(rows), {"thresholds": thresholds1, "verdict": {"verdict": "OLD"}})
    assert phase1["new_decision"] == "STRONG"

    thresholds8 = {
        "auroc_gain": 0.02, "auroc_gain_strong": 0.03, "e_aurc_reduction": 0.10,
        "e_aurc_reduction_strong": 0.20, "rer50_gain_pp": 5.0,
        "rer50_gain_pp_strong": 10.0, "diff_of_diffs_min": 0.005,
        "seed_sig_min": 2, "seed_total": 3,
    }
    jobf = "phase1f_k5_same4"
    hard = _estimate(0.025, 0.01, seed_points={"b3_seed1": 0.024, "b3_seed2": 0.025, "b3_seed3": 0.026})
    rowsf = {
        (jobf, "semantic_minus_stats__hard5::auroc_correct"): hard,
        (jobf, "e_aurc_reduction__hard5"): _estimate(0.15, 0.08),
        (jobf, "semantic_minus_stats__hard5::rer_at_50"): _estimate(4.0, 1.0),
        (jobf, "gate_dod_samecat5::auroc_correct"): _estimate(0.01, 0.005),
    }
    phase1f = _phase1f(_index(rowsf), {"verdict": {"thresholds": thresholds8, "verdict": "OLD"}})
    assert phase1f["new_decision"] == "CONFIRMED"


def test_d2_c1_gate_uses_registered_d2_estimate_names() -> None:
    job = "d2_c1_common_k50"
    auc = _estimate(0.046, 0.03, seed_points={
        "b3_seed1": 0.045, "b3_seed2": 0.047, "b3_seed3": 0.046,
    })
    eaurc = _estimate(0.30, 0.08, seed_points={
        "b3_seed1": 0.29, "b3_seed2": 0.31, "b3_seed3": 0.30,
    })
    rer50 = _estimate(46.8, 20.0, seed_points={
        "b3_seed1": 46.0, "b3_seed2": 47.0, "b3_seed3": 47.4,
    })
    rer50["scale"] = 100.0
    rows = {
        (job, "d2_c1_auroc_drop_K5_K50"): auc,
        (job, "d2_c1_eaurc_relative_worsening_K5_K50"): eaurc,
        (job, "d2_c1_rer50_drop_K5_K50"): rer50,
        ("d2_c4_random_hard5", "delta_hard_auroc"): _estimate(0.036, 0.02),
        ("d2_c4_random_hard5", "amplification_auroc"): _estimate(0.035, 0.02),
    }

    c1, c4 = _d2_gate_rows(_index(rows))

    assert c1["status"] == "EVALUATED"
    assert c1["old_decision"] is True
    assert c1["new_decision"] is True
    assert c1["predicate_source"].startswith("scripts/run_d2_c1_c4.py::c1_gate")
    assert {ref["estimate_name"] for ref in c1["new_estimates"]} == {
        "d2_c1_auroc_drop_K5_K50",
        "d2_c1_eaurc_relative_worsening_K5_K50",
        "d2_c1_rer50_drop_K5_K50",
    }
    assert c4["status"] == "EVALUATED"

