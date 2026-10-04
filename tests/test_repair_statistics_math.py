"""Counterexamples that constrain temperature and normalized-risk claims."""

from __future__ import annotations

import math
import json
import hashlib

import numpy as np
import pytest
import scripts.repair_statistics as repair_statistics

from ccg.metrics.calibration import (
    brier_binary,
    confidence_accuracy_gap,
    confidence_from_scores,
    nll_binary,
    top_label_ece,
    top_label_ece_adaptive,
)
from ccg.metrics.discrimination import auprc_correct, auroc_correct
from ccg.metrics.selective import (
    aurc,
    e_aurc,
    oracle_aurc,
    rer_at_coverage,
    risk_at_coverage,
    risk_coverage_curve,
)
from ccg.repairs.statistics import _estimate_value, joint_prediction_bootstrap, metric_bundle, relative_effect
from scripts.repair_statistics import (
    _canonical_indices, _cohort_metadata_from_counts, _derive_legacy_points, _estimate_spec_metadata,
)


def _oracle_rer_ceiling(accuracy: float, coverage: float) -> float:
    if accuracy >= coverage:
        return 1.0
    return accuracy * (1.0 - coverage) / (coverage * (1.0 - accuracy))


def test_temperature_keeps_each_argmax_but_can_reverse_cross_sample_msp_order() -> None:
    # At T=1, A has higher MSP; at T=10, B has higher MSP. Both samples keep
    # their candidate argmax, yet the correctness AUROC changes from 1 to 0.
    logits = np.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, -100.0]])
    correct = np.asarray([1, 0])
    confidence_t1, argmax_t1 = confidence_from_scores(logits, temperature=1.0)
    confidence_t10, argmax_t10 = confidence_from_scores(logits, temperature=10.0)
    assert np.array_equal(argmax_t1, argmax_t10)
    assert confidence_t1[0] > confidence_t1[1]
    assert confidence_t10[0] < confidence_t10[1]
    assert auroc_correct(confidence_t1, correct) == pytest.approx(1.0)
    assert auroc_correct(confidence_t10, correct) == pytest.approx(0.0)


def test_coverage_refresh_reads_repair_owned_file_and_requires_failure_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(repair_statistics, "ROOT", tmp_path)
    output = tmp_path / "results" / "research_repair_v1" / "statistics"
    output.mkdir(parents=True)
    monkeypatch.setattr(repair_statistics, "OUT", output)
    coverage = output / "required_scope_coverage.json"
    job_ids = ("m3_b0_severity_macro", "m3_b1_severity_macro", "m3_b2_severity_macro")
    records = [{
        "job_id": job_id, "required": True, "status": "UNVERIFIABLE",
        "evidence_path": f"results/research_repair_v1/statistics/recovery/{job_id}_failure.json",
    } for job_id in job_ids]
    coverage.write_text(json.dumps({"scopes": [{
        "scope_id": "m3", "required": True, "job_ids": [], "unverifiable_jobs": records,
    }]}), encoding="utf-8")
    summary = {"jobs": []}
    repair_statistics._refresh_required_scope_coverage({"m3": []}, summary)
    row = json.loads(coverage.read_text(encoding="utf-8"))["scopes"][0]
    assert row["status"] == "UNVERIFIABLE_EVIDENCE_PENDING"

    evidence_dir = tmp_path / "results/research_repair_v1/statistics/recovery"
    evidence_dir.mkdir(parents=True)
    for item in records:
        evidence = tmp_path / item["evidence_path"]
        evidence.write_text(json.dumps({"strict_anchor_failure": True, "job_id": item["job_id"]}), encoding="utf-8")
        item["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    coverage.write_text(json.dumps({"scopes": [{
        "scope_id": "m3", "required": True, "job_ids": [], "unverifiable_jobs": records,
    }]}), encoding="utf-8")
    repair_statistics._refresh_required_scope_coverage({"m3": []}, summary)
    row = json.loads(coverage.read_text(encoding="utf-8"))["scopes"][0]
    assert row["status"] == "UNVERIFIABLE"
    assert row["anchor_status"] == "STRICT_FAILURE_EVIDENCE_FOR_ALL_REQUIRED_JOBS"


def test_random_rank_eaurc_and_oracle_rer_ceiling_depend_on_accuracy() -> None:
    # In the population limit, random confidence has E-AURC = -a*ln(a) despite
    # AUROC remaining at chance; this exposes its remaining accuracy dependence.
    accuracy_easy, accuracy_hard = 0.8, 0.2
    random_eaurc_easy = -accuracy_easy * math.log(accuracy_easy)
    random_eaurc_hard = -accuracy_hard * math.log(accuracy_hard)
    assert random_eaurc_easy != pytest.approx(random_eaurc_hard)

    # A finite symmetric example has the same chance AUROC at both accuracies,
    # while the implemented E-AURC still differs.
    confidence = np.linspace(0.0, 1.0, 5)
    correct_easy = np.asarray([1, 1, 0, 1, 1])
    correct_hard = 1 - correct_easy
    assert auroc_correct(confidence, correct_easy) == pytest.approx(0.5)
    assert auroc_correct(confidence, correct_hard) == pytest.approx(0.5)

    def measured_eaurc(correct: np.ndarray) -> float:
        coverage, risk = risk_coverage_curve(confidence, correct)
        accuracy = float(np.mean(correct))
        return e_aurc(aurc(coverage, risk), accuracy)

    assert measured_eaurc(correct_easy) != pytest.approx(measured_eaurc(correct_hard))

    # RER's oracle ceiling is 1 when accuracy already exceeds target coverage,
    # and falls below 1 when the base accuracy is lower than that coverage.
    ceiling_easy = _oracle_rer_ceiling(0.5349, 0.50)
    ceiling_hard = _oracle_rer_ceiling(0.1879, 0.50)
    assert ceiling_easy == pytest.approx(1.0)
    assert ceiling_hard == pytest.approx(0.2314, abs=1e-4)
    assert ceiling_hard < ceiling_easy


def test_fast_metric_bundle_matches_repository_metric_contracts_with_ties() -> None:
    confidence = np.asarray([0.8, 0.8, 0.3, 0.3, 0.3, 0.05, 0.95, 0.8])
    correct = np.asarray([1, 0, 0, 1, 0, 1, 1, 1])
    coverage, risk = risk_coverage_curve(confidence, correct)
    raw_aurc = aurc(coverage, risk)
    bundle = metric_bundle(confidence, correct)
    assert bundle["accuracy"] == pytest.approx(np.mean(correct))
    assert bundle["auroc_correct"] == pytest.approx(auroc_correct(confidence, correct))
    assert bundle["auprc_correct"] == pytest.approx(auprc_correct(confidence, correct))
    assert bundle["aurc"] == pytest.approx(raw_aurc)
    assert bundle["e_aurc"] == pytest.approx(e_aurc(raw_aurc, float(np.mean(correct))))
    for coverage_level, name in ((0.5, "50"), (0.8, "80"), (0.9, "90"), (0.95, "95")):
        assert bundle[f"rer_at_{name}"] == pytest.approx(rer_at_coverage(confidence, correct, coverage_level))
        if name != "95":
            assert bundle[f"risk_at_{name}"] == pytest.approx(risk_at_coverage(confidence, correct, coverage_level))
    assert bundle["ece_adaptive"] == pytest.approx(top_label_ece_adaptive(confidence, correct))
    assert bundle["ece_equal_width"] == pytest.approx(top_label_ece(confidence, correct))
    assert bundle["brier_binary"] == pytest.approx(brier_binary(confidence, correct))
    assert bundle["nll_binary"] == pytest.approx(nll_binary(confidence, correct))
    assert bundle["confidence_accuracy_gap"] == pytest.approx(confidence_accuracy_gap(confidence, correct))


def test_historical_point_order_preserves_stable_tie_anchor_without_changing_draws() -> None:
    image_ids = np.asarray([1, 2, 3, 4])
    confidence = np.full(4, 0.5)
    correct = np.asarray([1, 1, 0, 0])
    cells = {"seed": {"native": (confidence, correct)}}
    specs = [{"name": "eaurc", "metric": "e_aurc", "operation": "condition", "condition": "native"}]
    canonical = joint_prediction_bootstrap(
        image_ids, cells, estimates=specs, metrics=("e_aurc",), n_replicates=24, seed=5,
    )["estimates"]["eaurc"]
    legacy_order = np.asarray([2, 3, 0, 1])
    ordered = joint_prediction_bootstrap(
        image_ids, cells, estimates=specs, metrics=("e_aurc",), n_replicates=24, seed=5,
        point_order=legacy_order,
    )["estimates"]["eaurc"]
    expected = metric_bundle(confidence[legacy_order], correct[legacy_order], metrics=("e_aurc",))["e_aurc"]
    assert ordered["point"] == pytest.approx(expected)
    assert ordered["point"] != pytest.approx(canonical["point"])
    assert np.array_equal(ordered["replicates"], canonical["replicates"])
    assert np.array_equal(ordered["per_seed_replicates"]["seed"], canonical["per_seed_replicates"]["seed"])
    with pytest.raises(ValueError, match="permutation"):
        joint_prediction_bootstrap(
            image_ids, cells, estimates=specs, metrics=("e_aurc",), n_replicates=2,
            point_order=np.asarray([0, 0, 2, 3]),
        )


def test_native_sentence_id_canonical_index_keeps_ref_and_image_rows_attached() -> None:
    raw_sentence_id = np.asarray([30, 10, 20])
    raw_ref_id = np.asarray([300, 100, 200])
    raw_image_id = np.asarray([33, 11, 22])
    canonical = np.asarray([10, 20, 30])
    order = _canonical_indices(raw_sentence_id, canonical, label="synthetic P archive")
    assert np.array_equal(raw_sentence_id[order], canonical)
    assert np.array_equal(raw_ref_id[order], np.asarray([100, 200, 300]))
    assert np.array_equal(raw_image_id[order], np.asarray([11, 22, 33]))
    with pytest.raises(ValueError, match="duplicate sentence IDs"):
        _canonical_indices(np.asarray([10, 10, 20]), canonical, label="duplicate synthetic archive")


def test_estimate_summary_records_signed_formula_scale_and_units() -> None:
    relative = _estimate_spec_metadata({
        "name": "relative_worsening", "metric": "e_aurc", "operation": "relative_contrast",
        "contrast_positive": "K50", "contrast_negative": "K5", "denominator_condition": "K5",
    })
    assert relative["operation"] == "relative_contrast"
    assert relative["scale"] == 1.0
    assert relative["formula"] == "1 * (e_aurc(K50) - e_aurc(K5)) / e_aurc(K5)"
    assert relative["unit"] == "relative_fraction"

    rer_gain = _estimate_spec_metadata({
        "name": "rer_gain_pp", "metric": "rer_at_50", "operation": "difference",
        "a": "hard_E1b", "b": "hard_R1", "scale": 100.0,
    })
    assert rer_gain["formula"] == "100 * (rer_at_50(hard_E1b) - rer_at_50(hard_R1))"
    assert rer_gain["unit"] == "percentage_points"


def test_cohort_metadata_distinguishes_validation_pools_from_test_only() -> None:
    validation_pool = _cohort_metadata_from_counts(
        {"val_select": 20, "val_calib": 10, "testA": 30, "testB": 40},
        historical_pool_identity="legacy all-common",
    )
    test_pool = _cohort_metadata_from_counts(
        {"testA": 30, "testB": 40}, historical_pool_identity="pooled test",
    )
    external = _cohort_metadata_from_counts(
        {"external_strict_test": 15}, historical_pool_identity="external set", heldout_test_only=True,
    )
    assert validation_pool["contains_validation"] is True
    assert validation_pool["heldout_test_only"] is False
    assert test_pool["contains_validation"] is False
    assert test_pool["heldout_test_only"] is True
    assert external["contains_validation"] is False
    assert external["heldout_test_only"] is True


def test_scaled_differences_and_relative_difference_in_differences_are_explicit() -> None:
    values = {
        "hard_semantic::rer_at_50": 0.61,
        "hard_stats::rer_at_50": 0.56,
        "hard_stats::e_aurc": 0.04,
        "hard_semantic::e_aurc": 0.03,
        "rand_stats::e_aurc": 0.08,
        "rand_semantic::e_aurc": 0.04,
    }
    scaled = {
        "metric": "rer_at_50", "operation": "difference", "a": "hard_semantic",
        "b": "hard_stats", "scale": 100.0,
    }
    assert _estimate_value(values, scaled) == pytest.approx(5.0)
    relative_dod = {
        "metric": "e_aurc", "operation": "difference_of_relative_contrasts",
        "positive_a": "hard_stats", "negative_a": "hard_semantic", "denominator_a": "hard_stats",
        "positive_b": "rand_stats", "negative_b": "rand_semantic", "denominator_b": "rand_stats",
        "scale": 100.0,
    }
    # (0.04 - 0.03)/0.04 - (0.08 - 0.04)/0.08 = -25 percentage points.
    assert _estimate_value(values, relative_dod) == pytest.approx(-25.0)
    values["hard_semantic::e_aurc"] = 0.02
    values["rand_semantic::e_aurc"] = 0.06
    # Reversing the cell effects reverses the signed hard-minus-random contrast.
    assert _estimate_value(values, relative_dod) == pytest.approx(25.0)


def test_legacy_relative_difference_in_differences_derivation_matches_point_operation() -> None:
    job = {
        "legacy_points": {
            "hard_stats::e_aurc": {"seed": 0.04},
            "hard_semantic::e_aurc": {"seed": 0.03},
            "rand_stats::e_aurc": {"seed": 0.08},
            "rand_semantic::e_aurc": {"seed": 0.04},
        },
        "estimates": [{
            "name": "hard_minus_rand_reduction",
            "metric": "e_aurc",
            "operation": "difference_of_relative_contrasts",
            "positive_a": "hard_stats", "negative_a": "hard_semantic", "denominator_a": "hard_stats",
            "positive_b": "rand_stats", "negative_b": "rand_semantic", "denominator_b": "rand_stats",
            "scale": 100.0,
        }],
    }
    _derive_legacy_points(job)
    assert job["legacy_points"]["hard_minus_rand_reduction"]["seed"] == pytest.approx(-25.0)
    job["legacy_points"]["hard_semantic::e_aurc"]["seed"] = 0.02
    job["legacy_points"]["rand_semantic::e_aurc"]["seed"] = 0.06
    job["legacy_points"].pop("hard_minus_rand_reduction")
    _derive_legacy_points(job)
    assert job["legacy_points"]["hard_minus_rand_reduction"]["seed"] == pytest.approx(25.0)


def test_joint_bootstrap_calculates_relative_did_and_severity_macro_per_seed() -> None:
    image_ids = np.asarray([1, 1, 2, 3, 3, 4, 5, 5])
    correct = {
        "seed1": np.asarray([1, 0, 1, 0, 0, 1, 0, 1]),
        "seed2": np.asarray([0, 0, 1, 1, 0, 0, 1, 1]),
    }
    conf = {
        "random": np.asarray([0.9, 0.2, 0.8, 0.3, 0.1, 0.7, 0.4, 0.6]),
        "hard": np.asarray([0.8, 0.3, 0.7, 0.2, 0.1, 0.6, 0.5, 0.4]),
        "m0": np.asarray([0.95, 0.9, 0.7, 0.5, 0.3, 0.8, 0.4, 0.2]),
        "m1": np.asarray([0.9, 0.8, 0.6, 0.45, 0.25, 0.65, 0.35, 0.15]),
    }
    seed_cells = {
        seed: {condition: (confidence, seed_correct) for condition, confidence in conf.items()}
        for seed, seed_correct in correct.items()
    }
    specs = [
        {"name": "hard_minus_random", "metric": "auroc_correct", "operation": "difference", "a": "hard", "b": "random"},
        {
            "name": "relative_hard_minus_random",
            "metric": "auroc_correct",
            "operation": "relative_contrast",
            "contrast_positive": "hard",
            "contrast_negative": "random",
            "denominator_condition": "random",
        },
        {
            "name": "two_path_interaction", "metric": "auroc_correct",
            "operation": "difference_of_differences", "a": "m1", "b": "m0", "c": "hard", "d": "random",
        },
        {
            "name": "relative_dod", "metric": "e_aurc", "operation": "difference_of_relative_contrasts",
            "positive_a": "m0", "negative_a": "m1", "denominator_a": "m0",
            "positive_b": "random", "negative_b": "hard", "denominator_b": "random",
        },
        {
            "name": "scaled_rer50_gain_pp", "metric": "rer_at_50", "operation": "difference",
            "a": "hard", "b": "random", "scale": 100.0,
        },
        {"name": "severity_macro", "metric": "e_aurc", "operation": "macro_mean", "conditions": ["m0", "m1"]},
    ]
    observed = joint_prediction_bootstrap(
        image_ids, seed_cells, estimates=specs, metrics=("accuracy", "auroc_correct", "e_aurc", "rer_at_50"),
        n_replicates=31, seed=0,
    )

    clusters = [np.flatnonzero(image_ids == image_id) for image_id in dict.fromkeys(image_ids.tolist())]
    rng = np.random.default_rng(0)
    expected_per_seed = {spec["name"]: {seed: [] for seed in correct} for spec in specs}
    for _ in range(31):
        draw = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(index)] for index in draw])
        for seed, seed_correct in correct.items():
            values = {
                condition: metric_bundle(confidence[rows], seed_correct[rows], metrics=("auroc_correct", "e_aurc"))
                for condition, confidence in conf.items()
            }
            derived = {
                "hard_minus_random": values["hard"]["auroc_correct"] - values["random"]["auroc_correct"],
                "relative_hard_minus_random": relative_effect(
                    values["hard"]["auroc_correct"] - values["random"]["auroc_correct"],
                    values["random"]["auroc_correct"],
                ),
                "two_path_interaction": (values["m1"]["auroc_correct"] - values["m0"]["auroc_correct"])
                - (values["hard"]["auroc_correct"] - values["random"]["auroc_correct"]),
                "relative_dod": relative_effect(
                    values["m0"]["e_aurc"] - values["m1"]["e_aurc"], values["m0"]["e_aurc"],
                ) - relative_effect(
                    values["random"]["e_aurc"] - values["hard"]["e_aurc"], values["random"]["e_aurc"],
                ),
                "scaled_rer50_gain_pp": 100.0 * (
                    metric_bundle(conf["hard"][rows], seed_correct[rows], metrics=("rer_at_50",))["rer_at_50"]
                    - metric_bundle(conf["random"][rows], seed_correct[rows], metrics=("rer_at_50",))["rer_at_50"]
                ),
                "severity_macro": np.mean([values["m0"]["e_aurc"], values["m1"]["e_aurc"]]),
            }
            for name, value in derived.items():
                expected_per_seed[name][seed].append(value)

    for spec in specs:
        name = spec["name"]
        for seed in correct:
            actual = observed["estimates"][name]["per_seed_replicates"][seed]
            assert np.allclose(actual, expected_per_seed[name][seed], equal_nan=True)
        aggregate = observed["estimates"][name]["replicates"]
        expected = np.mean([expected_per_seed[name][seed] for seed in correct], axis=0)
        assert np.allclose(aggregate, expected, equal_nan=True)


def test_joint_bootstrap_macro_difference_is_paired_inside_each_draw_and_seed() -> None:
    image_ids = np.asarray([11, 12, 13, 14])
    confidence = np.full(4, 0.5)
    correctness = {
        "seed1": {
            "m0_lcr": np.asarray([1, 1, 0, 0]),
            "m0_ref": np.asarray([1, 0, 0, 0]),
            "m2_lcr": np.asarray([1, 1, 1, 0]),
            "m2_ref": np.asarray([1, 0, 1, 0]),
        },
        "seed2": {
            "m0_lcr": np.asarray([1, 0, 0, 0]),
            "m0_ref": np.asarray([0, 0, 0, 0]),
            "m2_lcr": np.asarray([1, 1, 0, 0]),
            "m2_ref": np.asarray([1, 0, 0, 0]),
        },
    }
    cells = {
        seed: {name: (confidence, correct) for name, correct in values.items()}
        for seed, values in correctness.items()
    }
    spec = {
        "name": "lcr_minus_ref_severity_macro::accuracy",
        "metric": "accuracy",
        "operation": "macro_difference",
        "conditions_a": ["m0_lcr", "m2_lcr"],
        "conditions_b": ["m0_ref", "m2_ref"],
    }
    result = joint_prediction_bootstrap(
        image_ids, cells, estimates=[spec], metrics=("accuracy",),
        n_replicates=37, seed=19,
    )["estimates"][spec["name"]]
    assert result["seed_estimates"] == pytest.approx({"seed1": 0.25, "seed2": 0.25})
    assert result["point"] == pytest.approx(0.25)

    clusters = [np.flatnonzero(image_ids == image_id) for image_id in dict.fromkeys(image_ids.tolist())]
    rng = np.random.default_rng(19)
    expected = {seed: [] for seed in correctness}
    for _ in range(37):
        draw = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(index)] for index in draw])
        for seed, conditions in correctness.items():
            by_condition = {name: float(values[rows].mean()) for name, values in conditions.items()}
            expected[seed].append(
                0.5 * ((by_condition["m0_lcr"] - by_condition["m0_ref"])
                       + (by_condition["m2_lcr"] - by_condition["m2_ref"]))
            )
    for seed, values in expected.items():
        assert np.allclose(result["per_seed_replicates"][seed], values)
    assert np.allclose(
        result["replicates"],
        (np.asarray(expected["seed1"]) + np.asarray(expected["seed2"])) / 2.0,
    )


def test_zero_error_denominator_is_invalid_for_repair_rer_bootstrap() -> None:
    all_correct = np.ones(4, dtype=np.int8)
    confidence = np.asarray([0.9, 0.8, 0.7, 0.6])
    assert np.isnan(metric_bundle(confidence, all_correct)["rer_at_50"])
    assert metric_bundle(confidence, all_correct, zero_error_rer="legacy_zero")["rer_at_50"] == 0.0
    # Historical scalar convention remains available to legacy readers.
    assert rer_at_coverage(confidence, all_correct, 0.5) == 0.0

    image_ids = np.asarray([10, 11, 12, 13])
    correctness = np.asarray([1, 1, 1, 0])
    result = joint_prediction_bootstrap(
        image_ids,
        {"seed1": {"native": (confidence, correctness)}},
        estimates=[{"name": "rer50", "metric": "rer_at_50", "operation": "condition", "condition": "native"}],
        metrics=("rer_at_50",),
        n_replicates=128,
        seed=4,
    )
    estimate = result["estimates"]["rer50"]
    assert estimate["point"] == pytest.approx(1.0)
    assert estimate["invalid_replicates"] > 0
    assert estimate["valid_replicates"] + estimate["invalid_replicates"] == 128

    point_only = joint_prediction_bootstrap(
        image_ids,
        {"seed1": {"native": (confidence, all_correct)}},
        estimates=[{"name": "rer50", "metric": "rer_at_50", "operation": "condition", "condition": "native"}],
        metrics=("rer_at_50",),
        n_replicates=8,
        seed=4,
    )["estimates"]["rer50"]
    assert point_only["point"] == 0.0  # old observed-point convention
    assert point_only["valid_replicates"] == 0
    assert point_only["invalid_replicates"] == 8


def test_metric_bundle_risk_only_at_50_and_80() -> None:
    confidence = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    correctness = np.asarray([1, 0, 1, 0, 0])

    result = metric_bundle(
        confidence,
        correctness,
        metrics=("risk_at_50", "risk_at_80"),
    )

    assert result == pytest.approx({"risk_at_50": 1.0 / 3.0, "risk_at_80": 0.5})


def test_metric_bundle_risk_and_rer_at_50_and_80() -> None:
    confidence = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    correctness = np.asarray([1, 0, 1, 0, 0])

    result = metric_bundle(
        confidence,
        correctness,
        metrics=("risk_at_50", "rer_at_50", "risk_at_80", "rer_at_80"),
    )

    assert result["risk_at_50"] == pytest.approx(1.0 / 3.0)
    assert result["rer_at_50"] == pytest.approx(4.0 / 9.0)
    assert result["risk_at_80"] == pytest.approx(0.5)
    assert result["rer_at_80"] == pytest.approx(1.0 / 6.0)


def test_metric_bundle_all_correct_keeps_risk_defined_and_rer_policy_local() -> None:
    confidence = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    correctness = np.ones(5, dtype=np.int8)

    risks = metric_bundle(confidence, correctness, metrics=("risk_at_50", "risk_at_80"))
    assert risks == {"risk_at_50": 0.0, "risk_at_80": 0.0}

    with_nan_rer = metric_bundle(
        confidence,
        correctness,
        metrics=("risk_at_50", "rer_at_50", "risk_at_80", "rer_at_80"),
    )
    assert with_nan_rer["risk_at_50"] == 0.0
    assert with_nan_rer["risk_at_80"] == 0.0
    assert np.isnan(with_nan_rer["rer_at_50"])
    assert np.isnan(with_nan_rer["rer_at_80"])

    with_legacy_zero = metric_bundle(
        confidence,
        correctness,
        metrics=("risk_at_50", "rer_at_50", "risk_at_80", "rer_at_80"),
        zero_error_rer="legacy_zero",
    )
    assert with_legacy_zero == {
        "risk_at_50": 0.0,
        "rer_at_50": 0.0,
        "risk_at_80": 0.0,
        "rer_at_80": 0.0,
    }


def test_relative_improvement_of_macro_uses_the_explicit_macro_denominator() -> None:
    values = {
        "static_m0::e_aurc": 0.4, "static_m2::e_aurc": 0.6,
        "adaptive_m0::e_aurc": 0.3, "adaptive_m2::e_aurc": 0.3,
    }
    spec = {
        "name": "macro_eaurc_relative_improvement",
        "metric": "e_aurc", "operation": "relative_contrast_of_macros", "scale": 100.0,
        "contrast_positive_conditions": ["static_m0", "static_m2"],
        "contrast_negative_conditions": ["adaptive_m0", "adaptive_m2"],
    }
    assert _estimate_value(values, spec) == pytest.approx(40.0)
    metadata = _estimate_spec_metadata(spec)
    assert metadata["unit"] == "relative_percent"
    assert "mean(e_aurc over [static_m0, static_m2])" in metadata["formula"]
    zero_denominator = {key: value for key, value in values.items()}
    zero_denominator["static_m0::e_aurc"] = 0.0
    zero_denominator["static_m2::e_aurc"] = 0.0
    assert np.isnan(_estimate_value(zero_denominator, spec))


def test_auxiliary_continuous_features_share_image_draws_and_seedwise_effects() -> None:
    image_ids = np.asarray([7, 7, 8, 9, 10])
    confidence = np.asarray([0.9, 0.7, 0.6, 0.4, 0.2])
    correctness = np.asarray([1, 0, 1, 0, 0])
    cells = {seed: {"prediction": (confidence, correctness)} for seed in ("seed1", "seed2")}
    auxiliary = {
        "seed1": {
            "hard_feature": np.asarray([0.8, 0.6, 0.7, 0.3, 0.2]),
            "rand_feature": np.asarray([0.4, 0.3, 0.4, 0.2, 0.1]),
        },
        "seed2": {
            "hard_feature": np.asarray([0.7, 0.5, 0.6, 0.2, 0.1]),
            "rand_feature": np.asarray([0.3, 0.2, 0.3, 0.1, 0.0]),
        },
    }
    estimates = [{
        "name": "hard_minus_random_feature", "metric": "mean", "operation": "difference",
        "a": "hard_feature", "b": "rand_feature",
    }]
    result = joint_prediction_bootstrap(
        image_ids, cells, estimates=[{
            "name": "prediction_accuracy", "metric": "accuracy", "operation": "condition", "condition": "prediction",
        }], metrics=("accuracy",), auxiliary_cells=auxiliary, auxiliary_estimates=estimates,
        n_replicates=53, seed=17,
    )["estimates"]["hard_minus_random_feature"]
    clusters = [np.flatnonzero(image_ids == image) for image in dict.fromkeys(image_ids.tolist())]
    rng = np.random.default_rng(17)
    expected = {seed: [] for seed in auxiliary}
    for _ in range(53):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(index)] for index in sampled])
        for seed, fields in auxiliary.items():
            expected[seed].append(float(fields["hard_feature"][rows].mean() - fields["rand_feature"][rows].mean()))
    for seed, values in expected.items():
        assert np.allclose(result["per_seed_replicates"][seed], values)
    assert np.allclose(result["replicates"], (np.asarray(expected["seed1"]) + np.asarray(expected["seed2"])) / 2)
    assert result["point"] == pytest.approx(np.mean([
        auxiliary[seed]["hard_feature"].mean() - auxiliary[seed]["rand_feature"].mean()
        for seed in auxiliary
    ]))

