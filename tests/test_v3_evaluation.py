from __future__ import annotations

import numpy as np
import pytest

from ccg.v3.evaluation import (
    correctness_reliability,
    fractional_boundary_risk,
    primary_point_estimates,
    summarize_prediction_rows,
)


def test_fractional_boundary_ties_are_order_invariant_and_cover_exact_budget() -> None:
    confidence = np.asarray([1.0, 1.0, 0.0, 0.0])
    correct = np.asarray([1, 0, 1, 0])
    assert fractional_boundary_risk(confidence, correct, 0.25) == pytest.approx(0.5)
    assert fractional_boundary_risk(confidence, correct, 0.5) == pytest.approx(0.5)
    permutation = np.asarray([3, 1, 0, 2])
    assert fractional_boundary_risk(confidence[permutation], correct[permutation], 0.5) == pytest.approx(0.5)


def test_fractional_boundary_risk_all_correct_all_wrong_and_empty() -> None:
    confidence = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert fractional_boundary_risk(confidence, np.ones(4, dtype=bool), 0.8) == 0.0
    assert fractional_boundary_risk(confidence, np.zeros(4, dtype=bool), 0.8) == 1.0
    assert np.isnan(fractional_boundary_risk([], [], 0.5))
    assert correctness_reliability(confidence, np.ones(4, bool))["auroc_correct"] is None
    assert correctness_reliability(confidence, np.zeros(4, bool))["auroc_correct"] is None
    assert correctness_reliability([], [])["n"] == 0


@pytest.mark.parametrize("labels", [[np.nan, 1], [2, 0], [None, 1]])
def test_correctness_rejects_nan_and_nonbinary_inputs(labels: list[object]) -> None:
    with pytest.raises(ValueError, match="correctness"):
        fractional_boundary_risk([0.1, 0.2], labels, 0.5)


def test_summary_keeps_natural_flow_and_reports_paired_missing_denominators() -> None:
    rows = [
        {
            "mode": "natural", "backbone": "b0", "seed": 1, "requested_k": 5, "k_eff": 5,
            "target_present": True, "bank_target_coverage": True, "correct": True,
            "confidence": {"Full": 0.9, "S+Q": 0.8},
        },
        {
            "mode": "natural", "backbone": "b0", "seed": 1, "requested_k": 5, "k_eff": 4,
            "target_present": False, "bank_target_coverage": True, "correct": False,
            "confidence": {},
        },
        {
            "mode": "natural", "backbone": "b0", "seed": 1, "requested_k": 5, "k_eff": 0,
            "target_present": False, "bank_target_coverage": False, "correct": None,
            "confidence": {},
        },
    ]
    summary = summarize_prediction_rows(rows)["5"]
    assert summary["n_rows"] == 3
    assert summary["n_basic_grounding_only"] == 1
    assert summary["n_empty_candidates"] == 1
    assert summary["n_reliability_eligible_k_ge_5"] == 1
    assert summary["target_coverage"] == pytest.approx(1 / 3)
    assert summary["bank_target_coverage"] == pytest.approx(2 / 3)
    assert summary["all_rows_grounding_accuracy"] == pytest.approx(1 / 3)
    assert summary["paired_model_common_rows"]["Full_vs_S+Q"]["n_common_rows"] == 1

    partial = [
        {
            "mode": "natural", "backbone": "b0", "seed": 1, "requested_k": 5, "k_eff": 5,
            "target_present": True, "correct": True,
            "confidence": {"Full": None, "S+Q": 0.2},
            "confidence_missing_reason": {"Full": "model_unavailable"},
        },
        {
            "mode": "natural", "backbone": "b0", "seed": 1, "requested_k": 5, "k_eff": 5,
            "target_present": False, "correct": False,
            "confidence": {"Full": 0.1, "S+Q": 0.3},
        },
    ]
    paired = summarize_prediction_rows(partial)["5"]["paired_model_common_rows"]["Full_vs_S+Q"]
    assert paired["n_common_rows"] == 1
    assert paired["left_missing_confidence_rows"] == 1
    assert paired["right_missing_confidence_rows"] == 0


def _primary_inputs(b16_labels: np.ndarray) -> tuple[dict, dict]:
    seeds = ("b3_seed1", "b3_seed2", "b3_seed3")
    confidences = {}
    correctness = {}
    for seed in seeds:
        confidences[seed] = {
            "B0_MSP": {
                5: np.asarray([0.9, 0.8, 0.2, 0.1]),
                50: np.asarray([0.1, 0.2, 0.8, 0.9]),
            },
            "B0_Full": {50: np.asarray([0.1, 0.2, 0.8, 0.9])},
            "B0_SQ": {50: np.asarray([0.9, 0.8, 0.2, 0.1])},
            "B16_Full": {50: np.asarray([0.9, 0.8, 0.2, 0.1])},
            "B16_SQ": {50: np.asarray([0.1, 0.2, 0.8, 0.9])},
            "B0_SDS_large": {50: np.asarray([0.9, 0.8, 0.2, 0.1])},
            "B0_SDS_small": {50: np.asarray([0.1, 0.2, 0.8, 0.9])},
        }
        correctness[seed] = {
            "b0": {5: np.asarray([False, False, True, True]), 50: np.asarray([False, False, True, True])},
            "b16": {5: b16_labels, 50: b16_labels},
        }
    return confidences, correctness


def test_c3_uses_b16_correctness_not_shared_b0_label_and_undefined_auc_is_null() -> None:
    # B0's labels are the complement of B16's labels. Reusing B0 here flips
    # the Full-SQ C3 effect from +1 to -1 despite identical candidate rows.
    b16_labels = np.asarray([True, True, False, False])
    confidence, correctness = _primary_inputs(b16_labels)
    estimate = primary_point_estimates(
        image_ids=[1, 1, 2, 3],
        common_mask=[True, True, True, True],
        confidence_by_seed=confidence,
        correctness_by_seed=correctness,
    )
    name = "C3_B16_Full_minus_SQ_AUROC_K50"
    assert estimate["point_contrasts"][name] == pytest.approx(1.0)

    one_class_confidence, one_class_correctness = _primary_inputs(np.ones(4, dtype=bool))
    undefined = primary_point_estimates(
        image_ids=[1, 2, 3, 4],
        common_mask=[True] * 4,
        confidence_by_seed=one_class_confidence,
        correctness_by_seed=one_class_correctness,
    )
    assert undefined["point_contrasts"][name] is None
    assert undefined["undefined_seed_counts"][name] == 3
