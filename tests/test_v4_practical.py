"""Synthetic checks for the V4 practical-utility metric contract."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccg.v4.practical import (
    AURC_VERSION,
    DEFAULT_COVERAGES,
    aurc_v4,
    e_aurc_v4,
    fractional_risk_curve,
    high_low_auroc_effect,
    image_draw_row_multiplicities,
    oracle_aurc_v4,
    paired_aurc_effect,
    pairwise_credit_counts,
    pairwise_transition_counts,
    practical_profile,
    weighted_auroc,
)


def test_aurc_analytic_integral_with_mixed_tie_groups():
    scores = [0.9, 0.9, 0.5, 0.1]
    correct = [1, 0, 1, 0]
    expected = (1.0 + 1.0 * np.log(3 / 2) + (1.0 - 3.0) * np.log(4 / 3)) / 4
    # The first group contributes p*g = 1 before normalization.
    expected += 1 / 4
    assert aurc_v4(scores, correct) == pytest.approx(expected)

    curve = fractional_risk_curve(scores, correct, [0.25, 0.5, 1.0])
    assert curve["risks"]["0.25"] == pytest.approx(0.5)
    assert curve["risks"]["0.5"] == pytest.approx(0.5)
    assert curve["risks"]["1.0"] == pytest.approx(0.5)
    assert curve["coverage_zero"]["risk"] is None
    assert curve["coverage_zero"]["right_limit"] == pytest.approx(0.5)
    assert curve["coverage_zero"]["status"] == "UNDEFINED_AT_ZERO"


def test_perfect_and_constant_rankings_match_closed_forms():
    correct = [1, 1, 1, 0, 0]
    accuracy = 3 / 5
    assert aurc_v4([0.9, 0.8, 0.7, 0.2, 0.1], correct) == pytest.approx(
        oracle_aurc_v4(accuracy)
    )
    assert aurc_v4(np.ones(5), correct) == pytest.approx(2 / 5)
    assert oracle_aurc_v4(0.0) == 1.0
    assert oracle_aurc_v4(1.0) == 0.0
    with pytest.raises(ValueError, match="accuracy"):
        oracle_aurc_v4(1.01)


def test_weighted_metrics_equal_explicit_image_cluster_resampling():
    images = np.array(["img-a", "img-a", "img-b", "img-c", "img-c"])
    sampled = np.array(["img-c", "img-a", "img-c", "img-b", "img-a"])
    weights = image_draw_row_multiplicities(images, sampled)
    assert weights.tolist() == [2, 2, 1, 2, 2]

    full = np.array([0.95, 0.5, 0.7, 0.4, 0.2])
    sq = np.array([0.7, 0.6, 0.2, 0.9, 0.1])
    labels = np.array([1, 0, 0, 1, 0])
    expanded = np.repeat(np.arange(labels.size), weights)

    weighted = practical_profile(full, labels, multiplicities=weights)
    explicit = practical_profile(full[expanded], labels[expanded])
    assert weighted["aurc_v4"] == pytest.approx(explicit["aurc_v4"])
    assert weighted["auroc_correct"] == pytest.approx(explicit["auroc_correct"])
    assert weighted["pair_counts"]["pair_count"] == explicit["pair_counts"]["pair_count"]
    for coverage in DEFAULT_COVERAGES:
        key = str(float(coverage))
        assert weighted["risk_by_coverage"][key] == pytest.approx(
            explicit["risk_by_coverage"][key]
        )

    weighted_pair = pairwise_transition_counts(
        full, sq, labels, multiplicities=weights, block_size=1
    )
    explicit_pair = pairwise_transition_counts(
        full[expanded], sq[expanded], labels[expanded], block_size=2
    )
    assert weighted_pair["pair_count"] == explicit_pair["pair_count"]
    assert weighted_pair["transitions"] == explicit_pair["transitions"]
    assert weighted_pair["net_credit_full_minus_sq"] == pytest.approx(
        explicit_pair["net_credit_full_minus_sq"]
    )


def test_eaurc_difference_cancels_for_the_same_labels():
    labels = np.array([1, 0, 1, 0, 1, 0])
    full = np.array([0.9, 0.1, 0.7, 0.8, 0.4, 0.3])
    sq = np.array([0.2, 0.8, 0.6, 0.4, 0.3, 0.5])
    contrast = paired_aurc_effect(full, sq, labels)
    assert contrast["delta_e_aurc_v4_full_minus_sq"] == pytest.approx(
        contrast["delta_aurc_full_minus_sq"], abs=1e-12
    )
    assert contrast["delta_auroc_full_minus_sq"] == pytest.approx(
        weighted_auroc(full, labels) - weighted_auroc(sq, labels)
    )
    assert practical_profile(full, labels)["aurc_version"] == AURC_VERSION


def _brute_transition(full, sq, labels, weights=None):
    weights = np.ones(len(labels), dtype=int) if weights is None else np.asarray(weights)
    matrix = np.zeros((3, 3), dtype=int)
    for i in range(len(labels)):
        if not labels[i] or weights[i] == 0:
            continue
        for j in range(len(labels)):
            if labels[j] or weights[j] == 0:
                continue
            sq_state = int(np.sign(sq[i] - sq[j])) + 1
            full_state = int(np.sign(full[i] - full[j])) + 1
            matrix[sq_state, full_state] += weights[i] * weights[j]
    return matrix


def test_blocked_transition_matrix_matches_explicit_pair_enumeration():
    full = np.array([0.2, 0.9, 0.8, 0.1, 0.5, 0.5])
    sq = np.array([0.8, 0.3, 0.6, 0.4, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0, 1, 0])
    weights = np.array([2, 1, 3, 1, 1, 2])
    matrix = _brute_transition(full, sq, labels, weights)

    result = pairwise_transition_counts(
        full, sq, labels, multiplicities=weights, block_size=1
    )
    observed = np.array(
        [
            [result["transition_matrix"][row][column] for column in ("loss", "tie", "win")]
            for row in ("loss", "tie", "win")
        ]
    )
    assert np.array_equal(observed, matrix)
    label_mask = labels.astype(bool)
    assert result["pair_count"] == int(
        weights[label_mask].sum() * weights[~label_mask].sum()
    )
    assert sum(result["transitions"].values()) == result["pair_count"]
    assert result["models"]["Full"]["auroc"] == pytest.approx(
        pairwise_credit_counts(full, labels, multiplicities=weights)["auroc"]
    )
    assert result["models"]["S+Q"]["auroc"] == pytest.approx(
        pairwise_credit_counts(sq, labels, multiplicities=weights)["auroc"]
    )
    assert result["net_credit_full_minus_sq"] == pytest.approx(
        result["models"]["Full"]["credit"] - result["models"]["S+Q"]["credit"]
    )
    assert result["net_credit_per_10000_pairs"] == pytest.approx(
        result["net_credit_full_minus_sq"] / result["pair_count"] * 10000
    )


def test_single_class_auc_is_undefined_but_aurc_remains_defined():
    profile = practical_profile([0.8, 0.1, 0.5], [1, 1, 1])
    assert profile["auroc_correct"] is None
    assert profile["pair_counts"]["pair_count"] == 0
    assert profile["aurc_v4"] == 0.0
    assert profile["e_aurc_v4"] == 0.0
    assert pairwise_transition_counts([0.8, 0.1], [0.1, 0.8], [1, 1])["pair_count"] == 0

    empty = practical_profile([], [])
    assert empty["aurc_v4"] is None
    assert empty["auroc_correct"] is None
    assert empty["aurc_right_limit_at_zero"]["right_limit"] is None


def test_high_low_effect_uses_caller_group_mask_and_marks_degenerate_auc():
    labels = np.array([1, 0, 1, 0])
    full = np.array([0.9, 0.1, 0.2, 0.8])
    sq = np.array([0.8, 0.2, 0.7, 0.3])
    groups = high_low_auroc_effect(full, sq, labels, [1, 1, 0, 0])
    assert groups["high"]["delta_auroc_full_minus_sq"] == pytest.approx(0.0)
    assert groups["low"]["delta_auroc_full_minus_sq"] == pytest.approx(-1.0)
    assert groups["high_minus_low_delta_auroc"] == pytest.approx(1.0)

    degenerate = high_low_auroc_effect(full, sq, labels, [1, 0, 1, 0])
    assert degenerate["high"]["delta_auroc_full_minus_sq"] is None
    assert degenerate["high_minus_low_delta_auroc"] is None


@pytest.mark.parametrize(
    "scores,labels,weights",
    [
        ([0.1], [2], None),
        ([0.1], [1], [-1]),
        ([0.1], [1], [0.5]),
        ([float("nan")], [1], None),
    ],
)
def test_input_validation(scores, labels, weights):
    with pytest.raises(ValueError):
        aurc_v4(scores, labels, multiplicities=weights)


def test_image_draw_requires_existing_image_ids():
    with pytest.raises(ValueError, match="absent"):
        image_draw_row_multiplicities(["a", "a", "b"], ["a", "missing"])

