"""Synthetic integrity checks for V3 B16 replication model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.reliability import features as rfeat  # noqa: E402
from ccg.repairs.information import (  # noqa: E402
    FEATURE_GROUPS,
    Q_NAMES,
    V_NAMES,
    feature_block,
    fit_group_normalization,
)
from ccg.semantic.features import SEMANTIC_STAT_NAMES  # noqa: E402
from ccg.v3.evaluation import fractional_boundary_risk  # noqa: E402
from ccg.v3.replication import (  # noqa: E402
    ANCHOR_ATOL,
    GROUPS,
    assert_candidate_identities,
    assert_score_anchor,
    load_model_json,
    portable_model_payload,
)


def test_v3_groups_use_the_frozen_s17_q8_v8_full33_order():
    assert GROUPS == ("S", "S+Q", "S+V", "Full")
    assert Q_NAMES == (
        "clip_top1", "clip_top2", "clip_margin12", "clip_entropy", "clip_normH",
        "clip_rank_top1", "q_top3", "q_margin13",
    )
    assert V_NAMES == (
        "cand_vmax", "cand_vmean", "cand_vstd", "cand_top12_sim", "cand_top15_mean",
        "density_070", "density_080", "cand_top13_sim",
    )
    stats = np.arange(34, dtype=np.float64).reshape(2, 17)
    sem = np.arange(32, dtype=np.float64).reshape(2, 16)
    assert [feature_block(stats, sem, group).shape[1] for group in GROUPS] == [17, 25, 25, 33]
    assert FEATURE_GROUPS["S+Q"][17:] == Q_NAMES
    assert FEATURE_GROUPS["S+V"][17:] == V_NAMES
    assert FEATURE_GROUPS["Full"][-16:] == tuple(SEMANTIC_STAT_NAMES)


def test_group_standardizer_uses_only_the_frozen_training_rows():
    rows = {
        5: np.asarray([[1.0, 10.0], [3.0, 30.0], [10000.0, -10000.0]]),
        10: np.asarray([[5.0, 50.0], [7.0, 70.0], [-10000.0, 10000.0]]),
    }
    train = {5: np.asarray([True, True, False]), 10: np.asarray([True, True, False])}
    fit, standardized = fit_group_normalization(rows, train, (5, 10), ("x", "y"))
    expected = np.vstack((rows[5][:2], rows[10][:2]))
    np.testing.assert_allclose(fit.mean, expected.mean(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(fit.std, expected.std(axis=0), rtol=0, atol=0)
    assert np.all(np.isfinite(standardized[5]))
    assert np.array_equal(fit.keys, ("x", "y"))


def test_candidate_manifest_requires_target_column_zero_and_unique_proposals():
    sentence = np.asarray([11, 12])
    refs = np.asarray([101, 102])
    images = np.asarray([201, 202])
    candidates = np.asarray([[4, 8, 12, 16, 20], [2, 3, 5, 7, 11]])
    targets = candidates[:, 0].copy()
    assert_candidate_identities(sentence, refs, images, candidates, targets, expected_k=5)
    with pytest.raises(AssertionError, match="column 0"):
        assert_candidate_identities(sentence, refs, images, candidates, targets + 1, expected_k=5)
    duplicate = candidates.copy()
    duplicate[0, -1] = duplicate[0, 1]
    with pytest.raises(AssertionError, match="duplicate"):
        assert_candidate_identities(sentence, refs, images, duplicate, targets, expected_k=5)


def test_frozen_scorer_forward_guard_keeps_original_tolerance():
    reference = np.asarray([[1.0, 0.2], [0.1, 0.3]], dtype=np.float32)
    assert assert_score_anchor(reference + ANCHOR_ATOL / 2, reference) <= ANCHOR_ATOL
    with pytest.raises(AssertionError, match="exceeds"):
        assert_score_anchor(reference + ANCHOR_ATOL * 2, reference)


def test_v3_risk_uses_fractional_acceptance_at_a_tied_boundary():
    confidence = np.asarray([0.9, 0.8, 0.8, 0.1])
    correct = np.asarray([False, True, False, True])
    assert fractional_boundary_risk(confidence, correct, 0.5) == pytest.approx(0.75)
    permutation = np.asarray([2, 0, 3, 1])
    assert fractional_boundary_risk(confidence[permutation], correct[permutation], 0.5) == pytest.approx(0.75)


def test_portable_checkpoint_round_trip_preserves_normalization_and_prediction(tmp_path: Path):
    group = "S"
    names = FEATURE_GROUPS[group]
    fit = rfeat.NormalizationFit(
        mean=np.zeros(len(names), dtype=np.float64),
        std=np.ones(len(names), dtype=np.float64),
        keys=tuple(names),
    )
    predictor = {
        "family": "stats_logistic", "chosen_C": 1.0, "tie_tolerance": 0.002,
        "coefficients": [0.01] * len(names), "intercept": -0.2,
    }
    payload = portable_model_payload(
        seed=1, group=group, temperature=1.1711449916929477,
        feature_names=names, normalization=fit, predictor=predictor,
        grid_records=(), train_images=np.arange(523), tune_images=np.arange(1000, 1224),
    )
    path = tmp_path / "S.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded_fit, loaded_predictor, loaded_payload = load_model_json(path)
    matrix = np.ones((2, len(names)), dtype=np.float64)
    np.testing.assert_array_equal(loaded_fit.mean, fit.mean)
    assert loaded_payload["temperature_corrected"] == pytest.approx(1.1711449916929477)
    np.testing.assert_allclose(
        loaded_predictor.predict_proba(matrix),
        1 / (1 + np.exp(-(-0.2 + 0.01 * len(names)))),
        rtol=0, atol=1e-15,
    )
