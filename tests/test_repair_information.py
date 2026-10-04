"""Regression coverage for the Research Repair information experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase05  # noqa: E402
from ccg.reliability.data import ReliabilitySplit  # noqa: E402
from ccg.repairs.information import (
    FEATURE_GROUPS,
    Q_NAMES,
    V_NAMES,
    assert_nested_correctness,
    choose_logistic_c,
    feature_block,
    fit_score_deepsets_normalization,
)
from ccg.semantic.features import SEMANTIC_STAT_NAMES


def test_phase05_score_deepsets_logk_scaler_uses_only_training_rows():
    """Both K regimes must contribute their actual train rows to logK fitting."""
    image_ids = np.asarray([101, 102, 103, 104], dtype=np.int64)
    eval_split = np.full(4, "val_calib", dtype="<U10")
    split = ReliabilitySplit(
        seed=20260928,
        train_frac=0.5,
        train_images=np.asarray([101, 102], dtype=np.int64),
        tune_images=np.asarray([103, 104], dtype=np.int64),
    )
    corpus = {}
    for k in (5, 10, 20, 50):
        rng = np.random.default_rng(k)
        scores = rng.normal(size=(4, k)).astype(np.float32)
        corpus[k] = {
            "sentence_id": np.arange(4, dtype=np.int64),
            "image_id": image_ids.copy(),
            "eval_split": eval_split.copy(),
            "target_local": np.zeros(4, dtype=np.int32),
            "scores": scores,
            "correct": (scores.argmax(axis=1) == 0),
        }

    bundle = run_phase05._build_features(
        "b3_seed1", corpus, 1.0, split, max_epochs=1, log=lambda _: None
    )
    fit = bundle["sds"]["normalisation"]["log_k_fit"]
    expected = np.log(np.asarray([5.0, 5.0, 10.0, 10.0]))

    assert fit["mean"][0] == np.mean(expected)
    assert fit["std"][0] == np.std(expected, ddof=0)
    assert fit["std"][0] > 0.0


def test_information_feature_groups_use_the_frozen_eight_eight_split():
    """Q and V are fixed disjoint eight-column blocks; Full retains source order."""
    assert Q_NAMES == (
        "clip_top1", "clip_top2", "clip_margin12", "clip_entropy", "clip_normH",
        "clip_rank_top1", "q_top3", "q_margin13",
    )
    assert V_NAMES == (
        "cand_vmax", "cand_vmean", "cand_vstd", "cand_top12_sim", "cand_top15_mean",
        "density_070", "density_080", "cand_top13_sim",
    )
    stats = np.arange(34, dtype=np.float64).reshape(2, 17)
    semantic = np.arange(32, dtype=np.float64).reshape(2, 16)
    assert feature_block(stats, semantic, "S").shape == (2, 17)
    assert feature_block(stats, semantic, "S+Q").shape == (2, 25)
    assert feature_block(stats, semantic, "S+V").shape == (2, 25)
    assert feature_block(stats, semantic, "Full").shape == (2, 33)
    assert FEATURE_GROUPS["Full"][-16:] == tuple(SEMANTIC_STAT_NAMES)


def test_logistic_tie_break_chooses_the_earlier_simpler_c():
    """A C candidate within 0.002 AUROC of the best prefers smaller C."""
    assert choose_logistic_c({0.1: 0.810, 1.0: 0.811, 10.0: 0.809}) == 0.1
    assert choose_logistic_c({0.1: 0.808, 1.0: 0.811, 10.0: 0.809}) == 1.0


def test_score_deepsets_logk_fit_uses_both_actual_train_slices():
    """The corrected score-model transform fits both K values, not a K5 prefix."""
    masks = {
        5: np.asarray([True, False, True, False]),
        10: np.asarray([False, True, True, False]),
    }
    scores = {
        5: np.arange(20, dtype=np.float32).reshape(4, 5),
        10: np.arange(40, dtype=np.float32).reshape(4, 10),
    }
    fit = fit_score_deepsets_normalization(scores, masks, (5, 10))
    expected_logk = np.log(np.asarray([5.0, 5.0, 10.0, 10.0]))
    assert fit["log_k_fit"].mean[0] == np.mean(expected_logk)
    assert fit["log_k_fit"].std[0] == np.std(expected_logk, ddof=0)


def test_nested_candidate_correctness_may_degrade_but_cannot_recover():
    """K expansion permits 1->0 and 0->0, but not 0->1 for aligned rows."""
    assert_nested_correctness({
        5: np.asarray([1, 1, 0], dtype=bool),
        50: np.asarray([0, 1, 0], dtype=bool),
    })
    try:
        assert_nested_correctness({
            5: np.asarray([1, 0, 0], dtype=bool),
            50: np.asarray([0, 1, 0], dtype=bool),
        })
    except AssertionError as exc:
        assert "0->1" in str(exc)
    else:
        raise AssertionError("nested K expansion incorrectly accepted a 0->1 correctness flip")

