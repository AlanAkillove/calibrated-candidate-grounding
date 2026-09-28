"""Tests for the Phase 1 candidate-semantic reliability models (protocol A7.3).

Covers the frozen E2 (:class:`TopCompetitorModel`) and E3
(:class:`SemanticDeepSets`) model zoo: shapes / range / determinism, parameter
budgets, variant handling, permutation invariance, variable-``K`` padding
invariance, rank / top-1 feature correctness and the training-loop contract.

All tests run on CPU with tiny epochs / batches so the whole module stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.semantic.models import SemanticDeepSets, TopCompetitorModel

EMB = 512


# --------------------------------------------------------------------------- #
# synthetic data builders
# --------------------------------------------------------------------------- #
def _e2_inputs(n: int, seed: int):
    """Build a synthetic E2 batch with a learnable ``margin12 > 0`` signal."""
    rng = np.random.default_rng(seed)
    z_q = rng.standard_normal((n, EMB)).astype(np.float32)
    z_top1 = rng.standard_normal((n, EMB)).astype(np.float32)
    z_top2 = rng.standard_normal((n, EMB)).astype(np.float32)
    sim1 = np.einsum("nd,nd->n", z_q, z_top1)
    sim2 = np.einsum("nd,nd->n", z_q, z_top2)
    top1_score = (sim1 + rng.standard_normal(n) * 0.5).astype(np.float32)
    top2_score = (sim2 + rng.standard_normal(n) * 0.5).astype(np.float32)
    margin12 = (top1_score - top2_score).astype(np.float32)
    msp = (1.0 / (1.0 + np.exp(-np.clip(margin12, -30.0, 30.0)))).astype(np.float32)
    score_stats = np.stack([top1_score, top2_score, margin12, msp], axis=1).astype(np.float32)
    inputs = {
        "z_q": z_q,
        "z_top1": z_top1,
        "z_top2": z_top2,
        "score_stats": score_stats,
    }
    y = (margin12 > 0).astype(np.float32)
    return inputs, y


def _e3_inputs(n: int, k: int, seed: int, *, logk: bool = False):
    """Build a synthetic E3 batch (all candidates valid) with a score signal."""
    rng = np.random.default_rng(seed)
    z_q = rng.standard_normal((n, EMB)).astype(np.float32)
    z_i = rng.standard_normal((n, k, EMB)).astype(np.float32)
    mask = np.ones((n, k), dtype=bool)
    score = rng.standard_normal((n, k)).astype(np.float32)
    prob = rng.random((n, k)).astype(np.float32)
    top1 = score.max(axis=1)
    y = (top1 > np.median(top1)).astype(np.float32)
    inputs = {"z_q": z_q, "z_i": z_i, "mask": mask, "score": score, "prob": prob}
    if logk:
        inputs["log_k"] = (np.log(k) + rng.standard_normal(n) * 0.05).astype(np.float32)
    return inputs, y


# --------------------------------------------------------------------------- #
# E2 -- TopCompetitorModel
# --------------------------------------------------------------------------- #
def test_e2_fit_predict_range_and_determinism():
    """E2 predicts ``[n]`` probabilities in range and is seed-reproducible."""
    inputs, y = _e2_inputs(256, seed=0)
    m1 = TopCompetitorModel(seed=11, device="cpu")
    res1 = m1.fit(inputs, y, lr=1e-3, epochs=5, batch_size=128)
    p1 = m1.predict_proba(inputs)
    assert p1.shape == (256,)
    assert p1.dtype == np.float64
    assert np.all(p1 >= 0.0) and np.all(p1 <= 1.0)

    m2 = TopCompetitorModel(seed=11, device="cpu")
    m2.fit(inputs, y, lr=1e-3, epochs=5, batch_size=128)
    p2 = m2.predict_proba(inputs)
    assert np.array_equal(p1, p2)

    for key in (
        "epochs_run",
        "best_epoch",
        "best_val_bce",
        "stopped_early",
        "n_parameters",
        "seed",
        "lr",
        "include_score",
        "include_semantic",
    ):
        assert key in res1


def test_e2_parameter_budget_and_manual_count():
    """E2 stays below the 150k budget and matches the hand-computed count."""
    m = TopCompetitorModel(
        seed=1, proj_dim=64, hidden=128, include_score=True, include_semantic=True, device="cpu"
    )
    assert m.n_parameters() < 150_000

    expected = 2 * (EMB * 64 + 64)  # P_q and the shared P_v
    expected += (7 * 64 + 4) * 128 + 128  # head first linear (448 interaction + 4 score)
    expected += 128 * 1 + 1  # head second linear
    assert m.n_parameters() == expected == 123_777


def test_e2_score_only_variant_and_errors():
    """E2-score fits/predicts and invalid configurations raise."""
    n = 64
    rng = np.random.default_rng(2)
    score_stats = rng.standard_normal((n, 4)).astype(np.float32)
    y = (score_stats[:, 2] > 0).astype(np.float32)

    m = TopCompetitorModel(seed=2, include_score=True, include_semantic=False, device="cpu")
    m.fit({"score_stats": score_stats}, y, lr=1e-3, epochs=3, batch_size=32)
    p = m.predict_proba({"score_stats": score_stats})
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))

    semantic = TopCompetitorModel(seed=2, include_score=True, include_semantic=True, device="cpu")
    with pytest.raises((KeyError, ValueError)):
        semantic.fit({"score_stats": score_stats}, y, lr=1e-3, epochs=1)

    with pytest.raises(ValueError):
        TopCompetitorModel(seed=2, include_score=False, include_semantic=False, device="cpu")


def test_e2_fit_fields_and_early_stopping():
    """E2 reports the A7.3 fields, stops early and handles the no-val path."""
    inputs, _ = _e2_inputs(256, seed=4)
    val_inputs, _ = _e2_inputs(128, seed=404)
    rng = np.random.default_rng(4)
    y_rand = (rng.random(256) < 0.5).astype(np.float32)  # no learnable signal
    y_val = (rng.random(128) < 0.5).astype(np.float32)

    m = TopCompetitorModel(seed=8, device="cpu")
    res = m.fit(
        inputs,
        y_rand,
        lr=1e-2,
        inputs_val=val_inputs,
        y_val=y_val,
        epochs=40,
        batch_size=256,
        patience=2,
    )
    assert set(res) == {
        "epochs_run",
        "best_epoch",
        "best_val_bce",
        "stopped_early",
        "n_parameters",
        "seed",
        "lr",
        "include_score",
        "include_semantic",
    }
    assert res["stopped_early"] is True
    assert res["epochs_run"] < 40
    assert res["best_val_bce"] is not None
    assert res["best_epoch"] <= res["epochs_run"]

    m2 = TopCompetitorModel(seed=8, device="cpu")
    res2 = m2.fit(inputs, y_rand, lr=1e-2, epochs=5, batch_size=256)
    assert res2["best_val_bce"] is None
    assert res2["epochs_run"] == 5
    assert res2["best_epoch"] == 5
    assert res2["stopped_early"] is False


# --------------------------------------------------------------------------- #
# E3 -- SemanticDeepSets
# --------------------------------------------------------------------------- #
def test_e3_permutation_invariance():
    """Permuting z_i / score / prob / mask together leaves the output invariant."""
    inputs, y = _e3_inputs(64, 7, seed=3)
    m = SemanticDeepSets(seed=5, device="cpu")
    m.fit(inputs, y, lr=1e-3, epochs=4, batch_size=32)
    p1 = m.predict_proba(inputs)

    perm = np.random.default_rng(9).permutation(7)
    permuted = {
        "z_q": inputs["z_q"],
        "z_i": inputs["z_i"][:, perm],
        "mask": inputs["mask"][:, perm],
        "score": inputs["score"][:, perm],
        "prob": inputs["prob"][:, perm],
    }
    p2 = m.predict_proba(permuted)
    assert p1.shape == p2.shape == (64,)
    assert np.allclose(p1, p2, atol=1e-6)


def test_e3_variable_k_padding_invariance():
    """A K=5 subset padded to K=10 matches the same rows predicted with no padding."""
    rng = np.random.default_rng(21)
    n5, n10, big_k = 32, 32, 10
    n = n5 + n10
    z_q = rng.standard_normal((n, EMB)).astype(np.float32)
    z_i = rng.standard_normal((n, big_k, EMB)).astype(np.float32)
    mask = np.zeros((n, big_k), dtype=bool)
    mask[:n5, :5] = True
    mask[n5:, :] = True
    score = rng.standard_normal((n, big_k)).astype(np.float32)
    prob = rng.random((n, big_k)).astype(np.float32)
    z_i = np.where(mask[..., None], z_i, 0.0).astype(np.float32)
    score = np.where(mask, score, 0.0).astype(np.float32)
    prob = np.where(mask, prob, 0.0).astype(np.float32)
    inputs = {"z_q": z_q, "z_i": z_i, "mask": mask, "score": score, "prob": prob}
    top1 = np.where(mask, score, -np.inf).max(axis=1)
    y = (top1 > np.median(top1)).astype(np.float32)

    m = SemanticDeepSets(seed=6, device="cpu")
    m.fit(inputs, y, lr=1e-3, epochs=4, batch_size=32)

    rows = np.arange(n5)
    padded = {k: v[rows] for k, v in inputs.items()}
    compact = {
        "z_q": z_q[rows],
        "z_i": z_i[rows, :5],
        "mask": mask[rows, :5],
        "score": score[rows, :5],
        "prob": prob[rows, :5],
    }
    p_pad = m.predict_proba(padded)
    p_cmp = m.predict_proba(compact)
    assert p_pad.shape == p_cmp.shape == (n5,)
    assert np.allclose(p_pad, p_cmp, atol=1e-6)


def test_e3_candidate_features_rank_and_top1():
    """``candidate_features`` encodes ``1/rank``, a single top-1 and zeroed padding."""
    inputs, _ = _e3_inputs(8, 7, seed=11)
    mask = inputs["mask"].copy()
    mask[0, 6] = False
    mask[3, 1] = False
    score = np.where(mask, inputs["score"], 0.0).astype(np.float32)
    prob = np.where(mask, inputs["prob"], 0.0).astype(np.float32)
    z_i = np.where(mask[..., None], inputs["z_i"], 0.0).astype(np.float32)

    m = SemanticDeepSets(seed=3, proj_dim=64, device="cpu")
    h = m.candidate_features(inputs["z_q"], z_i, mask, score, prob)
    assert h.shape == (8, 7, 132)

    rank_col = h[..., 130]
    top1_col = h[..., 131]
    for r in range(8):
        valid = np.nonzero(mask[r])[0]
        order = valid[np.argsort(-score[r, valid], kind="stable")]
        for pos, idx in enumerate(order):
            assert rank_col[r, idx] == pytest.approx(1.0 / (pos + 1), abs=1e-6)
        assert top1_col[r].sum() == pytest.approx(1.0)
        assert top1_col[r, order[0]] == 1.0
        for c in range(7):
            if not mask[r, c]:
                assert np.all(h[r, c] == 0.0)


def test_e3_parameter_budget():
    """E3 stays below the 250k budget; ``include_logk`` adds one head input unit."""
    m = SemanticDeepSets(seed=1, proj_dim=64, phi_hidden=128, head_hidden=128, device="cpu")
    assert m.n_parameters() < 250_000
    m_logk = SemanticDeepSets(
        seed=1, proj_dim=64, phi_hidden=128, head_hidden=128, include_logk=True, device="cpu"
    )
    assert m_logk.n_parameters() == m.n_parameters() + 128


def test_e3_logk_required_and_trainable():
    """``include_logk=True`` requires ``log_k`` and trains once it is supplied."""
    inputs, y = _e3_inputs(32, 5, seed=12)
    m = SemanticDeepSets(seed=1, include_logk=True, device="cpu")
    with pytest.raises(ValueError):
        m.fit(inputs, y, lr=1e-3, epochs=2, batch_size=32)

    inputs_logk, y_logk = _e3_inputs(32, 5, seed=12, logk=True)
    res = m.fit(inputs_logk, y_logk, lr=1e-3, epochs=3, batch_size=32)
    assert res["include_logk"] is True
    p = m.predict_proba(inputs_logk)
    assert p.shape == (32,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_e3_mask_accepts_bool_or_int():
    """A 0/1 mask is equivalent to the boolean mask."""
    inputs, y = _e3_inputs(16, 6, seed=13)
    m = SemanticDeepSets(seed=2, device="cpu")
    m.fit(inputs, y, lr=1e-3, epochs=2, batch_size=16)
    p_bool = m.predict_proba(inputs)
    inputs_int = dict(inputs)
    inputs_int["mask"] = inputs["mask"].astype(np.int64)
    p_int = m.predict_proba(inputs_int)
    assert np.array_equal(p_bool, p_int)


def test_to_dict_kinds():
    """``to_dict`` reports the frozen model kind and parameter count."""
    e2 = TopCompetitorModel(seed=1, device="cpu")
    assert e2.to_dict()["kind"] == "top_competitor"
    assert e2.to_dict()["n_parameters"] == e2.n_parameters()

    e3 = SemanticDeepSets(seed=1, device="cpu")
    assert e3.to_dict()["kind"] == "semantic_deepsets"
    assert e3.to_dict()["n_parameters"] == e3.n_parameters()


def test_predict_before_fit_raises():
    """Both models refuse to predict before fitting."""
    inputs_e2, _ = _e2_inputs(8, seed=0)
    with pytest.raises(RuntimeError):
        TopCompetitorModel(seed=1, device="cpu").predict_proba(inputs_e2)

    inputs_e3, _ = _e3_inputs(8, 5, seed=0)
    with pytest.raises(RuntimeError):
        SemanticDeepSets(seed=1, device="cpu").predict_proba(inputs_e3)
