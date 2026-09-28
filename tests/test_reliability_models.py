"""Tests for :mod:`ccg.reliability.models` (Phase 0.5 frozen zoo, Amendment A6).

The reliability models are deliberately *score-only*: the guards below assert
that no ``fit`` / ``predict_proba`` parameter is named for an embedding,
geometry, objectness or query feature, and that the module never pulls in HDF5
(``h5py``) - a proxy for "only scalar scores are touched".
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from ccg.reliability.models import LogisticModel, ScoreDeepSets, TinyMLP

# --------------------------------------------------------------------------- #
# synthetic data helpers
# --------------------------------------------------------------------------- #
def _separable_2d(n: int = 200, *, seed: int = 0, offset: float = 6.0):
    """Two well-separated Gaussian blobs: positive class at ``+offset``."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pos = rng.normal(loc=offset, scale=1.0, size=(half, 2))
    neg = rng.normal(loc=-offset, scale=1.0, size=(n - half, 2))
    X = np.vstack([pos, neg]).astype(np.float64)
    y = np.concatenate([np.ones(half), np.zeros(n - half)]).astype(np.int64)
    return X, y


def _noisy_logistic(n: int = 800, d: int = 10, *, seed: int = 1, scale: float = 2.0):
    """Learnable Bernoulli labels (so training NLL can drop well below log 2)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    prob = 1.0 / (1.0 + np.exp(-scale * (X @ w)))
    y = (rng.uniform(size=n) < prob).astype(np.float32)
    return X, y


def _score_sets(n: int, K: int, *, seed: int = 0, pad_frac: float = 0.0):
    """Random score-set batch with optional right-side padding."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n, K)).astype(np.float32)
    mask = np.ones((n, K), dtype=bool)
    if pad_frac > 0.0:
        n_pad = int(round(K * pad_frac))
        if n_pad > 0:
            mask[:, K - n_pad :] = False
            scores[:, K - n_pad :] = 0.0
    log_k = rng.normal(size=n).astype(np.float32)
    z_top1 = rng.normal(size=n).astype(np.float32)
    return scores, mask, log_k, z_top1


# --------------------------------------------------------------------------- #
# 1. LogisticModel
# --------------------------------------------------------------------------- #
def test_logistic_separable_margins_budget_and_dict():
    X, y = _separable_2d()
    model = LogisticModel(C=1.0).fit(X, y)

    p = model.predict_proba(X)
    assert p.shape == (X.shape[0],)
    assert p.dtype == np.float64
    assert p[y == 1].mean() > 0.8
    assert p[y == 0].mean() < 0.2

    assert model.n_parameters() == X.shape[1] + 1  # d + intercept

    coef, intercept = model.coefficients()
    assert coef.shape == (X.shape[1],)
    assert isinstance(intercept, float)

    payload = model.to_dict()
    assert {"C", "coef", "intercept"} <= set(payload)
    assert payload["C"] == 1.0
    assert len(payload["coef"]) == X.shape[1]
    assert isinstance(payload["intercept"], float)


def test_logistic_requires_fit_before_use():
    with pytest.raises(RuntimeError):
        LogisticModel().predict_proba(np.zeros((3, 2)))


# --------------------------------------------------------------------------- #
# 2. Parameter budgets
# --------------------------------------------------------------------------- #
def test_parameter_budgets():
    mlp = TinyMLP(17, seed=0)
    n_mlp = mlp.n_parameters()
    ds_logk = ScoreDeepSets(seed=0, include_logk=True)
    ds_no_logk = ScoreDeepSets(seed=0, include_logk=False)
    print(f"[budget] TinyMLP(17)={n_mlp}  ScoreDeepSets(logk)={ds_logk.n_parameters()}  "
          f"ScoreDeepSets(no-logk)={ds_no_logk.n_parameters()}")

    assert 0 < n_mlp < 2000
    assert 0 < ds_logk.n_parameters() < 5000
    assert 0 < ds_no_logk.n_parameters() < 5000


# --------------------------------------------------------------------------- #
# 3. TinyMLP training + early stopping
# --------------------------------------------------------------------------- #
def test_tinymlp_train_nll_decreases_and_proba_in_range():
    X, y = _noisy_logistic()
    model = TinyMLP(X.shape[1], seed=0)
    result = model.fit(X, y, lr=1e-2, epochs=60, batch_size=256, patience=10000)

    history = result["history"]
    assert len(history) >= 2
    assert result["epochs_run"] == 60
    assert result["stopped_early"] is False
    assert history[-1]["train_nll"] < history[0]["train_nll"]

    p = model.predict_proba(X)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert np.all(np.isfinite(p))


def test_tinymlp_early_stop_on_non_improving_val():
    X, y = _noisy_logistic(seed=2)
    rng = np.random.default_rng(99)
    X_val = rng.normal(size=(200, X.shape[1])).astype(np.float32)
    y_val = (rng.uniform(size=200) < 0.5).astype(np.float32)  # pure noise -> val NLL flat

    model = TinyMLP(X.shape[1], seed=0)
    result = model.fit(
        X, y, lr=1e-2, X_val=X_val, y_val=y_val, epochs=300, batch_size=256, patience=3
    )
    assert result["stopped_early"] is True
    assert result["epochs_run"] < 300
    assert result["best_epoch"] >= 1
    assert result["best_val_nll"] is not None


def test_tinymlp_val_requires_y_val():
    X, y = _noisy_logistic(n=64, seed=0)
    with pytest.raises(ValueError):
        TinyMLP(X.shape[1], seed=0).fit(X, y, lr=1e-3, X_val=X)


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #
def test_tinymlp_determinism_same_seed():
    X, y = _noisy_logistic(seed=3)
    a = TinyMLP(X.shape[1], seed=7)
    a.fit(X, y, lr=3e-3, epochs=20, batch_size=128)
    b = TinyMLP(X.shape[1], seed=7)
    b.fit(X, y, lr=3e-3, epochs=20, batch_size=128)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-9, rtol=0.0)


def test_scoreds_determinism_same_seed():
    scores, mask, log_k, z_top1 = _score_sets(64, 8, seed=4)
    y = (np.arange(64) % 2).astype(np.float32)
    a = ScoreDeepSets(seed=11)
    a.fit(scores, mask, log_k, z_top1, y, lr=3e-3, epochs=15, batch_size=32)
    b = ScoreDeepSets(seed=11)
    b.fit(scores, mask, log_k, z_top1, y, lr=3e-3, epochs=15, batch_size=32)
    pa = a.predict_proba(scores, mask, log_k, z_top1)
    pb = b.predict_proba(scores, mask, log_k, z_top1)
    assert np.allclose(pa, pb, atol=1e-9, rtol=0.0)


# --------------------------------------------------------------------------- #
# 5. Permutation invariance
# --------------------------------------------------------------------------- #
def test_scoreds_permutation_invariance_untrained():
    scores, mask, log_k, z_top1 = _score_sets(16, 8, seed=0, pad_frac=0.25)
    model = ScoreDeepSets(seed=0)
    perm = np.random.default_rng(42).permutation(scores.shape[1])

    p0 = model.predict_proba(scores, mask, log_k, z_top1)
    p1 = model.predict_proba(scores[:, perm], mask[:, perm], log_k, z_top1)
    assert np.allclose(p0, p1, atol=1e-6, rtol=0.0)


def test_scoreds_permutation_invariance_fitted():
    scores, mask, log_k, z_top1 = _score_sets(128, 10, seed=5, pad_frac=0.2)
    rng = np.random.default_rng(5)
    y = (rng.uniform(size=128) < 0.5).astype(np.float32)
    model = ScoreDeepSets(seed=3)
    model.fit(scores, mask, log_k, z_top1, y, lr=3e-3, epochs=20, batch_size=64)

    perm = rng.permutation(scores.shape[1])
    p0 = model.predict_proba(scores, mask, log_k, z_top1)
    p1 = model.predict_proba(scores[:, perm], mask[:, perm], log_k, z_top1)
    assert np.allclose(p0, p1, atol=1e-6, rtol=0.0)


# --------------------------------------------------------------------------- #
# 6. Variable K
# --------------------------------------------------------------------------- #
def test_scoreds_variable_k_shapes_and_finiteness():
    model = ScoreDeepSets(seed=0)
    for K in (5, 10, 20, 50):
        scores, mask, log_k, z_top1 = _score_sets(8, K, seed=K)
        p = model.predict_proba(scores, mask, log_k, z_top1)
        assert p.shape == (8,)
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


# --------------------------------------------------------------------------- #
# 7. K=1 and all-padding rows stay finite (safe, documented behaviour)
# --------------------------------------------------------------------------- #
def test_scoreds_k1_and_all_padding_are_finite():
    model = ScoreDeepSets(seed=0)

    # K=1 singleton (valid)
    p1 = model.predict_proba(
        np.array([[0.5]], dtype=np.float32),
        np.array([[True]]),
        np.array([0.0], dtype=np.float32),
        np.array([0.3], dtype=np.float32),
    )
    assert p1.shape == (1,)
    assert np.all(np.isfinite(p1))

    # an all-padding row (no valid candidate) must not produce NaN
    p2 = model.predict_proba(
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 4), dtype=bool),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    assert p2.shape == (2,)
    assert np.all(np.isfinite(p2))
    assert np.all(p2 >= 0.0) and np.all(p2 <= 1.0)


# --------------------------------------------------------------------------- #
# 8. Score-only input guard
# --------------------------------------------------------------------------- #
def test_score_only_input_guard():
    forbidden = ("embed", "geometry", "objectness", "query")
    for cls in (LogisticModel, TinyMLP, ScoreDeepSets):
        for method in ("fit", "predict_proba"):
            sig = inspect.signature(getattr(cls, method))
            for name in sig.parameters:
                low = name.lower()
                assert not any(token in low for token in forbidden), (
                    f"{cls.__name__}.{method} exposes forbidden parameter {name!r}"
                )

    import ccg.reliability.models as models_module

    assert "h5py" not in inspect.getsource(models_module)


# --------------------------------------------------------------------------- #
# 9. include_logk / include_ztop1 head-dim wiring
# --------------------------------------------------------------------------- #
def test_scoreds_head_dim_flag_wiring():
    both = ScoreDeepSets(seed=0, include_logk=True, include_ztop1=True)
    no_logk = ScoreDeepSets(seed=0, include_logk=False, include_ztop1=True)
    no_ztop1 = ScoreDeepSets(seed=0, include_logk=True, include_ztop1=False)

    assert both.head_in == 3 * both.hidden + 1 + 1
    assert no_logk.head_in == 3 * no_logk.hidden + 1
    assert no_ztop1.head_in == 3 * no_ztop1.hidden + 1

    # dropping either scalar removes exactly one head input -> 32 fewer head params
    assert both.n_parameters() - no_logk.n_parameters() == 32
    assert both.n_parameters() - no_ztop1.n_parameters() == 32
    assert 0 < both.n_parameters() < 5000


# --------------------------------------------------------------------------- #
# 10. Padding must not affect the output
# --------------------------------------------------------------------------- #
def test_scoreds_mask_ignores_padding_values():
    scores, mask, log_k, z_top1 = _score_sets(12, 10, seed=5, pad_frac=0.4)
    model = ScoreDeepSets(seed=1)

    p_before = model.predict_proba(scores, mask, log_k, z_top1)

    scores_poisoned = scores.copy()
    scores_poisoned[~mask] = 1e4  # extreme value dumped into every padded slot

    p_after = model.predict_proba(scores_poisoned, mask, log_k, z_top1)
    assert np.allclose(p_before, p_after, atol=1e-6, rtol=0.0)
