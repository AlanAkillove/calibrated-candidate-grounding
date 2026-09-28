"""Score-only reliability feature extraction (Phase 0.5, Amendment A6).

Every assertion here doubles as a protocol guard: the feature module must consume
*only* ``[n, K]`` score matrices and must never reach into embeddings, geometry,
objectness, GT or a feature store.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ccg.reliability import features
from ccg.reliability.features import (
    SCALAR_NAMES,
    NormalizationFit,
    entry_moments,
    normalize_apply,
    normalize_fit,
    scalar_confidence,
    sds_pack,
    stat_feature_names,
    stat_features,
    top1_column,
    top_scores_feature_names,
    top_scores_features,
)

EPS = 1e-8


def _softmax_row(row, temperature=1.0):
    scaled = np.asarray(row, dtype=np.float64) / temperature
    shifted = scaled - scaled.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# 1. source-level guard: score arrays only, no feature store / heavy deps
# ---------------------------------------------------------------------------
def test_module_consumes_scores_only():
    source = inspect.getsource(features)
    for banned in ("h5py", "FeatureCache", "torch", "cache/features", "open_clip"):
        assert banned not in source, f"features.py must not reference {banned!r}"


# ---------------------------------------------------------------------------
# 2. hand example (T=1) + 3. K / temperature validation + stability
# ---------------------------------------------------------------------------
def test_scalar_confidence_hand_example():
    scores = np.array([[2.0, 0.0]])
    out = scalar_confidence(scores, temperature=1.0)

    assert set(out) == set(SCALAR_NAMES)
    for key in SCALAR_NAMES:
        assert out[key].shape == (1,)
        assert out[key].dtype == np.float64

    probs = _softmax_row([2.0, 0.0])
    expected_entropy = -float(np.sum(probs * np.log(probs)))

    assert out["msp"][0] == pytest.approx(0.880797, abs=1e-6)
    assert out["msp"][0] == pytest.approx(float(probs.max()), abs=1e-12)
    assert out["top1_score"][0] == pytest.approx(2.0)
    assert out["margin"][0] == pytest.approx(2.0)
    assert out["neg_entropy"][0] == pytest.approx(-expected_entropy, rel=1e-9)  # -H(P), section 7
    assert out["norm_entropy"][0] == pytest.approx(1.0 - expected_entropy / np.log(2.0), rel=1e-9)


def test_scalar_confidence_validation_and_stability():
    with pytest.raises(ValueError, match="K >= 2"):
        scalar_confidence(np.array([[1.0]]), temperature=1.0)
    with pytest.raises(ValueError, match="temperature"):
        scalar_confidence(np.array([[1.0, 2.0]]), temperature=0.0)

    # extreme logits must not overflow: msp saturates at 1, everything finite
    extreme = scalar_confidence(np.array([[1000.0, -1000.0]]), temperature=1.0)
    assert np.all(np.isfinite(extreme["msp"]))
    assert extreme["msp"][0] == pytest.approx(1.0)
    assert extreme["norm_entropy"][0] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. stat_features: dimension/order == stat_feature_names, spot-check values
# ---------------------------------------------------------------------------
def test_stat_feature_columns_match_names_and_values():
    scores = np.array([[2.0, 0.0, -1.0, 0.5, 1.0]])
    names = stat_feature_names()
    matrix = stat_features(scores, temperature=1.0)

    assert matrix.shape == (1, len(names))
    row = dict(zip(names, matrix[0]))

    assert row["top1"] == pytest.approx(2.0)
    assert row["top2"] == pytest.approx(1.0)
    assert row["top3"] == pytest.approx(0.5)
    assert row["margin_12"] == pytest.approx(1.0)
    assert row["margin_23"] == pytest.approx(0.5)
    assert row["log_k"] == pytest.approx(np.log(5.0))
    assert row["mean"] == pytest.approx(float(scores[0].mean()))
    assert row["std"] == pytest.approx(float(scores[0].std()))
    assert row["logsumexp"] == pytest.approx(float(np.log(np.exp(scores[0]).sum())))
    assert row["msp"] == pytest.approx(float(_softmax_row(scores[0]).max()))

    denom = float(scores[0].std()) + EPS
    assert row["z_top1"] == pytest.approx((2.0 - scores[0].mean()) / denom)
    assert row["z_margin"] == pytest.approx(1.0 / denom)


def test_stat_features_requires_k_ge_3():
    with pytest.raises(ValueError, match="K >= 3"):
        stat_features(np.array([[1.0, 2.0]]), temperature=1.0)


# ---------------------------------------------------------------------------
# 4. per-set statistics are permutation invariant
# ---------------------------------------------------------------------------
def test_stat_features_permutation_invariant():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(6, 7))
    reference = stat_features(scores, temperature=2.0)
    permuted = np.stack([rng.permutation(row) for row in scores])
    other = stat_features(permuted, temperature=2.0)
    np.testing.assert_allclose(reference, other, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. variant column combinations
# ---------------------------------------------------------------------------
def test_stat_feature_names_variants():
    stripped = stat_feature_names(include_logk=False, include_entropy=False, include_quantiles=False)
    assert "log_k" not in stripped
    assert "entropy" not in stripped and "entropy_over_logk" not in stripped
    assert not ({"q25", "q50", "q75"} & set(stripped))

    full = stat_feature_names()
    assert full[0] == "log_k"
    assert full[-3:] == ("q25", "q50", "q75")
    assert "entropy" in full and "entropy_over_logk" in full

    scores = np.array([[1.0, 0.0, 2.0, -1.0, 3.0]])
    assert stat_features(scores, temperature=1.0).shape[1] == len(full)
    variant = stat_features(
        scores,
        temperature=1.0,
        include_logk=False,
        include_entropy=False,
        include_quantiles=False,
    )
    assert variant.shape[1] == len(stripped)


# ---------------------------------------------------------------------------
# numerical hygiene: no NaN on degenerate / empty inputs
# ---------------------------------------------------------------------------
def test_stat_features_is_nan_free_on_degenerate_rows():
    scores = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],  # constant row: std == 0
            [1e6, -1e6, 0.0, 5.0, 2.0],  # huge dynamic range
        ]
    )
    matrix = stat_features(scores, temperature=1.0)
    assert np.all(np.isfinite(matrix))


def test_empty_batch_is_handled():
    empty = np.zeros((0, 5))
    out = scalar_confidence(empty, temperature=1.0)
    assert out["msp"].shape == (0,)
    assert stat_features(empty, temperature=1.0).shape == (0, len(stat_feature_names()))


# ---------------------------------------------------------------------------
# top1_column
# ---------------------------------------------------------------------------
def test_top1_column():
    scores = np.array([[2.0, 0.0, 1.0], [-1.0, 5.0, 3.0]])
    column = top1_column(scores)
    np.testing.assert_allclose(column, [2.0, 5.0])
    assert column.dtype == np.float64


# ---------------------------------------------------------------------------
# 6. normalize_fit reads only fit_rows
# ---------------------------------------------------------------------------
def test_normalize_fit_uses_only_fit_rows():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(10, 4))
    fit_rows = np.array([0, 1, 2, 3, 4])
    keys = ("a", "b", "c", "d")

    fit = normalize_fit(matrix, fit_rows=fit_rows, keys=keys)
    subset = matrix[fit_rows]
    np.testing.assert_allclose(fit.mean, subset.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(fit.std, subset.std(axis=0, ddof=0), atol=1e-12)
    assert fit.keys == keys

    # NaN injected into the non-fit rows must not change the fit at all
    dirty_nan = matrix.copy()
    dirty_nan[5:] = np.nan
    fit_nan = normalize_fit(dirty_nan, fit_rows=fit_rows, keys=keys)
    np.testing.assert_allclose(fit_nan.mean, fit.mean, atol=1e-12)
    np.testing.assert_allclose(fit_nan.std, fit.std, atol=1e-12)

    # ... nor must huge junk values
    dirty_junk = matrix.copy()
    dirty_junk[5:] = 1e9
    fit_junk = normalize_fit(dirty_junk, fit_rows=fit_rows, keys=keys)
    np.testing.assert_allclose(fit_junk.mean, fit.mean, atol=1e-12)
    np.testing.assert_allclose(fit_junk.std, fit.std, atol=1e-12)

    # boolean masks behave like index arrays
    boolean_fit = normalize_fit(matrix, fit_rows=np.arange(10) < 5, keys=keys)
    np.testing.assert_allclose(boolean_fit.mean, fit.mean, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. normalize_apply standardises the fit rows
# ---------------------------------------------------------------------------
def test_normalize_apply_standardizes_fit_rows():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(20, 3)) + np.array([1.0, -2.0, 5.0])
    fit_rows = np.arange(10)
    fit = normalize_fit(matrix, fit_rows=fit_rows, keys=("x", "y", "z"))

    normalized = normalize_apply(matrix, fit)
    assert normalized.dtype == np.float64
    np.testing.assert_allclose(normalized[fit_rows].mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(normalized[fit_rows].std(axis=0), 1.0, atol=1e-6)

    with pytest.raises(ValueError, match="does not match"):
        normalize_apply(matrix[:, :2], fit)


def test_normalization_fit_roundtrip():
    fit = NormalizationFit(
        mean=np.array([1.0, 2.0]), std=np.array([0.5, 0.25]), keys=("a", "b")
    )
    payload = fit.to_dict()
    assert payload["keys"] == ["a", "b"]
    assert payload["dim"] == 2
    restored = NormalizationFit.from_dict(payload)
    np.testing.assert_allclose(restored.mean, fit.mean)
    np.testing.assert_allclose(restored.std, fit.std)
    assert restored.keys == fit.keys


# ---------------------------------------------------------------------------
# 8. top_scores_features: shape 2*top_m + 3, K < top_m raises
# ---------------------------------------------------------------------------
def test_top_scores_features_shape_and_values():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(4, 5))
    out = top_scores_features(scores, top_m=5)

    assert out.shape == (4, 13)
    assert out.shape[1] == 2 * 5 + 3
    names = top_scores_feature_names(top_m=5)
    assert len(names) == out.shape[1]
    assert names[:5] == ("ts_z1", "ts_z2", "ts_z3", "ts_z4", "ts_z5")
    assert names[5:9] == ("ts_g12", "ts_g23", "ts_g34", "ts_g45")
    assert names[-3:] == ("ts_logk", "ts_mean", "ts_std")

    row = dict(zip(names, out[0]))
    ordered = np.sort(scores[0])[::-1]
    mean, std = float(scores[0].mean()), float(scores[0].std())
    z = (ordered - mean) / (std + EPS)
    assert row["ts_z1"] == pytest.approx(z[0], rel=1e-9)
    assert row["ts_g12"] == pytest.approx(z[0] - z[1], rel=1e-9)
    assert row["ts_logk"] == pytest.approx(np.log(5.0))
    assert row["ts_mean"] == pytest.approx(mean)
    assert row["ts_std"] == pytest.approx(std)

    # default top_m == 5
    assert top_scores_features(scores).shape == (4, 13)


def test_top_scores_features_requires_k_ge_top_m():
    with pytest.raises(ValueError, match="K >= top_m"):
        top_scores_features(np.zeros((2, 4)), top_m=5)


# ---------------------------------------------------------------------------
# 9. sds_pack: mixed K=5 / K=10 -> padded float32 + boolean mask
# ---------------------------------------------------------------------------
def test_sds_pack_mixed_cardinality():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)  # (n=2, K=3)
    b = (np.arange(20, dtype=np.float32).reshape(2, 10)) + 100.0  # (n=2, K=10)

    padded, mask = sds_pack([a, b])
    assert padded.shape == (4, 10)
    assert padded.dtype == np.float32
    assert mask.shape == (4, 10)
    assert mask.dtype == bool

    # first block: 3 valid columns, the rest are zero padding
    assert mask[:2, :3].all() and not mask[:2, 3:].any()
    assert (padded[:2, 3:] == 0.0).all()
    # second block: all 10 valid
    assert mask[2:, :10].all()
    # mask counts exactly the real candidates
    assert mask.sum() == 2 * 3 + 2 * 10

    np.testing.assert_allclose(padded[:2, :3], a)
    np.testing.assert_allclose(padded[2:], b)

    # order of the matrices is preserved (K=10 first, K=3 padded last)
    padded_rev, mask_rev = sds_pack([b, a])
    assert padded_rev.shape == (4, 10)
    assert (padded_rev[2:, 3:] == 0.0).all()
    assert mask_rev.sum() == mask.sum()


# ---------------------------------------------------------------------------
# 10. entry_moments == pooled numpy mean/std (ddof=0)
# ---------------------------------------------------------------------------
def test_entry_moments_matches_numpy_pooled():
    rng = np.random.default_rng(4)
    mats = [rng.normal(size=(3, 5)), rng.normal(size=(2, 7)) * 3.0 + 1.0]
    mean, std = entry_moments(mats)
    pooled = np.concatenate([m.reshape(-1) for m in mats])
    assert mean == pytest.approx(float(pooled.mean()))
    assert std == pytest.approx(float(pooled.std(ddof=0)))
