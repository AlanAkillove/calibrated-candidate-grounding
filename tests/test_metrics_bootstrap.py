"""Tests for :mod:`ccg.metrics.bootstrap` (paired / image-clustered bootstrap).

Covers the protocol requirements of section 23: paired comparison on identical
samples, image-level resampling, 95% CI, reproducibility from a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.metrics.bootstrap import bootstrap_ci, paired_bootstrap
from ccg.metrics.calibration import brier_binary, confidence_from_scores, top_label_ece
from ccg.metrics.selective import risk_at_coverage

KEYS = {"diff", "ci_low", "ci_high", "p_two_sided_hint"}


def _paired_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model A beats model B by a known margin; 4 correlated samples share one image."""
    rng = np.random.default_rng(seed)
    n_images = n // 4
    image = np.repeat(np.arange(n_images), 4)  # 4 expressions per image
    effect = rng.normal(0.35, 1.0, size=n_images)
    stat_b = rng.normal(0.0, 1.0, size=n)
    stat_a = stat_b + effect[image]
    return stat_a, stat_b, image


def test_paired_bootstrap_ci_covers_point_estimate() -> None:
    a, b, image = _paired_data()
    out = paired_bootstrap(a, b, cluster_ids=image, n_replicates=2000, seed=0)
    assert KEYS <= set(out)
    assert out["diff"] == pytest.approx(float(np.mean(a - b)))
    assert out["ci_low"] <= out["diff"] <= out["ci_high"]
    assert out["ci_low"] < out["ci_high"]
    assert out["resample_unit"] == "cluster" and out["n_clusters"] == 100
    assert out["n"] == a.shape[0] and out["n_replicates"] == 2000
    assert out["p_two_sided_hint"] < 0.05  # a real effect must be flagged
    assert 0.0 <= out["p_two_sided_hint"] <= 1.0


def test_paired_bootstrap_null_case_includes_zero() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    out = paired_bootstrap(a, a.copy(), n_replicates=1000, seed=5)
    assert out["diff"] == pytest.approx(0.0, abs=1e-15)
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["p_two_sided_hint"] > 0.05
    assert out["resample_unit"] == "sample" and out["n_clusters"] is None


def test_reproducibility_same_seed_twice() -> None:
    a, b, image = _paired_data()
    for cluster in (None, image):
        first = paired_bootstrap(a, b, cluster_ids=cluster, n_replicates=800, seed=17)
        second = paired_bootstrap(a, b, cluster_ids=cluster, n_replicates=800, seed=17)
        assert set(first) == set(second)
        for key in first:
            if key == "replicates":
                assert np.array_equal(first[key], second[key])
            else:
                assert first[key] == second[key]
        # different seed -> different replicate draws (same point estimate)
        other = paired_bootstrap(a, b, cluster_ids=cluster, n_replicates=800, seed=23)
        assert not np.array_equal(first["replicates"], other["replicates"])
        assert other["diff"] == pytest.approx(first["diff"])
        assert np.isfinite(other["ci_low"]) and np.isfinite(other["ci_high"])


def test_bootstrap_ci_basic_behaviour() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(loc=2.0, scale=1.0, size=500)
    out = bootstrap_ci(values, n_replicates=1500, seed=2)
    assert out["point"] == pytest.approx(float(np.mean(values)))
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["ci_high"] - out["ci_low"] < 0.5  # a tight interval for n = 500
    assert out["replicates"].shape == (1500,)
    again = bootstrap_ci(values, n_replicates=1500, seed=2)
    assert (again["ci_low"], again["ci_high"]) == (out["ci_low"], out["ci_high"])
    # a custom statistic works too (median of a symmetric distribution)
    med = bootstrap_ci(values, stat_fn=np.median, n_replicates=500, seed=4)
    assert abs(med["point"] - float(np.median(values))) < 1e-12
    # symmetric 90% interval is narrower than the 95% one
    narrow = bootstrap_ci(values, n_replicates=1500, seed=2, ci=0.90)
    assert narrow["ci_high"] - narrow["ci_low"] < out["ci_high"] - out["ci_low"]


def test_cluster_bootstrap_accounts_for_within_image_correlation() -> None:
    """Identical samples per image: the naive bootstrap must look overly certain."""
    n_images = 60
    per_image = 10
    rng = np.random.default_rng(8)
    image = np.repeat(np.arange(n_images), per_image)
    cluster_effect = rng.normal(0.6, 0.8, size=n_images)
    a = cluster_effect[image] + rng.normal(0.0, 0.05, size=image.shape[0])
    b = rng.normal(0.0, 0.05, size=image.shape[0])
    naive = paired_bootstrap(a, b, n_replicates=1500, seed=0)
    clustered = paired_bootstrap(a, b, cluster_ids=image, n_replicates=1500, seed=0)
    assert clustered["diff"] == pytest.approx(naive["diff"])
    width_naive = naive["ci_high"] - naive["ci_low"]
    width_cluster = clustered["ci_high"] - clustered["ci_low"]
    assert width_cluster > 1.5 * width_naive  # correlated samples -> honest, wider CI
    assert clustered["n_clusters"] == n_images


def test_bootstrap_on_real_metric_streams() -> None:
    """Per-sample contributions of ECE / Brier / risk can be bootstrapped directly."""
    rng = np.random.default_rng(21)
    n = 600
    scores_a = rng.normal(0.0, 1.0, size=(n, 10))
    scores_b = scores_a + rng.normal(0.3, 0.6, size=(n, 10))
    target = rng.integers(0, 10, size=n)
    image = rng.integers(0, 40, size=n)

    def per_sample_brier(scores: np.ndarray) -> np.ndarray:
        conf, pred = confidence_from_scores(scores)
        return (conf - (pred == target).astype(float)) ** 2

    brier_a, brier_b = per_sample_brier(scores_a), per_sample_brier(scores_b)
    out = paired_bootstrap(brier_a, brier_b, cluster_ids=image, n_replicates=800, seed=1)
    assert out["diff"] == pytest.approx(float(np.mean(brier_a) - np.mean(brier_b)))
    assert brier_binary(brier_a, np.ones_like(brier_a)) >= 0.0  # sanity on the metric itself
    assert np.isfinite(out["ci_low"]) and np.isfinite(out["ci_high"])

    conf, _pred = confidence_from_scores(scores_a)
    corr = (np.argmax(scores_a, axis=1) == target).astype(float)
    # per-sample |conf - correctness| is the ECE contribution of a one-bin table
    ece_like = np.abs(conf - corr)
    ece_out = paired_bootstrap(ece_like, np.abs(conf - corr), n_replicates=400, seed=3)
    assert ece_out["diff"] == pytest.approx(0.0, abs=1e-15)
    assert top_label_ece(conf, corr) >= 0.0
    assert 0.0 <= risk_at_coverage(conf, corr, 0.5) <= 1.0


def test_input_validation_and_errors() -> None:
    a = np.arange(10.0)
    with pytest.raises(ValueError, match="differ"):
        paired_bootstrap(a, a[:5])
    with pytest.raises(ValueError, match="NaN"):
        paired_bootstrap(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="n_replicates"):
        paired_bootstrap(a, a, n_replicates=0)
    with pytest.raises(ValueError, match="ci"):
        paired_bootstrap(a, a, ci=1.5)
    with pytest.raises(ValueError, match="cluster_ids"):
        paired_bootstrap(a, a, cluster_ids=np.arange(4))
    with pytest.raises(ValueError, match="two clusters"):
        paired_bootstrap(a, a, cluster_ids=np.zeros(10, dtype=int))
    with pytest.raises(ValueError, match="ci"):
        bootstrap_ci(a, ci=0.0)
