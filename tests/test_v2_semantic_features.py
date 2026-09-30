"""Tests for ccg.v2.semantic_features — backbone-neutral V2 primary features.

Verifies:
1. Feature count <= 16 and correct column names.
2. All features are finite for valid inputs.
3. No dependence on a hard-coded CLIP temperature.
4. Backbone-neutral: results are dimension-agnostic (512 vs 768 pass same logic).
5. Secondary descriptive features exist but are separate from primary.
6. Manipulation metrics reference valid primary feature names.
7. Monotonicity: q_margin12 = q_top1 - q_top2 (per-definition consistency).
8. Winner self-similarity is excluded from competitor stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.v2.semantic_features import (  # noqa: E402
    MANIPULATION_METRICS,
    V2_PRIMARY_SEMANTIC_NAMES,
    V2_SECONDARY_NAMES,
    secondary_descriptive,
    v2_primary_semantic_stats,
)


def _l2_normalize(arr):
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _make_inputs(n=10, k=5, d=512, seed=42):
    rng = np.random.default_rng(seed)
    z_q = _l2_normalize(rng.standard_normal((n, d)).astype(np.float64))
    z_i = _l2_normalize(rng.standard_normal((n, k, d)).astype(np.float64))
    scores = rng.standard_normal((n, k)).astype(np.float64)
    return z_q, z_i, scores


# ---------------------------------------------------------------------------
# 1. Shape and naming
# ---------------------------------------------------------------------------
class TestPrimaryShape:
    def test_output_columns_match_names(self):
        z_q, z_i, sc = _make_inputs(n=8, k=5, d=512)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        assert out.shape == (8, len(V2_PRIMARY_SEMANTIC_NAMES))

    def test_dim_at_most_16(self):
        assert len(V2_PRIMARY_SEMANTIC_NAMES) <= 16

    def test_all_finite(self):
        z_q, z_i, sc = _make_inputs(n=20, k=10, d=768)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        assert np.all(np.isfinite(out))

    def test_dtype_float64(self):
        z_q, z_i, sc = _make_inputs(n=3, k=5, d=512)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# 2. Dimension-agnostic (backbone-neutral)
# ---------------------------------------------------------------------------
class TestBackboneNeutral:
    @pytest.mark.parametrize("d", [512, 768, 1024])
    def test_same_logic_across_embedding_dims(self, d):
        """The function never inspects d, so same geometry -> same feature values."""
        z_q, z_i, sc = _make_inputs(n=6, k=5, d=d)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        assert out.shape == (6, len(V2_PRIMARY_SEMANTIC_NAMES))
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 3. No hard-coded CLIP_TEMPERATURE
# ---------------------------------------------------------------------------
class TestNoHardcodedTemperature:
    def test_primary_does_not_import_clip_temperature(self):
        """The primary module source must not contain CLIP_TEMPERATURE."""
        import inspect
        source = inspect.getsource(v2_primary_semantic_stats)
        assert "CLIP_TEMPERATURE" not in source
        assert "0.01" not in source


# ---------------------------------------------------------------------------
# 4. Consistency checks
# ---------------------------------------------------------------------------
class TestConsistency:
    def test_margin12_equals_top1_minus_top2(self):
        z_q, z_i, sc = _make_inputs(n=15, k=10, d=512)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        idx = {name: i for i, name in enumerate(V2_PRIMARY_SEMANTIC_NAMES)}
        margin12 = out[:, idx["q_margin12"]]
        top1 = out[:, idx["q_top1_cos"]]
        top2 = out[:, idx["q_top2_cos"]]
        np.testing.assert_allclose(margin12, top1 - top2, atol=1e-12)

    def test_margin13_equals_top1_minus_top3(self):
        z_q, z_i, sc = _make_inputs(n=15, k=10, d=512)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        idx = {name: i for i, name in enumerate(V2_PRIMARY_SEMANTIC_NAMES)}
        margin13 = out[:, idx["q_margin13"]]
        top1 = out[:, idx["q_top1_cos"]]
        top3 = out[:, idx["q_top3_cos"]]
        np.testing.assert_allclose(margin13, top1 - top3, atol=1e-12)

    def test_margin23_equals_top2_minus_top3(self):
        z_q, z_i, sc = _make_inputs(n=15, k=10, d=512)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        idx = {name: i for i, name in enumerate(V2_PRIMARY_SEMANTIC_NAMES)}
        margin23 = out[:, idx["q_margin23"]]
        top2 = out[:, idx["q_top2_cos"]]
        top3 = out[:, idx["q_top3_cos"]]
        np.testing.assert_allclose(margin23, top2 - top3, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Winner self-similarity excluded from competitor stats
# ---------------------------------------------------------------------------
class TestWinnerExclusion:
    def test_winner_comp_max_less_than_one(self):
        """Self-similarity = 1; if winner is excluded, max competitor cos < 1."""
        z_q, z_i, sc = _make_inputs(n=10, k=8, d=512)
        out = v2_primary_semantic_stats(z_q, z_i, sc)
        idx = {name: i for i, name in enumerate(V2_PRIMARY_SEMANTIC_NAMES)}
        # With random embeddings, the max inter-candidate cos should be << 1
        assert out[:, idx["winner_competitor_max_cos"]].max() < 0.99


# ---------------------------------------------------------------------------
# 6. Secondary features separate
# ---------------------------------------------------------------------------
class TestSecondary:
    def test_secondary_shape(self):
        z_q, z_i, sc = _make_inputs(n=5, k=5, d=512)
        out = secondary_descriptive(z_q, z_i, sc, temperature=0.05)
        assert out.shape == (5, len(V2_SECONDARY_NAMES))

    def test_secondary_no_temperature_default(self):
        """Without explicit temperature, T=1 is used (not V1's 0.01)."""
        z_q, z_i, sc = _make_inputs(n=5, k=5, d=512)
        out = secondary_descriptive(z_q, z_i, sc, temperature=None)
        assert np.all(np.isfinite(out))

    def test_secondary_rejects_negative_temperature(self):
        z_q, z_i, sc = _make_inputs(n=3, k=5, d=512)
        with pytest.raises(ValueError, match="positive"):
            secondary_descriptive(z_q, z_i, sc, temperature=-1.0)


# ---------------------------------------------------------------------------
# 7. Manipulation metrics reference valid primary features
# ---------------------------------------------------------------------------
class TestManipulationMetrics:
    def test_all_metric_names_in_primary(self):
        primary_set = set(V2_PRIMARY_SEMANTIC_NAMES)
        for name, direction in MANIPULATION_METRICS:
            assert name in primary_set, f"{name!r} not in primary features"
            assert direction in ("up", "down")


# ---------------------------------------------------------------------------
# 8. Input validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_rejects_k_less_than_5(self):
        z_q, z_i, sc = _make_inputs(n=3, k=4, d=512)
        with pytest.raises(ValueError, match="K must be >="):
            v2_primary_semantic_stats(z_q, z_i, sc)

    def test_rejects_dim_mismatch(self):
        z_q = _l2_normalize(np.random.randn(3, 512))
        z_i = _l2_normalize(np.random.randn(3, 5, 768))
        sc = np.random.randn(3, 5)
        with pytest.raises(ValueError, match="embedding dim mismatch"):
            v2_primary_semantic_stats(z_q, z_i, sc)
