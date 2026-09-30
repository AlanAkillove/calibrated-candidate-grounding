"""M0 implementation tests for the V2-M LCR stage (protocol section 31).

The twelve required pre-training checks, in frozen protocol order:

1. relation feature correctness (naive per-row reference + hand-computed case)
2. top-M stable ordering (tie resolution, contiguous competitor ranks)
3. candidate permutation handling (index remapping + feature invariance)
4. no GT category input (signature audit + runtime forbidden-key rejection)
5. R1 frozen (b enters additively, is never a learned parameter, no input mutation)
6. grounding scores untouched (bitwise no-write + fingerprint audit)
7. LCR param count (LCR 1333 / LCR-noGate 1329, under the 10k target)
8. gate range [0, 1]
9. MLP capacity control parameter matching (Aggregate-MLP vs LCR)
10. deterministic feature extraction (and same-seed training)
11. K5/K10/K20/K50 support
12. no hard data in M1 train/tune (zero-hard-exposure guards)

All tests must pass before M1 training starts.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.lcr import (  # noqa: E402
    GATE_INPUT_NAMES,
    M_COMPETITORS,
    RELATION_FEATURE_NAMES,
    AggregateMLP,
    LCR,
    LCRNoGate,
    RelationBatch,
    relation_features,
)
from ccg.lcr.audit import (  # noqa: E402
    assert_grounding_unchanged,
    assert_no_hard_rows,
    grounding_fingerprint,
    m1_training_guard,
)
from ccg.lcr.features import EPS, winning_index  # noqa: E402
from ccg.lcr.models import (  # noqa: E402
    AGG_INPUT_DIM,
    M_COMPETITORS_USED,
    REL_DIM,
)

PROTOCOL_PATH = _REPO_ROOT / "results" / "v2_local_competition" / "protocol.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _make_inputs(n: int = 12, k: int = 5, d: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    z_q = _l2_normalize(rng.standard_normal((n, d)))
    z_i = _l2_normalize(rng.standard_normal((n, k, d)))
    scores = rng.standard_normal((n, k))
    return z_q, z_i, scores


def _valid_lcr_inputs(n: int = 8, k: int = 5, d: int = 8, seed: int = 3):
    z_q, z_i, scores = _make_inputs(n=n, k=k, d=d, seed=seed)
    out = relation_features(scores, z_q, z_i, 1.0)
    return {"b": np.zeros(n), "r": out.r, "gate": out.gate}


def _naive_features(scores, z_q, z_i, temperature, m: int = M_COMPETITORS):
    """Per-row reference implementation written in plain loops."""
    sc = np.asarray(scores, dtype=np.float64)
    zq = np.asarray(z_q, dtype=np.float64)
    zi = np.asarray(z_i, dtype=np.float64)
    n, k = sc.shape
    r = np.empty((n, m, 7), dtype=np.float64)
    gate = np.empty((n, 3), dtype=np.float64)
    win = np.empty(n, dtype=np.int64)
    comp = np.empty((n, m), dtype=np.int64)
    for i in range(n):
        s, zi_i, zq_i = sc[i], zi[i], zq[i]
        order = np.argsort(-s, kind="stable")
        t = int(order[0])
        c = order[1 : m + 1]
        win[i] = t
        comp[i] = c
        std = np.std(s)  # ddof=0
        e = np.exp(s / temperature - np.max(s / temperature))
        p = e / e.sum()
        a = zi_i @ zq_i
        v = zi_i @ zi_i[t]
        for jj, j in enumerate(c):
            ds = (s[t] - s[j]) / (std + EPS)
            r[i, jj] = [ds, p[t] - p[j], a[t], a[j], a[t] - a[j], v[j], (jj + 2) / k]
        gate[i] = [v[c].max(), v[c].mean(), (s[t] - s[c[0]]) / (std + EPS)]
    return r, gate, win, comp


# ---------------------------------------------------------------------------
# 1. relation feature correctness
# ---------------------------------------------------------------------------
class TestRelationFeatureCorrectness:
    @pytest.mark.parametrize("k", [5, 10])
    def test_matches_naive_reference(self, k):
        n = 16
        z_q, z_i, scores = _make_inputs(n=n, k=k, d=12, seed=71 + k)
        out = relation_features(scores, z_q, z_i, 0.85)
        r_ref, gate_ref, win_ref, comp_ref = _naive_features(scores, z_q, z_i, 0.85)
        np.testing.assert_allclose(out.r, r_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out.gate, gate_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(out.winner_idx, win_ref)
        np.testing.assert_array_equal(out.competitor_idx, comp_ref)

    def test_hand_computed_k5_case(self):
        # scores strictly descending; embeddings chosen so cosines are exact decimals
        scores = np.array([[0.8, 0.6, 0.4, 0.2, 0.0]])
        z_q = np.array([[1.0, 0.0]])
        z_i = np.array(
            [[[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0], [0.8, 0.6]]]
        )
        out = relation_features(scores, z_q, z_i, 1.0)
        np.testing.assert_array_equal(out.winner_idx, [0])
        np.testing.assert_array_equal(out.competitor_idx, [[1, 2, 3, 4]])
        row = out.r[0]  # [4 competitors, 7]
        exp_p = np.exp(scores[0]) / np.exp(scores[0]).sum()
        exp_std = np.std(scores[0])
        np.testing.assert_allclose(row[0, 0], (0.8 - 0.6) / (exp_std + EPS), atol=1e-15)
        # ds_norm of the rank-2 competitor is exactly sqrt(0.08) up to the eps
        np.testing.assert_allclose(row[0, 0], 0.7071067811865475, rtol=0, atol=1e-6)
        np.testing.assert_allclose(row[0, 1], exp_p[0] - exp_p[1], atol=1e-12)
        np.testing.assert_allclose(row[:, 2], 1.0, atol=1e-12)  # a_t = cos(q, winner)
        np.testing.assert_allclose(row[:, 3], [0.6, 0.0, -1.0, 0.8], atol=1e-12)
        np.testing.assert_allclose(row[:, 4], 1.0 - row[:, 3], atol=1e-12)
        np.testing.assert_allclose(row[:, 5], [0.6, 0.0, -1.0, 0.8], atol=1e-12)
        np.testing.assert_allclose(row[:, 6], np.arange(2, 6) / 5.0, atol=0)
        np.testing.assert_allclose(out.gate[0, 0], 0.8, atol=1e-12)
        np.testing.assert_allclose(out.gate[0, 1], 0.1, atol=1e-12)
        np.testing.assert_allclose(out.gate[0, 2], row[0, 0], atol=1e-15)

    def test_temperature_only_affects_dp(self):
        z_q, z_i, scores = _make_inputs(n=6, k=5, d=8, seed=79)
        a = relation_features(scores, z_q, z_i, 0.5)
        b = relation_features(scores, z_q, z_i, 2.0)
        non_dp = [0, 2, 3, 4, 5, 6]
        np.testing.assert_array_equal(a.r[:, :, non_dp], b.r[:, :, non_dp])
        assert not np.allclose(a.r[:, :, 1], b.r[:, :, 1])


# ---------------------------------------------------------------------------
# 2. top-M stable ordering
# ---------------------------------------------------------------------------
class TestTopMStableOrdering:
    def test_tie_resolution_prefers_lower_index(self):
        z_q, z_i, _ = _make_inputs(n=1, k=5, d=8, seed=51)
        scores = np.array([[0.5, 0.7, 0.7, 0.3, 0.7]])
        out = relation_features(scores, z_q, z_i, 1.0)
        assert int(out.winner_idx[0]) == 1  # lowest index among the tied maxima
        np.testing.assert_array_equal(out.competitor_idx[0], [2, 4, 0, 3])
        np.testing.assert_array_equal(out.order[0], [1, 2, 4, 0, 3])
        assert int(out.winner_idx[0]) == int(np.argmax(scores[0]))

    def test_all_tied_scores_give_stable_ranks(self):
        z_q, z_i, _ = _make_inputs(n=1, k=5, d=8, seed=53)
        scores = np.full((1, 5), 0.25)
        out = relation_features(scores, z_q, z_i, 1.0)
        assert int(out.winner_idx[0]) == 0
        np.testing.assert_array_equal(out.competitor_idx[0], [1, 2, 3, 4])
        np.testing.assert_allclose(out.r[0, :, 0], 0.0)  # 0 / (0 + eps)
        np.testing.assert_allclose(out.gate[0, 2], 0.0)

    def test_ranks_are_contiguous_and_descending(self):
        z_q, z_i, scores = _make_inputs(n=10, k=10, d=8, seed=55)
        out = relation_features(scores, z_q, z_i, 1.0)
        for i in range(out.n):
            top = np.concatenate([[out.winner_idx[i]], out.competitor_idx[i]])
            assert len(set(top.tolist())) == M_COMPETITORS + 1
            s = scores[i][out.order[i]]
            assert np.all(np.diff(s) <= 0.0)

    def test_winning_index_matches_stable_argmax_under_ties(self):
        rng = np.random.default_rng(57)
        scores = rng.integers(0, 3, size=(50, 5)).astype(np.float64)  # ties guaranteed
        w = winning_index(scores)
        for i in range(scores.shape[0]):
            assert int(w[i]) == int(np.argmax(scores[i]))
            assert int(w[i]) == int(np.argsort(-scores[i], kind="stable")[0])


# ---------------------------------------------------------------------------
# 3. candidate permutation handling
# ---------------------------------------------------------------------------
class TestCandidatePermutationHandling:
    def test_features_invariant_and_indices_remap(self):
        n, k, d = 12, 5, 16
        rng = np.random.default_rng(61)
        z_q = _l2_normalize(rng.standard_normal((n, d)))
        z_i = _l2_normalize(rng.standard_normal((n, k, d)))
        # each row is a permutation of 1..k -> strictly distinct scores, no ties
        scores = rng.permuted(np.tile(np.arange(1, k + 1), (n, 1)), axis=1).astype(np.float64)
        assert np.all(np.diff(np.sort(scores, axis=1), axis=1) > 0.0)

        base = relation_features(scores, z_q, z_i, 1.3)
        p = rng.permutation(k)
        inv = np.argsort(p)  # new index -> old index is p; old -> new is inv
        perm = relation_features(scores[:, p], z_q, z_i[:, p], 1.3)

        np.testing.assert_array_equal(perm.winner_idx, inv[base.winner_idx])
        np.testing.assert_array_equal(perm.competitor_idx, inv[base.competitor_idx])
        np.testing.assert_array_equal(perm.order, inv[base.order])
        np.testing.assert_allclose(perm.r, base.r, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(perm.gate, base.gate, rtol=1e-12, atol=1e-12)

    def test_double_permutation_returns_to_identity(self):
        n, k, d = 6, 5, 8
        z_q, z_i, scores = _make_inputs(n=n, k=k, d=d, seed=63)
        p = np.random.default_rng(63).permutation(k)
        inv = np.argsort(p)
        assert np.array_equal(inv[p], np.arange(k))  # inv is the exact inverse of p
        out = relation_features(
            scores[:, p][:, inv], z_q, z_i[:, p][:, inv], 1.0
        )
        base = relation_features(scores, z_q, z_i, 1.0)
        np.testing.assert_array_equal(out.competitor_idx, base.competitor_idx)
        np.testing.assert_allclose(out.r, base.r, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. no GT category input
# ---------------------------------------------------------------------------
class TestNoGTCategoryInput:
    def test_relation_features_signature_has_no_forbidden_inputs(self):
        sig = inspect.signature(relation_features)
        names = set(sig.parameters)
        assert names == {"scores", "z_q", "z_i", "temperature", "m"}
        assert sig.parameters["m"].default == M_COMPETITORS
        banned = ("categ", "gt", "object", "iou", "regime", "hard", "label")
        for name in names:
            assert not any(token in name.lower() for token in banned)

    def test_relation_batch_fields_frozen(self):
        fields = set(RelationBatch.__dataclass_fields__)
        assert fields == {"order", "winner_idx", "competitor_idx", "r", "gate", "k"}

    @pytest.mark.parametrize(
        "variant,forbidden",
        [
            ("LCR", "category"),
            ("LCR-noGate", "objectness"),
            ("Aggregate-MLP", "raw_embedding"),
        ],
    )
    def test_models_reject_unknown_keys(self, variant, forbidden):
        if variant == "LCR":
            model, inputs = LCR(seed=0), _valid_lcr_inputs()
        elif variant == "LCR-noGate":
            model, inputs = LCRNoGate(seed=0), _valid_lcr_inputs()
        else:
            model, inputs = AggregateMLP(seed=0), {"x": np.zeros((8, AGG_INPUT_DIM))}
        with pytest.raises(ValueError, match="forbidden input key"):
            model.predict_proba({**inputs, forbidden: np.zeros(8)})

    def test_regime_label_rejected(self):
        inputs = _valid_lcr_inputs()
        with pytest.raises(ValueError, match="forbidden input key"):
            LCR(seed=0).predict_proba({**inputs, "regime": np.array(["hard"] * 8)})

    def test_missing_contract_key_rejected(self):
        inputs = _valid_lcr_inputs()
        incomplete = {"b": inputs["b"], "r": inputs["r"]}  # gate missing
        with pytest.raises(KeyError):
            LCR(seed=0).predict_proba(incomplete)


# ---------------------------------------------------------------------------
# 5. R1 frozen
# ---------------------------------------------------------------------------
class TestR1Frozen:
    _LCR_PARAM_KEYS = {
        "phi__0__weight",
        "phi__0__bias",
        "phi__2__weight",
        "phi__2__bias",
        "gate__weight",
        "gate__bias",
        "delta__0__weight",
        "delta__0__bias",
        "delta__2__weight",
        "delta__2__bias",
    }

    def test_no_learned_parameter_named_b(self):
        keys = set(LCR(seed=0).state_arrays())
        assert keys == self._LCR_PARAM_KEYS
        assert not any(k == "b" or k.startswith("b__") for k in keys)

    def test_nogate_parameter_keys(self):
        keys = set(LCRNoGate(seed=0).state_arrays())
        assert keys == self._LCR_PARAM_KEYS - {"gate__weight", "gate__bias"}

    def test_fit_does_not_mutate_inputs(self):
        n = 96
        z_q, z_i, scores = _make_inputs(n=n, k=5, d=8, seed=21)
        out = relation_features(scores, z_q, z_i, 1.0)
        rng = np.random.default_rng(21)
        b = rng.standard_normal(n)
        y = (rng.random(n) < 0.5).astype(np.float64)
        b0, r0, g0, y0 = b.copy(), out.r.copy(), out.gate.copy(), y.copy()
        LCR(seed=3).fit({"b": b, "r": out.r, "gate": out.gate}, y, lr=1e-3, epochs=5)
        np.testing.assert_array_equal(b, b0)
        np.testing.assert_array_equal(out.r, r0)
        np.testing.assert_array_equal(out.gate, g0)
        np.testing.assert_array_equal(y, y0)

    def test_b_enters_additively_after_training(self):
        n = 128
        z_q, z_i, scores = _make_inputs(n=n, k=5, d=8, seed=23)
        out = relation_features(scores, z_q, z_i, 0.9)
        rng = np.random.default_rng(23)
        b = rng.standard_normal(n)
        y = (rng.random(n) < 0.55).astype(np.float64)
        assert 0 < int(y.sum()) < n
        model = LCR(seed=5)
        model.fit({"b": b, "r": out.r, "gate": out.gate}, y, lr=1e-3, epochs=10)
        p1 = model.predict_proba({"b": b, "r": out.r, "gate": out.gate})
        p2 = model.predict_proba({"b": b + 0.7, "r": out.r, "gate": out.gate})
        assert np.all((p1 > 1e-4) & (p1 < 1 - 1e-4))
        assert np.all((p2 > 1e-4) & (p2 < 1 - 1e-4))
        z1 = np.log(p1 / (1.0 - p1))
        z2 = np.log(p2 / (1.0 - p2))
        np.testing.assert_allclose(z2 - z1, 0.7, rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# 6. grounding scores untouched
# ---------------------------------------------------------------------------
class TestGroundingScoresUntouched:
    def test_relation_features_does_not_write_scores(self):
        z_q, z_i, scores = _make_inputs(seed=7)
        before = scores.copy()
        relation_features(scores, z_q, z_i, 1.0)
        np.testing.assert_array_equal(scores, before)

    def test_fingerprint_identity_and_accuracy(self):
        z_q, z_i, scores = _make_inputs(n=32, k=5, d=8, seed=11)
        fp = grounding_fingerprint(scores)
        argmax = np.argmax(scores, axis=1)
        assert fp["accuracy"] == float(np.mean(argmax == 0))
        assert fp["n"] == 32 and fp["k"] == 5
        assert_grounding_unchanged(fp, grounding_fingerprint(scores.copy()))

    def test_fingerprint_detects_any_score_change(self):
        z_q, z_i, scores = _make_inputs(n=16, k=5, d=8, seed=13)
        before = grounding_fingerprint(scores)
        changed = scores.copy()
        changed[0, -1] += 1e-9
        with pytest.raises(AssertionError, match="GROUNDING INVARIANCE VIOLATION"):
            assert_grounding_unchanged(before, grounding_fingerprint(changed))

    def test_fingerprint_tracks_argmax_flips(self):
        a = np.array([[0.5000000001, 0.5, 0.1, 0.0, -0.1]])
        b = np.array([[0.5, 0.5000000001, 0.1, 0.0, -0.1]])
        fp_a, fp_b = grounding_fingerprint(a), grounding_fingerprint(b)
        assert fp_a["argmax_sha256"] != fp_b["argmax_sha256"]
        assert fp_a["accuracy"] == 1.0 and fp_b["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 7 + 9. parameter counts and capacity-control matching
# ---------------------------------------------------------------------------
class TestParamCountsAndCapacityMatching:
    def test_exact_frozen_parameter_counts(self):
        assert LCR(seed=0).n_parameters() == 1333
        assert LCRNoGate(seed=0).n_parameters() == 1329
        assert AggregateMLP(seed=0).n_parameters() == 1569

    def test_budget_and_capacity_matching(self):
        n_lcr = LCR(seed=0).n_parameters()
        n_agg = AggregateMLP(seed=0).n_parameters()
        assert n_lcr < 10_000  # protocol target
        assert n_agg < 50_000  # protocol hard max
        # capacity control must stay the same order of magnitude as LCR
        assert abs(n_agg - n_lcr) / n_lcr <= 0.25

    def test_frozen_dimension_constants(self):
        assert M_COMPETITORS == 4 == M_COMPETITORS_USED
        assert REL_DIM == 7 == len(RELATION_FEATURE_NAMES)
        assert len(GATE_INPUT_NAMES) == 3
        assert AGG_INPUT_DIM == 31

    def test_no_architecture_zoo(self):
        import ccg.lcr as lcr_pkg

        assert set(lcr_pkg.__all__) == {
            "M_COMPETITORS",
            "GATE_INPUT_NAMES",
            "RELATION_FEATURE_NAMES",
            "RelationBatch",
            "relation_features",
            "LCR",
            "LCRNoGate",
            "AggregateMLP",
        }


# ---------------------------------------------------------------------------
# 8. gate range [0, 1]
# ---------------------------------------------------------------------------
class TestGateRange:
    def test_gate_values_strictly_inside_unit_interval(self):
        n = 64
        z_q, z_i, scores = _make_inputs(n=n, k=5, d=8, seed=41)
        out = relation_features(scores, z_q, z_i, 1.0)
        g = LCR(seed=0).gate_values({"b": np.zeros(n), "r": out.r, "gate": out.gate})
        assert g.shape == (n,)
        assert np.all((g > 0.0) & (g < 1.0))

    def test_gate_bounded_for_extreme_inputs(self):
        n = 4
        z_q, z_i, scores = _make_inputs(n=n, k=5, d=8, seed=43)
        out = relation_features(scores, z_q, z_i, 1.0)
        model = LCR(seed=1)
        for scale in (1e6, -1e6):
            gate = np.full((n, 3), scale)
            g = model.gate_values({"b": np.zeros(n), "r": out.r, "gate": gate})
            assert np.all((g >= 0.0) & (g <= 1.0))


# ---------------------------------------------------------------------------
# 10. determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_relation_features_bit_identical_across_calls(self):
        z_q, z_i, scores = _make_inputs(n=24, k=10, d=32, seed=31)
        a = relation_features(scores, z_q, z_i, 1.2)
        b = relation_features(scores, z_q, z_i, 1.2)
        for attr in ("order", "winner_idx", "competitor_idx", "r", "gate"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))

    def test_same_seed_training_bit_identical(self):
        n = 96
        z_q, z_i, scores = _make_inputs(n=n, k=5, d=8, seed=33)
        out = relation_features(scores, z_q, z_i, 1.0)
        rng = np.random.default_rng(33)
        b = rng.standard_normal(n)
        y = (rng.random(n) < 0.5).astype(np.float64)
        inputs = {"b": b, "r": out.r, "gate": out.gate}
        m1 = LCR(seed=11)
        m1.fit(inputs, y, lr=1e-3, epochs=6)
        torch.randn(10)  # unrelated global RNG draw must not matter
        m2 = LCR(seed=11)
        m2.fit(inputs, y, lr=1e-3, epochs=6)
        np.testing.assert_array_equal(m1.predict_proba(inputs), m2.predict_proba(inputs))

    def test_same_seed_training_with_validation_bit_identical(self):
        z_q, z_i, scores = _make_inputs(n=96, k=5, d=8, seed=35)
        out = relation_features(scores, z_q, z_i, 1.0)
        rng = np.random.default_rng(35)
        b = rng.standard_normal(96)
        y = (rng.random(96) < 0.5).astype(np.float64)
        zq_v, zi_v, sc_v = _make_inputs(n=48, k=5, d=8, seed=36)
        out_v = relation_features(sc_v, zq_v, zi_v, 1.0)
        b_v = rng.standard_normal(48)
        y_v = (rng.random(48) < 0.5).astype(np.float64)
        inputs = {"b": b, "r": out.r, "gate": out.gate}
        val = {"b": b_v, "r": out_v.r, "gate": out_v.gate}
        m1 = LCR(seed=13)
        m1.fit(inputs, y, lr=1e-3, inputs_val=val, y_val=y_v, epochs=10, patience=3)
        m2 = LCR(seed=13)
        m2.fit(inputs, y, lr=1e-3, inputs_val=val, y_val=y_v, epochs=10, patience=3)
        np.testing.assert_array_equal(m1.predict_proba(val), m2.predict_proba(val))


# ---------------------------------------------------------------------------
# 11. K5 / K10 / K20 / K50 support
# ---------------------------------------------------------------------------
class TestCandidateSizeSupport:
    @pytest.mark.parametrize("k", [5, 10, 20, 50])
    def test_supported_sizes(self, k):
        n, d = 7, 16
        z_q, z_i, scores = _make_inputs(n=n, k=k, d=d, seed=k)
        out = relation_features(scores, z_q, z_i, 1.5)
        assert out.r.shape == (n, M_COMPETITORS, len(RELATION_FEATURE_NAMES))
        assert out.gate.shape == (n, len(GATE_INPUT_NAMES))
        assert out.r.dtype == np.float64 and out.gate.dtype == np.float64
        assert out.winner_idx.dtype == np.int64 and out.competitor_idx.dtype == np.int64
        assert out.k == k
        for i in range(n):
            assert int(out.winner_idx[i]) not in set(out.competitor_idx[i].tolist())
            assert int(out.order[i, 0]) == int(out.winner_idx[i])
        np.testing.assert_allclose(
            out.r[:, :, 6],
            np.broadcast_to(
                np.arange(2, M_COMPETITORS + 2) / float(k), (n, M_COMPETITORS)
            ),
        )

    def test_k_below_five_rejected(self):
        z_q, z_i, scores = _make_inputs(n=4, k=4, d=8, seed=2)
        with pytest.raises(ValueError, match="K must be >= 5"):
            relation_features(scores, z_q, z_i, 1.0)

    def test_m_out_of_bounds_rejected(self):
        z_q, z_i, scores = _make_inputs(n=4, k=5, d=8, seed=2)
        with pytest.raises(ValueError, match=r"M must be in \[1, K-1\]"):
            relation_features(scores, z_q, z_i, 1.0, m=5)


# ---------------------------------------------------------------------------
# 12. no hard data in M1 train/tune
# ---------------------------------------------------------------------------
class TestZeroHardExposureGuard:
    def test_guard_passes_on_pure_val_calib_random(self):
        splits = np.array(["val_calib"] * 6)
        sids = np.array([10, 11, 12, 13, 14, 15])
        res = m1_training_guard(splits, sids, np.array([100, 101, 102]))
        assert res["pass"] is True
        assert res["n_rows"] == 6
        assert res["hard_overlap"] == 0
        assert res["ks"] == [5, 10]
        assert res["splits"] == ["val_calib"]

    def test_guard_rejects_non_val_calib_rows(self):
        splits = np.array(["val_calib", "testA", "val_calib"])
        with pytest.raises(AssertionError, match="outside val_calib"):
            m1_training_guard(splits, np.array([10, 11, 12]), np.array([100]))

    def test_guard_rejects_hard_sentence_overlap(self):
        splits = np.array(["val_calib"] * 3)
        with pytest.raises(AssertionError, match="SameCategory universe"):
            m1_training_guard(splits, np.array([10, 100, 12]), np.array([100]))

    def test_guard_rejects_unseen_k(self):
        splits = np.array(["val_calib"] * 3)
        sids = np.array([10, 11, 12])
        with pytest.raises(AssertionError, match="outside the M1 training regimes"):
            m1_training_guard(splits, sids, np.array([100]), train_ks=(5, 10), seen_ks=(5, 10, 20))

    def test_assert_no_hard_rows_disjoint_returns_zero(self):
        assert assert_no_hard_rows(np.array([1, 2, 3]), np.array([7, 8])) == 0
        with pytest.raises(AssertionError, match="ZERO-HARD-EXPOSURE"):
            assert_no_hard_rows(np.array([1, 7]), np.array([7, 8]))


# ---------------------------------------------------------------------------
# extra: the frozen protocol artifact stays consistent with the code
# ---------------------------------------------------------------------------
class TestProtocolArtifactFreeze:
    @pytest.mark.skipif(not PROTOCOL_PATH.exists(), reason="protocol artifact not present")
    def test_key_frozen_fields(self):
        doc = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        assert doc["base_commit"] == "0aaa29d"
        assert doc["method"]["local_competition_set"]["M"] == M_COMPETITORS
        assert doc["method"]["relation_vector"]["dim"] == REL_DIM
        assert doc["method"]["parameter_budget"] == {"hard_max": 50000, "target": 10000}
        assert doc["training_protocol"]["seeds"] == [1, 2, 3]
        assert doc["training_protocol"]["lr_grid"] == [0.0001, 0.0003, 0.001]
        assert doc["training_protocol"]["frozen_hyperparameters"]["M"] == 4
        assert (
            doc["training_protocol"]["frozen_hyperparameters"][
                "no_hyperparameter_search_beyond_lr_grid"
            ]
            is True
        )
        assert doc["statistics"]["replicates"] == 5000
        assert doc["M1_gates"]["GO_gate"]["delta_auroc_LCR_minus_E1b_ge"] == 0.01
        assert doc["M2_preregistration"]["status"] == "PREREGISTERED_NOT_RUN"
        assert len(doc["m0_tests_required_before_M1"]) == 12
