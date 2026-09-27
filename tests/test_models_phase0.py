"""Phase 0 baselines: cosine, independent MLP, score statistics, C3 structure."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.models.base import (
    BaseScorer,
    Scorer,
    as_candidate_matrix,
    cosine_similarity,
    softmax_np,
)
from ccg.models.cosine import CosineScorer
from ccg.models.deepsets import GATE_MESSAGE as DEEPSETS_GATE
from ccg.models.deepsets import DeepSetsReliability, DeepSetsReranker
from ccg.models.independent import (
    PARAM_BUDGET,
    IndependentMLPScorer,
    ScoringExample,
    build_candidate_inputs,
    candidate_input_dim,
    geometry_features,
)
from ccg.models.presence import (
    FlatNoneClassifier,
    FactorizedPresenceModel,
    SetAwarePresenceModel,
    StatsPresenceModel,
    factorized_probabilities,
)
from ccg.models.stats_calibrator import (
    STATS_FIELDS,
    StatsOnlyCalibrator,
    extract_score_stats,
    extract_score_stats_batch,
    stats_dim,
)

QUERY = np.array([1.0, 0.0, 0.0, 2.0], dtype=np.float32)
CANDIDATES = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# shared interface helpers
# ---------------------------------------------------------------------------
def test_softmax_is_stable_and_temperature_aware():
    p = softmax_np(np.array([1000.0, 1001.0, 999.0]))
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)
    assert p.argmax() == 1
    T2 = softmax_np(np.array([2.0, 0.0]), temperature=2.0)
    assert T2[0] == pytest.approx(np.exp(1.0) / (np.exp(1.0) + 1.0), rel=1e-5)
    uniform = softmax_np(np.zeros(5))
    np.testing.assert_allclose(uniform, 0.2, atol=1e-6)
    with pytest.raises(ValueError, match="positive"):
        softmax_np(np.zeros(3), temperature=0.0)


def test_cosine_similarity_and_shape_coercion():
    sims = cosine_similarity(QUERY, CANDIDATES)
    norm_q = float(np.linalg.norm(QUERY))
    assert sims[0] == pytest.approx(1.0 / norm_q, rel=1e-5)
    assert sims[2] == pytest.approx(-1.0 / norm_q, rel=1e-5)
    assert sims[1] == pytest.approx(0.0, abs=1e-6)
    assert sims[3] == pytest.approx(6.0 / (norm_q * 3.0), rel=1e-5)
    assert sims.shape == (4,)
    # a 1-D candidate matrix is one candidate
    assert as_candidate_matrix(CANDIDATES[0]).shape == (1, 4)
    with pytest.raises(ValueError, match="near-zero norm"):
        cosine_similarity(np.zeros(4), CANDIDATES)
    with pytest.raises(ValueError, match="feature dim mismatch"):
        cosine_similarity(np.zeros(5), CANDIDATES)
    with pytest.raises(ValueError, match=r"\[K,D\]"):
        as_candidate_matrix(np.zeros((2, 2, 2)))


# ---------------------------------------------------------------------------
# B1 cosine scorer
# ---------------------------------------------------------------------------
def test_cosine_scorer_shape_and_values():
    scorer = CosineScorer()
    logits = scorer.score(QUERY, CANDIDATES)
    assert logits.shape == (4,) and logits.dtype == np.float32
    np.testing.assert_allclose(logits, cosine_similarity(QUERY, CANDIDATES), atol=1e-6)
    assert isinstance(scorer, Scorer) and isinstance(scorer, BaseScorer)
    assert scorer.name == "b1_clip_cosine"

    hot = CosineScorer(temperature=2.0)
    np.testing.assert_allclose(hot.score(QUERY, CANDIDATES), logits / 2.0, atol=1e-6)
    np.testing.assert_allclose(hot.similarities(QUERY, CANDIDATES), logits, atol=1e-6)
    # geometry is accepted for interface parity and must be ignored
    np.testing.assert_allclose(
        scorer.score(QUERY, CANDIDATES, geometry=np.zeros((4, 5))), logits, atol=1e-7
    )
    with pytest.raises(ValueError, match="positive"):
        CosineScorer(temperature=0.0)


def test_cosine_probabilities_pick_the_best_candidate():
    probs = CosineScorer().probabilities(QUERY, CANDIDATES)
    assert probs.sum() == pytest.approx(1.0)
    # candidate 3 shares the query direction and has the largest cosine
    assert probs.argmax() == 3


# ---------------------------------------------------------------------------
# B3 independent MLP
# ---------------------------------------------------------------------------
def test_mlp_parameter_count_is_under_the_budget():
    scorer = IndependentMLPScorer()
    count = scorer.num_parameters()
    assert count < 300_000, "Phase 0 freezes the trainable decision module below 0.3M params"
    assert count == 214_145 == PARAM_BUDGET - 85_855
    # analytic form: (1542*128+128) + (128*128+128) + (128+1)
    assert candidate_input_dim(512, 5) == 1542
    assert (1542 * 128 + 128) + (128 * 128 + 128) + (128 + 1) == count
    usage = scorer.param_budget_usage()
    assert usage["usage"] < 1.0 and usage["analytic"] == usage["num_parameters"]
    assert scorer.config["num_parameters"] == count


def test_mlp_rejects_an_oversized_variant():
    with pytest.raises(ValueError, match="budget"):
        IndependentMLPScorer(hidden_dim=256)
    with pytest.raises(ValueError, match="positive"):
        IndependentMLPScorer(temperature=0.0)


def test_mlp_forward_shape_and_geometry():
    scorer = IndependentMLPScorer(feature_dim=4, hidden_dim=8, geo_dim=5)
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 25], [1, 2, 3, 4], [0, 0, 1, 1]], np.float32)
    geometry = geometry_features(boxes, image_size=(640, 480))
    assert geometry.shape == (4, 5)
    np.testing.assert_allclose(geometry[1, 2], 10.0 / 640.0, rtol=1e-5)
    np.testing.assert_allclose(geometry[1, 3], 20.0 / 480.0, rtol=1e-5)
    np.testing.assert_allclose(geometry[1, 4], (10.0 / 640.0) * (20.0 / 480.0), rtol=1e-5)
    with_obj = geometry_features(boxes, objectness=np.full(4, 0.5, np.float32))
    assert with_obj.shape == (4, 6)
    with pytest.raises(ValueError, match="image_size must be positive"):
        geometry_features(boxes, image_size=(0, 480))
    with pytest.raises(ValueError, match=r"\[K,4\]"):
        geometry_features(np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError, match="expected 4"):
        geometry_features(boxes, objectness=np.full(3, 0.5, np.float32))

    logits = scorer.score(QUERY, CANDIDATES, geometry)
    assert logits.shape == (4,) and np.isfinite(logits).all()
    probs = scorer.probabilities(QUERY, CANDIDATES, geometry)
    assert probs.shape == (4,) and probs.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"\[K=4,5\]"):
        scorer.score(QUERY, CANDIDATES, geometry[:, :4])


def test_mlp_is_candidate_independent_under_permutation():
    """score(x)[pi] == score(x[pi]): the network may not read its neighbours."""
    rng = np.random.default_rng(0)
    scorer = IndependentMLPScorer(feature_dim=8, hidden_dim=16, geo_dim=5)
    query = rng.normal(size=8).astype(np.float32)
    candidates = rng.normal(size=(6, 8)).astype(np.float32)
    boxes = rng.uniform(0, 100, size=(6, 4)).astype(np.float32)
    boxes[:, 2:] += boxes[:, :2]
    geometry = geometry_features(boxes, image_size=(640, 480))

    base = scorer.score(query, candidates, geometry)
    permutation = rng.permutation(6)
    shuffled = scorer.score(
        query, candidates[permutation], geometry[permutation]
    )
    np.testing.assert_allclose(base[permutation], shuffled, atol=1e-6)

    # adding/removing a distractor leaves the other candidates' scores untouched
    extra = scorer.score(query, np.concatenate([candidates, rng.normal(size=(1, 8)).astype(np.float32)]),
                         np.concatenate([geometry, geometry[:1]]))
    np.testing.assert_allclose(extra[:6], base, atol=1e-6)

    # no BatchNorm / dropout-style coupling may appear inside the module
    kinds = {type(layer).__name__ for layer in scorer.net}
    assert kinds <= {"Linear", "ReLU"}, kinds


def test_candidate_inputs_layout():
    inputs = build_candidate_inputs(QUERY, CANDIDATES, geo_dim=5)
    d_in = candidate_input_dim(4, 5)
    assert inputs.shape == (4, d_in) and d_in == 3 * 4 + 1 + 5
    d = 4
    np.testing.assert_allclose(inputs[:, 0:d], np.tile(QUERY, (4, 1)), atol=1e-6)
    np.testing.assert_allclose(inputs[:, d : 2 * d], CANDIDATES, atol=1e-6)
    np.testing.assert_allclose(inputs[:, 2 * d : 3 * d], CANDIDATES * QUERY, atol=1e-6)
    np.testing.assert_allclose(
        inputs[:, 3 * d], cosine_similarity(QUERY, CANDIDATES), atol=1e-6
    )
    np.testing.assert_allclose(inputs[:, 3 * d + 1 :], np.zeros((4, 5)), atol=1e-7)


def test_mlp_training_skeleton_runs_and_reports_skips():
    rng = np.random.default_rng(3)
    scorer = IndependentMLPScorer(feature_dim=8, hidden_dim=8, geo_dim=5)
    examples = []
    for i in range(6):
        candidates = rng.normal(size=(4, 8)).astype(np.float32)
        target = int(i % 4)
        examples.append(
            ScoringExample(
                ref_id=i,
                query_feature=rng.normal(size=8).astype(np.float32),
                candidate_features=candidates,
                geometry=geometry_features(
                    rng.uniform(0, 100, size=(4, 4)).astype(np.float32),
                    image_size=(640, 480),
                ),
                target_index=target,
            )
        )
    absent = ScoringExample(
        ref_id=99,
        query_feature=rng.normal(size=8).astype(np.float32),
        candidate_features=rng.normal(size=(4, 8)).astype(np.float32),
        geometry=None,
        target_index=None,
    )
    history = scorer.train(examples + [absent], epochs=2, lr=1e-2, seed=0)
    assert len(history["loss_per_epoch"]) == 2
    assert history["num_skipped_absent"] == 1, "target-absent sets are reported, not hidden"
    assert history["num_examples"] == 7
    assert history["num_parameters"] == scorer.num_parameters()
    assert np.isfinite(history["loss_per_epoch"]).all()


def test_mlp_state_roundtrip_is_exact(tmp_path):
    scorer = IndependentMLPScorer(feature_dim=4, hidden_dim=8, geo_dim=5)
    logits = scorer.score(QUERY, CANDIDATES)
    path = tmp_path / "b3.npz"
    scorer.save(path)
    rebuilt = IndependentMLPScorer(feature_dim=4, hidden_dim=8, geo_dim=5, seed=1234)
    assert not np.allclose(rebuilt.score(QUERY, CANDIDATES), logits)
    rebuilt.load(path)
    np.testing.assert_allclose(rebuilt.score(QUERY, CANDIDATES), logits, atol=1e-7)


# ---------------------------------------------------------------------------
# score statistics (C3 input, pure numpy, fully implemented)
# ---------------------------------------------------------------------------
def test_score_stats_uniform_logit_hand_example():
    stats = extract_score_stats(np.zeros(4))
    assert stats.shape == (stats_dim(),) and stats.dtype == np.float32
    expected = {
        "K": 4.0,
        "max_score": 0.0,
        "top1_top2_margin": 0.0,
        "entropy": float(np.log(4)),
        "mean_score": 0.0,
        "std_score": 0.0,
        "max_softmax": 0.25,
    }
    for i, field in enumerate(STATS_FIELDS):
        assert stats[i] == pytest.approx(expected[field], abs=1e-6), field


def test_score_stats_two_candidate_hand_example():
    scores = np.array([1.0, 0.0], dtype=np.float32)
    p1 = float(np.exp(1.0) / (np.exp(1.0) + 1.0))
    p2 = 1.0 - p1
    stats = extract_score_stats(scores)
    np.testing.assert_allclose(
        stats,
        [
            2.0,
            1.0,  # max_score
            1.0,  # top1_top2_margin
            -(p1 * np.log(p1) + p2 * np.log(p2)),  # entropy
            0.5,  # mean_score
            0.5,  # std_score (population)
            p1,  # max_softmax
        ],
        atol=1e-6,
    )
    # the temperature knob rescales the softmax, not the raw statistics
    hot = extract_score_stats(scores, temperature=2.0)
    assert hot[3] > stats[3] and hot[6] < stats[6]
    assert hot[1] == pytest.approx(1.0) and hot[4] == pytest.approx(0.5)


def test_score_stats_top3_block():
    scores = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    probs = softmax_np(scores)
    stats = extract_score_stats(scores, include_top3=True)
    assert stats.shape == (stats_dim(include_top3=True),)
    assert stats.shape[0] == 10
    np.testing.assert_allclose(stats[7:], [0.0, probs[2], probs[0] - probs[2]], atol=1e-6)

    small = extract_score_stats(np.array([1.0, 0.0]), include_top3=True)
    p = softmax_np(np.array([1.0, 0.0]))
    # K < 3 duplicates the weakest entry instead of inventing probability mass
    np.testing.assert_allclose(small[7:], [0.0, p[1], p[0] - p[1]], atol=1e-6)
    assert small[0] == pytest.approx(2.0), "the real K stays visible"

    single = extract_score_stats(np.array([3.0]), include_top3=True)
    assert single[2] == pytest.approx(0.0) and single[6] == pytest.approx(1.0)
    assert single[7] == pytest.approx(3.0)


def test_score_stats_errors_and_batch():
    with pytest.raises(ValueError, match="at least one"):
        extract_score_stats(np.empty(0))
    with pytest.raises(ValueError, match="non-finite"):
        extract_score_stats(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="positive"):
        extract_score_stats(np.array([1.0]), temperature=-1.0)
    batch = extract_score_stats_batch([np.zeros(2), np.array([1.0, 0.0, -1.0])])
    assert batch.shape == (2, stats_dim())
    assert batch[0, 0] == 2.0 and batch[1, 0] == 3.0
    with pytest.raises(ValueError, match="empty sequence"):
        extract_score_stats_batch([])


# ---------------------------------------------------------------------------
# C3 structure (forward only, training is gated)
# ---------------------------------------------------------------------------
def test_stats_only_calibrator_structure_and_gate():
    model = StatsOnlyCalibrator(hidden_dim=16)
    assert model.input_dim == stats_dim() == 7
    stats = extract_score_stats_batch([np.zeros(3), np.array([1.0, 2.0, 3.0])])
    assert model.forward_raw(stats).shape == (2,)
    p_correct = model.predict(stats)
    assert p_correct.shape == (2,) and ((p_correct >= 0) & (p_correct <= 1)).all()
    # a single set is accepted as a 1-D vector
    assert model.predict(extract_score_stats(np.zeros(3))).shape == (1,)
    via_scores = model.predict_from_scores([np.zeros(3), np.array([1.0, 2.0, 3.0])])
    np.testing.assert_allclose(via_scores, p_correct, atol=1e-7)

    temp_model = StatsOnlyCalibrator(hidden_dim=16, output_mode="temperature")
    temps = temp_model.predict(stats)
    assert (temps > 0).all() and temps.min() >= temp_model.temperature_floor
    with pytest.raises(ValueError, match="output_mode"):
        StatsOnlyCalibrator(output_mode="logits")
    with pytest.raises(ValueError, match="columns"):
        model.predict(np.zeros((2, 5)))
    with pytest.raises(NotImplementedError, match="Gate Q1"):
        model.fit(stats)
    description = model.description
    assert description["input_dim"] == 7 and description["num_parameters"] == model.num_parameters()
    assert description["stats_fields"] == list(STATS_FIELDS)
    top3 = StatsOnlyCalibrator(hidden_dim=8, include_top3=True)
    assert top3.input_dim == stats_dim(include_top3=True) == 10
    assert top3.description["stats_fields"][-1] == "top3_gap"


# ---------------------------------------------------------------------------
# gated placeholders must stay unimplemented
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls",
    [DeepSetsReliability, DeepSetsReranker, FlatNoneClassifier, StatsPresenceModel,
     SetAwarePresenceModel, FactorizedPresenceModel],
)
def test_phase1_and_phase2_models_are_gated_stubs(cls):
    with pytest.raises(NotImplementedError, match="Gate|Phase 2"):
        cls()
    assert hasattr(cls, "name")


def test_gated_modules_do_not_import_torch_heavily():
    """The placeholder docstrings freeze the design; importing them is always safe."""
    import ccg.models.deepsets as deepsets
    import ccg.models.presence as presence

    assert "Phase 1" in deepsets.__doc__ and "Phase 2" in presence.__doc__
    assert DEEPSETS_GATE.startswith("DeepSets is a Phase 1 model")
    conditional = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    probs, p_none = factorized_probabilities(conditional, 0.8)
    np.testing.assert_allclose(probs, 0.8 * conditional, atol=1e-6)
    assert p_none == pytest.approx(0.2)
    assert probs.sum() + p_none == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        factorized_probabilities(conditional, 1.5)
    with pytest.raises(ValueError, match="at least one"):
        factorized_probabilities(np.empty(0), 0.5)
