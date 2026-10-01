"""V2-M3 mixer tests (amendment sections 5-9 / 27-28).

Required checks, mapped to test classes:

1.  expert weights frozen / pure arithmetic .......... ``TestPurityAndInvariance``
2.  CDF fit train-only ............................... ``TestCompetitionIndex``
3.  mixer fit tune-only (m8 rejected) ................ ``TestMixerContract``
4.  m8 absent from fit ............................... ``TestMixerContract``
5.  beta >= 0, alpha in [0, 1], A in [0, 1] .......... ``TestAdaptiveMix`` / ``TestCompetitionIndex``
6.  StaticMix single constant / parameter budget ..... ``TestStaticMix`` / ``TestProtocolConsistency``
7.  same rows / shared bootstrap draws ............... ``TestMacroBootstrap``
8.  B0/B1/B2 formula identical (protocol vs module) .. ``TestProtocolConsistency``
9.  grounding invariant (inputs never modified) ...... ``TestPurityAndInvariance``

All tests must pass before the B0 developmental audit runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.experiment.phase0a import SampleStats, _ClusterSampler, paired_cluster_bootstrap  # noqa: E402
from ccg.mixture import (  # noqa: E402
    COMPETITION_DIRECTIONS,
    COMPETITION_FEATURES,
    MIXER_LEVELS,
    AdaptiveMix,
    CompetitionIndex,
    EqualMix,
    StaticMix,
    competition_features_from_sem14,
    macro_paired_cluster_bootstrap,
    softplus,
)
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

PROTOCOL_PATH = _REPO_ROOT / "results/v2_local_competition/m3_mixture/protocol_m3.json"


def _synthetic_index_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "winner_competitor_max_cos": rng.normal(0.5, 0.1, n),
        "winner_top2_cos": rng.normal(0.4, 0.1, n),
        "q_margin12": rng.normal(0.2, 0.05, n),
    }


def _synthetic_groups(
    n_per_level: int = 120, seed: int = 1, *, levels=(0, 2, 4), with_a: bool = True
):
    rng = np.random.default_rng(seed)
    groups = {}
    for level in levels:
        y = (rng.random(n_per_level) < 0.5).astype(np.float64)
        groups[int(level)] = {
            "p_r": rng.random(n_per_level),
            "p_c": rng.random(n_per_level),
            "y": y,
        }
        if with_a:
            groups[int(level)]["a"] = rng.random(n_per_level)
    return groups


class TestCompetitionIndex:
    def test_transform_in_unit_interval(self):
        index = CompetitionIndex().fit(_synthetic_index_data())
        a = index.transform(_synthetic_index_data(n=250, seed=3))
        assert a.shape == (250,)
        assert np.all(a >= 0.0) and np.all(a <= 1.0)

    def test_directions_are_frozen_and_inverted_correctly(self):
        assert COMPETITION_FEATURES == (
            "winner_competitor_max_cos", "winner_top2_cos", "q_margin12",
        )
        assert COMPETITION_DIRECTIONS == (+1, +1, -1)
        index = CompetitionIndex().fit(_synthetic_index_data())
        low = {"winner_competitor_max_cos": np.array([-10.0]),
               "winner_top2_cos": np.array([-10.0]),
               "q_margin12": np.array([10.0])}
        high = {"winner_competitor_max_cos": np.array([10.0]),
                "winner_top2_cos": np.array([10.0]),
                "q_margin12": np.array([-10.0])}
        a_low = index.transform(low)
        a_high = index.transform(high)
        assert a_low[0] == 0.0 and a_high[0] == 1.0
        # q_margin12 alone: smaller gap -> larger u3
        small = index.transform({"winner_competitor_max_cos": np.array([0.5]),
                                 "winner_top2_cos": np.array([0.4]),
                                 "q_margin12": np.array([-10.0])})
        large = index.transform({"winner_competitor_max_cos": np.array([0.5]),
                                 "winner_top2_cos": np.array([0.4]),
                                 "q_margin12": np.array([10.0])})
        assert small[0] > large[0]

    def test_transform_is_pure_and_train_only(self):
        train = _synthetic_index_data(seed=5)
        test = _synthetic_index_data(n=150, seed=6)
        index = CompetitionIndex().fit(train)
        first = index.transform(test)
        second = index.transform(test)
        np.testing.assert_array_equal(first, second)
        # refitting on other train data changes the map -> transform only reads the CDF
        other = CompetitionIndex().fit({k: v + 100.0 for k, v in train.items()})
        assert not np.allclose(first, other.transform(test))

    def test_feature_contract_and_transform_before_fit(self):
        index = CompetitionIndex()
        with pytest.raises(RuntimeError, match="before fit"):
            index.transform(_synthetic_index_data(n=10))
        with pytest.raises(ValueError, match="exactly"):
            CompetitionIndex().fit({"winner_top2_cos": np.zeros(4)})
        with pytest.raises(ValueError):
            CompetitionIndex().fit({**{k: np.zeros(4) for k in COMPETITION_FEATURES[:-1]},
                                    "other": np.zeros(4)})

    def test_sem14_extraction(self):
        names = list(v2feat.V2_PRIMARY_SEMANTIC_NAMES)
        block = np.random.default_rng(0).random((7, len(names)))
        features = competition_features_from_sem14(block, names)
        for name in COMPETITION_FEATURES:
            np.testing.assert_array_equal(features[name], block[:, names.index(name)])


class TestMixerContract:
    def test_m8_never_enters_fitting(self):
        groups = _synthetic_groups(levels=(0, 2, 4, 8))
        with pytest.raises(ValueError, match="m=8"):
            StaticMix().fit(groups)
        with pytest.raises(ValueError, match="m=8"):
            AdaptiveMix().fit(groups)

    def test_levels_must_be_subset_of_frozen_set(self):
        assert MIXER_LEVELS == (0, 2, 4)
        groups = _synthetic_groups(levels=(2,))
        mix = StaticMix().fit(groups)  # single level is allowed (subset)
        assert mix.fitted

    def test_level_sizes_may_differ_between_levels(self):
        # real severity cells differ in size; only the rows *within* a level must match
        groups = _synthetic_groups(n_per_level=100)
        padded = {m: dict(block) for m, block in groups.items()}
        for key in ("p_r", "p_c", "a"):
            padded[2][key] = np.append(padded[2][key], padded[2][key][0])
        padded[2]["y"] = np.append(padded[2]["y"], padded[2]["y"][0])
        assert StaticMix().fit(padded).fitted
        assert AdaptiveMix().fit(padded).fitted

    def test_row_count_and_probability_validation(self):
        groups = _synthetic_groups()
        broken = {m: dict(block) for m, block in groups.items()}
        broken[2]["p_c"] = broken[2]["p_c"][:-1]
        with pytest.raises(ValueError, match="row count"):
            StaticMix().fit(broken)
        out_of_range = {m: dict(block) for m, block in groups.items()}
        out_of_range[0]["p_r"] = out_of_range[0]["p_r"] + 5.0
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            StaticMix().fit(out_of_range)

    def test_y_must_be_binary(self):
        groups = _synthetic_groups()
        groups[4]["y"] = np.full_like(groups[4]["y"], 0.3)
        with pytest.raises(ValueError, match="binary"):
            StaticMix().fit(groups)


class TestAdaptiveMix:
    def test_beta_nonnegative_for_extreme_theta(self):
        for theta in (-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6):
            beta = float(softplus(theta))
            assert np.isfinite(beta) and beta >= 0.0
        assert float(softplus(-100.0)) > 0.0

    def test_alpha_in_unit_interval(self):
        for theta, tau in ((0.0, 0.5), (50.0, 0.5), (1e6, 0.0), (-1e6, 0.0), (2.0, 10.0)):
            mix = AdaptiveMix(theta=theta, tau=tau)
            alphas = mix.alpha(np.array([0.0, 0.25, 0.5, 1.0]))
            assert np.all(alphas >= 0.0) and np.all(alphas <= 1.0)

    def test_alpha_is_monotone_increasing_in_a(self):
        mix = AdaptiveMix(theta=10.0, tau=0.5)
        alphas = mix.alpha(np.array([0.0, 0.5, 1.0]))
        assert alphas[0] < alphas[1] < alphas[2]

    def test_predict_interpolates_the_experts(self):
        mix = AdaptiveMix(theta=50.0, tau=0.5)  # steep: alpha(0) ~ 0, alpha(1) ~ 1
        p_r = np.array([0.1, 0.9])
        p_c = np.array([0.9, 0.1])
        out = mix.predict_proba(p_r, p_c, np.array([0.0, 1.0]))
        # a = 0 -> alpha ~ 0 -> the random expert; a = 1 -> alpha ~ 1 -> the curriculum expert
        np.testing.assert_allclose(out, np.array([p_r[0], p_c[1]]), atol=1e-6)
        assert out[0] == pytest.approx(p_r[0], abs=1e-6)
        assert out[1] == pytest.approx(p_c[1], abs=1e-6)

    def test_predict_requires_a(self):
        mix = AdaptiveMix()
        with pytest.raises(ValueError, match="competition index"):
            mix.predict_proba(np.array([0.5]), np.array([0.5]), None)

    def test_n_parameters_and_fit_picks_up_adaptation(self):
        mix = AdaptiveMix()
        assert mix.n_parameters() == 2
        rng = np.random.default_rng(7)
        n = 240
        a = rng.random(n)
        low = a < 0.5
        y = (rng.random(n) < 0.5).astype(np.float64)
        # low competition: p_R exact; high competition: p_C exact (reversed otherwise)
        p_r = np.where(low, y, 1.0 - y)
        p_c = np.where(low, 1.0 - y, y)
        groups = {
            m: {"p_r": p_r[s], "p_c": p_c[s], "y": y[s], "a": a[s]}
            for m, s in zip((0, 2, 4), np.array_split(np.arange(n), 3))
        }
        mix.fit(groups)
        assert mix.fitted and mix.success
        assert np.isfinite(mix.nll)
        assert mix.nll <= _balanced_reference_nll(groups, 0.0, 0.5) + 1e-9
        alphas = mix.alpha(np.array([0.0, 1.0]))
        assert alphas[1] > alphas[0]


def _balanced_reference_nll(groups, theta: float, tau: float) -> float:
    beta = float(softplus(theta))
    parts = []
    for block in groups.values():
        weight = 1.0 / (1.0 + np.exp(-beta * (block["a"] - tau)))
        p = (1.0 - weight) * block["p_r"] + weight * block["p_c"]
        p = np.clip(p, 1e-12, 1 - 1e-12)
        y = block["y"]
        parts.append(float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))))
    return float(np.mean(parts))


class TestStaticMix:
    def test_parameter_budget(self):
        assert EqualMix().n_parameters() == 0
        assert StaticMix().n_parameters() == 1
        assert AdaptiveMix().n_parameters() == 2

    def test_equal_mix_is_exactly_half_half(self):
        p_r = np.array([0.0, 0.2, 1.0])
        p_c = np.array([1.0, 0.4, 0.0])
        np.testing.assert_allclose(EqualMix().predict_proba(p_r, p_c), [0.5, 0.3, 0.5])

    def test_fit_recovers_direction(self):
        rng = np.random.default_rng(11)
        n = 300
        y = (rng.random(n) < 0.5).astype(np.float64)
        good = np.clip(y + rng.normal(0, 0.01, n), 0, 1)
        bad = rng.random(n)
        splits = np.array_split(np.arange(n), 3)
        groups_bad_r = {
            m: {"p_r": bad[s], "p_c": good[s], "y": y[s]} for m, s in zip((0, 2, 4), splits)
        }
        up = StaticMix().fit(groups_bad_r)
        assert up.fitted and 0.0 <= up.c <= 1.0
        assert up.c > 0.8
        groups_bad_c = {
            m: {"p_r": good[s], "p_c": bad[s], "y": y[s]} for m, s in zip((0, 2, 4), splits)
        }
        down = StaticMix().fit(groups_bad_c)
        assert down.c < 0.2

    def test_alpha_is_constant(self):
        mix = StaticMix(c=0.3)
        alphas = mix.alpha(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(alphas, 0.3)

    def test_invalid_c_rejected(self):
        with pytest.raises(ValueError, match="c must lie"):
            StaticMix(c=1.5).predict_proba(np.array([0.5]), np.array([0.5]))


class TestMacroBootstrap:
    def _pair(self, rng, n):
        return rng.random(n), rng.random(n), (rng.random(n) < 0.5).astype(np.float64), rng.integers(0, 40, n)

    def test_single_severity_equals_paired_bootstrap_bit_for_bit(self):
        rng = np.random.default_rng(21)
        n = 200
        conf_a, conf_b, correct, clusters = self._pair(rng, n)
        metric = reval._metric_fn("auroc_correct")
        macro = macro_paired_cluster_bootstrap(
            metric, [((conf_a, correct), (conf_b, correct))], clusters,
            n_replicates=200, seed=0,
        )
        reference = paired_cluster_bootstrap(
            metric,
            SampleStats.from_conf_correct(conf_a, correct),
            SampleStats.from_conf_correct(conf_b, correct),
            clusters, n_replicates=200, seed=0,
        )
        np.testing.assert_array_equal(macro["replicates"], reference["replicates"])
        assert macro["diff"] == reference["diff"]
        assert macro["ci_low"] == reference["ci_low"]
        assert macro["ci_high"] == reference["ci_high"]

    def test_draws_are_shared_across_severity(self):
        rng = np.random.default_rng(22)
        n = 180
        conf_a1, conf_b1, correct1, clusters = self._pair(rng, n)
        conf_a2, conf_b2, correct2, _ = self._pair(rng, n)
        metric = reval._metric_fn("auroc_correct")
        result = macro_paired_cluster_bootstrap(
            metric,
            [((conf_a1, correct1), (conf_b1, correct1)),
             ((conf_a2, correct2), (conf_b2, correct2))],
            clusters, n_replicates=50, seed=0,
        )
        sampler = _ClusterSampler(np.asarray(clusters))
        manual_rng = np.random.default_rng(0)
        manual = np.empty(50)
        for index in range(50):
            draw = sampler.draw(manual_rng)
            d1 = float(metric(SampleStats.from_conf_correct(conf_a1, correct1).take(draw))) - float(
                metric(SampleStats.from_conf_correct(conf_b1, correct1).take(draw))
            )
            d2 = float(metric(SampleStats.from_conf_correct(conf_a2, correct2).take(draw))) - float(
                metric(SampleStats.from_conf_correct(conf_b2, correct2).take(draw))
            )
            manual[index] = 0.5 * (d1 + d2)
        # the two-severity pairs are (level1, level2) = ((a1,b1), (a2,b2)) sequenced twice;
        # the first two entries are the (a,b) pairs of level 1, the next two of level 2
        assert result["n_levels"] == 2
        np.testing.assert_allclose(result["replicates"], manual, rtol=0, atol=0)

    def test_row_mismatch_rejected(self):
        rng = np.random.default_rng(23)
        conf_a1, conf_b1, correct1, clusters = self._pair(rng, 100)
        conf_a2, conf_b2, correct2, _ = self._pair(rng, 90)
        metric = reval._metric_fn("auroc_correct")
        with pytest.raises(ValueError, match="row count"):
            macro_paired_cluster_bootstrap(
                metric,
                [((conf_a1, correct1), (conf_b1, correct1)),
                 ((conf_a2, correct2), (conf_b2, correct2))],
                clusters, n_replicates=10, seed=0,
            )


class TestProtocolConsistency:
    def test_protocol_constants_match_the_module(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        index = protocol["competition_index"]
        assert [f["name"] for f in index["features"]] == list(COMPETITION_FEATURES)
        expected = ["up" if d > 0 else "down" for d in COMPETITION_DIRECTIONS]
        assert [f["direction"].split()[0] for f in index["features"]] == expected
        assert protocol["adaptive_mix"]["n_trainable"] == AdaptiveMix().n_parameters()
        assert protocol["controls"]["static_mix"]["n_trainable"] == StaticMix().n_parameters()
        assert protocol["controls"]["equal_mix"]["n_trainable"] == EqualMix().n_parameters()
        assert protocol["fitting"]["m8_exclusions"]
        bootstrap = protocol["metrics"]["bootstrap"]
        assert bootstrap["shared_draws_across_severity"] is True
        assert bootstrap["replicates"] == 5000 and bootstrap["seed"] == 0
        assert protocol["methodological_boundary"]["b0_label"] == "POST-HOC DEVELOPMENTAL"
        assert protocol["parameter_budget"]["hidden_layers"] == 0


class TestPurityAndInvariance:
    def test_predict_never_modifies_its_inputs(self):
        rng = np.random.default_rng(31)
        p_r = rng.random(20)
        p_c = rng.random(20)
        a = rng.random(20)
        copies = (p_r.copy(), p_c.copy(), a.copy())
        for mix in (EqualMix(), StaticMix(c=0.4), AdaptiveMix(theta=1.0, tau=0.5)):
            out = mix.predict_proba(p_r, p_c, a)
            assert out.shape == p_r.shape
        np.testing.assert_array_equal(p_r, copies[0])
        np.testing.assert_array_equal(p_c, copies[1])
        np.testing.assert_array_equal(a, copies[2])

    def test_predictions_are_deterministic_pure_arithmetic(self):
        rng = np.random.default_rng(32)
        p_r, p_c, a = rng.random(30), rng.random(30), rng.random(30)
        mix = AdaptiveMix(theta=2.5, tau=0.4)
        first = mix.predict_proba(p_r, p_c, a)
        second = mix.predict_proba(p_r, p_c, a)
        np.testing.assert_array_equal(first, second)

    def test_mixture_output_stays_in_unit_interval(self):
        rng = np.random.default_rng(33)
        p_r, p_c, a = rng.random(50), rng.random(50), rng.random(50)
        for mix in (EqualMix(), StaticMix(c=0.7), AdaptiveMix(theta=5.0, tau=0.5)):
            out = mix.predict_proba(p_r, p_c, a)
            assert np.all(out >= 0.0) and np.all(out <= 1.0)
