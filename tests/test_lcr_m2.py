"""M2 implementation tests for the V2-M curriculum stage (round-request section 22).

The required checks, mapped to test classes:

1. train / tune / test image disjointness ...... ``TestManifestFreeze`` / ``TestSplitIsolation``
2. level manifests deterministic ............... ``TestManifestFreeze`` (re-derivation + hashes)
3. GT category absent from model input ......... ``TestGTCategoryBoundary``
4. regime-balanced loss ........................ ``TestRegimeBalancedLoss``
5. m=8 never enters model selection ............ ``TestM8NeverInSelection``
6. all four models share the same curriculum ... ``TestSharedCurriculum``
7. architecture hash identical to M1 ........... ``TestArchitectureFrozen``
8. hyperparameter grid unchanged ............... ``TestHyperparameterGrid``
9. bootstrap draws shared ...................... ``TestBootstrapDrawsShared``
10. grounding scores unchanged ................. ``TestGroundingScoresUnchanged``

All tests must pass before the M2 run starts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.lcr import AggregateMLP, LCR, LCRNoGate, relation_features  # noqa: E402
from ccg.lcr.audit import grounding_fingerprint  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import models as rmodels  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M2 = _load_module("run_v2m_m2", "scripts/run_v2m_m2.py")
BUILD = _load_module("build_v2m_m2_manifests", "scripts/build_v2m_m2_manifests.py")
M1 = _load_module("run_v2m_m1", "scripts/run_v2m_m1.py")

FREEZE_PATH = _REPO_ROOT / "results/v2_local_competition/m2_curriculum/manifests/manifest_freeze.json"
M0_PATH = _REPO_ROOT / "results/v2_local_competition/m0_architecture.json"
PROTOCOL_PATH = _REPO_ROOT / "results/v2_local_competition/protocol.json"
EXPB_PREDICTIONS = _REPO_ROOT / "results/phase1f_hard_semantic/predictions"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _lcr_inputs(n: int = 8, seed: int = 3):
    rng = np.random.default_rng(seed)
    z_q = _l2_normalize(rng.standard_normal((n, 8)))
    z_i = _l2_normalize(rng.standard_normal((n, 5, 8)))
    scores = rng.standard_normal((n, 5))
    out = relation_features(scores, z_q, z_i, 1.0)
    return {"b": np.zeros(n), "r": out.r, "gate": out.gate}


# ---------------------------------------------------------------------------
# 1 + 2. manifest freeze: determinism, K mapping, isolation
# ---------------------------------------------------------------------------
class TestManifestFreeze:
    def test_freeze_hash_verification_passes(self):
        doc = M2._load_freeze(lambda message: None)
        assert doc["k_level"] == 10
        assert doc["levels_train"] == [0, 2, 4]
        assert doc["levels_test"] == [0, 2, 4, 8]

    def test_k_mapping_reconciliation(self):
        doc = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        for level in (0, 2, 4):
            assert doc["k_mapping"]["train"][f"m{level}"] == M2.K_LEVEL == 10
        for level in (0, 2, 4, 8):
            assert doc["k_mapping"]["test"][f"m{level}"] == M2.K_LEVEL == 10

    def test_level_tables_deterministic_rederivation(self):
        """``--verify`` path: re-derive the val tables and compare to the freeze."""
        assert BUILD.verify() == 0

    def test_freeze_isolation_zero(self):
        doc = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        assert all(int(v) == 0 for v in doc["disjointness"].values())


class TestSplitIsolation:
    def test_train_tune_test_images_pairwise_disjoint(self):
        derived = BUILD.derive_level_tables()
        cohort = derived["cohort"]
        split = derived["split"]
        train_mask = split.row_mask(cohort.eval_split, cohort.image_id, kind="reliability_train")
        tune_mask = split.row_mask(cohort.eval_split, cohort.image_id, kind="reliability_tune")
        train_images = set(int(i) for i in np.asarray(cohort.image_id)[train_mask])
        tune_images = set(int(i) for i in np.asarray(cohort.image_id)[tune_mask])
        cohort_hc = shard.load_hard_cohort(manifests_root=_REPO_ROOT / "cache/manifests")
        rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
        test_images = set(int(i) for i in np.asarray(cohort_hc.image_id)[rows_same8])
        assert not (train_images & tune_images)
        assert not (train_images & test_images)
        assert not (tune_images & test_images)

    def test_curriculum_rows_stay_inside_val_calib(self):
        derived = BUILD.derive_level_tables()
        cohort = derived["cohort"]
        for m in (0, 2, 4):
            rows = derived["tables"][f"m{m}_train"]["rows"]
            splits = set(str(s) for s in np.asarray(cohort.eval_split)[rows])
            assert splits == {"val_calib"}
            rows_tune = derived["tables"][f"m{m}_tune"]["rows"]
            splits_tune = set(str(s) for s in np.asarray(cohort.eval_split)[rows_tune])
            assert splits_tune == {"val_calib"}


# ---------------------------------------------------------------------------
# 3. GT category boundary
# ---------------------------------------------------------------------------
class TestGTCategoryBoundary:
    @pytest.mark.parametrize(
        "variant,forbidden",
        [
            ("LCR", "category"),
            ("LCR", "category_id"),
            ("LCR-noGate", "gt_category"),
            ("Aggregate-MLP", "regime"),
        ],
    )
    def test_models_reject_forbidden_keys(self, variant, forbidden):
        if variant == "LCR":
            model, inputs = LCR(seed=0), _lcr_inputs()
        elif variant == "LCR-noGate":
            model, inputs = LCRNoGate(seed=0), {"b": np.zeros(8), "r": _lcr_inputs()["r"]}
        else:
            model, inputs = AggregateMLP(seed=0), {"x": np.zeros((8, 31))}
        with pytest.raises(ValueError, match="forbidden input key"):
            model.predict_proba({**inputs, forbidden: np.zeros(8)})

    def test_m2_input_contract_guard(self):
        ok = {m: {"b": np.zeros(4), "r": np.zeros((4, 4, 7)), "gate": np.zeros((4, 3))} for m in (0, 2, 4)}
        M2._assert_model_inputs(ok, ("b", "r", "gate"))
        bad = {m: dict(v, category=np.zeros(4)) for m, v in ok.items()}
        with pytest.raises(AssertionError, match="contract violated"):
            M2._assert_model_inputs(bad, ("b", "r", "gate"))
        extra = {m: {"b": np.zeros(4)} for m in (0, 2, 4)}
        with pytest.raises(AssertionError, match="contract violated"):
            M2._assert_model_inputs(extra, ("b", "r", "gate"))


# ---------------------------------------------------------------------------
# 4. regime-balanced loss
# ---------------------------------------------------------------------------
class TestRegimeBalancedLoss:
    def test_train_and_val_weight_formulas(self):
        sizes = [10, 20, 30]
        s = M2.balanced_train_weights(sizes)
        v = M2.balanced_val_weights(sizes)
        expected_s = np.concatenate([np.full(n, 60.0 / (3.0 * n)) for n in sizes])
        expected_v = np.concatenate([np.full(n, 1.0 / (3.0 * n)) for n in sizes])
        np.testing.assert_allclose(s, expected_s, rtol=0, atol=0)
        np.testing.assert_allclose(v, expected_v, rtol=0, atol=0)
        assert np.isclose(float(s.mean()), 1.0)
        assert np.isclose(float(v.sum()), 1.0)

    def test_batch_mean_estimator_equals_balanced_mean(self):
        rng = np.random.default_rng(0)
        sizes = [37, 41, 53]
        losses = rng.random(int(np.sum(sizes)))
        s = M2.balanced_train_weights(sizes)
        v = M2.balanced_val_weights(sizes)
        reference = M2.balanced_mean_loss(losses, sizes)
        assert np.isclose(float(np.mean(s * losses)), reference, atol=1e-12)
        assert np.isclose(float(np.sum(v * losses)), reference, atol=1e-12)

    def test_non_uniform_weights_change_the_fit(self):
        """Two regimes with opposite label balance: balancing must move the model."""
        rng = np.random.default_rng(1)
        n_a, n_b = 60, 240
        x = rng.standard_normal((n_a + n_b, 3))
        y = np.concatenate(
            [
                (rng.random(n_a) < 0.9).astype(np.float64),
                (rng.random(n_b) < 0.1).astype(np.float64),
            ]
        )
        plain = rmodels.LogisticModel(C=1.0).fit(x, y)
        weighted = rmodels.LogisticModel(C=1.0).fit(
            x, y, sample_weight=M2.balanced_train_weights([n_a, n_b])
        )
        coef_plain, _ = plain.coefficients()
        coef_weighted, _ = weighted.coefficients()
        assert not np.allclose(coef_plain, coef_weighted, atol=1e-3)
        # ones == exactly the unweighted path
        ones = rmodels.LogisticModel(C=1.0).fit(x, y, sample_weight=np.ones(x.shape[0]))
        np.testing.assert_allclose(ones.coefficients()[0], coef_plain, atol=1e-12)

    def test_torch_fit_applies_sample_weight(self):
        inputs = _lcr_inputs(n=90, seed=5)
        rng = np.random.default_rng(5)
        y = np.concatenate(
            [(rng.random(30) < 0.9).astype(np.float64), (rng.random(60) < 0.1).astype(np.float64)]
        )
        s = M2.balanced_train_weights([30, 60])
        v = M2.balanced_val_weights([30, 60])
        m_plain = LCR(seed=7)
        m_plain.fit(inputs, y, lr=1e-3, epochs=4)
        m_weighted = LCR(seed=7)
        info = m_weighted.fit(
            inputs, y, lr=1e-3, epochs=4,
            inputs_val=inputs, y_val=y, patience=30,
            sample_weight=s, val_sample_weight=v,
        )
        assert np.isfinite(info["best_val_nll"])
        p_plain = m_plain.predict_proba(inputs)
        p_weighted = m_weighted.predict_proba(inputs)
        assert not np.allclose(p_plain, p_weighted, atol=1e-6)

    def test_weight_validation(self):
        with pytest.raises(ValueError):
            M2.balanced_train_weights([0, 10])
        model = LCR(seed=0)
        inputs = _lcr_inputs(n=8)
        with pytest.raises(ValueError):
            model.fit(inputs, np.zeros(8), lr=1e-3, epochs=1, sample_weight=np.zeros(7))


# ---------------------------------------------------------------------------
# 5. m=8 never enters model selection
# ---------------------------------------------------------------------------
class TestM8NeverInSelection:
    def test_level_guard_rejects_m8(self):
        M2.assert_selection_levels([0, 2, 4])
        with pytest.raises(AssertionError, match="completely unseen"):
            M2.assert_selection_levels([0, 2, 4, 8])

    def test_tune_balanced_auroc_guards(self):
        conf = {m: np.random.default_rng(m).random(10) for m in (0, 2, 4, 8)}
        correct = {m: (np.random.default_rng(m).random(10) < 0.5) for m in (0, 2, 4, 8)}
        with pytest.raises(AssertionError):
            M2._tune_balanced_auroc(conf, correct, [0, 2, 4, 8])

    def test_fit_functions_reject_selection_on_test_levels(self):
        rng = np.random.default_rng(2)
        x = {m: rng.standard_normal((12, 3)) for m in (0, 2, 8)}
        y = {m: (rng.random(12) < 0.5).astype(np.float64) for m in (0, 2, 8)}
        with pytest.raises(AssertionError):
            M2._fit_logistic_curriculum(x, y, x, y, name="bad", log=lambda message: None)
        inputs = {m: {"x": rng.standard_normal((12, 3))} for m in (0, 2, 8)}
        with pytest.raises(AssertionError):
            M2._fit_torch_curriculum(
                AggregateMLP, inputs, y, inputs, y,
                seed=0, name="bad", log=lambda message: None,
            )


# ---------------------------------------------------------------------------
# 6. all four models share the same curriculum
# ---------------------------------------------------------------------------
class TestSharedCurriculum:
    def test_single_curriculum_identity(self):
        sha_a = M2._curriculum_sha256_from_tables()
        sha_b = M2._curriculum_sha256_from_tables()
        assert sha_a == sha_b and len(sha_a) == 64

    def test_shared_curriculum_rows_match_freeze_counts(self):
        doc = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        sentence_id = {}
        for side in ("train", "tune"):
            sentence_id[side] = {}
            for m in (0, 2, 4):
                with np.load(BUILD.MAN_DIR / f"val_level_m{m}.npz") as npz:
                    sentences = np.asarray(npz[f"{side}__sentence_id"], dtype=np.int64)
                sentence_id[side][m] = sentences
                key = f"{side}_rows"
                assert sentences.size == doc["val_tables"][f"m{m}"][key]
        assert M2._curriculum_sha256(sentence_id) == M2._curriculum_sha256_from_tables()


# ---------------------------------------------------------------------------
# 7. architecture frozen (identical to M0/M1)
# ---------------------------------------------------------------------------
class TestArchitectureFrozen:
    def test_params_match_m0_record(self):
        m0 = json.loads(M0_PATH.read_text(encoding="utf-8"))
        counts = m0["architecture"]["parameter_counts"]
        assert LCR(seed=0).n_parameters() == counts["LCR"] == M2.EXPECTED_PARAMS["LCR"] == 1333
        assert LCRNoGate(seed=0).n_parameters() == counts["LCR-noGate"] == 1329
        assert AggregateMLP(seed=0).n_parameters() == counts["Aggregate-MLP"] == 1569

    def test_state_keys_and_deterministic_build(self):
        m0 = json.loads(M0_PATH.read_text(encoding="utf-8"))
        keys_expected = sorted(m0["architecture"]["state_keys"]["LCR"])
        first = LCR(seed=0).state_arrays()
        second = LCR(seed=0).state_arrays()
        assert sorted(first) == keys_expected
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_runner_expected_params_unchanged(self):
        assert M2.EXPECTED_PARAMS == {"LCR": 1333, "LCR-noGate": 1329, "Aggregate-MLP": 1569}


# ---------------------------------------------------------------------------
# 8. hyperparameter grid unchanged
# ---------------------------------------------------------------------------
class TestHyperparameterGrid:
    def test_grids_match_m1_and_protocol(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        assert list(M2.LR_GRID) == list(M1.LR_GRID) == [0.0001, 0.0003, 0.001]
        assert protocol["training_protocol"]["lr_grid"] == list(M2.LR_GRID)
        assert M2.C_GRID == M1.C_GRID == (0.1, 1.0, 10.0)
        assert M2.SELECTION_TIE == M1.SELECTION_TIE == 0.002
        assert (M2.EPOCHS, M2.BATCH_SIZE, M2.PATIENCE) == (M1.EPOCHS, M1.BATCH_SIZE, M1.PATIENCE) == (300, 256, 30)
        assert M2.WEIGHT_DECAY == M1.WEIGHT_DECAY == 1e-4
        assert protocol["training_protocol"]["seeds"] == list(M2.SEEDS)

    def test_m2_gate_constants_match_protocol(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        gate = protocol["M2_preregistration"]["gate"]
        assert gate["vs_curriculum_E1b"]["delta_auroc_ge"] == M2.GO_AUROC_MIN == 0.010
        assert gate["vs_curriculum_E1b"]["ci_low_gt"] == 0.0
        assert gate["vs_Aggregate_MLP"]["ci_low_gt"] == 0.0
        assert protocol["M2_preregistration"]["training"]["levels"] == list(M2.LEVELS_TRAIN)


# ---------------------------------------------------------------------------
# 9. bootstrap draws shared
# ---------------------------------------------------------------------------
class TestBootstrapDrawsShared:
    def _rows(self, a, b, correct, clusters, model_ab):
        return reval.model_vs_model_bootstrap_row(
            a, correct, b, correct, clusters,
            eval_split=shard.POOLED, K=5,
            model_a=model_ab[0], model_b=model_ab[1],
            metrics=("auroc_correct",),
            replicates=200, seed=0, ci=0.95,
        )[0]

    def test_pair_order_exact_antisymmetry(self):
        rng = np.random.default_rng(7)
        n = 240
        correct = rng.random(n) < 0.6
        a = rng.random(n)
        b = rng.random(n) + 0.05 * correct
        clusters = np.repeat(np.arange(60), 4)
        ab = self._rows(a, b, correct, clusters, ("A", "B"))
        ba = self._rows(b, a, correct, clusters, ("B", "A"))
        assert np.isclose(float(ab["diff"]), -float(ba["diff"]), atol=1e-12)
        assert np.isclose(float(ab["ci_low"]), -float(ba["ci_high"]), atol=1e-12)
        assert np.isclose(float(ab["ci_high"]), -float(ba["ci_low"]), atol=1e-12)

    def test_repeated_call_deterministic(self):
        rng = np.random.default_rng(8)
        n = 200
        correct = rng.random(n) < 0.5
        a = rng.random(n)
        b = rng.random(n)
        clusters = np.repeat(np.arange(50), 4)
        first = self._rows(a, b, correct, clusters, ("A", "B"))
        second = self._rows(a, b, correct, clusters, ("A", "B"))
        for key in ("diff", "ci_low", "ci_high", "mean_a", "mean_b"):
            assert float(first[key]) == float(second[key])


# ---------------------------------------------------------------------------
# 10. grounding scores unchanged
# ---------------------------------------------------------------------------
class TestGroundingScoresUnchanged:
    def test_frozen_expb_self_consistency(self):
        path = EXPB_PREDICTIONS / "expb_m8__b3_seed1.npz"
        with np.load(path) as frozen:
            scores = np.asarray(frozen["scores"], dtype=np.float64)
            correct = np.asarray(frozen["correct"], dtype=bool)
        assert np.array_equal(np.argmax(scores, axis=1) == 0, correct)
        fp = grounding_fingerprint(scores)
        assert fp["accuracy"] == float(correct.mean())
        assert fp["n"] == scores.shape[0] and fp["k"] == 10

    def test_level_rescore_reproduces_frozen_slice(self):
        """A8.4: the frozen checkpoint re-scores a slice of expb_m8 exactly."""
        from ccg.models.b3_data import B3Corpus
        from ccg.semantic import hard_scores as hscores

        cohort_hc = shard.load_hard_cohort(manifests_root=_REPO_ROOT / "cache/manifests")
        rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
        take = rows_same8[:16]
        samples = [cohort_hc.level_sample(int(row), 8, 10) for row in take.tolist()]
        corpus = B3Corpus(
            _REPO_ROOT / "cache/features",
            _REPO_ROOT / "cache/manifests",
            _REPO_ROOT / "data/raw/refcoco+/refcoco+/refs(unc).p",
            _REPO_ROOT / "cache/proposals.h5",
            image_sizes_path=_REPO_ROOT / "cache/image_sizes.npz",
            ks=(5,),
            regime="random",
        )
        try:
            batch = hscores.materialise_examples(corpus, samples, 10, text_cache={})
        finally:
            corpus.close()
        scorers = hscores.load_frozen_scorers(seeds=(1,))
        rescored = np.asarray(
            hscores.score_examples(scorers["b3_seed1"], batch, batch_size=64), dtype=np.float64
        )
        with np.load(EXPB_PREDICTIONS / "expb_m8__b3_seed1.npz") as frozen:
            reference = np.asarray(frozen["scores"], dtype=np.float64)[: take.size]
        delta = float(np.max(np.abs(rescored - reference)))
        assert delta <= 1e-4
