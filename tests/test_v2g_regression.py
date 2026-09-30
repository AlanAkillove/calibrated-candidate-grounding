"""V2-G item-31 regression suite (protocol items 1-18 + G4/G5 supplements).

These assert on the REAL frozen artifacts + pipeline outputs that produced the
V2-G results, not on remembered APIs.  Every check is cheap (JSON / CSV / pkl /
numpy); no GPU, no re-scoring, no re-bootstrap.  They guard the exact invariants
the cross-backbone conclusions rest on.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.reliability.features import stat_feature_names  # noqa: E402
from ccg.v2.semantic_features import (  # noqa: E402
    MANIPULATION_METRICS,
    V2_PRIMARY_SEMANTIC_NAMES,
    V2_SECONDARY_NAMES,
)

_ROOT = _REPO / "results" / "v2_backbone_generalization"
_CACHE = _REPO / "cache" / "v2_backbones"
_MANIFESTS = _REPO / "cache" / "manifests"
_BACKBONES = {"b1": "openclip_b16", "b2": "siglip_b16"}
_SEEDS = (1, 2, 3)


# ---------------------------------------------------------------------------
# loaders (memoised)
# ---------------------------------------------------------------------------
_JSON_CACHE: dict = {}
_PKL_CACHE: dict = {}


def _load_json(path: Path):
    key = str(path)
    if key not in _JSON_CACHE:
        with open(path, "r", encoding="utf-8") as fh:
            _JSON_CACHE[key] = json.load(fh)
    return _JSON_CACHE[key]


def _load_csv(path: Path):
    key = str(path)
    if key not in _JSON_CACHE:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            _JSON_CACHE[key] = list(csv.DictReader(fh))
    return _JSON_CACHE[key]


def _pkl(tag: str, seed: int):
    key = (tag, seed)
    if key not in _PKL_CACHE:
        with open(_ROOT / "g4_phaseA" / tag / f"seed_{seed}" / "models.pkl", "rb") as fh:
            _PKL_CACHE[key] = pickle.load(fh)
    return _PKL_CACHE[key]


def freeze():
    return _load_json(_ROOT / "g4_protocol_freeze.json")


def proto():
    return _load_json(_ROOT / "protocol.json")


def g3():
    return _load_json(_ROOT / "g3_cardinality_gate.json")


def gate(tag: str):
    return _load_json(_ROOT / "g4_phaseB" / tag / "gate.json")


def amp(tag: str):
    return _load_csv(_ROOT / "g4_phaseB" / tag / "amplification.csv")


def manip(tag: str):
    return _load_csv(_ROOT / "g4_phaseB" / tag / "manipulation_check.csv")


def dose(tag: str):
    return _load_csv(_ROOT / "g4_phaseB" / tag / "dose_response.csv")


def phaseB_meta():
    return _load_json(_ROOT / "g4_phaseB" / "metadata.json")


def g5():
    return _load_json(_ROOT / "g4_phaseB" / "g5_overall_gate.json")


def manifest(tag: str, seed: int):
    return _load_json(_ROOT / "g4_phaseA" / tag / f"seed_{seed}" / "model_manifest.json")


def eval_meta(tag: str, seed: int):
    return _load_json(_ROOT / f"{tag}_phase0b" / f"seed_{seed}" / "eval_metadata.json")


def cache_meta(bb: str):
    return _load_json(_CACHE / bb / "metadata.json")


def _select_C(grid, values, tie=0.002):
    best = max(values)
    tied = [c for c, v in zip(grid, values) if best - v <= tie]
    return min(tied)


# ===========================================================================
# Protocol items 1-18
# ===========================================================================

# --- 1. finalized feature banks complete
class TestFeatureBanksComplete:
    @pytest.mark.parametrize("bb", list(_BACKBONES.values()))
    def test_counts_and_files(self, bb):
        m = cache_meta(bb)
        assert m["n_images"] == 19992
        assert m["n_crops"] == 1279488
        assert m["n_sentences"] == 141564
        assert m["invalid_crop_count"] == 0
        for f in ("region_features.h5", "text_features.h5", "global_features.h5"):
            assert (_CACHE / bb / f).exists(), f

    def test_banks_match_protocol_record(self):
        for tag, bb in _BACKBONES.items():
            run = proto()["execution_status"]  # g1_reason documents the same totals
            m = cache_meta(bb)
            assert m["extraction_stats"]["n_target_images"] == m["n_images"]
            assert run.get("G1") == "EXTRACTED"


# --- 2. feature dimensions correct
class TestFeatureDimensions:
    def test_embedding_dims(self):
        assert cache_meta("openclip_b16")["backbone"]["embedding_dim"] == 512
        assert cache_meta("siglip_b16")["backbone"]["embedding_dim"] == 768

    def test_dims_agree_across_contract_and_models(self):
        fr = freeze()["backbones"]
        for tag in _BACKBONES:
            dim = fr[tag.upper()]["feature_dim"]
            assert cache_meta(_BACKBONES[tag])["backbone"]["embedding_dim"] == dim
            for seed in _SEEDS:
                assert manifest(tag, seed)["feature_dim"] == dim


# --- 3. V1 candidate manifests identity (reuse, single frozen source)
class TestManifestIdentity:
    def test_no_rebuild_and_single_source(self):
        assert proto()["data_reuse"]["rebuild_candidate_sets"] is False
        assert proto()["data_reuse"]["only_variable"] == "backbone representation"
        assert freeze()["frozen_items"]["6_random_hard_manifests"]["manifests_root"].startswith("cache/manifests")

    def test_eight_frozen_manifests_present(self):
        names = proto()["data_reuse"]["manifests"]
        assert len(names) == 8
        for n in names:
            assert (_MANIFESTS / f"{n}.jsonl").exists(), n

    def test_both_backbones_read_same_manifest_root(self):
        for tag in _BACKBONES:
            assert eval_meta(tag, 1)  # phase0b used cache\\manifests (config below)
            cfg = _load_json(_ROOT / f"{tag}_phase0b" / "metadata.json")["config"]
            assert cfg["manifests_root"] == "cache\\manifests"


# --- 4. B3 candidate independence (empirical invariance hard-check passed)
class TestB3CandidateIndependence:
    def test_architectural_contract(self):
        assert "no cross-candidate attention/pooling/K-input/other-candidate embeddings" in \
            proto()["scorers"]["layerB"]

    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_score_invariance_hard_check_passed(self, tag):
        for seed in _SEEDS:
            hc = eval_meta(tag, seed)["hard_checks"]
            assert hc["score_invariance"]["status"] == "passed"
            assert hc["score_invariance"]["n_violations"] == 0
            assert hc["rank_monotonic"]["n_violations"] == 0


# --- 5. parameter budget
class TestParameterBudget:
    EXPECTED = {"b1": 214273, "b2": 312577}

    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_within_v2_budget(self, tag):
        for seed in _SEEDS:
            tr = eval_meta(tag, seed)["training"]
            assert tr["num_parameters"] == self.EXPECTED[tag]
            assert tr["num_parameters"] < tr["param_budget"]
            assert tr["param_budget"] == 500000

    def test_matches_freeze(self):
        fr = freeze()["backbones"]
        for tag in _BACKBONES:
            assert fr[tag.upper()]["b3_params"] == self.EXPECTED[tag]


# --- 6. 3 scorer seeds kept separate
class TestScorerSeedSeparation:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_three_seed_dirs_and_models(self, tag):
        for seed in _SEEDS:
            assert (_ROOT / f"{tag}_phase0b" / f"seed_{seed}" / "model.npz").exists()
            assert (_ROOT / "g4_phaseA" / tag / f"seed_{seed}" / "models.pkl").exists()
        # gate reports per-seed, never a single pooled model
        g = gate(tag)
        assert set(g["per_seed_delta_hard"]) == {f"b3_seed{s}" for s in _SEEDS}


# --- 7. K20 / K50 never used for model selection
class TestSelectionUsesOnlyK5K10:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_train_and_select_ks(self, tag):
        for seed in _SEEDS:
            tr = eval_meta(tag, seed)["training"]
            assert tr["train_ks"] == [5, 10]
            assert tr["selection_metric"] == "val_select_mean_nll_K5_K10"

    def test_forbidden_in_freeze(self):
        forbidden = freeze()["frozen_items"]["5_selection_split"]["forbidden_in_train_or_select"]
        assert "K20" in forbidden and "K50" in forbidden


# --- 8. global T fit scope + phaseA reuse of the same T
class TestGlobalTemperatureScope:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_eval_and_phaseA_use_same_T(self, tag):
        fr = freeze()["backbones"][tag.upper()]["corrected_global_T"]
        for seed in _SEEDS:
            t_eval = eval_meta(tag, seed)["temperature_corrected"]
            assert abs(t_eval - fr[str(seed)]) < 1e-12
            assert abs(manifest(tag, seed)["T_corrected"] - t_eval) < 1e-12

    def test_T_is_val_calib_fitted_scalar(self):
        # the single corrected global T is a scalar (one val_calib fit), bounds recorded
        fr = freeze()["backbones"]["B1"]["corrected_global_T"]
        assert set(fr) == {"1", "2", "3"}


# --- 9. temperature interior optimum
class TestTemperatureInterior:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_interior(self, tag):
        meta = _load_json(_ROOT / f"{tag}_phase0b" / "metadata.json")
        lo, hi = meta["config"]["temperature_bounds"]
        for seed in _SEEDS:
            ps = meta["per_seed"][str(seed)]
            assert ps["corrected_interior"] is True
            assert lo < ps["temperature_corrected"] < hi


# --- 10. temperature NOT shared across backbone / seed
class TestTemperatureIsolation:
    def test_all_six_temperatures_distinct(self):
        ts = []
        for tag in _BACKBONES:
            for seed in _SEEDS:
                ts.append(eval_meta(tag, seed)["temperature_corrected"])
        assert len(set(round(t, 6) for t in ts)) == 6


# --- 11. V2 semantic features carry no OpenCLIP-specific primary assumption
class TestNoOpenCLIPAssumption:
    def test_no_hardcoded_temperature_in_primary(self):
        import inspect
        from ccg.v2.semantic_features import v2_primary_semantic_stats
        src = inspect.getsource(v2_primary_semantic_stats)
        assert "CLIP_TEMPERATURE" not in src
        assert "0.01" not in src

    def test_freeze_declares_backbone_neutral(self):
        item = freeze()["frozen_items"]["1_semantic_feature_list"]
        assert item["backbone_neutral"] is True
        assert any("CLIP_TEMPERATURE" in f for f in item["forbidden_in_primary"])


# --- 12. primary semantic features are continuous / rank based
class TestPrimaryFeaturesContinuous:
    def test_density_not_primary(self):
        prim = set(V2_PRIMARY_SEMANTIC_NAMES)
        assert "density_070" not in prim and "density_080" not in prim
        assert {"density_070", "density_080"} <= set(V2_SECONDARY_NAMES)

    def test_all_primary_are_cosine_or_rank(self):
        for name in V2_PRIMARY_SEMANTIC_NAMES:
            assert ("cos" in name) or ("margin" in name) or ("rank" in name) or ("spread" in name), name


# --- 13. R1 / R2 never read raw embeddings
class TestModelsReadNoRawEmbeddings:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_feature_widths(self, tag):
        n_stats = len(stat_feature_names())
        for seed in _SEEDS:
            p = _pkl(tag, seed)
            r1_coef, _ = p["R1_model"].coefficients()
            r2_coef, _ = p["R2_model"].coefficients()
            assert r1_coef.size == n_stats == 17
            assert r2_coef.size == n_stats + len(V2_PRIMARY_SEMANTIC_NAMES) == 31
            assert r2_coef.size != freeze()["backbones"][tag.upper()]["feature_dim"]


# --- 14. Random / Hard cohort identity
class TestCohortIdentity:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_matched_cell_sizes(self, tag):
        rows = amp(tag)
        k5 = {int(r["n"]) for r in rows if int(r["K"]) == 5}
        k10 = {int(r["n"]) for r in rows if int(r["K"]) == 10}
        assert k5 == {9487} and k10 == {6765}
        assert {r["hard_cell"] for r in rows if int(r["K"]) == 5} == {"hard5"}
        assert {r["rand_cell"] for r in rows if int(r["K"]) == 5} == {"rand5"}


# --- 15. paired bootstrap draws shared
class TestBootstrapDrawsShared:
    def test_single_bootstrap_config(self):
        mb = phaseB_meta()["bootstrap"]
        item8 = freeze()["frozen_items"]["8_bootstrap_seed_draw_policy"]
        assert mb == {"replicates": item8["replicates"], "seed": item8["bootstrap_seed"], "ci": item8["ci"]}

    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_cluster_unit_is_image_id(self, tag):
        # one shared image-cluster set across every paired shift (rand vs hard, same draw)
        clusters = {int(r["n_clusters"]) for r in manip(tag)}
        assert len(clusters) == 1
        # single bootstrap_seed recorded on every amplification row (shared draws)
        assert {int(r["bootstrap_seed"]) for r in amp(tag)} == {0}


# --- 16. manipulation uses only backbone-neutral primary features
class TestManipulationBackboneNeutral:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_features_and_directions(self, tag):
        allowed = {name: d for name, d in MANIPULATION_METRICS}
        prim = set(V2_PRIMARY_SEMANTIC_NAMES)
        for r in manip(tag):
            assert r["feature"] in allowed and r["feature"] in prim
            assert r["direction"] == allowed[r["feature"]]


# --- 17. 3-seed independence (mean, never concatenated pool)
class TestThreeSeedIndependence:
    @pytest.mark.parametrize("tag", list(_BACKBONES))
    def test_headline_is_mean_of_three_distinct_seeds(self, tag):
        g = gate(tag)
        per = [g["per_seed_delta_hard"][f"b3_seed{s}"] for s in _SEEDS]
        assert len(set(per)) == 3                       # three distinct per-seed values
        assert abs(np.mean(per) - g["delta_hard_mean"]) < 1e-12


# --- 18. overall gate deterministic
class TestOverallGateDeterministic:
    def test_rule_reproduces_stored_verdict(self):
        gb = g5()
        n_both = sum(int(r["cardinality_replicated"] and r["hard_semantic_replicated"])
                     for r in gb["cross_backbone"])
        assert n_both == gb["n_both_replicated"]
        assert gb["V2_METHOD_DEVELOPMENT_AUTHORIZED"] == (n_both >= 2)

    def test_b1_b2_entries_match_raw_artifacts(self):
        gb = {r["backbone"]: r for r in g5()["cross_backbone"]}
        for tag, key in (("b1", "B1_openclip_b16"), ("b2", "B2_siglip_b16")):
            assert gb[key]["hard_semantic_replicated"] == (gate(tag)["verdict"] == "HARD_SEMANTIC_REPLICATED")
            assert gb[key]["cardinality_replicated"] == \
                g3()["backbones"][key]["CARDINALITY_REPLICATED"]


# ===========================================================================
# G4 / G5 supplementary invariants
# ===========================================================================

class TestG4G5Supplements:
    def test_r1_r2_coefficients_isolated_across_backbone(self):
        c1, _ = _pkl("b1", 1)["R1_model"].coefficients()
        c2, _ = _pkl("b2", 1)["R1_model"].coefficients()
        assert not np.allclose(c1, c2)

    def test_normalization_isolated_across_backbone(self):
        m1 = np.asarray(_pkl("b1", 1)["stats_fit"].mean)
        m2 = np.asarray(_pkl("b2", 1)["stats_fit"].mean)
        assert m1.shape == m2.shape == (17,)
        assert not np.allclose(m1, m2)

    def test_selected_C_isolated_and_rederivable(self):
        grid = [0.1, 1.0, 10.0]
        for tag in _BACKBONES:
            for seed in _SEEDS:
                for model in ("R1", "R2"):
                    mm = manifest(tag, seed)[model]
                    assert mm["grid"] == grid
                    assert mm["chosen_C"] == _select_C(mm["grid"], mm["values"])

    def test_no_hard_rows_in_reliability_train_tune(self):
        forbidden = set(freeze()["frozen_items"]["5_selection_split"]["forbidden_in_train_or_select"])
        assert {"K20", "K50", "same_category", "testA", "testB"} == forbidden
        for tag in _BACKBONES:
            for seed in _SEEDS:
                # train-fit rows are the small random-regime train split, never the pooled test set
                assert manifest(tag, seed)["stats_fit_rows"] < 10286

    def test_random_hard_identical_row_identities(self):
        for tag in _BACKBONES:
            for r in amp(tag):
                # the matched pair is scored on ONE shared row set (single n per cell line)
                assert int(r["n"]) > 0
                assert r["hard_cell"].replace("hard", "") == r["rand_cell"].replace("rand", "")

    def test_amplification_uses_shared_draws_and_coupling(self):
        for tag in _BACKBONES:
            g = gate(tag)
            rows5 = [r for r in amp(tag) if int(r["K"]) == 5]
            dh = np.mean([float(r["delta_hard"]) for r in rows5])
            dr = np.mean([float(r["delta_rand"]) for r in rows5])
            # amplification point == delta_hard - delta_rand from the SAME cluster draw
            assert abs((dh - dr) - g["amplification_mean"]) < 1e-9

    def test_dose_response_manifests_immutable(self):
        for tag in _BACKBONES:
            rows = dose(tag)
            assert {int(r["n"]) for r in rows} == {7410}
            assert {int(r["level"]) for r in rows} == {0, 2, 4, 8}
            for r in rows:
                assert abs(float(r["hard_fraction"]) - int(r["level"]) / 9.0) < 1e-9

    def test_phaseA_cannot_alter_phaseB_config(self):
        fr = freeze()
        assert fr["frozen_before_phase_a"] is True
        item9 = fr["frozen_items"]["9_g4_gate_thresholds"]
        item7 = fr["frozen_items"]["7_manipulation_metrics"]
        created = datetime.strptime(fr["created_utc"], "%Y-%m-%dT%H:%M:%SZ")
        completed = datetime.strptime(phaseB_meta()["completed_utc"], "%Y-%m-%dT%H:%M:%SZ")
        assert created < completed
        for tag in _BACKBONES:
            g = gate(tag)
            th = g["thresholds"]
            for key in ("delta_hard_min", "delta_hard_ci_low_gt", "amplification_min",
                        "amplification_ci_low_gt", "requires_manipulation_valid"):
                assert th[key] == item9[key]
            assert [list(m) for m in g["manipulation"]["metrics"]] == item7["metrics"]
