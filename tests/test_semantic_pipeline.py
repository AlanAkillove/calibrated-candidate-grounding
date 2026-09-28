"""Phase 1 (protocol A7) pipeline tests - the 15 required checks of section 41.

Half of the suite runs against the frozen on-disk artifacts (score matrices,
embedding archives, phase 0.5 split manifest); the other half uses small
synthetic inputs.  Real-data tests skip when an artifact is absent so the suite
still runs on a fresh checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _extra in (_REPO / "src", _REPO / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import run_phase1 as R  # noqa: E402

from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import models as smodels  # noqa: E402

_EMB_ROOT = Path("cache/semantic_phase1")
_EMB_READY = sdata.embedding_file(5, out_root=_EMB_ROOT).exists()
_SPLIT_MANIFEST = Path("results/phase05_score_sufficiency/split_manifest.json")
_SCORE_READY = (
    Path("results/phase0b_independent/seed_1/raw_scores/K5.npz").exists()
)

pytestmark = pytest.mark.skipif(
    not (_EMB_READY and _SPLIT_MANIFEST.exists() and _SCORE_READY),
    reason="phase-1 frozen artifacts (embeddings / split manifest / scores) not built",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _seed1(k: int):
    return sdata.load_scorer_canonical("b3_seed1", k)


def _store(k: int) -> sdata.EmbeddingStore:
    return sdata.load_embedding_store(k, out_root=_EMB_ROOT)


def _synthetic_scores(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    return (rng.normal(size=(n, k)) + np.linspace(1.0, 0.0, k)[None, :]).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. frozen B3 ranking unchanged
# ---------------------------------------------------------------------------
def test_scores_roundtrip_is_frozen() -> None:
    npz = np.load("results/phase0b_independent/seed_1/raw_scores/K5.npz")
    order = np.argsort(npz["sentence_id"], kind="stable")
    canonical = _seed1(5)
    assert np.array_equal(canonical.sentence_id, npz["sentence_id"][order])
    assert np.array_equal(canonical.scores, npz["scores"][order])  # bit-identical
    assert np.array_equal(canonical.correct, np.argmax(npz["scores"][order], axis=1) == npz["target_local"][order])


# ---------------------------------------------------------------------------
# 2. reliability models cannot modify candidate scores / their inputs
# ---------------------------------------------------------------------------
def test_models_do_not_modify_inputs() -> None:
    rng = np.random.default_rng(0)
    n, k = 96, 5
    scores = _synthetic_scores(rng, n, k)
    z_q = rng.normal(size=(n, 512)).astype(np.float32)
    z_i = rng.normal(size=(n, k, 512)).astype(np.float32)
    mask = np.ones((n, k), dtype=bool)
    prob = np.full((n, k), 1.0 / k, dtype=np.float32)
    y = (rng.random(n) < 0.5).astype(np.float64)
    snapshots = {name: arr.copy() for name, arr in
                 (("scores", scores), ("z_q", z_q), ("z_i", z_i), ("mask", mask), ("prob", prob))}
    e2 = smodels.TopCompetitorModel(seed=1)
    e2.fit({"z_q": z_q, "z_top1": z_i[:, 0], "z_top2": z_i[:, 1],
            "score_stats": np.column_stack([scores[:, 0], scores[:, 1],
                                            scores[:, 0] - scores[:, 1], np.full(n, 0.5)])},
           y, lr=3e-4, epochs=2, patience=2)
    e2.predict_proba({"z_q": z_q, "z_top1": z_i[:, 0], "z_top2": z_i[:, 1],
                      "score_stats": np.column_stack([scores[:, 0], scores[:, 1],
                                                      scores[:, 0] - scores[:, 1], np.full(n, 0.5)])})
    e3 = smodels.SemanticDeepSets(seed=1)
    e3.fit({"z_q": z_q, "z_i": z_i, "mask": mask, "score": scores.astype(np.float32), "prob": prob},
           y, lr=3e-4, epochs=2, patience=2)
    e3.predict_proba({"z_q": z_q, "z_i": z_i, "mask": mask, "score": scores.astype(np.float32),
                      "prob": prob})
    for name, arr in (("scores", scores), ("z_q", z_q), ("z_i", z_i), ("mask", mask), ("prob", prob)):
        assert np.array_equal(arr, snapshots[name]), f"{name} was modified"


# ---------------------------------------------------------------------------
# 3/4. no GT metadata, no geometry / objectness inputs anywhere in phase 1
# ---------------------------------------------------------------------------
def test_input_builders_expose_only_allowed_keys() -> None:
    scores = _seed1(5).scores
    rows = slice(0, 64)
    store = _store(5)
    temperature = 1.0
    e2 = R._e2_inputs_for_k(5, store, np.asarray(scores, dtype=np.float64), temperature)
    e3 = R._e3_native_for_k(5, store, np.asarray(scores, dtype=np.float64), temperature, top5=False)
    allowed_e2 = {"z_q", "z_top1", "z_top2", "score_stats"}
    allowed_e3 = {"z_q", "z_i", "score", "prob"}
    assert set(e2) == allowed_e2
    assert set(e3) == allowed_e3
    forbidden = ("geometry", "object", "iou", "category", "gt_", "target_index", "ref_id")
    for keys in (set(e2), set(e3)):
        for key in keys:
            assert not any(bad in key.lower() for bad in forbidden)
    # E1 statistics take embeddings + frozen scores only
    stats = sfeat.semantic_stats(np.asarray(store.z_q[rows], dtype=np.float32),
                                 np.asarray(store.z_i[rows], dtype=np.float32),
                                 np.asarray(scores[rows], dtype=np.float64))
    assert stats.shape == (64, len(sfeat.SEMANTIC_STAT_NAMES))


# ---------------------------------------------------------------------------
# 5. the candidate projection is shared between top1 / top2
# ---------------------------------------------------------------------------
def test_e2_single_shared_pv_projection() -> None:
    model = smodels.TopCompetitorModel(seed=1, include_score=True, include_semantic=True)
    model.fit({"z_q": np.zeros((4, 512), np.float32), "z_top1": np.zeros((4, 512), np.float32),
               "z_top2": np.zeros((4, 512), np.float32), "score_stats": np.zeros((4, 4), np.float32)},
              np.asarray([0, 1, 0, 1], dtype=np.float64), lr=1e-3, epochs=1, patience=1)
    state = model._net.state_dict()
    pv_like = [key for key, value in state.items() if value.shape == (64, 512)]
    assert len(pv_like) == 2  # exactly p_q and p_v - no per-role copy
    assert sum("p_v" in key and key.endswith("weight") for key in state) == 1


# ---------------------------------------------------------------------------
# 6/7/8. E3 invariance: permutation, variable K, top1 identity
# ---------------------------------------------------------------------------
def test_e3_permutation_invariance_and_variable_k() -> None:
    rng = np.random.default_rng(1)
    n, k = 48, 6
    scores = _synthetic_scores(rng, n, k)
    z_q = rng.normal(size=(n, 512)).astype(np.float32)
    z_i = rng.normal(size=(n, k, 512)).astype(np.float32)
    mask = np.ones((n, k), dtype=bool)
    prob = np.full((n, k), 1.0 / k, dtype=np.float32)
    y = (scores[:, 0] > scores[:, 1]).astype(np.float64)
    model = smodels.SemanticDeepSets(seed=3)
    model.fit({"z_q": z_q, "z_i": z_i, "mask": mask, "score": scores.astype(np.float32), "prob": prob},
              y, lr=1e-3, epochs=3, patience=2)
    base = model.predict_proba({"z_q": z_q, "z_i": z_i, "mask": mask,
                                "score": scores.astype(np.float32), "prob": prob})
    perm = rng.permutation(k)
    permuted = model.predict_proba({"z_q": z_q, "z_i": z_i[:, perm], "mask": mask[:, perm],
                                    "score": scores.astype(np.float32)[:, perm],
                                    "prob": prob[:, perm]})
    assert np.allclose(base, permuted, atol=1e-6)

    # variable K: pad the same set to K+4 with a zero mask -> identical predictions
    k2 = k + 4
    padded = {
        "z_q": z_q,
        "z_i": np.concatenate([z_i, np.zeros((n, 4, 512), np.float32)], axis=1),
        "mask": np.concatenate([mask, np.zeros((n, 4), dtype=bool)], axis=1),
        "score": np.concatenate([scores.astype(np.float32), np.zeros((n, 4), np.float32)], axis=1),
        "prob": np.concatenate([prob, np.zeros((n, 4), np.float32)], axis=1),
    }
    assert padded["z_i"].shape[1] == k2
    padded_pred = model.predict_proba(padded)
    assert np.allclose(base, padded_pred, atol=1e-6)

    # top1 identity: after permutation the top1 indicator still marks the max-score candidate
    features = model.candidate_features(z_q, z_i[:, perm], mask[:, perm],
                                        scores.astype(np.float32)[:, perm], prob[:, perm])
    top1_col = features[..., -1]
    winners = np.argmax(scores[:, perm], axis=1)
    assert np.array_equal(top1_col.argmax(axis=1), winners)


# ---------------------------------------------------------------------------
# 9. E3-top5 is exactly the five highest B3-score candidates
# ---------------------------------------------------------------------------
def test_e3_top5_selection_exact() -> None:
    k = 20
    scores = np.asarray(_seed1(k).scores, dtype=np.float64)
    store = _store(k)
    top5 = R._e3_native_for_k(k, store, scores, 1.0, top5=True)
    assert top5["z_i"].shape[1] == 5
    idx = sfeat.top5_indices(scores)
    z_i = np.asarray(store.z_i, dtype=np.float32)
    rows = np.arange(scores.shape[0])[:, None]
    probe = slice(0, 256)
    assert np.array_equal(top5["z_i"][probe], z_i[probe][rows[probe], idx[probe]])
    assert np.array_equal(top5["score"][probe], scores[probe][rows[probe], idx[probe]].astype(np.float32))
    expected = np.sort(scores, axis=1)[:, ::-1][:, :5]
    assert np.allclose(top5["score"], expected)


# ---------------------------------------------------------------------------
# 10. K20 / K50 never enter training or tuning
# ---------------------------------------------------------------------------
def test_k20_k50_excluded_from_training_and_tuning() -> None:
    assert R.TRAIN_KS == (5, 10)
    assert set(R.TRAIN_KS).isdisjoint(set(R.OOD_KS))
    split = sdata.load_split_from_manifest(_SPLIT_MANIFEST)
    sample = _seed1(50)
    masks = R._row_masks(np.asarray(sample.eval_split), np.asarray(sample.image_id), split)
    for kind in ("train", "tune"):
        assert not np.any(masks[kind] & masks["testA"])
        assert not np.any(masks[kind] & masks["testB"])
        assert not np.any(masks[kind] & masks["val_select"])
        assert np.all(np.asarray(sample.eval_split)[masks[kind]] == "val_calib")
    # primary selection metric only ever reads TRAIN_KS masks
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "for k in TRAIN_KS" in source
    assert "tune_flags" in source


# ---------------------------------------------------------------------------
# 11. split reuse is identical to phase 0.5
# ---------------------------------------------------------------------------
def test_split_reuse_identical_to_phase05() -> None:
    split = sdata.load_split_from_manifest(_SPLIT_MANIFEST)
    sample = _seed1(5)
    sdata.assert_split_matches_fresh(split, np.asarray(sample.eval_split), np.asarray(sample.image_id))
    assert (split.train_images.size, split.tune_images.size) == (523, 224)


# ---------------------------------------------------------------------------
# 12. scorer seed separation
# ---------------------------------------------------------------------------
def test_scorer_seed_separation() -> None:
    a = sdata.load_scorer_canonical("b3_seed1", 5)
    b = sdata.load_scorer_canonical("b3_seed2", 5)
    assert np.array_equal(a.sentence_id, b.sentence_id)
    assert a.scores.shape == b.scores.shape
    assert not np.allclose(a.scores, b.scores, atol=1e-4)
    seeds = {int(name.split("b3_seed")[1]) for name in R.B3_SEEDS}
    assert seeds == {1, 2, 3}


# ---------------------------------------------------------------------------
# 13. bootstrap uses the same image clusters for both models (paired design)
# ---------------------------------------------------------------------------
def test_bootstrap_rows_share_image_clusters() -> None:
    rng = np.random.default_rng(2)
    n = 400
    image_id = np.repeat(np.arange(40), 10)
    correct = (rng.random(n) < 0.4).astype(bool)
    conf_a = np.clip(correct + rng.normal(0, 0.3, n), 0, 1)
    conf_b = np.clip(correct + rng.normal(0, 0.4, n), 0, 1)
    zeros = np.zeros(10_000, dtype=bool)
    res = {
        "scorer": "b3_seed1",
        "conf": {"sem": {5: conf_a}, "score": {5: conf_b}},
        "correct": {5: correct},
        "image_id": image_id,
        "masks": {R.POOLED: np.ones(n, dtype=bool), "x": zeros[:n]},
    }
    cache: dict = {}
    entry = R._pair_entry(res, cache, "sem", "score", 5, replicates=200, seed=0, ci=0.95)
    assert entry["abs"]["auroc_correct"]["n_clusters"] == 40
    assert entry["ratio"]["n_clusters"] == 40
    assert entry["abs"]["auroc_correct"]["n"] == n
    again = R._pair_entry(res, cache, "sem", "score", 5, replicates=200, seed=0, ci=0.95)
    assert again is entry  # cached, deterministic
    fresh_cache: dict = {}
    recomputed = R._pair_entry(res, fresh_cache, "sem", "score", 5, replicates=200, seed=0, ci=0.95)
    assert recomputed["abs"]["auroc_correct"]["diff"] == entry["abs"]["auroc_correct"]["diff"]
    assert recomputed["ratio"]["reduction"] == entry["ratio"]["reduction"]


# ---------------------------------------------------------------------------
# 14. semantic statistics are deterministic
# ---------------------------------------------------------------------------
def test_semantic_stats_deterministic() -> None:
    store = _store(5)
    scores = np.asarray(_seed1(5).scores, dtype=np.float64)[:256]
    first = sfeat.semantic_stats(np.asarray(store.z_q[:256], dtype=np.float32),
                                 np.asarray(store.z_i[:256], dtype=np.float32), scores)
    second = sfeat.semantic_stats(np.asarray(store.z_q[:256], dtype=np.float32),
                                  np.asarray(store.z_i[:256], dtype=np.float32), scores)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# 15. matched-score diagnostic does not use correctness for matching
# ---------------------------------------------------------------------------
def test_matched_score_matching_ignores_labels() -> None:
    rng = np.random.default_rng(3)
    n = 300
    mask = np.ones(n, dtype=bool)
    sim = rng.random(n)
    res = {
        "scorer": "b3_seed1",
        "masks": {R.POOLED: mask},
        "image_id": np.arange(n) // 3,
        "scalars": {50: {"msp": rng.random(n), "margin": rng.random(n), "neg_entropy": -rng.random(n)}},
    }
    stats = {"b3_seed1": {50: np.zeros((n, len(sfeat.SEMANTIC_STAT_NAMES)))}}
    stats["b3_seed1"][50][:, sfeat.SEMANTIC_STAT_NAMES.index("cand_top12_sim")] = sim
    y_a = (rng.random(n) < 0.5).astype(bool)
    y_b = ~y_a
    res["correct"] = {50: y_a}
    rows_a = R._matched_score_rows([res], stats, replicates=50, seed=0, ci=0.95)
    res["correct"] = {50: y_b}
    rows_b = R._matched_score_rows([res], stats, replicates=50, seed=0, ci=0.95)
    assert rows_a and rows_b
    pair_a, pair_b = rows_a[0], rows_b[0]
    assert pair_a["n_matched"] == pair_b["n_matched"]
    assert pair_a["treat_threshold_top12_sim"] == pair_b["treat_threshold_top12_sim"]
    assert pair_a["mean_match_distance"] == pair_b["mean_match_distance"]
    assert pair_a["delta_correct"] == pytest.approx(-pair_b["delta_correct"])
