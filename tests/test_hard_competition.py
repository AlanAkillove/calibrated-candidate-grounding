"""Phase 1F hard-competition (protocol Amendment A8) frozen-artifact tests.

Twelve checks pin the Phase 1F *confirmatory* stress test against the frozen
on-disk artifacts:

* the matched ``random`` / ``same_category`` cohorts and the
  ``C_K = [target] + order[:K-1]`` candidate construction (A8.2 / A8.3);
* the A8.4 frozen Stats-Logistic / E1b recovery (coefficients, normalisations,
  no hard-regime label leakage into training);
* the A8.5 image-clustered paired diff-of-diffs bootstrap; and
* the pure A7.3 semantic statistics the E1b block consumes.

The heavy lifts (``load_hard_cohort`` / frozen-bundle verification) are loaded
once per module.  The whole module skips when a required frozen artifact is
absent, so a fresh checkout still collects cleanly.
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")  # noqa: E402  (the proposal bank is HDF5)

_REPO = Path(__file__).resolve().parents[1]
for _extra in (_REPO / "src", _REPO / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from ccg.data.audit import assign_gt_category  # noqa: E402
from ccg.data.coco import build_index as build_coco_index  # noqa: E402
from ccg.data.manifests import ManifestFile  # noqa: E402
from ccg.data.proposals import DEFAULT_IOU_THRESH, iou_matrix  # noqa: E402
from ccg.experiment import phase0a  # noqa: E402
from ccg.reliability import evaluate as relev  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import hard_eval  # noqa: E402
from ccg.semantic.frozen_load import (  # noqa: E402
    load_models,
    verify_against_frozen_artifacts,
)
from ccg.semantic.hard import (  # noqa: E402
    HARD_FRACTION_LEVELS,
    REGIMES,
    REPORT_SPLITS,
    load_hard_cohort,
)

# ---------------------------------------------------------------------------
# frozen artifact roots (anchored to the repo, never the CWD)
# ---------------------------------------------------------------------------
_MANIFEST_ROOT = _REPO / "cache" / "manifests"
_BANK = _REPO / "cache" / "proposals.h5"
_PHASE05 = _REPO / "results" / "phase05_score_sufficiency"
_PHASE1 = _REPO / "results" / "phase1_semantic_sufficiency"
_FEATURES_DIR = _PHASE05 / "features"
_EMB_ROOT = _REPO / "cache" / "semantic_phase1"
_B3_ROOT = _REPO / "results" / "phase0b_independent"
_FROZEN_BUNDLE = _REPO / "results" / "phase1e_refcocog_external" / "frozen_models"
_SPLIT_MANIFEST = _PHASE05 / "split_manifest.json"
_COCO_INSTANCES = _REPO / "data" / "raw" / "annotations" / "instances_train2014.json"

_SCORER = "b3_seed1"
_COMPARE_SPLITS = ("val_select", "testA", "testB")
_TOL = 1e-9

_REQUIRED: list[Path] = [
    _BANK,
    _SPLIT_MANIFEST,
    _FEATURES_DIR / f"{_SCORER}.npz",
    _FEATURES_DIR / f"{_SCORER}_normalisation.json",
    _PHASE05 / "stats_logistic" / "selection.json",
    _PHASE05 / "predictions" / _SCORER / "stats_logistic.csv.gz",
    _PHASE1 / "e1_logistic" / "coefficients.csv",
    _PHASE1 / "e1_logistic" / "selection.json",
    _FROZEN_BUNDLE / "models.json",
    _FROZEN_BUNDLE / "models.sha256",
    _FROZEN_BUNDLE / "bundle_verification.json",
    _FROZEN_BUNDLE / "frozen_artifact_manifest.json",
    _EMB_ROOT / "embeddings_K5.npz",
    _EMB_ROOT / "embeddings_K10.npz",
]
for _seed in (1, 2, 3):
    _REQUIRED.append(_B3_ROOT / f"seed_{_seed}" / "eval_metadata.json")
    for _k in (5, 10):
        _REQUIRED.append(_B3_ROOT / f"seed_{_seed}" / "raw_scores" / f"K{_k}.npz")
for _regime in REGIMES:
    for _split in REPORT_SPLITS:
        _REQUIRED.append(_MANIFEST_ROOT / f"{_regime}_{_split}.jsonl")
        _REQUIRED.append(_MANIFEST_ROOT / f"{_regime}_{_split}.meta.json")

_MISSING = [str(path.relative_to(_REPO)) for path in _REQUIRED if not path.exists()]

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"phase-1F frozen artifacts missing: {_MISSING[:6]}",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _first_distinct_ref_rows(cohort, mask_name: str, count: int = 10) -> np.ndarray:
    """Row indices of the first ``count`` distinct refs (row order) in one mask."""
    rows = np.flatnonzero(cohort.masks[mask_name])
    if rows.size == 0:
        return rows
    _, first_pos = np.unique(cohort.ref_id[rows], return_index=True)
    return rows[np.sort(first_pos)[:count]]


def _bank_boxes(image_ids) -> dict:
    """``{image_id: boxes[N,4] float32}`` straight from the frozen bank."""
    wanted = sorted({int(x) for x in image_ids})
    with h5py.File(_BANK, "r") as handle:
        return {
            image_id: np.asarray(handle[f"image_{image_id}"]["boxes"], dtype=np.float32)
            for image_id in wanted
        }


def _load_manifest_table(regime: str) -> dict:
    """``{ref_id: ManifestEntry}`` of one regime over the pooled test splits."""
    table: dict = {}
    for split in REPORT_SPLITS:
        manifest = ManifestFile.load(_MANIFEST_ROOT / f"{regime}_{split}.jsonl")
        for entry in manifest.entries:
            table[int(entry.ref_id)] = entry
    return table


def _metric():
    return relev._metric_fn("auroc_correct")


# ---------------------------------------------------------------------------
# module fixtures (the two ~10 s lifts, reused by every test)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cohort():
    return load_hard_cohort(features_dir=_FEATURES_DIR, manifests_root=_MANIFEST_ROOT)


@pytest.fixture(scope="module")
def frozen():
    # Historical exact-refit errors remain recorded in the original pytest and
    # recovery evidence.  The fixture now verifies the existing frozen bundle
    # directly at the unchanged 1e-9 artifact tolerance; it does not refit.
    bundle = load_models(_FROZEN_BUNDLE, verify_checksum=True)
    verification = verify_against_frozen_artifacts(
        bundle,
        phase05_root=_PHASE05,
        phase1_root=_PHASE1,
        phase1f_root=_REPO / "results" / "phase1f_hard_semantic",
        b3_root=_B3_ROOT,
        features_dir=_FEATURES_DIR,
        scorers=bundle.scorers,
    )
    return SimpleNamespace(seeds=bundle.seeds, verification=verification, bundle=bundle)


# ---------------------------------------------------------------------------
# 1. same-category candidates share the target's GT category
# ---------------------------------------------------------------------------
def test_same_category_candidates_share_gt_category(cohort) -> None:
    rows = np.flatnonzero(cohort.masks["same4"])
    assert rows.size > 0, "the A8.2 same4 cohort must not be empty"

    # structural: the C5 prefix is fully same-category by construction
    for row in rows.tolist():
        entry = cohort.entries["same_category"][int(cohort.ref_id[row])]
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        n_avail = int(entry.n_same_category_available)
        assert n_avail >= 4
        assert abs(float(entry.hard_fraction_by_K[5]) - 1.0) <= 1e-12
        assert int(entry.n_same_used_by_K[5]) == min(n_avail, 4) == 4
        ci = cohort.candidate_indices(row, "same_category", 5)
        # the 4 distractors of C5 are the same-category ordering prefix
        np.testing.assert_array_equal(ci[1:], order[:4])

    # small-sample real GT-category check (same loader as build_manifests)
    if not _COCO_INSTANCES.exists():
        pytest.skip(f"COCO GT annotations not present: {_COCO_INSTANCES.name}")
    sample = _first_distinct_ref_rows(cohort, "same4", 10)
    images = [int(x) for x in cohort.image_id[sample]]
    boxes = _bank_boxes(images)
    payload = json.loads(_COCO_INSTANCES.read_text(encoding="utf-8"))
    index = build_coco_index(payload, source=str(_COCO_INSTANCES), only_image_ids=images)

    for row in sample.tolist():
        image_id = int(cohort.image_id[row])
        gt = index.objects(image_id)
        categories, _ = assign_gt_category(
            boxes[image_id], gt.boxes, gt.categories, iou_thresh=DEFAULT_IOU_THRESH
        )
        ci = cohort.candidate_indices(row, "same_category", 5)
        target_category = int(categories[ci[0]])
        assert target_category != -1, f"row {row}: target has no GT category"
        assert np.all(categories[ci[1:]] == target_category), (
            f"row {row}: C5 distractor categories {categories[ci[1:]].tolist()} "
            f"!= target {target_category}"
        )


# ---------------------------------------------------------------------------
# 2. no target-equivalent candidate (IoU < 0.5 against the target box)
# ---------------------------------------------------------------------------
def test_no_target_equivalent_candidates(cohort) -> None:
    sample = _first_distinct_ref_rows(cohort, "base", 10)
    images = [int(x) for x in cohort.image_id[sample]]
    boxes = _bank_boxes(images)

    for row in sample.tolist():
        image_id = int(cohort.image_id[row])
        bank = boxes[image_id]
        for regime in REGIMES:
            entry = cohort.entries[regime][int(cohort.ref_id[row])]
            target_index = int(entry.target_index)
            order = np.asarray(entry.distractor_order, dtype=np.int64)
            assert target_index not in order.tolist(), f"row {row}/{regime}: target in order"
            ci = cohort.candidate_indices(row, regime, 5)
            assert np.unique(ci).size == ci.size, f"row {row}/{regime}: duplicate candidates"
            ious = iou_matrix(bank[ci[1:]], bank[[target_index]]).reshape(-1)
            assert float(ious.max()) < DEFAULT_IOU_THRESH, (
                f"row {row}/{regime}: target-equivalent candidate (max IoU "
                f"{float(ious.max()):.4f} >= {DEFAULT_IOU_THRESH})"
            )


# ---------------------------------------------------------------------------
# 3. the two regimes are strictly matched per ref (positive + negative)
# ---------------------------------------------------------------------------
def test_matched_random_hard_cohorts_identical(cohort, tmp_path) -> None:
    random_table = _load_manifest_table("random")
    hard_table = _load_manifest_table("same_category")

    for ref in np.unique(cohort.ref_id).tolist():
        rand = random_table[int(ref)]
        hard = hard_table[int(ref)]
        assert rand.target_index == hard.target_index, f"ref {ref}: target mismatch"
        np.testing.assert_array_equal(
            np.sort(np.asarray(rand.distractor_order)),
            np.sort(np.asarray(hard.distractor_order)),
            err_msg=f"ref {ref}: valid distractor pool differs across regimes",
        )
        assert int(rand.n_same_category_available) == int(hard.n_same_category_available)

    # negative: a perturbed ordering must trip the matched-control assertion.
    for regime in REGIMES:
        for split in REPORT_SPLITS:
            for suffix in (".jsonl", ".meta.json"):
                shutil.copy2(
                    _MANIFEST_ROOT / f"{regime}_{split}{suffix}",
                    tmp_path / f"{regime}_{split}{suffix}",
                )

    ref0 = int(cohort.ref_id[0])
    split0 = str(cohort.eval_split[0])
    corrupted = tmp_path / f"random_{split0}.jsonl"
    lines: list[str] = []
    touched = False
    for line in corrupted.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if int(payload["ref_id"]) == ref0:
            order = list(payload["distractor_order"])[:-1]  # drop one distractor
            payload["distractor_order"] = order
            payload["n_valid_distractors"] = len(order)  # keep the entry self-consistent
            lines.append(json.dumps(payload, ensure_ascii=False))
            touched = True
        else:
            lines.append(line)
    assert touched, f"ref {ref0} not found in random_{split0}.jsonl"
    corrupted.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AssertionError):
        load_hard_cohort(features_dir=_FEATURES_DIR, manifests_root=tmp_path)


# ---------------------------------------------------------------------------
# 4. K is fixed: C_K / level orderings always hold exactly K candidates
# ---------------------------------------------------------------------------
def test_k_fixed(cohort) -> None:
    # candidate_indices: C_K always holds exactly K entries = [target] + order[:K-1]
    for mask_name in ("same4", "same8", "same9"):
        for row in _first_distinct_ref_rows(cohort, mask_name, 5).tolist():
            for regime in REGIMES:
                entry = cohort.entries[regime][int(cohort.ref_id[row])]
                order = np.asarray(entry.distractor_order, dtype=np.int64)
                ci5 = cohort.candidate_indices(row, regime, 5)
                ci10 = cohort.candidate_indices(row, regime, 10)
                assert ci5.size == 5 and ci10.size == 10
                np.testing.assert_array_equal(
                    ci5, np.concatenate(([entry.target_index], order[:4]))
                )
                np.testing.assert_array_equal(
                    ci10, np.concatenate(([entry.target_index], order[:9]))
                )

    # level_indices: level m needs >= m same-category distractors, so probe the
    # "same8" mask (proves n_same_category_available >= 8 for every row); its
    # stored rest pool is >= 16 for every row (verified from the frozen cohort),
    # so ``k-1-m <= 9`` entries are always available for m in HARD_FRACTION_LEVELS.
    for row in _first_distinct_ref_rows(cohort, "same8", 5).tolist():
        entry = cohort.entries["same_category"][int(cohort.ref_id[row])]
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        n_avail = int(entry.n_same_category_available)
        assert n_avail >= max(HARD_FRACTION_LEVELS)
        for level in HARD_FRACTION_LEVELS:
            li = cohort.level_indices(row, level, 10)
            assert li.size == 10
            expected = np.concatenate(
                (
                    [entry.target_index],
                    order[:level],
                    order[n_avail : n_avail + (10 - 1 - level)],
                )
            )
            np.testing.assert_array_equal(li, expected)


# ---------------------------------------------------------------------------
# 5. frozen E1b coefficients unchanged
# ---------------------------------------------------------------------------
def test_frozen_e1b_coefficients_unchanged(frozen) -> None:
    assert frozen.verification["all"]["e1b_coefficients_max_abs"] <= _TOL

    expected = list(rfeat.stat_feature_names()) + list(sfeat.SEMANTIC_STAT_NAMES)
    path = _PHASE1 / "e1_logistic" / "coefficients.csv"
    table: dict = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["feature"]).lower() in ("intercept", "bias", "const"):
                continue
            table.setdefault(str(row["scorer"]), {})[str(row["feature"])] = float(row["coefficient"])

    for scorer, models in frozen.seeds.items():
        coef = models.e1b_coef
        reference = np.asarray([table[scorer][name] for name in expected], dtype=np.float64)
        assert coef.shape == reference.shape
        assert float(np.abs(coef - reference).max()) <= _TOL, f"{scorer}: E1b coef drift"


# ---------------------------------------------------------------------------
# 6. frozen Stats Logistic unchanged (prediction multiset + normalisation)
# ---------------------------------------------------------------------------
def test_frozen_stats_logistic_unchanged(frozen) -> None:
    assert frozen.verification["all"]["stats_logistic_pred_max_abs"] <= _TOL
    assert frozen.verification["all"]["stats17_normalisation_max_abs"] <= _TOL

    models = frozen.seeds[_SCORER]
    grouped: dict = {}
    with gzip.open(
        _PHASE05 / "predictions" / _SCORER / "stats_logistic.csv.gz", "rt", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            key = (int(row["K"]), str(row["eval_split"]))
            grouped.setdefault(key, []).append(float(row["reliability_score"]))

    for k in (5, 10):
        core = sdata.load_scorer_canonical(_SCORER, k, b3_root=_B3_ROOT)
        eval_split = np.asarray(core.eval_split)
        features = rfeat.stat_features(
            np.asarray(core.scores, dtype=np.float64), temperature=float(models.temperature)
        )
        confidence = models.stats_probability(rfeat.normalize_apply(features, models.stats_fit))
        for split in _COMPARE_SPLITS:
            ours = np.sort(confidence[eval_split == split])
            reference = np.sort(np.asarray(grouped[(k, split)], dtype=np.float64))
            assert ours.size == reference.size
            if ours.size:
                assert float(np.abs(ours - reference).max()) <= _TOL, (
                    f"stats logistic K={k} split={split}: prediction multiset drift"
                )


# ---------------------------------------------------------------------------
# 7. no hard-regime labels enter the reliability-training rows
# ---------------------------------------------------------------------------
def test_no_hard_regime_labels_enter_training(frozen) -> None:
    split = sdata.load_split_from_manifest(_SPLIT_MANIFEST)
    core = sdata.load_scorer_canonical(_SCORER, 5, b3_root=_B3_ROOT)
    eval_split = np.asarray(core.eval_split)
    image_id = np.asarray(core.image_id, dtype=np.int64)
    train = split.row_mask(eval_split, image_id, kind="reliability_train")

    assert train.any()
    assert np.all(eval_split[train] == "val_calib"), "train rows must all be val_calib"
    assert not np.any(np.isin(eval_split[train], np.asarray(REPORT_SPLITS)))

    # The original train-row provenance remains checked independently of the
    # load-only recovery path: training inputs are exactly the frozen
    # reliability_train rows, all val_calib and outside either report split.
    assert all(value <= _TOL for value in frozen.verification["all"].values())

    stats_selection = json.loads((_PHASE05 / "stats_logistic" / "selection.json").read_text(encoding="utf-8"))
    e1b_selection = json.loads((_PHASE1 / "e1_logistic" / "selection.json").read_text(encoding="utf-8"))
    for scorer, models in frozen.seeds.items():
        expected_stats_c = float(stats_selection["per_scorer"][scorer]["chosen_hp"])
        expected_e1b_c = float(e1b_selection["per_scorer"][scorer]["models"]["e1b_stats_semantic"]["selected_hp"])
        assert abs(float(models.stats_C) - expected_stats_c) <= _TOL, f"{scorer}: Stats C does not match frozen selection"
        assert abs(float(models.e1b_C) - expected_e1b_c) <= _TOL, f"{scorer}: E1b C does not match frozen selection"


# ---------------------------------------------------------------------------
# 8. the same frozen normalisation is reused (stats + semantic blocks)
# ---------------------------------------------------------------------------
def test_same_normalization_reused(frozen) -> None:
    models = frozen.seeds[_SCORER]

    payload = json.loads((_FEATURES_DIR / f"{_SCORER}_normalisation.json").read_text(encoding="utf-8"))
    variant = payload["variants"]["stats_logK"]
    assert float(np.abs(np.asarray(models.stats_fit.mean) - np.asarray(variant["mean"])).max()) <= _TOL
    assert float(np.abs(np.asarray(models.stats_fit.std) - np.asarray(variant["std"])).max()) <= _TOL

    # semantic block: re-fit the train-only normalisation from the frozen stores
    split = sdata.load_split_from_manifest(_SPLIT_MANIFEST)
    anchor = sdata.load_scorer_canonical(_SCORER, 5, b3_root=_B3_ROOT)
    eval_split = np.asarray(anchor.eval_split)
    image_id = np.asarray(anchor.image_id, dtype=np.int64)
    train_flags = {
        k: split.row_mask(eval_split, image_id, kind="reliability_train") for k in (5, 10)
    }
    raw_train = {}
    for k in (5, 10):
        store = sdata.load_embedding_store(k, out_root=_EMB_ROOT)
        core = sdata.load_scorer_canonical(_SCORER, k, b3_root=_B3_ROOT)
        flags = train_flags[k]
        raw_train[k] = sfeat.semantic_stats(
            np.asarray(store.z_q, dtype=np.float32)[flags],
            np.asarray(store.z_i, dtype=np.float32)[flags],
            np.asarray(core.scores, dtype=np.float64)[flags],
        )
    combined = np.vstack([raw_train[5], raw_train[10]])
    refit = rfeat.normalize_fit(
        combined, fit_rows=np.arange(combined.shape[0]), keys=sfeat.SEMANTIC_STAT_NAMES
    )
    assert float(np.abs(np.asarray(models.sem_fit.mean) - refit.mean).max()) <= _TOL
    assert float(np.abs(np.asarray(models.sem_fit.std) - refit.std).max()) <= _TOL

    # Frozen load-and-predict equals normalization + hstack + the frozen
    # closed-form probabilities (no re-fit).
    rng = np.random.default_rng(0)
    stats17_raw = rng.normal(size=(8, len(rfeat.stat_feature_names())))
    sem16_raw = rng.normal(size=(8, len(sfeat.SEMANTIC_STAT_NAMES)))
    stats_conf, e1b_conf = frozen.bundle.predict(_SCORER, stats17_raw, sem16_raw)
    stats_std = rfeat.normalize_apply(stats17_raw, models.stats_fit)
    sem_std = rfeat.normalize_apply(sem16_raw, models.sem_fit)
    np.testing.assert_allclose(stats_conf, models.stats_probability(stats_std), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        e1b_conf, models.e1b_probability(np.hstack([stats_std, sem_std])), rtol=0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 9. the reliability model never mutates or reorders its inputs
# ---------------------------------------------------------------------------
def test_candidate_ranking_unchanged_by_reliability_model(frozen) -> None:
    models = frozen.seeds[_SCORER]
    rng = np.random.default_rng(1)
    n = 6
    stats17_raw = rng.normal(size=(n, len(rfeat.stat_feature_names())))
    sem16_raw = rng.normal(size=(n, len(sfeat.SEMANTIC_STAT_NAMES)))
    snapshot_stats = stats17_raw.copy()
    snapshot_sem = sem16_raw.copy()

    stats_conf, e1b_conf = frozen.bundle.predict(_SCORER, stats17_raw, sem16_raw)
    np.testing.assert_array_equal(stats17_raw, snapshot_stats, err_msg="stats17 inputs mutated")
    np.testing.assert_array_equal(sem16_raw, snapshot_sem, err_msg="sem16 inputs mutated")

    stats_again, e1b_again = frozen.bundle.predict(_SCORER, stats17_raw, sem16_raw)
    np.testing.assert_array_equal(stats_conf, stats_again)
    np.testing.assert_array_equal(e1b_conf, e1b_again)

    perm = rng.permutation(n)
    stats_perm, e1b_perm = frozen.bundle.predict(_SCORER, stats17_raw[perm], sem16_raw[perm])
    np.testing.assert_allclose(stats_perm, stats_conf[perm], rtol=0, atol=1e-12)
    np.testing.assert_allclose(e1b_perm, e1b_conf[perm], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 10. paired diff-of-diffs bootstrap uses same-image clusters
# ---------------------------------------------------------------------------
def test_paired_bootstrap_same_image_clusters() -> None:
    metric = _metric()
    n_images, per_image = 10, 4
    image_id = np.repeat(np.arange(n_images), per_image)
    correct = np.tile([1.0, 1.0, 0.0, 0.0], n_images)
    base = np.tile([0.9, 0.8, 0.1, 0.2], n_images)

    def stats(conf: np.ndarray):
        return phase0a.SampleStats.from_conf_correct(conf, correct)

    hard_sem = stats(base)
    hard_score = stats(np.clip(base - 0.1, 0.0, 1.0))
    rand_sem = stats(np.clip(base * 0.9, 0.0, 1.0))
    rand_score = stats(np.clip(base * 0.5 + 0.1, 0.0, 1.0))

    first = hard_eval.paired_diff_of_diffs_bootstrap(
        metric, hard_sem, hard_score, rand_sem, rand_score,
        cluster_ids=image_id, n_replicates=100, seed=0,
    )
    second = hard_eval.paired_diff_of_diffs_bootstrap(
        metric, hard_sem, hard_score, rand_sem, rand_score,
        cluster_ids=image_id, n_replicates=100, seed=0,
    )
    np.testing.assert_array_equal(first["replicates"], second["replicates"])
    assert first["resample_unit"] == "image"
    assert first["n_clusters"] == n_images
    assert first["n"] == n_images * per_image

    # A == B and C == D -> every replicate is exactly zero
    zeros = hard_eval.paired_diff_of_diffs_bootstrap(
        metric, hard_sem, hard_sem, rand_sem, rand_sem,
        cluster_ids=image_id, n_replicates=100, seed=0,
    )
    assert np.all(zeros["replicates"] == 0.0)
    assert zeros["diff"] == 0.0

    # without clusters the resample unit is the sample
    unclustered = hard_eval.paired_diff_of_diffs_bootstrap(
        metric, hard_sem, hard_score, rand_sem, rand_score, n_replicates=100, seed=0
    )
    assert unclustered["resample_unit"] == "sample"
    assert unclustered["n_clusters"] == unclustered["n"]

    # manual replay of the first replicate with the shared cluster sampler
    sampler = phase0a._ClusterSampler(image_id)
    rng = np.random.default_rng(0)
    draw = sampler.draw(rng)
    expected = (float(metric(hard_sem.take(draw))) - float(metric(hard_score.take(draw)))) - (
        float(metric(rand_sem.take(draw))) - float(metric(rand_score.take(draw)))
    )
    np.testing.assert_allclose(first["replicates"][0], expected, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 11. manifest generation is deterministic
# ---------------------------------------------------------------------------
def test_deterministic_manifest_generation(cohort) -> None:
    path = _MANIFEST_ROOT / "same_category_testA.jsonl"
    left = ManifestFile.load(path)
    right = ManifestFile.load(path)
    assert len(left.entries) == len(right.entries)
    for a, b in zip(left.entries, right.entries):
        assert a.ref_id == b.ref_id
        np.testing.assert_array_equal(a.distractor_order, b.distractor_order)

    for row in _first_distinct_ref_rows(cohort, "same4", 5).tolist():
        for regime in REGIMES:
            np.testing.assert_array_equal(
                cohort.candidate_indices(row, regime, 5),
                cohort.candidate_indices(row, regime, 5),
            )
        entry = cohort.entries["same_category"][int(cohort.ref_id[row])]
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        n_prefix = min(int(entry.n_same_category_available), 5 - 1)
        assert n_prefix >= 1
        # the same-category prefix is the ascending bank-index block
        assert np.all(np.diff(order[:n_prefix]) > 0), f"row {row}: same-cat prefix not ascending"


# ---------------------------------------------------------------------------
# 12. the A7.3 semantic statistics are computed exactly as documented
# ---------------------------------------------------------------------------
def test_manipulation_features_correctly_computed() -> None:
    rng = np.random.default_rng(0)
    n, k, d = 4, 5, 8
    z_q = rng.normal(size=(n, d))
    z_q /= np.linalg.norm(z_q, axis=1, keepdims=True)
    z_i = rng.normal(size=(n, k, d))
    z_i /= np.linalg.norm(z_i, axis=2, keepdims=True)
    scores = rng.normal(size=(n, k))

    stats = sfeat.semantic_stats(z_q, z_i, scores)
    assert stats.shape == (n, len(sfeat.SEMANTIC_STAT_NAMES))

    order = np.argsort(-scores, axis=1, kind="stable")
    rows = np.arange(n)
    winner = order[:, 0]
    second = order[:, 1]

    a = np.einsum("nd,nkd->nk", z_q, z_i)  # cos(z_q, z_i)
    z_winner = z_i[rows, winner]
    v = np.einsum("nd,nkd->nk", z_winner, z_i)  # cos(z_winner, z_i)
    masked = np.where(np.arange(k)[None, :] != winner[:, None], v, -np.inf)

    clip_margin12 = a[rows, winner] - a[rows, second]
    cand_vmax = masked.max(axis=1)
    cand_top12_sim = v[rows, second]

    np.testing.assert_allclose(stats[:, 2], clip_margin12, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats[:, 6], cand_vmax, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats[:, 9], cand_top12_sim, rtol=1e-12, atol=1e-12)

    assert sfeat.SEMANTIC_STAT_NAMES.index("clip_margin12") == 2
    assert sfeat.SEMANTIC_STAT_NAMES.index("cand_vmax") == 6
    assert sfeat.SEMANTIC_STAT_NAMES.index("cand_top12_sim") == 9
