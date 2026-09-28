"""Phase 0B / B3 data-pipeline tests (CPU, synthetic, fast).

Built on the *real* frozen writers - :func:`ccg.features.cache.write_feature_cache`,
:class:`ccg.data.manifests.ManifestFile`, :func:`ccg.data.bank.write_bank_entry` -
so the synthetic corpus exercises exactly the on-disk contracts the production
run uses.  Layout (6 images x 12 proposals, 2 sentences per ref):

* image 1 -> ``val_select``, image 2 -> ``val_calib``,
* image 3 -> ``testA``, image 4 -> ``testB``, images 5-6 -> ``train``.

What is asserted:

1. geometry normalisation (centre / area / objectness) and the hard error on a
   missing true image size;
2. the ``C_K`` prefix identity - the B3 candidate bank rows equal
   ``phase0a.score_sets``'s ``candidate_indices[K]``;
3. bit-identical per-candidate rows across ``K`` (`atol=0`);
4. model independence: the same ``(q, c_i)`` scores identically under different
   ``K`` and under a permutation of the candidate order (`atol=0`);
5. training iterators only emit ``train`` rows, eval iterators match the
   Phase 0A row set (count and sentence ids).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccg.data import bank as bank_module
from ccg.data.manifests import SCHEMA_VERSION, ManifestEntry, ManifestFile
from ccg.experiment import phase0a
from ccg.features.cache import FeatureCache, write_feature_cache, write_text_index
from ccg.models.b3_data import (
    B3Corpus,
    build_image_sizes,
    geometry_features_centered,
    load_image_sizes,
)
from ccg.models.independent import IndependentMLPScorer

N_IMAGES = 6
N_PROPOSALS = 12
FEATURES_DIM = 512
KS = (5, 10)
TRAIN_IMAGES = (5, 6)
VAL_SELECT_IMAGES = (1,)
IMG_SIZE = {image_id: (100 + image_id, 80 + image_id) for image_id in range(1, N_IMAGES + 1)}
#: image_id -> UNC split stored in text_index.csv
SPLIT_OF = {1: "val", 2: "val", 3: "testA", 4: "testB", 5: "train", 6: "train"}
EVAL_SPLIT_OF = {1: "val_select", 2: "val_calib"}


# ---------------------------------------------------------------------------
# synthetic dataset
# ---------------------------------------------------------------------------
def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    v = rng.normal(size=(n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _boxes(image_id: int) -> np.ndarray:
    """Deterministic ``[12, 4]`` xyxy boxes (centre ``(j+5, j+10)``, size ``(10, 20)``)."""
    rows = []
    for j in range(N_PROPOSALS):
        rows.append([j + image_id, j + image_id, j + 10 + image_id, j + 20 + image_id])
    return np.asarray(rows, dtype=np.float32)


def _objectness(image_id: int) -> np.ndarray:
    return np.linspace(0.1, 1.0, N_PROPOSALS, dtype=np.float32) + np.float32(image_id) * 0.01


def _make_entry(rng: np.random.Generator, ref_id: int, image_id: int, split: str, eval_split=None):
    target = int(rng.integers(0, N_PROPOSALS))
    order = [i for i in range(N_PROPOSALS) if i != target]
    return ManifestEntry(
        ref_id=int(ref_id),
        image_id=int(image_id),
        split=str(split),
        target_index=target,
        distractor_order=np.asarray(order, dtype=np.int32),
        n_valid_distractors=len(order),
        n_target_equiv_removed=0,
        n_same_category_available=0,
        eligible={k: True for k in (5, 10, 20, 50)},
        n_same_used_by_K={k: 0 for k in (5, 10, 20, 50)},
        hard_fraction_by_K={k: 0.0 for k in (5, 10, 20, 50)},
        target_max_iou=0.8,
        eval_split=None if eval_split is None else str(eval_split),
    )


def _write_images(images_root: Path) -> None:
    from PIL import Image

    (images_root / "train2014").mkdir(parents=True, exist_ok=True)
    for image_id, (width, height) in IMG_SIZE.items():
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        path = images_root / "train2014" / f"COCO_train2014_{image_id:012d}.jpg"
        Image.fromarray(arr).save(path)


def _write_image_manifest(path: Path) -> None:
    lines = ["image_id,file_name,coco_split,ref_count"]
    for image_id in range(1, N_IMAGES + 1):
        lines.append(f"{image_id},COCO_train2014_{image_id:012d}.jpg,train2014,1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_dataset(tmp_path: Path) -> dict:
    """Write the synthetic corpus; return the paths / refs a B3Corpus needs."""
    features_root = tmp_path / "features"
    images_root = tmp_path / "mscoco"
    _write_images(images_root)
    image_manifest = tmp_path / "full_image_manifest.csv"
    _write_image_manifest(image_manifest)

    rng = np.random.default_rng(0)
    region = {image_id: _unit_rows(rng, N_PROPOSALS, FEATURES_DIM) for image_id in range(1, N_IMAGES + 1)}
    text = {sid: _unit_rows(rng, 1, FEATURES_DIM)[0] for sid in range(2 * N_IMAGES)}
    global_ = {image_id: _unit_rows(rng, 1, FEATURES_DIM)[0] for image_id in range(1, N_IMAGES + 1)}
    write_feature_cache(
        features_root, region, text, global_, {"native_logit_scale": 100.0, "source": "b3-test"}
    )
    text_rows = []
    for image_id in range(1, N_IMAGES + 1):
        for j in (0, 1):
            sid = 2 * (image_id - 1) + j
            text_rows.append(
                {
                    "sentence_id": sid,
                    "ref_id": image_id,
                    "sent_id": j,
                    "image_id": image_id,
                    "split": SPLIT_OF[image_id],
                    "text": f"text {sid}",
                }
            )
    text_rows.sort(key=lambda row: row["sentence_id"])
    write_text_index(features_root, text_rows)

    bank_path = tmp_path / "proposals.h5"
    import h5py

    with h5py.File(str(bank_path), "w") as handle:
        for image_id in range(1, N_IMAGES + 1):
            bank_module.write_bank_entry(
                handle, image_id, _boxes(image_id), _objectness(image_id)
            )

    image_sizes_path = tmp_path / "image_sizes.npz"
    with open(image_sizes_path, "wb") as handle:
        np.savez(handle, **{str(k): np.asarray(v, dtype=np.int64) for k, v in IMG_SIZE.items()})

    refs = [
        {"ref_id": image_id, "image_id": image_id, "sent_ids": [2 * (image_id - 1), 2 * (image_id - 1) + 1]}
        for image_id in range(1, N_IMAGES + 1)
    ]

    mrng = np.random.default_rng(1)
    meta = {"schema_version": SCHEMA_VERSION, "regime": "random", "source": "b3-test"}
    ManifestFile(
        regime="random",
        meta=meta,
        entries=[_make_entry(mrng, 5, 5, "train"), _make_entry(mrng, 6, 6, "train")],
    ).save(tmp_path / "random_train.jsonl")
    ManifestFile(
        regime="random",
        meta=meta,
        entries=[
            _make_entry(mrng, 1, 1, "val", eval_split="val_select"),
            _make_entry(mrng, 2, 2, "val", eval_split="val_calib"),
        ],
    ).save(tmp_path / "random_val.jsonl")
    ManifestFile(regime="random", meta=meta, entries=[_make_entry(mrng, 3, 3, "testA")]).save(
        tmp_path / "random_testA.jsonl"
    )
    ManifestFile(regime="random", meta=meta, entries=[_make_entry(mrng, 4, 4, "testB")]).save(
        tmp_path / "random_testB.jsonl"
    )

    return {
        "features_root": features_root,
        "manifests_root": tmp_path,
        "refs": refs,
        "bank_path": bank_path,
        "image_sizes_path": image_sizes_path,
        "images_root": images_root,
        "image_manifest": image_manifest,
    }


def _corpus(dataset: dict) -> B3Corpus:
    return B3Corpus(
        dataset["features_root"],
        dataset["manifests_root"],
        dataset["refs"],
        dataset["bank_path"],
        image_sizes_path=dataset["image_sizes_path"],
        region_cache_size=4,
        bank_cache_size=4,
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_geometry_is_centred_normalised_and_raises_without_size():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 25]], dtype=np.float32)
    objectness = np.array([0.3, 0.7], dtype=np.float32)
    geo = geometry_features_centered(boxes, (640, 480), objectness)
    assert geo.shape == (2, 6) and geo.dtype == np.float32
    np.testing.assert_allclose(
        geo[0], [5 / 640, 5 / 480, 10 / 640, 10 / 480, (10 * 10) / (640 * 480), 0.3], atol=1e-6
    )
    np.testing.assert_allclose(
        geo[1], [10 / 640, 15 / 480, 10 / 640, 20 / 480, (10 * 20) / (640 * 480), 0.7], atol=1e-6
    )
    # no objectness -> the sixth column is exactly zero (never a set statistic)
    no_obj = geometry_features_centered(boxes, (640, 480))
    assert no_obj.shape == (2, 6) and np.array_equal(no_obj[:, 5], np.zeros(2, np.float32))

    with pytest.raises(ValueError, match="image_size"):
        geometry_features_centered(boxes, None)
    with pytest.raises(ValueError, match="positive"):
        geometry_features_centered(boxes, (0, 480))
    with pytest.raises(ValueError, match=r"\[K,4\]"):
        geometry_features_centered(np.zeros((2, 3), np.float32), (640, 480))
    with pytest.raises(ValueError, match="expected 2"):
        geometry_features_centered(boxes, (640, 480), np.zeros(3, np.float32))


# ---------------------------------------------------------------------------
# image sizes
# ---------------------------------------------------------------------------
def test_build_image_sizes_is_correct_idempotent_and_strict(tmp_path):
    images_root = tmp_path / "mscoco"
    _write_images(images_root)
    manifest = tmp_path / "full_image_manifest.csv"
    _write_image_manifest(manifest)
    out = tmp_path / "cache" / "image_sizes.npz"

    built = build_image_sizes(images_root, manifest, out, workers=4)
    assert built == out and out.exists()
    sizes = load_image_sizes(out)
    assert sizes == IMG_SIZE
    assert all(isinstance(w, int) for w, _ in sizes.values())

    before = out.read_bytes()
    assert build_image_sizes(images_root, manifest, out, workers=2) == out
    assert out.read_bytes() == before, "an existing cache must be returned untouched (idempotent)"

    # every requested image is covered; a missing image aborts the build
    with pytest.raises(FileNotFoundError):
        load_image_sizes(tmp_path / "does_not_exist.npz")
    broken = tmp_path / "broken_manifest.csv"
    broken.write_text("image_id,file_name,coco_split\n999,COCO_missing.jpg,train2014\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not be measured"):
        build_image_sizes(images_root, broken, tmp_path / "missing.npz", workers=2)


def test_corpus_image_size_lookup_is_strict(tmp_path):
    dataset = _build_dataset(tmp_path)
    corpus = _corpus(dataset)
    try:
        assert corpus.image_size(1) == IMG_SIZE[1]
        with pytest.raises(KeyError, match="no entry"):
            corpus.image_size(9999)
    finally:
        corpus.close()


# ---------------------------------------------------------------------------
# C_K prefix identity vs Phase 0A
# ---------------------------------------------------------------------------
def test_b3_candidate_rows_match_phase0a_score_sets(tmp_path):
    dataset = _build_dataset(tmp_path)
    corpus = _corpus(dataset)
    cache = FeatureCache.open(dataset["features_root"])
    try:
        records = phase0a.load_cosine_inputs(
            dataset["features_root"],
            dataset["manifests_root"],
            dataset["refs"],
            "val_select",
            regime="random",
            ks=KS,
        )
        bundle = phase0a.score_sets(cache, records, ks=KS)
        assert bundle.n_rows(5) == len(VAL_SELECT_IMAGES) * 2
        for k in KS:
            for row, record in enumerate(bundle.records_by_k[k]):
                expected = np.asarray(bundle.candidate_indices[k][row], dtype=np.int64)
                got = corpus.candidate_bank_indices(record, k)
                np.testing.assert_array_equal(got, expected)
                assert got[0] == record.target_index, "the target must stay at local index 0"
                example = corpus.example_for(record, k)
                assert example.target_index == 0 and example.K == k
                region = cache.region_features(record.image_id).astype(np.float32)
                np.testing.assert_array_equal(example.candidate_features, region[expected])
                np.testing.assert_array_equal(
                    example.query_feature, cache.text_features(record.sentence_id).astype(np.float32)
                )
    finally:
        cache.close()
        corpus.close()


# ---------------------------------------------------------------------------
# per-candidate bit-identity across K
# ---------------------------------------------------------------------------
def test_shared_candidate_rows_are_bit_identical_across_k(tmp_path):
    dataset = _build_dataset(tmp_path)
    corpus = _corpus(dataset)
    try:
        sample = corpus.train_samples()[0]
        small = corpus.example_for(sample, 5)
        large = corpus.example_for(sample, 10)
        np.testing.assert_array_equal(small.query_feature, large.query_feature)
        np.testing.assert_array_equal(small.candidate_features, large.candidate_features[:5])
        np.testing.assert_array_equal(small.geometry, large.geometry[:5])
        assert small.target_index == large.target_index == 0

        record = next(iter(corpus.eval_records("val_select", KS)))
        r5 = corpus.example_for(record, 5)
        r10 = corpus.example_for(record, 10)
        np.testing.assert_array_equal(r5.candidate_features, r10.candidate_features[:5])
        np.testing.assert_array_equal(r5.geometry, r10.geometry[:5])
    finally:
        corpus.close()


# ---------------------------------------------------------------------------
# model independence (atol=0)
# ---------------------------------------------------------------------------
def test_model_scores_a_candidate_independently_of_the_set(tmp_path):
    dataset = _build_dataset(tmp_path)
    corpus = _corpus(dataset)
    try:
        sample = corpus.train_samples()[0]
        example = corpus.example_for(sample, 10)
        query = example.query_feature
        candidates = example.candidate_features
        geometry = example.geometry

        scorer = IndependentMLPScorer(feature_dim=FEATURES_DIM, hidden_dim=128, geo_dim=6, seed=0)

        def per_row(indices):
            return np.asarray(
                [scorer.score(query, candidates[i : i + 1], geometry[i : i + 1])[0] for i in indices],
                dtype=np.float32,
            )

        # same (q, c_i) under K=5 and K=10 -> exactly equal
        np.testing.assert_array_equal(per_row(range(5)), per_row(range(10))[:5])

        # permutation of the candidate order -> element-wise identity
        perm = np.array([7, 0, 9, 3, 1, 8, 2, 6, 4, 5], dtype=np.int64)
        np.testing.assert_array_equal(per_row(perm), per_row(range(10))[perm])

        # the batch path is only numerically equivalent (BLAS/gemm row-blocking);
        # the exact statement is the per-row one above.
        batch5 = scorer.score(query, candidates[:5], geometry[:5])
        batch10 = scorer.score(query, candidates[:10], geometry[:10])
        np.testing.assert_allclose(batch5, batch10[:5], atol=1e-6)

        kinds = {type(layer).__name__ for layer in scorer.net}
        assert kinds <= {"Linear", "ReLU"}, kinds
    finally:
        corpus.close()


# ---------------------------------------------------------------------------
# training / evaluation iterators
# ---------------------------------------------------------------------------
def test_train_iterator_only_emits_train_rows_and_is_deterministic(tmp_path):
    dataset = _build_dataset(tmp_path)
    corpus = _corpus(dataset)
    try:
        samples = corpus.train_samples()
        assert len(samples) == 2 * len(TRAIN_IMAGES)
        assert all(s.split == "train" for s in samples)
        assert {s.sentence_id for s in samples} == {8, 9, 10, 11}

        first = list(corpus.iter_examples("train", KS, seed=1, epoch=0))
        again = list(corpus.iter_examples("train", KS, seed=1, epoch=0))
        assert [e.sentence_id for e in first] == [e.sentence_id for e in again]
        assert all(e.split == "train" for e in first)
        assert {e.K for e in first} <= set(KS)

        # K is drawn per (seed, epoch, sentence_id): the assignment moves across epochs
        assignments = set()
        for epoch in range(10):
            assignments.add(
                tuple(
                    (e.sentence_id, e.K)
                    for e in corpus.iter_examples("train", KS, seed=1, epoch=epoch)
                )
            )
        assert len(assignments) > 1, "the training K must vary across epochs"

        # the planned bucket sizes match the yielded counts
        sizes = corpus.train_bucket_sizes(KS, seed=1, epoch=0)
        yielded = {5: 0, 10: 0}
        for example in corpus.iter_examples("train", KS, seed=1, epoch=0):
            yielded[example.K] += 1
        assert {k: v for k, v in yielded.items() if v} == {k: v for k, v in sizes.items() if v}

        with pytest.raises(ValueError, match="cannot be mixed"):
            list(corpus.iter_examples(("train", "val_select"), KS, seed=1, epoch=0))
    finally:
        corpus.close()


def test_eval_iterator_matches_phase0a_rows(tmp_path):
    dataset = _build_dataset(tmp_path)
    corpus = _corpus(dataset)
    cache = FeatureCache.open(dataset["features_root"])
    try:
        records = corpus.eval_records("val_select", KS)
        direct = phase0a.load_cosine_inputs(
            dataset["features_root"],
            dataset["manifests_root"],
            dataset["refs"],
            "val_select",
            regime="random",
            ks=KS,
        )
        assert [r.sentence_id for r in records] == [r.sentence_id for r in direct]

        bundle = phase0a.score_sets(cache, records, ks=KS)
        for k in KS:
            b3_ids = [e.sentence_id for e in corpus.iter_eval("val_select", KS) if e.K == k]
            pa_ids = [int(s) for s in bundle.sentence_ids(k)]
            assert sorted(b3_ids) == sorted(pa_ids)
            assert len(b3_ids) == bundle.n_rows(k)

        labelled = list(corpus.iter_examples("val_select", KS, seed=0))
        assert labelled and all(e.split == "val_select" for e in labelled)
        assert {e.K for e in labelled} == set(KS)

        available = corpus.eval_available_counts("val_select", KS)
        assert available == {k: bundle.n_rows(k) for k in KS}
    finally:
        cache.close()
        corpus.close()
