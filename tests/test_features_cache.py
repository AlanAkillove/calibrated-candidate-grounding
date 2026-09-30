"""CPU-only tests for the frozen feature cache (no torch, no weights).

Covers the frozen interface of ``ccg.features.cache`` (random access, offsets,
metadata roundtrip, error semantics), the crash-resume semantics of the
streaming writers, and the structural audit functions (NaN / Inf / zero-norm
counting, invalid-crop accounting) on synthetic data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.features.cache import (  # noqa: E402
    CACHE_VERSION,
    GLOBAL_FILENAME,
    REGION_FILENAME,
    TEXT_FILENAME,
    FeatureCache,
    StreamingGlobalWriter,
    StreamingRegionWriter,
    feature_health,
    merge_extraction_stats,
    read_metadata,
    read_text_index,
    update_metadata,
    write_feature_cache,
    write_global_cache,
    write_region_cache,
    write_text_cache,
    write_text_index,
)
from ccg.features.extract_regions import (  # noqa: E402
    INVALID_CROP_COLUMNS,
    read_invalid_crop_rows,
    resume_missing_ids,
    run_region_extraction,
)

FEATURE_DIM = 512


def _load_audit():
    path = _REPO_ROOT / "scripts" / "audit_cache.py"
    spec = importlib.util.spec_from_file_location("audit_cache_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


audit_cache = _load_audit()


# ---------------------------------------------------------------------------
# synthetic data builders
# ---------------------------------------------------------------------------
def _unit(rng, n, dim=FEATURE_DIM):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _build_synth(root: Path, *, seed=0):
    rng = np.random.default_rng(seed)
    region = {1: _unit(rng, 5), 2: _unit(rng, 3), 7: _unit(rng, 1)}
    region[2][0] = 0.0  # one invalid crop -> stored as an all-zero row
    text = {0: _unit(rng, 1)[0], 1: _unit(rng, 1)[0], 5: _unit(rng, 1)[0]}
    global_ = {1: _unit(rng, 1)[0], 2: _unit(rng, 1)[0], 7: _unit(rng, 1)[0]}
    metadata = {
        "cache_version": CACHE_VERSION,
        "backbone": {"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
        "native_logit_scale": 100.0,
        "invalid_crop_count": 1,
    }
    write_feature_cache(root, region, text, global_, metadata)
    # the frozen one-shot writer stores empty h5 texts; the CLI path (and the
    # fixture) fills the real sentence strings through write_text_cache
    write_text_cache(root, text, [f"sentence {sid}" for sid in sorted(text)])
    # a real CLI run always writes the full text index and the invalid-crop log
    write_text_index(
        root,
        [
            {
                "sentence_id": sid,
                "ref_id": 100 + sid,
                "sent_id": sid % 3,
                "image_id": 1 + sid,
                "split": "train2014",
                "text": f"sentence {sid}",
            }
            for sid in sorted(text)
        ],
    )
    _write_invalid_csv(root, [(2, 0, 10, 10, 10, 40)])
    return region, text, global_


def _write_invalid_csv(root: Path, rows):
    path = root / "invalid_crops.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(INVALID_CROP_COLUMNS) + "\n")
        for row in rows:
            handle.write(",".join(str(int(v)) for v in row) + "\n")
    return path


@pytest.fixture()
def synth(tmp_path):
    region, text, global_ = _build_synth(tmp_path)
    return tmp_path, region, text, global_


# ---------------------------------------------------------------------------
# random access + offsets
# ---------------------------------------------------------------------------
def test_random_access_shapes_dtypes(synth):
    root, region, text, global_ = synth
    with FeatureCache.open(root) as cache:
        for image_id, block in region.items():
            got = cache.region_features(image_id)
            assert got.shape == (block.shape[0], FEATURE_DIM)
            assert got.dtype == np.float16
            assert np.array_equal(got, np.asarray(block, dtype=np.float16))
        for sentence_id, vector in text.items():
            got = cache.text_features(sentence_id)
            assert got.shape == (FEATURE_DIM,)
            assert got.dtype == np.float16
            assert np.array_equal(got, np.asarray(vector, dtype=np.float16))
        for image_id, vector in global_.items():
            got = cache.global_feature(image_id)
            assert got.shape == (FEATURE_DIM,)
            assert got.dtype == np.float16
            assert np.array_equal(got, np.asarray(vector, dtype=np.float16))


def test_region_offsets_contiguous_and_sorted(synth):
    root, region, text, _global = synth
    with FeatureCache.open(root) as cache:
        offsets = cache.region_offsets
        ids = offsets[:, 0]
        assert list(ids) == sorted(region)  # frozen: ascending id order
        assert np.array_equal(offsets[:, 2], [region[i].shape[0] for i in sorted(region)])
        expected_starts = np.concatenate([[0], np.cumsum(offsets[:, 2])[:-1]])
        assert np.array_equal(offsets[:, 1], expected_starts)
        assert cache.region_ids().dtype == np.int64
        assert list(cache.region_ids()) == sorted(region)
        assert cache.n_images == len(region)
        assert cache.n_sentences == len(text)
        rows = {int(i): int(c) for i, _s, c in offsets}
        assert sum(rows.values()) == 5 + 3 + 1


def test_text_offsets_rows_are_arange(synth):
    root, _region, text, _global = synth
    with FeatureCache.open(root) as cache:
        offsets = cache.sentence_offsets
        assert list(offsets[:, 0]) == sorted(text)
        assert np.array_equal(offsets[:, 1], np.arange(len(text), dtype=np.int64))
        assert [cache.text(sid) for sid in sorted(text)] == [f"sentence {sid}" for sid in sorted(text)]


def test_missing_ids_raise_keyerror(synth):
    root, _region, _text, _global = synth
    with FeatureCache.open(root) as cache:
        for call in (cache.region_features, cache.global_feature):
            with pytest.raises(KeyError):
                call(999)
        with pytest.raises(KeyError):
            cache.text_features(999)


def test_open_requires_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        FeatureCache.open(tmp_path)


def test_partial_cache_raises_on_missing_part(tmp_path):
    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})
    with FeatureCache.open(tmp_path) as cache:
        assert cache.n_images == 0
        assert cache.n_sentences == 0
        with pytest.raises(FileNotFoundError):
            cache.region_features(1)
        with pytest.raises(FileNotFoundError):
            cache.text_features(0)
        with pytest.raises(FileNotFoundError):
            cache.global_feature(1)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
def test_metadata_roundtrip_and_recomputed_counts(synth):
    root, region, text, _global = synth
    meta = read_metadata(root)
    assert meta["cache_version"] == CACHE_VERSION
    assert meta["n_images"] == len(region)
    assert meta["n_sentences"] == len(text)
    assert meta["n_crops"] == sum(block.shape[0] for block in region.values())
    assert meta["native_logit_scale"] == 100.0
    assert meta["backbone"]["model_name"] == "ViT-B-32"
    with FeatureCache.open(root) as cache:
        assert cache.metadata == meta


def test_write_feature_cache_recomputes_drifting_counts(tmp_path):
    rng = np.random.default_rng(3)
    write_feature_cache(
        tmp_path,
        {1: _unit(rng, 2)},
        {0: _unit(rng, 1)[0]},
        {1: _unit(rng, 1)[0]},
        {"cache_version": CACHE_VERSION, "n_crops": 9999, "n_images": 123, "n_sentences": 42},
    )
    meta = read_metadata(tmp_path)
    assert (meta["n_images"], meta["n_crops"], meta["n_sentences"]) == (1, 2, 1)


def test_merge_extraction_stats_keeps_other_keys(tmp_path):
    update_metadata(tmp_path, {"cache_version": CACHE_VERSION, "n_images": 3})
    merge_extraction_stats(tmp_path, {"text": {"n_sentences": 7}})
    merge_extraction_stats(tmp_path, {"region": {"region_crops_encoded": 11}})
    meta = read_metadata(tmp_path)
    assert meta["cache_version"] == CACHE_VERSION
    assert meta["n_images"] == 3
    assert meta["extraction_stats"] == {
        "text": {"n_sentences": 7},
        "region": {"region_crops_encoded": 11},
    }


def test_text_index_roundtrip_and_validation(tmp_path):
    rows = [
        {"sentence_id": 0, "ref_id": 5, "sent_id": 0, "image_id": 9, "split": "train2014", "text": "a"},
        {"sentence_id": 1, "ref_id": 5, "sent_id": 1, "image_id": 9, "split": "train2014", "text": "b"},
    ]
    write_text_index(tmp_path, rows)
    parsed = read_text_index(tmp_path)
    assert parsed == rows
    with pytest.raises(ValueError):
        write_text_index(tmp_path, [{"sentence_id": 0}])  # missing columns


# ---------------------------------------------------------------------------
# feature_health / structural audit
# ---------------------------------------------------------------------------
def test_feature_health_counts():
    dim = 4
    good = np.ones((2, dim), dtype=np.float16)
    nan_row = np.zeros((1, dim), dtype=np.float16)
    nan_row[0, 1] = np.nan
    inf_row = np.zeros((1, dim), dtype=np.float16)
    inf_row[0, 2] = np.inf
    zero = np.zeros((1, dim), dtype=np.float16)
    matrix = np.concatenate([good, nan_row, inf_row, zero]).astype(np.float16)
    health = feature_health(matrix)
    assert health == {"total_rows": 5, "nan_rows": 1, "inf_rows": 1, "zero_norm_rows": 1}
    assert feature_health(np.zeros((0, dim), np.float16)) == {
        "total_rows": 0, "nan_rows": 0, "inf_rows": 0, "zero_norm_rows": 0
    }
    with pytest.raises(ValueError):
        feature_health(np.ones(dim))


def test_audit_structure_passes_on_well_formed_synth(synth):
    root, region, text, _global = synth
    report = audit_cache.audit_structure(root, bank_path=None)
    assert report["passed"], report["failures"]
    assert report["region"]["n_crops"] == sum(b.shape[0] for b in region.values())
    assert report["region"]["zero_norm_rows"] == 1
    assert report["region"]["invalid_csv_rows"] == 1
    assert report["text"]["n_sentences"] == len(text)


def test_audit_structure_flags_nan_and_zero_rows(tmp_path):
    rng = np.random.default_rng(4)
    region = {1: _unit(rng, 3)}
    region[1][0, 0] = np.nan
    region[1][1] = 0.0
    write_feature_cache(
        tmp_path,
        region,
        {0: _unit(rng, 1)[0]},
        {1: _unit(rng, 1)[0]},
        {"cache_version": CACHE_VERSION, "invalid_crop_count": 1},
    )
    write_text_index(
        tmp_path,
        [{"sentence_id": 0, "ref_id": 1, "sent_id": 0, "image_id": 1, "split": "train2014", "text": "x"}],
    )
    # one zero row but no invalid_crops.csv -> flagged; the NaN row -> flagged
    report = audit_cache.audit_structure(tmp_path, bank_path=None)
    assert not report["passed"]
    joined = " | ".join(report["failures"])
    assert "region.nan_rows" in joined
    assert "zero_norm_without_csv" in joined
    assert report["region"]["nan_rows"] == 1
    assert report["region"]["zero_norm_rows"] == 1


def test_audit_structure_flags_metadata_count_drift(synth):
    root, _region, _text, _global = synth
    update_metadata(root, {"invalid_crop_count": 7})
    report = audit_cache.audit_structure(root, bank_path=None)
    assert not report["passed"]
    assert any("metadata_invalid_count" in f for f in report["failures"])


def test_read_invalid_crop_rows_validates(tmp_path):
    path = _write_invalid_csv(tmp_path, [(2, 0, 10, 10, 10, 40)])
    assert read_invalid_crop_rows(path) == [(2, 0, 10, 10, 10, 40)]
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_invalid_crop_rows(bad)


# ---------------------------------------------------------------------------
# streaming writers (crash resume)
# ---------------------------------------------------------------------------
def test_streaming_region_writer_resume_prefix(tmp_path):
    rng = np.random.default_rng(5)
    blocks = {1: _unit(rng, 2), 2: _unit(rng, 3), 3: _unit(rng, 1)}
    path = tmp_path / REGION_FILENAME

    writer = StreamingRegionWriter(path, resume=False)
    writer.append(1, blocks[1])
    writer.append(2, blocks[2])
    writer.flush()
    writer.abandon()  # simulated crash: tmp survives, final never promoted
    assert not path.exists()
    assert (tmp_path / (REGION_FILENAME + ".tmp")).exists()

    resumed = StreamingRegionWriter(path, resume=True)
    assert resumed.source == "tmp"
    assert sorted(resumed.processed) == [1, 2]
    resumed.append(3, blocks[3])
    resumed.finish()

    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})
    with FeatureCache.open(tmp_path) as cache:
        assert list(cache.region_ids()) == [1, 2, 3]
        for image_id, block in blocks.items():
            assert np.array_equal(cache.region_features(image_id), np.asarray(block, dtype=np.float16))


def test_streaming_region_writer_rejects_bad_order_and_duplicates(tmp_path):
    rng = np.random.default_rng(6)
    writer = StreamingRegionWriter(tmp_path / REGION_FILENAME, resume=False)
    writer.append(5, _unit(rng, 2))
    with pytest.raises(ValueError):
        writer.append(5, _unit(rng, 1))  # duplicate
    with pytest.raises(ValueError):
        writer.append(4, _unit(rng, 1))  # descending
    writer.append(6, _unit(rng, 1))
    writer.abandon()


def test_streaming_global_writer_adopt_final(tmp_path):
    rng = np.random.default_rng(7)
    first = {1: _unit(rng, 1)[0], 2: _unit(rng, 1)[0]}
    write_global_cache(tmp_path, first)
    path = tmp_path / GLOBAL_FILENAME

    writer = StreamingGlobalWriter(path, resume=True)
    assert writer.source == "final"
    assert sorted(writer.processed) == [1, 2]
    writer.adopt_final()
    third = _unit(rng, 1)[0]
    writer.append(3, third)
    writer.finish()

    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})
    with FeatureCache.open(tmp_path) as cache:
        assert list(cache.global_ids()) == [1, 2, 3]
        for image_id, vector in first.items():
            assert np.array_equal(cache.global_feature(image_id), np.asarray(vector, dtype=np.float16))
        assert np.array_equal(cache.global_feature(3), np.asarray(third, dtype=np.float16))


def test_streaming_writer_skips_complete_final(tmp_path):
    rng = np.random.default_rng(8)
    write_global_cache(tmp_path, {1: _unit(rng, 1)[0], 2: _unit(rng, 1)[0]})
    writer = StreamingGlobalWriter(tmp_path / GLOBAL_FILENAME, resume=True)
    assert writer.covers([1, 2])
    writer.abandon()
    assert (tmp_path / GLOBAL_FILENAME).exists()  # untouched


def test_streaming_writer_handles_more_images_than_the_hdf5_attribute_limit(tmp_path):
    """Regression (live crash, 2026-09-27): ``flush()`` mirrored the per-image
    counts into a file *attribute*; HDF5 object-header attributes cap near
    64 KiB, so the int64 ``counts`` array raised "object header message is too
    large" at 8192 images and killed the real extraction at ~41% of the bank.
    The resumable index is the offsets dataset and must carry any image count.
    """
    import h5py

    rng = np.random.default_rng(9)
    n_images = 9000  # > 8192 int64 counts == the old attribute ceiling
    path = tmp_path / REGION_FILENAME

    # zero-row blocks: the failing array is one entry *per image*, so the bulk
    # appends cost nothing while still driving the counts length past 8192
    empty = np.zeros((0, FEATURE_DIM), dtype=np.float16)
    writer = StreamingRegionWriter(path, resume=False, flush_every=257)
    for image_id in range(1, n_images + 1):
        writer.append(image_id, empty)
    writer.flush()  # index far past the old limit, must not raise
    writer.abandon()  # simulated crash: the tmp survives

    resumed = StreamingRegionWriter(path, resume=True)
    assert resumed.source == "tmp"
    assert len(resumed.processed) == n_images
    resumed.append(n_images + 1, _unit(rng, 1))
    resumed.finish()

    with h5py.File(str(path), "r") as handle:
        assert "counts" not in handle.attrs  # the fix: never an attribute again
        offsets = handle["image_offsets"][...]
        assert offsets.shape == (n_images + 1, 3)
        assert handle["features"].shape == (1, FEATURE_DIM)
        assert int(offsets[:, 2].sum()) == 1


def test_resume_missing_ids_prefix_semantics():
    assert resume_missing_ids([10, 20], [10, 20, 30]) == [30]
    assert resume_missing_ids([], [10]) == [10]
    assert resume_missing_ids([10, 20, 30], [10, 20, 30]) == []
    with pytest.raises(RuntimeError):
        resume_missing_ids([20], [10, 20])  # not a prefix
    with pytest.raises(RuntimeError):
        resume_missing_ids([10, 20, 30, 40], [10, 20, 30])  # cache is longer


# ---------------------------------------------------------------------------
# extraction orchestration (deterministic fake encoder, no torch / no weights)
# ---------------------------------------------------------------------------
class _FakeEncoder:
    """Content-addressed deterministic embeddings - enough to test the
    batching/ordering/invalid-stash logic of the ``run_*_extraction`` helpers."""

    context_length = 77
    feature_dim = FEATURE_DIM

    def __init__(self):
        self.image_batches = []
        self.text_calls = 0

    @staticmethod
    def _vector(seed: int) -> np.ndarray:
        rng = np.random.default_rng(abs(int(seed)) % (2**31 - 1))
        return _unit(rng, 1)[0].astype(np.float16)

    def encode_images(self, images, batch_size=None):
        images = list(images)
        self.image_batches.append(len(images))
        rows = []
        for img in images:
            arr = np.asarray(img.convert("RGB"), dtype=np.int64)
            rows.append(self._vector(int(arr.sum()) % (2**31 - 1)))
        if not rows:
            return np.zeros((0, FEATURE_DIM), dtype=np.float16)
        return np.asarray(rows, dtype=np.float16)

    def encode_texts(self, texts, batch_size=None):
        self.text_calls += 1
        rows = []
        for text in texts:
            seed = sum((i + 1) * ord(ch) for i, ch in enumerate(str(text)))
            rows.append(self._vector(seed % (2**31 - 1)))
        if not rows:
            return np.zeros((0, FEATURE_DIM), dtype=np.float16)
        return np.asarray(rows, dtype=np.float16)

    def token_counts(self, texts):
        return np.asarray([len(str(t).split()) + 2 for t in texts], dtype=np.int64)


def _make_jpeg(path: Path, width: int, height: int, seed: int) -> Path:
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=95)
    return path


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_run_region_extraction_bank_order_and_invalid_stash(tmp_path, batch_size):
    from ccg.features.clip_encoder import crop_image_at_boxes, load_rgb_image

    img_a = _make_jpeg(tmp_path / "a.jpg", 64, 48, 1)
    img_b = _make_jpeg(tmp_path / "b.jpg", 80, 60, 2)
    boxes_a = np.asarray(
        [
            [2.0, 3.0, 33.0, 20.0],          # valid
            [10.0, 10.0, 10.0, 30.0],        # zero width -> invalid
            [0.0, 0.0, 64.0, 48.0],          # full image -> valid
            [5.4, 6.6, 8.2, 9.4],            # rounds to 3x2 px -> valid
        ],
        dtype=np.float32,
    )
    boxes_b = np.asarray([[1.0, 1.0, 40.0, 30.0], [70.0, 40.0, 60.0, 50.0]], dtype=np.float32)
    entries = [(11, boxes_a, img_a), (22, boxes_b, img_b)]

    invalid_seen = {}
    encoder = _FakeEncoder()
    writer = StreamingRegionWriter(tmp_path / REGION_FILENAME, resume=False)
    stats = run_region_extraction(
        encoder,
        entries,
        writer,
        batch_size=batch_size,
        on_invalid=lambda image_id, rows: invalid_seen.__setitem__(image_id, rows),
        log=lambda _msg: None,
    )
    writer.finish()
    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})

    assert stats["n_images_encoded"] == 2
    assert stats["n_crops_total"] == 6
    assert stats["region_crops_encoded"] == 4  # 6 rows minus 2 invalid
    assert stats["invalid_crop_count"] == 2
    assert invalid_seen[11] == [(1, 10, 10, 10, 30)]
    assert invalid_seen[22] == [(1, 70, 40, 60, 50)]

    with FeatureCache.open(tmp_path) as cache:
        assert list(cache.region_ids()) == [11, 22]
        for image_id, boxes, path in entries:
            stored = cache.region_features(image_id)
            assert stored.shape == (boxes.shape[0], FEATURE_DIM)
            assert stored.dtype == np.float16
            crops, valid, _ = crop_image_at_boxes(load_rgb_image(path), boxes)
            expected = np.zeros((boxes.shape[0], FEATURE_DIM), dtype=np.float16)
            if valid.any():
                expected[valid] = encoder.encode_images(crops)
            assert np.array_equal(stored, expected), f"row order/batching broken for image {image_id}"


def test_run_global_and_text_extraction_roundtrip(tmp_path):
    from ccg.features.clip_encoder import load_rgb_image
    from ccg.features.extract_image import run_global_extraction
    from ccg.features.extract_text import run_text_extraction

    encoder = _FakeEncoder()
    img_a = _make_jpeg(tmp_path / "a.jpg", 40, 30, 3)
    img_b = _make_jpeg(tmp_path / "b.jpg", 50, 40, 4)

    global_writer = StreamingGlobalWriter(tmp_path / GLOBAL_FILENAME, resume=False)
    global_stats = run_global_extraction(
        encoder, [(7, img_a), (9, img_b)], global_writer, batch_size=1, log=lambda _m: None
    )
    global_writer.finish()
    assert global_stats["n_images_encoded"] == 2

    corpus = [
        {
            "sentence_id": sid,
            "ref_id": sid // 2,
            "sent_id": sid % 2,
            "image_id": sid,
            "split": "train2014",
            "text": f"text number {sid}",
        }
        for sid in range(4)
    ]
    text_stats = run_text_extraction(encoder, corpus, tmp_path, batch_size=2, log=lambda _m: None)
    assert text_stats["n_sentences"] == 4
    assert text_stats["skipped"] is False
    assert encoder.text_calls == 1

    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})
    with FeatureCache.open(tmp_path) as cache:
        assert cache.n_images == 2  # no region cache -> falls back to global
        for image_id, path in ((7, img_a), (9, img_b)):
            expected = encoder.encode_images([load_rgb_image(path)])[0]
            assert np.array_equal(cache.global_feature(image_id), expected)
        for row in corpus:
            expected = encoder.encode_texts([row["text"]])[0]
            assert np.array_equal(cache.text_features(row["sentence_id"]), expected)
        index = read_text_index(tmp_path)
        assert [int(r["sentence_id"]) for r in index] == [0, 1, 2, 3]

    calls_before_resume = encoder.text_calls
    resumed = run_text_extraction(
        encoder, corpus, tmp_path, batch_size=2, resume=True, log=lambda _m: None
    )
    assert resumed["skipped"] is True
    assert encoder.text_calls == calls_before_resume  # nothing re-encoded on resume


# ---------------------------------------------------------------------------
# dynamic feature_dim (V2-G): a cache carries its backbone's own width (768 etc.)
# ---------------------------------------------------------------------------
D768 = 768


def _synth_dim(root: Path, dim: int, *, seed=0):
    rng = np.random.default_rng(seed)
    region = {1: _unit(rng, 5, dim), 2: _unit(rng, 3, dim)}
    region[2][0] = 0.0
    text = {0: _unit(rng, 1, dim)[0], 1: _unit(rng, 1, dim)[0]}
    global_ = {1: _unit(rng, 1, dim)[0], 2: _unit(rng, 1, dim)[0]}
    metadata = {"backbone": {"model_name": "siglip-base-patch16-224"}}
    write_feature_cache(root, region, text, global_, metadata)
    return region, text, global_


def test_write_and_read_768_feature_cache(tmp_path):
    region, text, global_ = _synth_dim(tmp_path, D768)
    with FeatureCache.open(tmp_path) as cache:
        assert cache.feature_dim == D768
        assert read_metadata(tmp_path)["feature_dim"] == D768
        for image_id, block in region.items():
            got = cache.region_features(image_id)
            assert got.shape == (block.shape[0], D768)
            assert np.array_equal(got, np.asarray(block, dtype=np.float16))
        for sid, vector in text.items():
            assert cache.text_features(sid).shape == (D768,)
        for image_id in global_:
            assert cache.global_feature(image_id).shape == (D768,)


def test_feature_dim_written_to_every_h5_attr(tmp_path):
    import h5py

    _synth_dim(tmp_path, D768)
    for name in (REGION_FILENAME, TEXT_FILENAME, GLOBAL_FILENAME):
        with h5py.File(str(tmp_path / name), "r") as handle:
            assert int(handle.attrs["feature_dim"]) == D768
            assert int(handle["features"].shape[1]) == D768


def test_v1_512_cache_still_reports_512(synth):
    root, _region, _text, _global = synth
    with FeatureCache.open(root) as cache:
        assert cache.feature_dim == FEATURE_DIM


def test_mixed_width_blocks_are_rejected(tmp_path):
    rng = np.random.default_rng(1)
    region = {1: _unit(rng, 2, FEATURE_DIM), 2: _unit(rng, 2, D768)}
    with pytest.raises(ValueError, match="mixed feature widths"):
        write_region_cache(tmp_path, region)


def test_explicit_feature_dim_conflicts_with_arrays(tmp_path):
    rng = np.random.default_rng(2)
    region = {1: _unit(rng, 2, D768)}
    with pytest.raises(ValueError, match="given for feature_dim"):
        write_region_cache(tmp_path, region, feature_dim=FEATURE_DIM)


def test_empty_cache_requires_explicit_feature_dim(tmp_path):
    with pytest.raises(ValueError, match="empty cache"):
        write_feature_cache(tmp_path, {}, {}, {}, {})
    write_feature_cache(tmp_path, {}, {}, {}, {"backbone": {}}, feature_dim=D768)
    assert read_metadata(tmp_path)["feature_dim"] == D768
    with FeatureCache.open(tmp_path) as cache:
        # the empty datasets still carry the explicit width, so the reader sees it
        assert cache.feature_dim == D768


def test_cross_part_width_mismatch_rejected_on_open(tmp_path):
    rng = np.random.default_rng(3)
    write_region_cache(tmp_path, {1: _unit(rng, 2, D768)})
    write_text_cache(tmp_path, {0: _unit(rng, 1, FEATURE_DIM)[0]}, ["s0"])
    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})
    with pytest.raises(ValueError, match="disagrees with the other parts"):
        FeatureCache.open(tmp_path)


def test_streaming_writer_768_roundtrip_and_resume(tmp_path):
    rng = np.random.default_rng(4)
    blocks = {1: _unit(rng, 2, D768), 2: _unit(rng, 3, D768), 3: _unit(rng, 1, D768)}
    path = tmp_path / REGION_FILENAME
    writer = StreamingRegionWriter(path, resume=False, feature_dim=D768)
    assert writer.feature_dim == D768
    writer.append(1, blocks[1])
    writer.flush()
    writer.abandon()

    resumed = StreamingRegionWriter(path, resume=True, feature_dim=D768)
    assert resumed.feature_dim == D768
    resumed.append(2, blocks[2])
    resumed.append(3, blocks[3])
    resumed.finish()
    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})
    with FeatureCache.open(tmp_path) as cache:
        assert cache.feature_dim == D768
        for image_id, block in blocks.items():
            assert np.array_equal(cache.region_features(image_id), np.asarray(block, dtype=np.float16))


def test_streaming_resume_rejects_width_change(tmp_path):
    rng = np.random.default_rng(7)
    path = tmp_path / REGION_FILENAME
    writer = StreamingRegionWriter(path, resume=False, feature_dim=D768)
    writer.append(1, _unit(rng, 2, D768))
    writer.finish()
    with pytest.raises(ValueError, match="stored feature width"):
        StreamingRegionWriter(path, resume=True, feature_dim=FEATURE_DIM)


class _FakeEncoder768(_FakeEncoder):
    """Same content-addressed fake at the SigLIP width (proves the a4 wiring)."""

    feature_dim = D768

    @staticmethod
    def _vector(seed: int) -> np.ndarray:
        rng = np.random.default_rng(abs(int(seed)) % (2**31 - 1))
        return _unit(rng, 1, D768)[0].astype(np.float16)


def test_run_region_extraction_768_through_encoder_dim(tmp_path):
    from ccg.features.clip_encoder import crop_image_at_boxes, load_rgb_image

    img_a = _make_jpeg(tmp_path / "a.jpg", 64, 48, 1)
    boxes_a = np.asarray(
        [[2.0, 3.0, 33.0, 20.0], [10.0, 10.0, 10.0, 30.0], [0.0, 0.0, 64.0, 48.0]],
        dtype=np.float32,
    )
    encoder = _FakeEncoder768()
    writer = StreamingRegionWriter(
        tmp_path / REGION_FILENAME, resume=False, feature_dim=encoder.feature_dim
    )
    run_region_extraction(encoder, [(11, boxes_a, img_a)], writer, batch_size=2, log=lambda _m: None)
    writer.finish()
    update_metadata(tmp_path, {"cache_version": CACHE_VERSION})
    with FeatureCache.open(tmp_path) as cache:
        assert cache.feature_dim == D768
        stored = cache.region_features(11)
        assert stored.shape == (3, D768)
        # the invalid middle box is a zero row; the two valid ones match the encoder
        assert np.array_equal(stored[1], np.zeros(D768, dtype=np.float16))
        crops, valid, _ = crop_image_at_boxes(load_rgb_image(img_a), boxes_a)
        assert np.array_equal(stored[valid], encoder.encode_images(crops))
