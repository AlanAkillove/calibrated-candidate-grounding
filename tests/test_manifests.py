"""Frozen candidate manifests: nesting, determinism, eligibility, same-category.

Everything runs on synthetic banks (hand-made boxes written with the exact
frozen h5 layout) - no real data, no real proposal bank.  The properties
pinned here are the frozen construction rules of :mod:`ccg.data.manifests`:

* one ordering per ref, prefixes are the candidate sets (C5 subset C10 subset
  C20 subset C50),
* determinism per ``(seed, ref_id)`` and independence of iteration order,
* eligibility = target present AND ``n_valid >= K - 1``,
* the same-category promotions / fractions and the target-equivalent removal.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")  # noqa: E402  (bank files are HDF5)

from ccg.data.manifests import (  # noqa: E402
    FILE_SPLITS,
    MANIFEST_SEED,
    PRIMARY_KS,
    ManifestEntry,
    ManifestFile,
    build_manifests,
    filter_entries,
    manifest_path,
)
from ccg.data.splits import SPLIT_VAL_CALIB, SPLIT_VAL_SELECT  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic data helpers (frozen h5 layout: image_{id}/boxes/objectness)
# ---------------------------------------------------------------------------
def tiled_boxes(n: int, size: float = 10.0, gap: float = 10.0, per_row: int = 8) -> np.ndarray:
    """``n`` pairwise-disjoint boxes on a grid (any two have IoU == 0)."""
    step = size + gap
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        x = (i % per_row) * step
        y = (i // per_row) * step
        boxes[i] = [x, y, x + size, y + size]
    return boxes


def shrunken(box, factor: float) -> np.ndarray:
    """Concentric shrink: IoU(box, result) == factor**2 (0.9 -> 0.81)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = (x2 - x1) * factor
    height = (y2 - y1) * factor
    return np.asarray([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
                      dtype=np.float32)


def as_xywh(box) -> list:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]


def descending_objectness(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.1, n, dtype=np.float32)


def write_bank(path, images) -> None:
    """Write ``{image_id: boxes}`` with the frozen per-image group layout."""
    with h5py.File(path, "w") as handle:
        for image_id, boxes in images.items():
            boxes = np.asarray(boxes, dtype=np.float32)
            group = handle.create_group(f"image_{int(image_id)}")
            group.create_dataset("boxes", data=boxes)
            group.create_dataset("objectness", data=descending_objectness(boxes.shape[0]))
            group.attrs["n_raw_post_nms"] = int(boxes.shape[0])


def make_record(ref_id: int, image_id: int, ann_id: int, split: str = "train") -> dict:
    return {
        "ref_id": int(ref_id),
        "image_id": int(image_id),
        "split": split,
        "ann_id": int(ann_id),
        "sentences": [f"expression {ref_id} on {image_id}"],
    }


def make_join(entries) -> dict:
    """``[(ann_id, image_id, target_box_xyxy)]`` -> instances join table."""
    return {
        str(ann_id): {
            "image_id": int(image_id),
            "bbox": as_xywh(box),
            "category_id": 1,
        }
        for ann_id, image_id, box in entries
    }


def single_ref_manifest(tmp_path, n, target_box, *, split="train", regime="random",
                        seed=MANIFEST_SEED, top_n=64, coco=None, image_id=11, ref_id=1):
    bank_path = tmp_path / "proposals.h5"
    write_bank(bank_path, {image_id: tiled_boxes(n)})
    records = [make_record(ref_id, image_id, ann_id=900 + ref_id, split=split)]
    join = make_join([(900 + ref_id, image_id, target_box)])
    return build_manifests(bank_path, records, join, regime, seed=seed, top_n=top_n, coco=coco)


def assert_entries_equal(left: ManifestEntry, right: ManifestEntry) -> None:
    """Field-by-field equality including the ordering array (bit-exact)."""
    assert left.ref_id == right.ref_id
    assert left.image_id == right.image_id
    assert left.split == right.split
    assert left.target_index == right.target_index
    assert left.distractor_order.dtype == np.int32
    np.testing.assert_array_equal(left.distractor_order, right.distractor_order)
    assert left.n_valid_distractors == right.n_valid_distractors
    assert left.n_target_equiv_removed == right.n_target_equiv_removed
    assert left.n_same_category_available == right.n_same_category_available
    assert left.eligible == right.eligible
    assert left.n_same_used_by_K == right.n_same_used_by_K
    assert left.hard_fraction_by_K == right.hard_fraction_by_K
    assert left.target_max_iou == right.target_max_iou
    assert left.eval_split == right.eval_split


def candidate_set(entry: ManifestEntry, K: int) -> np.ndarray:
    """``C_K = target + order[:K-1]`` exactly as a consumer would slice it."""
    return np.concatenate(([entry.target_index], entry.distractor_order[: K - 1]))


# ---------------------------------------------------------------------------
# nesting: one ordering, prefixes are the candidate sets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [16, 64])
def test_candidate_sets_are_prefix_nested(tmp_path, n):
    boxes = tiled_boxes(n)
    manifest = single_ref_manifest(tmp_path, n, shrunken(boxes[7], 0.9))
    assert len(manifest.entries) == 1
    entry = manifest.entries[0]

    assert entry.target_index == 7
    assert entry.target_max_iou == pytest.approx(0.81, abs=1e-4)
    assert entry.distractor_order.dtype == np.int32
    assert entry.distractor_order.size == min(n - 1, 63)
    assert set(entry.distractor_order.tolist()) == set(range(n)) - {7}

    # ordered prefix nesting: C_K is a prefix of every larger C_K'
    for smaller, larger in zip(PRIMARY_KS[:-1], PRIMARY_KS[1:]):
        small, large = candidate_set(entry, smaller), candidate_set(entry, larger)
        assert small.size <= large.size
        np.testing.assert_array_equal(small, large[: small.size])

    # set nesting (the protocol requirement) and the constant target
    sets = {K: set(candidate_set(entry, K).tolist()) for K in PRIMARY_KS}
    assert sets[5] <= sets[10] <= sets[20] <= sets[50]
    assert entry.target_index in sets[5]
    for K in PRIMARY_KS:
        assert candidate_set(entry, K)[0] == entry.target_index


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def _multi_ref_fixture(tmp_path, image_id=11):
    boxes = tiled_boxes(64)
    write_bank(tmp_path / "proposals.h5", {image_id: boxes})
    records = []
    join_entries = []
    for ref_id in range(1, 6):
        ann_id = 900 + ref_id
        records.append(make_record(ref_id, image_id, ann_id))
        join_entries.append((ann_id, image_id, shrunken(boxes[ref_id - 1], 0.9)))
    return records, make_join(join_entries)


def test_same_seed_regenerates_identical_entries(tmp_path):
    records, join = _multi_ref_fixture(tmp_path)
    bank = tmp_path / "proposals.h5"

    first = build_manifests(bank, records, join, "random", seed=MANIFEST_SEED)
    second = build_manifests(bank, records, join, "random", seed=MANIFEST_SEED)
    assert len(first.entries) == len(second.entries) == 5
    for left, right in zip(first.entries, second.entries):
        assert_entries_equal(left, right)


def test_other_seed_changes_the_orderings(tmp_path):
    records, join = _multi_ref_fixture(tmp_path)
    bank = tmp_path / "proposals.h5"

    base = build_manifests(bank, records, join, "random", seed=MANIFEST_SEED)
    other = build_manifests(bank, records, join, "random", seed=MANIFEST_SEED + 1)
    changed = [
        not np.array_equal(a.distractor_order, b.distractor_order)
        for a, b in zip(base.entries, other.entries)
    ]
    assert any(changed), "a different seed must change at least one ordering"


def test_iteration_order_does_not_affect_per_ref_ordering(tmp_path):
    records, join = _multi_ref_fixture(tmp_path)
    bank = tmp_path / "proposals.h5"

    forward = build_manifests(bank, records, join, "random", seed=MANIFEST_SEED)
    reversed_records = list(reversed(records))
    backward = build_manifests(bank, reversed_records, join, "random", seed=MANIFEST_SEED)

    # output follows the input order ...
    assert [entry.ref_id for entry in backward.entries] == [5, 4, 3, 2, 1]
    assert [entry.ref_id for entry in forward.entries] == [1, 2, 3, 4, 5]
    # ... but every ref's ordering is identical (rng = f(seed, ref_id) only)
    by_ref = {entry.ref_id: entry for entry in forward.entries}
    for entry in backward.entries:
        assert_entries_equal(entry, by_ref[entry.ref_id])


# ---------------------------------------------------------------------------
# eligibility vs pool size / target presence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n, expected",
    [
        (5, {5: True, 10: False, 20: False, 50: False}),
        (8, {5: True, 10: False, 20: False, 50: False}),
        (10, {5: True, 10: True, 20: False, 50: False}),
        (49, {5: True, 10: True, 20: True, 50: False}),
        (64, {5: True, 10: True, 20: True, 50: True}),
    ],
)
def test_eligibility_tracks_pool_size(tmp_path, n, expected):
    boxes = tiled_boxes(n)
    manifest = single_ref_manifest(tmp_path, n, shrunken(boxes[0], 0.9))
    entry = manifest.entries[0]

    assert entry.target_index == 0
    assert entry.n_target_equiv_removed == 0
    assert entry.n_valid_distractors == n - 1
    assert entry.distractor_order.size == min(n - 1, 63)
    assert entry.eligible == expected
    if entry.n_valid_distractors >= 4:
        assert entry.eligible[5] is True


@pytest.mark.parametrize("n", [8, 64])
def test_target_miss_keeps_the_full_ordering(tmp_path, n):
    far = np.asarray([10_000.0, 10_000.0, 10_020.0, 10_020.0], dtype=np.float32)
    manifest = single_ref_manifest(tmp_path, n, far)
    entry = manifest.entries[0]

    assert entry.target_index is None
    assert entry.target_max_iou < 0.5
    assert entry.eligible == {5: False, 10: False, 20: False, 50: False}
    assert entry.n_target_equiv_removed == 0
    assert entry.n_valid_distractors == n
    assert entry.distractor_order.size == min(n, 63)
    order = entry.distractor_order.tolist()
    assert len(set(order)) == len(order)  # still a full permutation of the pool
    assert set(order) <= set(range(n))
    if n <= 63:
        assert set(order) == set(range(n))


# ---------------------------------------------------------------------------
# same-category regime
# ---------------------------------------------------------------------------
def _same_category_fixture(tmp_path):
    """Target on boxes[2] with a duplicate on boxes[7]; GT cats 3/3/3/7 on 2/3/4/5."""
    image_id = 21
    boxes = tiled_boxes(12)
    boxes[7] = shrunken(boxes[2], 0.8)  # target-equivalent (IoU 0.79 >= 0.5)
    write_bank(tmp_path / "proposals.h5", {image_id: boxes})

    target_box = shrunken(boxes[2], 0.9)  # IoU 0.81 with boxes[2], 0.79 with boxes[7]
    record = make_record(1, image_id, ann_id=601)
    join = make_join([(601, image_id, target_box)])
    coco = {
        "images": [
            {
                "id": image_id,
                "file_name": "COCO_train2014_000000000021.jpg",
                "height": 400,
                "width": 400,
            }
        ],
        "annotations": [
            {"id": 501, "image_id": image_id, "bbox": as_xywh(boxes[2]), "category_id": 3},
            {"id": 502, "image_id": image_id, "bbox": as_xywh(boxes[3]), "category_id": 3},
            {"id": 503, "image_id": image_id, "bbox": as_xywh(boxes[4]), "category_id": 3},
            {"id": 504, "image_id": image_id, "bbox": as_xywh(boxes[5]), "category_id": 7},
        ],
        "categories": [{"id": 3, "name": "cat_a"}, {"id": 7, "name": "cat_b"}],
    }
    return record, join, coco


def test_same_category_promotes_hard_distractors(tmp_path):
    record, join, coco = _same_category_fixture(tmp_path)
    bank = tmp_path / "proposals.h5"
    manifest = build_manifests(bank, [record], join, "same_category", coco=coco)
    entry = manifest.entries[0]

    # unique target on row 2, the equivalent on row 7 removed
    assert entry.target_index == 2
    assert entry.target_max_iou == pytest.approx(0.81, abs=1e-4)
    assert entry.n_target_equiv_removed == 1
    assert entry.n_valid_distractors == 10
    assert 7 not in entry.distractor_order.tolist()

    # same-category supply: boxes[3], boxes[4] (cat 3); boxes[5] is cat 7
    assert entry.n_same_category_available == 2
    assert entry.distractor_order[:2].tolist() == [3, 4]  # ascending bank index
    assert set(entry.distractor_order[2:].tolist()) == {0, 1, 5, 6, 8, 9, 10, 11}

    assert entry.n_same_used_by_K == {5: 2, 10: 2, 20: 2, 50: 2}
    assert entry.hard_fraction_by_K == pytest.approx(
        {5: 2 / 4, 10: 2 / 9, 20: 2 / 19, 50: 2 / 49}
    )
    assert entry.eligible == {5: True, 10: True, 20: False, 50: False}


def test_random_regime_records_zero_hard_metadata(tmp_path):
    record, join, coco = _same_category_fixture(tmp_path)
    bank = tmp_path / "proposals.h5"
    manifest = build_manifests(bank, [record], join, "random", coco=coco)
    entry = manifest.entries[0]

    # the pool / removal bookkeeping is identical ...
    assert entry.target_index == 2
    assert entry.n_target_equiv_removed == 1
    assert entry.n_valid_distractors == 10
    assert entry.n_same_category_available == 2  # metadata is available ...
    # ... but the random ordering never promotes same-category distractors
    assert entry.n_same_used_by_K == {5: 0, 10: 0, 20: 0, 50: 0}
    assert entry.hard_fraction_by_K == {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}


def test_unknown_target_category_yields_zero_hard_supply(tmp_path):
    image_id = 22
    boxes = tiled_boxes(8)
    write_bank(tmp_path / "proposals.h5", {image_id: boxes})
    target_box = shrunken(boxes[0], 0.9)  # no GT object overlaps boxes[0]
    record = make_record(1, image_id, ann_id=701)
    join = make_join([(701, image_id, target_box)])
    coco = {
        "images": [
            {
                "id": image_id,
                "file_name": "COCO_train2014_000000000022.jpg",
                "height": 400,
                "width": 400,
            }
        ],
        "annotations": [
            {"id": 801, "image_id": image_id, "bbox": as_xywh(boxes[3]), "category_id": 3},
            {"id": 802, "image_id": image_id, "bbox": as_xywh(boxes[4]), "category_id": 3},
        ],
        "categories": [{"id": 3, "name": "cat_a"}],
    }
    manifest = build_manifests(
        tmp_path / "proposals.h5", [record], join, "same_category", coco=coco
    )
    entry = manifest.entries[0]

    assert entry.target_index == 0
    assert entry.n_same_category_available == 0  # unknown (-1) never matches
    assert entry.n_same_used_by_K == {5: 0, 10: 0, 20: 0, 50: 0}
    assert entry.hard_fraction_by_K == {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
    assert entry.distractor_order.size == 7
    assert entry.eligible[5] is True


def test_same_category_regime_requires_coco(tmp_path):
    boxes = tiled_boxes(8)
    bank = tmp_path / "proposals.h5"
    write_bank(bank, {11: boxes})
    records = [make_record(1, 11, 901)]
    join = make_join([(901, 11, shrunken(boxes[0], 0.9))])
    with pytest.raises(ValueError, match="coco"):
        build_manifests(bank, records, join, "same_category")


# ---------------------------------------------------------------------------
# top_n truncation
# ---------------------------------------------------------------------------
def test_top_n_truncates_bank_and_caps_the_order(tmp_path):
    boxes = tiled_boxes(16)
    manifest = single_ref_manifest(tmp_path, 16, shrunken(boxes[2], 0.9), top_n=8)
    entry = manifest.entries[0]

    assert manifest.meta["top_n"] == 8
    assert entry.target_index == 2
    assert entry.n_valid_distractors == 7
    assert entry.distractor_order.size == 7
    assert set(entry.distractor_order.tolist()) == {0, 1, 3, 4, 5, 6, 7}
    assert entry.eligible == {5: True, 10: False, 20: False, 50: False}


# ---------------------------------------------------------------------------
# persistence: jsonl + meta, the val sub-split column, filter_entries
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_val_subsplits_and_filter(tmp_path):
    boxes = tiled_boxes(64)
    image_train, image_val_a, image_val_b = 11, 12, 13
    write_bank(
        tmp_path / "proposals.h5",
        {image_train: boxes, image_val_a: boxes, image_val_b: boxes},
    )
    records = [
        make_record(1, image_train, 901, split="train"),
        make_record(2, image_val_a, 902, split="val"),
        make_record(3, image_val_a, 903, split="val"),
        make_record(4, image_val_b, 904, split="val"),
    ]
    join = make_join(
        [
            (901, image_train, shrunken(boxes[0], 0.9)),
            (902, image_val_a, shrunken(boxes[1], 0.9)),
            (903, image_val_a, shrunken(boxes[2], 0.9)),
            (904, image_val_b, shrunken(boxes[3], 0.9)),
        ]
    )
    manifest = build_manifests(tmp_path / "proposals.h5", records, join, "random")
    assert len(manifest.entries) == 4

    # write exactly like the CLI: one file per UNC split
    for split in FILE_SPLITS:
        subset = manifest.subset(filter_entries(manifest, split))
        subset.save(manifest_path(tmp_path, "random", split))

    val_path = manifest_path(tmp_path, "random", "val")
    assert val_path == tmp_path / "manifests" / "random_val.jsonl"
    assert val_path.with_suffix(".meta.json").exists()
    assert manifest_path(tmp_path, "random", "train").exists()

    loaded = ManifestFile.load(val_path)
    expected_val = filter_entries(manifest, "val")
    assert [entry.ref_id for entry in loaded.entries] == [2, 3, 4]
    for left, right in zip(loaded.entries, expected_val):
        assert_entries_equal(left, right)

    # meta provenance block
    meta = loaded.meta
    assert meta["schema_version"] == "manifests-v1"
    assert meta["regime"] == "random"
    assert meta["seed"] == MANIFEST_SEED
    assert meta["top_n"] == 64
    assert meta["n_refs"] == 3
    assert meta["n_images"] == 2
    assert meta["split_counts"] == {"val": 3}
    assert meta["val_split_counts"] == {"val_select": 1, "val_calib": 1}
    fingerprint = meta["bank_fingerprint"]
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    int(fingerprint, 16)  # hex

    # the val cut is by image: both refs of image 12 share one sub-split
    by_image = {}
    for entry in loaded.entries:
        assert entry.eval_split in (SPLIT_VAL_SELECT, SPLIT_VAL_CALIB)
        by_image.setdefault(entry.image_id, set()).add(entry.eval_split)
    assert all(len(splits) == 1 for splits in by_image.values())
    image_splits = {next(iter(splits)) for splits in by_image.values()}
    assert image_splits == {SPLIT_VAL_SELECT, SPLIT_VAL_CALIB}

    # filter_entries understands both the UNC splits and the val sub-splits
    assert len(filter_entries(loaded, "train")) == 0
    assert len(filter_entries(loaded, "val")) == 3
    n_select = len(filter_entries(loaded, "val_select"))
    n_calib = len(filter_entries(loaded, "val_calib"))
    assert n_select + n_calib == 3
    assert {entry.ref_id for entry in filter_entries(loaded, "val_select")} <= {2, 3, 4}
    with pytest.raises(ValueError, match="unknown split"):
        filter_entries(loaded, "banana")

    # jsonl format: one JSON object per line, arrays as lists, str-keyed dicts
    lines = [line for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert isinstance(first["distractor_order"], list)
    assert set(first["eligible"]) == {"5", "10", "20", "50"}


def test_train_entries_have_no_eval_split(tmp_path):
    boxes = tiled_boxes(8)
    manifest = single_ref_manifest(tmp_path, 8, shrunken(boxes[0], 0.9), split="train")
    assert manifest.entries[0].eval_split is None
    payload = manifest.entries[0].to_dict()
    assert "eval_split" not in payload


def test_meta_json_and_jsonl_are_reloadable_after_subset(tmp_path):
    boxes = tiled_boxes(8)
    manifest = single_ref_manifest(tmp_path, 8, shrunken(boxes[0], 0.9))
    subset = manifest.subset(manifest.entries)
    out = tmp_path / "nested" / "random_train.jsonl"
    subset.save(out)
    reloaded = ManifestFile.load(out)
    assert reloaded.meta["n_refs"] == 1
    assert reloaded.meta["n_images"] == 1
    assert reloaded.meta["split_counts"] == {"train": 1}
    assert_entries_equal(reloaded.entries[0], manifest.entries[0])


# ---------------------------------------------------------------------------
# entry validation
# ---------------------------------------------------------------------------
def _entry_kwargs(**overrides):
    payload = dict(
        ref_id=1,
        image_id=1,
        split="train",
        target_index=3,
        distractor_order=np.asarray([0, 1, 2], dtype=np.int32),
        n_valid_distractors=3,
        n_target_equiv_removed=0,
        n_same_category_available=0,
        eligible={5: False, 10: False, 20: False, 50: False},
        n_same_used_by_K={5: 0, 10: 0, 20: 0, 50: 0},
        hard_fraction_by_K={5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
        target_max_iou=0.0,
    )
    payload.update(overrides)
    return payload


def test_manifest_entry_validates_the_ordering():
    entry = ManifestEntry(**_entry_kwargs())
    assert entry.distractor_order.dtype == np.int32

    with pytest.raises(ValueError, match="duplicate"):
        ManifestEntry(**_entry_kwargs(distractor_order=np.asarray([1, 1, 2])))
    with pytest.raises(ValueError, match="target"):
        ManifestEntry(**_entry_kwargs(distractor_order=np.asarray([2, 3, 4])))
    with pytest.raises(ValueError, match="non-negative"):
        ManifestEntry(**_entry_kwargs(distractor_order=np.asarray([-1, 0, 1])))
    with pytest.raises(ValueError, match="stored ordering"):
        ManifestEntry(**_entry_kwargs(n_valid_distractors=1))


def test_manifest_file_rejects_unknown_regime_and_bad_entry_types(tmp_path):
    with pytest.raises(ValueError):
        ManifestFile(regime="", meta={}, entries=[])
    with pytest.raises(TypeError):
        ManifestFile(regime="random", meta={}, entries=["nope"])


def test_build_manifests_rejects_unknown_regime_and_missing_bank(tmp_path):
    boxes = tiled_boxes(8)
    bank = tmp_path / "proposals.h5"
    write_bank(bank, {11: boxes})
    records = [make_record(1, 11, 901)]
    join = make_join([(901, 11, shrunken(boxes[0], 0.9))])

    with pytest.raises(ValueError, match="regime"):
        build_manifests(bank, records, join, "clip_hard")
    with pytest.raises(FileNotFoundError, match="proposal bank"):
        build_manifests(tmp_path / "missing.h5", records, join, "random")


def test_build_manifests_rejects_bad_join(tmp_path):
    boxes = tiled_boxes(8)
    bank = tmp_path / "proposals.h5"
    write_bank(bank, {11: boxes})
    records = [make_record(1, 11, 901)]
    join = make_join([(901, 11, shrunken(boxes[0], 0.9))])

    # ann_id missing from the join table
    with pytest.raises(ValueError, match="join table"):
        build_manifests(bank, [make_record(1, 11, 999)], join, "random")
    # ann_id belongs to a different image
    wrong = make_join([(901, 12, shrunken(boxes[0], 0.9))])
    with pytest.raises(ValueError, match="belongs to image"):
        build_manifests(bank, records, wrong, "random")
