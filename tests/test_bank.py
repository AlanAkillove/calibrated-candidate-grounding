"""Tests for the frozen ``bank-v1`` HDF5 schema (:mod:`ccg.data.bank`).

Everything here runs on synthetic ``tmp_path`` banks - no GPU, no real data:

* write/read round-trip, dtype/shape enforcement, ``n_raw_post_nms`` attrs;
* resume semantics (an existing group is skipped; ``overwrite`` rewrites);
* ascending traversal (``image_ids`` / ``iter_bank``) regardless of write order;
* corrupt half-written groups: detection, ``--repair`` cleanup, and reads that
  fail loudly instead of silently skipping;
* ``finalize_bank`` root attributes (frozen key set, forced
  ``schema_version`` / ``n_images``, generated ``created_utc``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.data.bank import (  # noqa: E402
    BANK_SCHEMA_VERSION,
    ROOT_ATTR_KEYS,
    bank_attrs,
    bank_group_name,
    bank_has_image,
    delete_bank_entry,
    finalize_bank,
    find_corrupt_images,
    image_ids,
    iter_bank,
    read_bank_image,
    repair_bank,
    write_bank_entry,
)

K3 = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 25.0, 30.0], [1.0, 2.0, 3.0, 4.0]], np.float32)
S3 = np.array([0.9, 0.5, 0.25], np.float32)


def _fresh_bank(tmp_path: Path) -> Path:
    return tmp_path / "bank.h5"


def _write(path: Path, image_id: int, boxes=K3, objectness=S3, extras=None, overwrite=False) -> None:
    with h5py.File(path, "a") as handle:
        write_bank_entry(
            handle, image_id, boxes, objectness, extras, overwrite=overwrite
        )


# ---------------------------------------------------------------------------
# round-trip / resume
# ---------------------------------------------------------------------------
def test_write_read_roundtrip(tmp_path):
    path = _fresh_bank(tmp_path)
    _write(path, 1234, extras={"n_raw_post_nms": 100, "source": "unit-test"})

    boxes, objectness = read_bank_image(path, 1234)
    np.testing.assert_array_equal(boxes, K3)
    np.testing.assert_array_equal(objectness, S3)
    assert boxes.dtype == np.float32 and boxes.shape == (3, 4)
    assert objectness.dtype == np.float32 and objectness.shape == (3,)

    with h5py.File(path, "r") as handle:
        group = handle["image_1234"]
        assert int(group.attrs["n_raw_post_nms"]) == 100
        assert str(group.attrs["source"]) == "unit-test"
        assert set(group.keys()) == {"boxes", "objectness"}

    # float64 in, float32 stored (the frozen dtype)
    _write(path, 7, boxes=K3.astype(np.float64), objectness=S3.astype(np.float64))
    boxes64, objectness64 = read_bank_image(path, 7)
    assert boxes64.dtype == np.float32 and objectness64.dtype == np.float32


def test_n_raw_post_nms_defaults_to_k(tmp_path):
    path = _fresh_bank(tmp_path)
    _write(path, 5)
    with h5py.File(path, "r") as handle:
        assert int(handle["image_5"].attrs["n_raw_post_nms"]) == 3


def test_resume_skips_existing_group(tmp_path):
    path = _fresh_bank(tmp_path)
    _write(path, 11, boxes=K3)
    second = np.zeros((1, 4), np.float32)
    _write(path, 11, boxes=second, objectness=np.array([0.1], np.float32))

    boxes, _ = read_bank_image(path, 11)
    np.testing.assert_array_equal(boxes, K3)  # first write won

    # overwrite=True (non-resume re-extraction) replaces the group
    _write(path, 11, boxes=second, objectness=np.array([0.1], np.float32), overwrite=True)
    boxes, objectness = read_bank_image(path, 11)
    np.testing.assert_array_equal(boxes, second)
    np.testing.assert_array_equal(objectness, np.array([0.1], np.float32))


def test_delete_bank_entry(tmp_path):
    path = _fresh_bank(tmp_path)
    _write(path, 2)
    assert delete_bank_entry(path, 2) is True
    assert delete_bank_entry(path, 2) is False
    assert not bank_has_image(path, 2)


# ---------------------------------------------------------------------------
# traversal order / membership
# ---------------------------------------------------------------------------
def test_iter_order_and_image_ids_sorted(tmp_path):
    path = _fresh_bank(tmp_path)
    for image_id in (500, 7, 42):  # deliberately unsorted
        _write(path, image_id, boxes=K3 + image_id, objectness=S3)

    ids = image_ids(path)
    assert ids.dtype == np.int64
    assert ids.tolist() == [7, 42, 500]

    seen = []
    for image_id, boxes, objectness in iter_bank(path):
        seen.append(image_id)
        np.testing.assert_array_equal(boxes, K3 + image_id)
        np.testing.assert_array_equal(objectness, S3)
    assert seen == [7, 42, 500]

    assert bank_has_image(path, 42) is True
    assert bank_has_image(path, 43) is False


def test_read_missing_image_raises(tmp_path):
    path = _fresh_bank(tmp_path)
    _write(path, 1)
    with pytest.raises(KeyError):
        read_bank_image(path, 999)


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------
def test_write_validation_raises(tmp_path):
    path = _fresh_bank(tmp_path)
    good_boxes = np.zeros((2, 4), np.float32)
    good_scores = np.zeros(2, np.float32)

    with h5py.File(path, "a") as handle:
        with pytest.raises(ValueError):  # wrong last dim
            write_bank_entry(handle, 1, np.zeros((2, 3), np.float32), good_scores)
        with pytest.raises(ValueError):  # non-numeric dtype (strings)
            write_bank_entry(handle, 1, np.array([["a", "b", "c", "d"]]), np.zeros(1, np.float32))
        with pytest.raises(ValueError):  # object dtype
            write_bank_entry(handle, 1, np.array([[None] * 4], dtype=object), np.zeros(1, np.float32))
        with pytest.raises(ValueError):  # misaligned K
            write_bank_entry(handle, 1, good_boxes, np.zeros(3, np.float32))
        with pytest.raises(ValueError):  # objectness not 1-D
            write_bank_entry(handle, 1, good_boxes, np.zeros((2, 1), np.float32))
        with pytest.raises(ValueError):  # empty bank
            write_bank_entry(handle, 1, np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        with pytest.raises(ValueError):  # non-finite boxes
            write_bank_entry(
                handle, 1, np.array([[np.nan, 0.0, 1.0, 1.0]], np.float32), np.zeros(1, np.float32)
            )
        with pytest.raises(ValueError):  # n_raw_post_nms < K
            write_bank_entry(handle, 1, good_boxes, good_scores, {"n_raw_post_nms": 1})
        with pytest.raises(ValueError):  # unstorable attr payload
            write_bank_entry(handle, 1, good_boxes, good_scores, {"bad": object()})

    # every rejected write left no group behind
    assert image_ids(path).tolist() == []


def test_non_integer_image_id_raises(tmp_path):
    path = _fresh_bank(tmp_path)
    with h5py.File(path, "a") as handle:
        with pytest.raises(ValueError):
            write_bank_entry(handle, "not-an-id", K3, S3)
    assert bank_group_name(123) == "image_123"


# ---------------------------------------------------------------------------
# corruption detection / repair
# ---------------------------------------------------------------------------
def _make_corrupt_bank(path: Path) -> None:
    """One valid image (1) plus five different half-written shapes."""
    _write(path, 1)
    with h5py.File(path, "a") as handle:
        only_boxes = handle.create_group("image_999")  # missing objectness
        only_boxes.create_dataset("boxes", data=np.zeros((2, 4), np.float32))
        only_scores = handle.create_group("image_998")  # missing boxes
        only_scores.create_dataset("objectness", data=np.zeros(2, np.float32))
        handle.create_group("image_997")  # empty group
        wrong = handle.create_group("image_996")  # wrong boxes shape
        wrong.create_dataset("boxes", data=np.zeros((3, 3), np.float32))
        wrong.create_dataset("objectness", data=np.zeros(3, np.float32))
        zero = handle.create_group("image_995")  # zero-length bank
        zero.create_dataset("boxes", data=np.zeros((0, 4), np.float32))
        zero.create_dataset("objectness", data=np.zeros(0, np.float32))


def test_corrupt_detection_and_repair(tmp_path):
    path = _fresh_bank(tmp_path)
    _make_corrupt_bank(path)

    assert find_corrupt_images(path) == [995, 996, 997, 998, 999]

    # reads hit the corrupt group loudly (never a silent skip)
    with pytest.raises(KeyError):
        read_bank_image(path, 999)
    with pytest.raises(KeyError):
        list(iter_bank(path))

    # dry-run reports without deleting
    assert repair_bank(path, dry_run=True) == [995, 996, 997, 998, 999]
    assert find_corrupt_images(path) == [995, 996, 997, 998, 999]

    # the real repair removes exactly the corrupt groups
    assert repair_bank(path) == [995, 996, 997, 998, 999]
    assert find_corrupt_images(path) == []
    assert image_ids(path).tolist() == [1]
    boxes, _ = read_bank_image(path, 1)
    np.testing.assert_array_equal(boxes, K3)

    # a repaired slot can be rewritten (the --repair -> re-extract path)
    _write(path, 999)
    assert find_corrupt_images(path) == []
    assert image_ids(path).tolist() == [1, 999]


# ---------------------------------------------------------------------------
# finalisation
# ---------------------------------------------------------------------------
FINAL_ATTRS = {
    "model_name": "fasterrcnn_resnet50_fpn",
    "weights": "COCO_V1",
    "proposal_type": "class-agnostic post-NMS RPN",
    "top_n": 64,
    "torchvision_version": "0.20.1+cu121",
    "images_root": "data/raw/mscoco",
}


def test_finalize_bank_attrs_complete(tmp_path):
    path = _fresh_bank(tmp_path)
    for image_id in (3, 8, 20):
        _write(path, image_id)
    finalize_bank(path, dict(FINAL_ATTRS))

    attrs = bank_attrs(path)
    for key in ROOT_ATTR_KEYS:
        assert key in attrs, f"root attr {key!r} missing after finalize_bank"
    assert attrs["schema_version"] == BANK_SCHEMA_VERSION == "bank-v1"
    assert attrs["model_name"] == "fasterrcnn_resnet50_fpn"
    assert attrs["weights"] == "COCO_V1"
    assert attrs["proposal_type"] == "class-agnostic post-NMS RPN"
    assert attrs["top_n"] == 64
    assert attrs["torchvision_version"] == "0.20.1+cu121"
    assert attrs["images_root"] == "data/raw/mscoco"
    assert attrs["n_images"] == 3
    assert isinstance(attrs["created_utc"], str) and attrs["created_utc"].endswith("Z")


def test_finalize_forces_schema_and_true_image_count(tmp_path):
    path = _fresh_bank(tmp_path)
    _write(path, 1)
    # a caller lying about version/count cannot poison the bank
    finalize_bank(path, dict(FINAL_ATTRS, schema_version="banana-v9", n_images=999))
    attrs = bank_attrs(path)
    assert attrs["schema_version"] == BANK_SCHEMA_VERSION
    assert attrs["n_images"] == 1


def test_bank_attrs_empty_before_finalize(tmp_path):
    path = _fresh_bank(tmp_path)
    _write(path, 1)
    assert bank_attrs(path) == {}
