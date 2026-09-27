"""RefCOCO+ annotation parsers: json layouts, box handling, split auditing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ccg.data.refcoco import (
    AnnotationFormatError,
    ParseReport,
    dump_refs_json,
    load_refcoco_plus,
    parse_refs_json,
    parse_unc_mats,
    refs_from_records,
)
from ccg.data.splits import SPLIT_TEST_A, SPLIT_TEST_B, SPLIT_TRAIN, SPLIT_VAL
from ccg.data.types import ReferringExample

RECORDS = [
    {
        "ref_id": 21,
        "split": "train",
        "image_id": 391893,
        "bbox": [10.0, 20.0, 30.0, 40.0],  # xywh
        "phrase": "the man in the black shirt",
    },
    {
        "ref_id": 22,
        "split": "val",
        "image_id": 100,
        "bbox": [1.0, 2.0, 4.0, 6.0],
        "phrase": "a red cup",
        "gt_object_id": 7,
    },
]


def as_xyxy(x, y, w, h):
    return [float(x), float(y), float(x + w), float(y + h)]


# ---------------------------------------------------------------------------
# refs_from_records (the shared core, no file needed)
# ---------------------------------------------------------------------------
def test_refs_from_records_primary_layout():
    examples = refs_from_records(RECORDS)
    assert [ex.ref_id for ex in examples] == [21, 22]
    assert [ex.image_id for ex in examples] == [391893, 100]
    assert examples[0].text == "the man in the black shirt"
    # bbox is stored as xywh on disk and normalised to xyxy in memory
    np.testing.assert_allclose(examples[0].gt_box, as_xyxy(10, 20, 30, 40))
    np.testing.assert_allclose(examples[1].gt_box, as_xyxy(1, 2, 4, 6))
    assert examples[0].gt_object_id is None
    assert examples[1].gt_object_id == 7
    assert all(isinstance(ex, ReferringExample) for ex in examples)


def test_split_labels_are_canonical():
    records = [
        {"ref_id": 1, "image_id": 1, "bbox": [0, 0, 2, 2], "phrase": "a", "split": "training"},
        {"ref_id": 2, "image_id": 2, "bbox": [0, 0, 2, 2], "phrase": "b", "split": "validation"},
        {"ref_id": 3, "image_id": 3, "bbox": [0, 0, 2, 2], "phrase": "c", "split": "test-A"},
        {"ref_id": 4, "image_id": 4, "bbox": [0, 0, 2, 2], "phrase": "d", "split": "testB"},
    ]
    examples = refs_from_records(records)
    assert [ex.split for ex in examples] == [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST_A, SPLIT_TEST_B]


def test_unknown_split_is_reported_not_dropped():
    report = ParseReport()
    records = [
        {"ref_id": 1, "image_id": 1, "bbox": [0, 0, 2, 2], "phrase": "a", "split": "testC"},
        {"ref_id": 2, "image_id": 2, "bbox": [0, 0, 2, 2], "phrase": "b", "split": "train"},
    ]
    examples = refs_from_records(records, report=report)
    assert len(examples) == 2, "an unrecognised spelling must never silently drop data"
    assert examples[0].split == "testc"
    assert report.to_dict()["unknown_splits"] == ["testc"]
    assert report.num_examples == 2 and report.num_records_seen == 2
    assert report.missing_object_ids == 2, "neither record carries a GT object id"


def test_explicit_split_overrides_missing_labels():
    records = [{"ref_id": 1, "image_id": 1, "bbox": [0, 0, 2, 2], "phrase": "a"}]
    examples = refs_from_records(records, split="testA")
    assert examples[0].split == SPLIT_TEST_A
    with pytest.raises(AnnotationFormatError, match="missing split"):
        refs_from_records(records)


def test_field_aliases_are_accepted():
    record = {
        "region_id": 5,
        "img_id": 6,
        "sentences": ["a cat", "another paraphrase"],
        "box": [0, 0, 10, 10],
        "split": "val",
        "annotation_id": 3,
    }
    examples = refs_from_records([record])
    assert (examples[0].ref_id, examples[0].image_id) == (5, 6)
    assert examples[0].text == "a cat", "the first paraphrase is used (待核对)"
    assert examples[0].gt_object_id == 3


def test_nested_sentence_mapping():
    record = {
        "ref_id": 1,
        "image_id": 1,
        "bbox": [0, 0, 4, 4],
        "split": "train",
        "sentences": [{"sentence_raw": "  the blue one  "}],
    }
    assert refs_from_records([record])[0].text == "the blue one"


def test_box_format_switch_and_dict_boxes():
    record = {"ref_id": 1, "image_id": 1, "split": "train", "phrase": "a"}
    corner = dict(record, bbox=[0.0, 0.0, 10.0, 20.0])
    np.testing.assert_allclose(
        refs_from_records([corner], box_format="xyxy")[0].gt_box, [0, 0, 10, 20]
    )
    nested = dict(record, bbox={"bbox": [1.0, 1.0, 2.0, 3.0]})
    np.testing.assert_allclose(refs_from_records([nested])[0].gt_box, as_xyxy(1, 1, 2, 3))
    kv = dict(record, bbox={"x": 1.0, "y": 1.0, "w": 2.0, "h": 3.0})
    np.testing.assert_allclose(refs_from_records([kv])[0].gt_box, as_xyxy(1, 1, 2, 3))
    # UNC BOXES rows can carry a trailing flag column
    padded = dict(record, bbox=[1.0, 1.0, 2.0, 3.0, 0.0, 0.0])
    np.testing.assert_allclose(refs_from_records([padded])[0].gt_box, as_xyxy(1, 1, 2, 3))


def test_bad_records_raise_with_actionable_messages():
    base = {"ref_id": 1, "image_id": 1, "split": "train", "phrase": "a"}
    with pytest.raises(AnnotationFormatError, match="no ref id"):
        refs_from_records([{"image_id": 1, "split": "train", "phrase": "a", "bbox": [0, 0, 1, 1]}])
    with pytest.raises(AnnotationFormatError, match="no image id"):
        refs_from_records([{"ref_id": 1, "split": "train", "phrase": "a", "bbox": [0, 0, 1, 1]}])
    with pytest.raises(AnnotationFormatError, match="no box"):
        refs_from_records([{"ref_id": 1, "image_id": 1, "split": "train", "phrase": "a"}])
    with pytest.raises(AnnotationFormatError, match="degenerate"):
        refs_from_records([dict(base, bbox=[0, 0, 0, 5])])
    with pytest.raises(AnnotationFormatError, match="4-value box"):
        refs_from_records([dict(base, bbox=[0, 0, 1])])
    with pytest.raises(AnnotationFormatError, match="empty expression text"):
        refs_from_records([dict(base, phrase="   ")])
    with pytest.raises(AnnotationFormatError, match="not a mapping"):
        refs_from_records(["nope"])
    with pytest.raises(ValueError, match="unknown box_format"):
        refs_from_records([dict(base, bbox=[0, 0, 1, 1])], box_format="cxcywh")


# ---------------------------------------------------------------------------
# json layouts
# ---------------------------------------------------------------------------
def write(tmp_path, payload, name="refs.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_plain_list_and_wrapped_list(tmp_path):
    path = write(tmp_path, RECORDS)
    assert len(parse_refs_json(path)) == 2
    path = write(tmp_path, {"refs": RECORDS}, "refs2.json")
    assert len(parse_refs_json(path)) == 2
    path = write(tmp_path, {"regions": RECORDS}, "refs3.json")
    assert len(parse_refs_json(path)) == 2


def test_parse_images_with_nested_regions(tmp_path):
    payload = {
        "images": [
            {
                "image_id": 500,
                "regions": [
                    {"ref_id": 1, "bbox": [0, 0, 5, 5], "phrase": "first", "split": "val"},
                    {"bbox": [1, 1, 2, 2], "phrase": "second", "split": "val"},
                ],
            }
        ]
    }
    examples = parse_refs_json(write(tmp_path, payload))
    assert [ex.image_id for ex in examples] == [500, 500], "image_id is inherited"
    assert [ex.ref_id for ex in examples] == [1, 2], "a deterministic id fills the gap"
    assert examples[1].text == "second"


def test_parse_refcoco2_triple_store(tmp_path):
    payload = {
        "images": [{"image_id": 700, "split": "val"}],
        "regions": [{"region_id": 11, "image_id": 700, "bbox": [2, 3, 4, 5]}],
        "sentences": [
            {"sentence_id": 1, "region_id": 11, "sentence_raw": "the small one", "split": "val"},
            {"sentence_id": 2, "region_id": 999, "sentence_raw": "dangling"},
        ],
    }
    examples = parse_refs_json(write(tmp_path, payload))
    assert len(examples) == 1, "a sentence with no resolvable region is skipped"
    assert examples[0].ref_id == 11 and examples[0].image_id == 700
    assert examples[0].text == "the small one"
    np.testing.assert_allclose(examples[0].gt_box, as_xyxy(2, 3, 4, 5))


def test_parse_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_refs_json(tmp_path / "absent.json")
    with pytest.raises(AnnotationFormatError, match="cannot find referring records"):
        parse_refs_json(write(tmp_path, {"foo": 1}))
    with pytest.raises(AnnotationFormatError, match="unexpected json root"):
        parse_refs_json(write(tmp_path, 42))


def test_dump_and_parse_roundtrip(tmp_path):
    examples = refs_from_records(RECORDS)
    path = dump_refs_json(examples, tmp_path / "out" / "prepared.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # dump converts back to the on-disk xywh convention, so the values are unchanged
    assert payload["refs"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    restored = parse_refs_json(path)
    assert [ex.ref_id for ex in restored] == [21, 22]
    for a, b in zip(examples, restored):
        np.testing.assert_allclose(a.gt_box, b.gt_box, atol=1e-5)
        assert a.split == b.split and a.text == b.text and a.gt_object_id == b.gt_object_id


def test_load_refcoco_plus_discovers_files_and_reports_missing(tmp_path):
    write(tmp_path, RECORDS, "refcoco+.json")
    assert len(load_refcoco_plus(tmp_path)) == 2
    (tmp_path / "refcoco+.json").unlink()
    write(tmp_path, [dict(r, split="unknown") for r in RECORDS], "refs_testA.json")
    examples = load_refcoco_plus(tmp_path)
    assert [ex.split for ex in examples] == [SPLIT_TEST_A] * 2, "inferred from the file name"
    with pytest.raises(FileNotFoundError, match="Pending data download"):
        load_refcoco_plus(tmp_path / "nothing_here")


# ---------------------------------------------------------------------------
# UNC .mat skeleton
# ---------------------------------------------------------------------------
def test_parse_unc_mats_skeleton(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    refs = [
        {
            "REF_ID": np.array([[21]]),
            "IMAGE_ID": np.array([[391893]]),
            "BOXES": np.array([[10.0, 20.0, 30.0, 40.0, 0.0]]),
            "SENTIENCE_PLACEHOLDER": None,
        }
    ]
    refs[0]["SENTENCE"] = np.array(["the man in the black shirt"])
    del refs[0]["SENTIENCE_PLACEHOLDER"]
    path = tmp_path / "refCOCOp_val.mat"
    scipy_io.savemat(str(path), {"refs": refs})
    examples = parse_unc_mats(path, split="val")
    assert len(examples) == 1
    assert examples[0].ref_id == 21 and examples[0].image_id == 391893
    np.testing.assert_allclose(examples[0].gt_box, as_xyxy(10, 20, 30, 40))
    assert examples[0].text == "the man in the black shirt"

    other = tmp_path / "other.mat"
    scipy_io.savemat(str(other), {"something_else": np.array([1])})
    with pytest.raises(AnnotationFormatError, match="absent"):
        parse_unc_mats(other)
    with pytest.raises(FileNotFoundError):
        parse_unc_mats(tmp_path / "missing.mat")
