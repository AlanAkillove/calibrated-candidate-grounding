"""Tests for the proposal-audit report aggregation and the audit driver.

Two layers, both CPU-only (no GPU, no detector, no real data files):

* :mod:`ccg.data.audit_report` - hand-computed aggregation, CSV/JSON writers,
  figures; the expected values below are computed by hand in the comments;
* ``scripts/run_proposal_audit.py`` - end-to-end with the frozen interfaces
  monkeypatched: a stubbed ``extract_proposals`` returns fixed boxes, a stubbed
  RefCOCO+/COCO loader returns fixed GT, so the full pipeline (cache -> audit ->
  artifacts) runs without torch.

The statistics module :mod:`ccg.data.audit` is frozen (ten-function contract,
2026-09-27); this file pins its documented conventions (exact Wilson endpoints,
``0.0`` rates for empty populations, ``available[K] = distractors >= K - 1``)
through :mod:`ccg.data.audit_report`, which re-exports ``wilson_ci``.

Run:  $env:PYTHONPATH="src"; conda run -n deepminer python -m pytest tests/test_audit_report.py -q
"""

from __future__ import annotations

import csv
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.audit_report import (  # noqa: E402
    CANDIDATE_AVAILABILITY_FIELDS,
    IOU_STATISTICS_FIELDS,
    NATURAL_OMISSION_FIELDS,
    RECALL_FIELDS,
    SAME_CATEGORY_FIELDS,
    aggregate_audit,
    candidate_availability_rows,
    group_refs_by_image,
    iou_statistics_rows,
    make_expression_record,
    make_image_record,
    make_jsonable,
    natural_omission_rows,
    recall_rows,
    same_category_rows,
    wilson_ci,
)
from ccg.data.coco import GtObjects  # noqa: E402
from ccg.data.types import ReferringExample  # noqa: E402


def _load_script_module():
    path = _REPO_ROOT / "scripts" / "run_proposal_audit.py"
    spec = importlib.util.spec_from_file_location("run_proposal_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_proposal_audit"] = module
    spec.loader.exec_module(module)
    return module


run_proposal_audit = _load_script_module()


# ---------------------------------------------------------------------------
# wilson_ci (frozen conventions)
# ---------------------------------------------------------------------------
def test_wilson_ci_frozen_conventions():
    low, high = wilson_ci(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    # k = 0 / k = n: endpoints are exact, never 0.9999999999999999
    low0, high0 = wilson_ci(0, 10)
    assert low0 == 0.0 and 0.2 < high0 < 0.35
    low_n, high_n = wilson_ci(10, 10)
    assert high_n == 1.0 and 0.65 < low_n < 0.8
    # empty denominator: no observations, no information
    assert wilson_ci(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(0, -1)


def test_wilson_ci_is_the_frozen_contract_object():
    from ccg.data.audit import wilson_ci as frozen_wilson_ci

    assert wilson_ci is frozen_wilson_ci  # re-exported, not a private copy


# ---------------------------------------------------------------------------
# record builders
# ---------------------------------------------------------------------------
def _expr(ref_id, max_iou, *, distractors, miss=False, removed=0, same=0, unknown=False):
    """One expression record; availability follows the frozen rule.

    ``distractors`` is ``remaining_candidate_count`` == ``N - |{target} U to_remove|``
    (the frozen contract uses the same pool for
    :func:`ccg.data.audit.target_availability`), so ``available[K]`` is
    ``distractors >= K - 1``.
    """
    available = {k: (distractors >= k - 1) for k in (5, 10, 20, 50)}
    return make_expression_record(
        ref_id=ref_id,
        max_iou=max_iou,
        is_miss=miss,
        num_remaining_candidates=distractors,
        num_equivalent_removed=removed,
        num_same_category_distractors=same,
        available=available,
        target_category_unknown=unknown,
    )


def test_make_expression_record_roundtrip_and_validation():
    record = _expr(1, 0.8, distractors=10, removed=1, same=3)
    assert record["num_remaining_candidates"] == 10
    assert record["num_equivalent_removed"] == 1
    assert record["available"] == {"5": True, "10": True, "20": False, "50": False}

    # natural miss: same-category counts / unknown flags would fabricate a target
    with pytest.raises(ValueError):
        _expr(1, 0.2, miss=True, distractors=64, same=1)
    with pytest.raises(ValueError):
        _expr(1, 0.2, miss=True, distractors=64, unknown=True)
    # a miss removes nothing (assign_target returns an empty to_remove)
    with pytest.raises(ValueError):
        _expr(1, 0.2, miss=True, distractors=64, removed=2)
    # an unknown target category cannot carry same-category distractors
    with pytest.raises(ValueError):
        _expr(1, 0.5, distractors=10, same=2, unknown=True)
    # same-category distractors cannot exceed the remaining candidates
    with pytest.raises(ValueError):
        _expr(1, 0.5, distractors=3, same=4)
    # availability flags contradicting the frozen rule are rejected
    with pytest.raises(ValueError):
        # distractors = 4 >= K - 1 = 4 -> available[5] must be True
        make_expression_record(1, 0.5, False, 4, 0, 0, {"5": False})
    with pytest.raises(ValueError):
        _expr(1, 1.5, distractors=10)
    with pytest.raises(ValueError):
        _expr(1, 0.5, distractors=-1)


def test_make_image_record_validation_and_normalisation():
    record = make_image_record(
        7,
        N_cap=64,
        num_proposals=60,
        gt_max_ious=[0.9, 0.3],
        gt_recall_counts={"0.5": [1, 2], "0.7": [1, 2]},
        redundancy={"n_pairs": 6, "frac_pairs_gt_mid": 0.5, "frac_pairs_gt_high": 0.25},
        expressions=[],
    )
    assert record["num_gt_objects"] == 2
    assert record["gt_recall_counts"] == {"0.5": [1, 2], "0.7": [1, 2]}
    assert record["redundancy"] == {
        "n_pairs": 6,
        "frac_pairs_gt_mid": 0.5,
        "frac_pairs_gt_high": 0.25,
    }
    # recall counts must agree with the raw per-object max-IoU sample ...
    with pytest.raises(ValueError):
        make_image_record(7, 64, 60, [0.9, 0.3], {"0.5": [2, 2]}, {}, [])
    # ... and with its own unit count
    with pytest.raises(ValueError):
        make_image_record(7, 64, 60, [0.9, 0.3], {"0.5": [1, 3]}, {}, [])
    # fractions outside [0, 1] are rejected
    with pytest.raises(ValueError):
        make_image_record(
            7, 64, 60, [0.9], {"0.5": [1, 1]}, {"n_pairs": 3, "frac_pairs_gt_mid": 1.5}, []
        )
    with pytest.raises(ValueError):
        make_image_record(7, N_cap=64, num_proposals=0, gt_max_ious=[],
                          gt_recall_counts={}, redundancy={}, expressions=[])


def test_group_refs_by_image_sorted():
    refs = [
        SimpleNamespace(image_id=2, ref_id=20),
        SimpleNamespace(image_id=1, ref_id=11),
        SimpleNamespace(image_id=2, ref_id=12),
    ]
    grouped = group_refs_by_image(refs)
    assert sorted(grouped) == [1, 2]
    assert [ex.ref_id for ex in grouped[2]] == [12, 20]


# ---------------------------------------------------------------------------
# aggregation (hand-computed expectations)
# ---------------------------------------------------------------------------
def _hand_records():
    """2 images x 2 levels; every aggregate below is hand-computable.

    img 1001: gt max IoUs [0.9, 0.3] -> gt recall @0.5 = 1/2, @0.7 = 1/2;
      expressions: covered target (max IoU 0.8, 10 remaining candidates after
      1 equivalent removal, 3 same-category distractors) + natural miss
      (0.2, 64 remaining, no removal); redundancy 3 pairs, 1 above 0.7.
    img 1002: gt [0.8] -> gt recall 1/1; expression (0.6, 6 remaining,
      1 same-category); redundancy 6 pairs, 2 above 0.7, 1 above 0.9.
    """

    def pair(image_id, gt_ious, counts, redundancy, exprs):
        return [
            make_image_record(
                image_id=image_id,
                N_cap=n_cap,
                num_proposals=64,
                gt_max_ious=gt_ious,
                gt_recall_counts=counts,
                redundancy=redundancy,
                expressions=exprs,
            )
            for n_cap in (64, 128)
        ]

    img1 = pair(
        1001,
        [0.9, 0.3],
        {"0.5": [1, 2], "0.7": [1, 2]},
        {"n_pairs": 3, "frac_pairs_gt_mid": 1 / 3, "frac_pairs_gt_high": 0.0},
        [
            _expr(1, 0.8, distractors=10, removed=1, same=3),
            _expr(2, 0.2, miss=True, distractors=64),
        ],
    )
    img2 = pair(
        1002,
        [0.8],
        {"0.5": [1, 1], "0.7": [1, 1]},
        {"n_pairs": 6, "frac_pairs_gt_mid": 2 / 6, "frac_pairs_gt_high": 1 / 6},
        [_expr(3, 0.6, distractors=6, same=1)],
    )
    return img1 + img2


def test_aggregate_audit_hand_computed_values():
    agg = aggregate_audit(_hand_records())
    assert agg["levels"] == [64, 128]
    level = agg["per_N"]["64"]
    assert level["num_images"] == 2
    assert level["num_expressions"] == 3
    assert level["num_gt_objects"] == 3

    # GT-object recall: ious [0.9, 0.3, 0.8] -> @0.5: 2/3, @0.7: 2/3
    gt = level["recall"]["gt_object"]["by_threshold"]
    assert gt["0.5"]["num_recalled"] == 2 and gt["0.5"]["recall"] == pytest.approx(2 / 3)
    assert gt["0.7"]["num_recalled"] == 2
    assert gt["0.5"]["ci_low"] <= 2 / 3 <= gt["0.5"]["ci_high"]
    # RefCOCO+ target recall: ious [0.8, 0.2, 0.6] -> @0.5: 2/3, @0.7: 1/3
    ref = level["recall"]["ref_target"]["by_threshold"]
    assert ref["0.5"]["num_recalled"] == 2
    assert ref["0.7"]["num_recalled"] == 1 and ref["0.7"]["recall"] == pytest.approx(1 / 3)

    # candidate availability over remaining candidates [10, 64, 6]
    avail = level["candidate_availability"]
    assert avail["5"]["num_available"] == 3 and avail["5"]["frac_available"] == 1.0
    assert avail["10"]["num_available"] == 2  # 10 >= 9, 64 >= 9, 6 < 9
    assert avail["20"]["num_available"] == 1
    assert avail["50"]["num_available"] == 1
    assert avail["50"]["frac_available"] == pytest.approx(1 / 3)

    # same-category counts [3, 0, 1]: >=1 -> 2/3, >=2 -> 1/3, >=4 -> 0
    same = level["same_category"]
    assert same["1"]["num_available"] == 2
    assert same["2"]["num_available"] == 1
    assert same["4"]["num_available"] == 0 and same["4"]["frac"] == 0.0
    assert same["9"]["num_available"] == 0
    assert same["num_target_category_unknown"] == 0

    # natural omission: one of three below 0.5 at both levels
    omission = level["natural_omission"]
    assert omission["expression"]["rate"] == pytest.approx(1 / 3)
    assert omission["gt_object"]["rate"] == pytest.approx(1 / 3)
    assert omission["expression"]["ci_low"] <= omission["expression"]["rate"]
    assert omission["expression"]["ci_high"] >= omission["expression"]["rate"]

    # iou statistics
    stats = level["iou_stats"]
    target = stats["target_max_iou"]
    assert target["median"] == pytest.approx(0.6)
    assert target["p25"] == pytest.approx(0.4)  # np.quantile linear interpolation
    assert target["p75"] == pytest.approx(0.7)
    gt_stats = stats["gt_max_iou"]
    assert gt_stats["p25"] == pytest.approx(0.55)
    assert gt_stats["median"] == pytest.approx(0.8)
    assert gt_stats["p75"] == pytest.approx(0.85)
    assert stats["remaining_count"]["median"] == pytest.approx(10)  # median of [10, 64, 6]
    # pooled redundancy: (1 + 2) / (3 + 6), (0 + 1) / 9
    redundancy = stats["redundancy"]
    assert redundancy["n_pairs"] == 9
    assert redundancy["frac_gt_0.7"] == pytest.approx(3 / 9)
    assert redundancy["frac_gt_0.9"] == pytest.approx(1 / 9)

    # both levels present, identical expression numbers by construction
    assert agg["per_N"]["128"]["recall"]["ref_target"]["by_threshold"]["0.5"]["num_recalled"] == 2


def test_aggregate_audit_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_audit([])


def test_aggregate_audit_handles_expression_free_level():
    # GT-only image (no RefCOCO+ expressions): counts stay 0 and rates report
    # 0.0 with the frozen (0.0, 1.0) convention instead of inventing capacity
    record = make_image_record(
        5,
        N_cap=64,
        num_proposals=64,
        gt_max_ious=[0.9],
        gt_recall_counts={"0.5": [1, 1], "0.7": [1, 1]},
        redundancy={"n_pairs": 1, "frac_pairs_gt_mid": 0.0, "frac_pairs_gt_high": 0.0},
        expressions=[],
    )
    agg = aggregate_audit([record])
    level = agg["per_N"]["64"]
    assert level["num_expressions"] == 0
    avail = level["candidate_availability"]["5"]
    assert avail["num_available"] == 0 and avail["num_expressions"] == 0
    assert avail["frac_available"] == 0.0
    assert avail["ci_low"] == 0.0 and avail["ci_high"] == 1.0
    assert level["same_category"]["1"]["frac"] == 0.0
    omission = level["natural_omission"]
    assert omission["expression"]["num_units"] == 0
    assert omission["expression"]["rate"] == 0.0
    assert omission["expression"]["ci_low"] == 0.0 and omission["expression"]["ci_high"] == 1.0
    assert omission["gt_object"]["num_units"] == 1
    assert omission["gt_object"]["rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CSV artifacts
# ---------------------------------------------------------------------------
def test_csv_row_builders_row_counts_and_values():
    agg = aggregate_audit(_hand_records())
    recall = recall_rows(agg)
    assert len(recall) == 2  # one row per N
    assert set(recall[0]) == set(RECALL_FIELDS)
    row64 = recall[0]
    assert row64["N"] == 64
    assert row64["gt_object_recall@0.5"] == pytest.approx(2 / 3)
    assert row64["ref_target_recall@0.7"] == pytest.approx(1 / 3)

    availability = candidate_availability_rows(agg)
    assert len(availability) == 2 * 4  # 2 levels x (5, 10, 20, 50)
    assert set(availability[0]) == set(CANDIDATE_AVAILABILITY_FIELDS)

    same = same_category_rows(agg)
    assert len(same) == 2 * 4
    assert set(same[0]) == set(SAME_CATEGORY_FIELDS)

    omission = natural_omission_rows(agg)
    assert len(omission) == 2 * 2  # 2 levels x (expression, gt_object)
    assert {row["level"] for row in omission} == {"expression", "gt_object"}
    assert set(omission[0]) == set(NATURAL_OMISSION_FIELDS)

    iou = iou_statistics_rows(agg)
    assert len(iou) == 2
    assert set(iou[0]) == set(IOU_STATISTICS_FIELDS)
    assert iou[0]["target_max_iou_median"] == pytest.approx(0.6)
    assert iou[0]["remaining_count_median"] == pytest.approx(10)


def test_write_csv_artifacts_roundtrip(tmp_path):
    from ccg.data.audit_report import write_csv_artifacts

    agg = aggregate_audit(_hand_records())
    written = write_csv_artifacts(tmp_path, agg)
    assert set(written) == {
        "recall_by_N.csv",
        "candidate_availability_by_N.csv",
        "same_category_availability.csv",
        "natural_omission.csv",
        "proposal_iou_statistics.csv",
    }
    with (tmp_path / "recall_by_N.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == RECALL_FIELDS
        rows = list(reader)
    assert [int(r["N"]) for r in rows] == [64, 128]
    assert float(rows[0]["gt_object_recall@0.5"]) == pytest.approx(2 / 3)
    assert float(rows[0]["ref_target_recall@0.7"]) == pytest.approx(1 / 3)
    with (tmp_path / "candidate_availability_by_N.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    row = next(r for r in rows if r["N"] == "64" and r["K"] == "10")
    assert int(row["num_available"]) == 2
    assert float(row["frac_available"]) == pytest.approx(2 / 3)


def test_write_summary_json_is_strict_json(tmp_path):
    from ccg.data.audit_report import write_summary_json

    payload = {
        "versions": {"numpy": np.__version__},
        "num": np.int64(3),
        "frac": np.float32(0.25),
        "nan_value": float("nan"),
        "nested": {"arr": np.asarray([1, 2], dtype=np.int64)},
    }
    path = write_summary_json(tmp_path / "summary.json", payload)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["num"] == 3 and isinstance(loaded["num"], int)
    assert loaded["frac"] == pytest.approx(0.25)
    assert loaded["nan_value"] is None
    assert loaded["nested"]["arr"] == [1, 2]
    assert make_jsonable({"k": np.bool_(True)}) == {"k": True}


def test_render_figures_writes_five_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    from ccg.data.audit_report import render_figures

    records = _hand_records()
    agg = aggregate_audit(records)
    written = render_figures(records, agg, tmp_path / "figures")
    assert len(written) == 5
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# driver: subset parsing / path resolution
# ---------------------------------------------------------------------------
def test_load_subset_rows_tolerates_extra_columns(tmp_path):
    subset = tmp_path / "subset.csv"
    subset.write_text(
        "image_id,file_name,coco_split,refcoco_split,ref_count\n"
        "909,COCO_train2014_000000000909.jpg,train2014,val_select,2\n"
        "1224,COCO_train2014_000000001224.jpg,train2014,train,\n",
        encoding="utf-8-sig",
    )
    rows = run_proposal_audit.load_subset_rows(subset)
    assert len(rows) == 2
    assert rows[0] == {
        "image_id": 909,
        "file_name": "COCO_train2014_000000000909.jpg",
        "coco_split": "train2014",
        "refcoco_split": "val_select",
        "ref_count": 2,
    }
    assert "ref_count" not in rows[1]
    with pytest.raises(FileNotFoundError):
        run_proposal_audit.load_subset_rows(tmp_path / "nope.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run_proposal_audit.load_subset_rows(bad)


def test_resolve_image_path_layouts(tmp_path):
    flat = tmp_path / "COCO_train2014_000000000909.jpg"
    nested = tmp_path / "train2014" / "COCO_train2014_000000001224.jpg"
    nested.parent.mkdir()
    flat.write_bytes(b"x")
    nested.write_bytes(b"x")
    assert run_proposal_audit.resolve_image_path(tmp_path, flat.name) == flat
    assert run_proposal_audit.resolve_image_path(tmp_path, nested.name) == nested
    assert run_proposal_audit.resolve_image_path(tmp_path, "missing.jpg") is None


# ---------------------------------------------------------------------------
# driver: end-to-end with stubbed frozen interfaces
# ---------------------------------------------------------------------------
_STUB_BOXES = np.asarray(
    [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]],
    dtype=np.float32,
)
_STUB_SCORES = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
_GT_BOXES = np.asarray([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)


class _StubIndex:
    def __init__(self, ids):
        self._objects = {
            int(i): GtObjects(
                image_id=int(i),
                boxes=_GT_BOXES,
                categories=np.asarray([1, 2], dtype=np.int64),
                object_ids=np.asarray([10, 11], dtype=np.int64),
            )
            for i in ids
        }

    def objects(self, image_id):
        return self._objects[int(image_id)]

    def __len__(self):
        return len(self._objects)


def _stub_refs():
    return [
        # found target (IoU 1.0 with the first proposal)
        ReferringExample(ref_id=1, image_id=111, text="the box", gt_box=np.asarray([0, 0, 10, 10]),
                         split="train"),
        # natural miss (no proposal anywhere near)
        ReferringExample(ref_id=2, image_id=222, text="another box",
                         gt_box=np.asarray([200, 200, 210, 210]), split="train"),
    ]


def _write_subset(tmp_path):
    subset = tmp_path / "subset.csv"
    subset.write_text(
        "image_id,file_name,coco_split,refcoco_split,ref_count\n"
        "999,COCO_train2014_000000000999.jpg,train2014,train,1\n"
        "111,COCO_train2014_000000000111.jpg,train2014,train,1\n"
        "222,COCO_train2014_000000000222.jpg,train2014,train,1\n",
        encoding="utf-8",
    )
    images = tmp_path / "images"
    images.mkdir()
    for name in ("COCO_train2014_000000000111.jpg", "COCO_train2014_000000000222.jpg"):
        (images / name).write_bytes(b"not-a-real-jpeg-but-never-decoded")
    return subset, images


def _base_argv(tmp_path, subset, images, cache, out, max_images):
    return [
        "--subset", str(subset),
        "--out", str(out),
        "--cache", str(cache),
        "--images-root", str(images),
        "--coco-annotations", str(tmp_path / "instances.json"),  # loaders are stubbed
        "--refcoco-pickle", str(tmp_path / "refs.p"),
        "--refcoco-instances", str(tmp_path / "refsinst.json"),
        "--max-images", str(max_images),
        "--device", "cpu",
    ]


def test_end_to_end_stubbed_pipeline(tmp_path, monkeypatch):
    subset, images = _write_subset(tmp_path)
    cache = tmp_path / "cache"
    out = tmp_path / "out"

    extract_calls = []

    def fake_extract(image_path, model, top_n, autocast):
        extract_calls.append((str(image_path), int(top_n), bool(autocast)))
        return _STUB_BOXES.copy(), _STUB_SCORES.copy(), {"stub": True, "num_raw_proposals": 4}

    monkeypatch.setattr(run_proposal_audit, "extract_for_image", fake_extract)
    monkeypatch.setattr(run_proposal_audit, "build_rpn_model_for", lambda device: None)
    monkeypatch.setattr(
        run_proposal_audit, "load_gt_index", lambda path, ids: _StubIndex(ids)
    )
    monkeypatch.setattr(
        run_proposal_audit, "load_refcoco_refs", lambda pickle_path, instances_path: _stub_refs()
    )

    # corrupt cache entry: must be detected and recomputed, not crash
    cache.mkdir()
    (cache / "111.npz").write_bytes(b"not-a-npz")

    # run 1: budget 1 -> the missing row 999 is skipped, image 111 is analysed
    rc = run_proposal_audit.main(_base_argv(tmp_path, subset, images, cache, out, max_images=1))
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_images"]["analyzed"] == 1
    assert summary["n_images"]["skipped_missing"] == 1
    assert summary["n_images"]["failed"] == 0
    assert summary["n_images"]["extracted"] == 1
    assert len(extract_calls) == 1
    assert summary["config"]["levels"] == [64, 128]
    assert summary["per_N"]["64"]["num_images"] == 1

    # run 2: budget 2 -> 111 comes from cache, 222 is extracted; 222's ref is a natural miss
    rc = run_proposal_audit.main(_base_argv(tmp_path, subset, images, cache, out, max_images=2))
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_images"]["analyzed"] == 2
    assert summary["n_images"]["cached_reused"] == 1
    assert summary["n_images"]["extracted"] == 1
    assert len(extract_calls) == 2  # cache hit never re-runs the detector
    level = summary["per_N"]["64"]
    assert level["recall"]["ref_target"]["by_threshold"]["0.5"]["num_recalled"] == 1
    assert level["recall"]["ref_target"]["by_threshold"]["0.5"]["num_units"] == 2
    assert level["natural_omission"]["expression"]["num_omitted"] == 1

    # artifacts exist for the five required CSVs
    for name in ("recall_by_N.csv", "candidate_availability_by_N.csv",
                 "same_category_availability.csv", "natural_omission.csv",
                 "proposal_iou_statistics.csv"):
        assert (out / name).exists()

    with (out / "recall_by_N.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["N"]) for r in rows] == [64, 128]
    assert float(rows[0]["ref_target_recall@0.5"]) == pytest.approx(0.5)
    assert float(rows[0]["gt_object_recall@0.5"]) == pytest.approx(0.5)  # 2 of 4 GT objects


def test_load_refcoco_refs_region_level(tmp_path):
    """The real loader path: one example per *region*, ann_id join, xywh -> xyxy."""
    records = [
        {
            "ref_id": 1,
            "image_id": 111,
            "split": "train",
            "ann_id": 10,
            "sentences": [{"raw": "the first box"}, {"raw": "a second sentence"}],
        },
        {  # ann_id absent from instances.json: skipped and counted, never fatal
            "ref_id": 2,
            "image_id": 111,
            "split": "train",
            "ann_id": 999,
            "sentences": [{"raw": "ghost region"}],
        },
    ]
    pickle_path = tmp_path / "refs(unc).p"
    pickle_path.write_bytes(pickle.dumps(records))
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps({
            "annotations": [
                {"id": 10, "image_id": 111, "bbox": [5, 6, 10, 20], "category_id": 3}
            ]
        }),
        encoding="utf-8",
    )

    refs = run_proposal_audit.load_refcoco_refs(pickle_path, instances_path)

    assert len(refs) == 1  # one example per region; the broken join is dropped
    example = refs[0]
    assert (example.ref_id, example.image_id) == (1, 111)
    assert example.text == "the first box"  # region text = its first sentence
    assert example.gt_object_id == 10  # ann_id join (box lives in the instances json)
    np.testing.assert_allclose(example.gt_box, [5.0, 6.0, 15.0, 26.0])  # xywh -> xyxy
    assert any("999" in message for message in run_proposal_audit._REF_LOAD_SKIPS)


def test_main_rejects_bad_arguments():
    assert run_proposal_audit.main(["--topn", "32"]) == 2
    assert run_proposal_audit.main(["--max-images", "-1"]) == 2


def test_main_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run_proposal_audit.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--subset" in out and "--topn" in out and "--max-images" in out
