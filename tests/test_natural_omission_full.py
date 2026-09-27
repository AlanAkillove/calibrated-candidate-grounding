"""Tests for the full-data natural-omission counter (scripts/natural_omission_full.py).

A synthetic world in ``tmp_path``: a small ``bank.h5`` written through the
frozen :mod:`ccg.data.bank` API and hand-made ``refs`` / ``instances`` tables,
so every expected number is computed by hand:

* three denominators (expression / unique_object / image) with a 3-sentence
  miss region, a single-sentence hit and a multi-sentence hit region;
* ``missing_bank_image`` and corrupt-group accounting (excluded from the
  denominators, listed in the report);
* Wilson CI equality with ``ccg.data.audit.wilson_ci``;
* IoU-threshold sensitivity rows (0.3/0.5/0.7) with the ``is_primary`` flag;
* CSV column contract of ``by_split.csv`` / ``threshold_sensitivity.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT / "src", ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import natural_omission_full as nof  # noqa: E402
from ccg.data.audit import wilson_ci  # noqa: E402
from ccg.data.bank import write_bank_entry  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic world helpers
# ---------------------------------------------------------------------------
def _ref(ref_id: int, image_id: int, split: str, ann_id: int, n_sentences: int = 1) -> dict:
    return {
        "ref_id": ref_id,
        "image_id": image_id,
        "split": split,
        "ann_id": ann_id,
        "sentences": [
            {"sent_id": index, "raw": f"expression {index}"} for index in range(n_sentences)
        ],
    }


def _instances(*entries) -> dict:
    """``entries``: ``(ann_id, image_id, bbox_xywh)`` tuples."""
    out = {}
    for ann_id, image_id, bbox in entries:
        out[str(ann_id)] = {"image_id": image_id, "bbox": [float(v) for v in bbox], "category_id": 1}
    return out


def _bank(tmp_path: Path, boxes_by_image: dict) -> Path:
    path = tmp_path / "bank.h5"
    with h5py.File(path, "a") as handle:
        for image_id, boxes in boxes_by_image.items():
            arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            write_bank_entry(
                handle, int(image_id), arr, np.full(arr.shape[0], 0.5, dtype=np.float32)
            )
    return path


def _score(refs, instances, bank_path):
    return nof.score_refs(refs, instances, bank_path)


def _counts(refs, instances, bank_path, thresholds=nof.THRESHOLDS):
    scoring = _score(refs, instances, bank_path)
    blob = nof.split_counts(scoring["per_ref"], thresholds)
    return scoring, blob


# ---------------------------------------------------------------------------
# the core scenario: miss / hit / multi-sentence + a missing bank image
# ---------------------------------------------------------------------------
def _core_world(tmp_path: Path):
    bank = _bank(
        tmp_path,
        {
            1: [[50, 50, 70, 70], [200, 200, 210, 210]],  # hits ref2's target exactly
            2: [[100, 100, 110, 110]],  # hits ref3's target exactly
        },
    )
    instances = _instances(
        (101, 1, [0, 0, 10, 10]),  # ref1 target: far from every bank box
        (102, 1, [50, 50, 20, 20]),  # ref2 target: exact bank box 1
        (103, 2, [100, 100, 10, 10]),  # ref3 target: exact bank box 2
        (104, 3, [0, 0, 5, 5]),  # ref4 target: image 3 has no bank group
    )
    refs = [
        _ref(1, 1, "val", 101, n_sentences=3),  # miss, multi-sentence
        _ref(2, 1, "val", 102, n_sentences=1),  # hit
        _ref(3, 2, "testA", 103, n_sentences=2),  # hit, multi-sentence
        _ref(4, 3, "val", 104, n_sentences=1),  # missing bank image
    ]
    return bank, instances, refs


def test_three_denominators_with_missing_image(tmp_path):
    bank, instances, refs = _core_world(tmp_path)
    scoring, blob = _counts(refs, instances, bank)
    counts = blob["counts"]["0.5"]

    # hand-computed expectations at the primary threshold 0.5
    assert counts["val"]["expression"] == (4, 3)  # 3+1 sentences, miss = 3
    assert counts["val"]["unique_object"] == (2, 1)  # ann 101 miss, ann 102 hit
    assert counts["val"]["image"] == (1, 1)  # image 1, contains a miss
    assert counts["testA"]["expression"] == (2, 0)  # multi-sentence hit counts fully
    assert counts["testA"]["unique_object"] == (1, 0)
    assert counts["testA"]["image"] == (1, 0)
    assert counts["train"]["expression"] == (0, 0)
    assert counts["testB"]["expression"] == (0, 0)

    # missing bank image: recorded with its ref count, excluded from the counts
    assert scoring["missing_bank_images"] == [{"image_id": 3, "n_refs": 1}]
    assert len(scoring["per_ref"]) == 3
    assert scoring["unresolved_refs"] == []

    # per-ref max IoU sanity (the raw sample behind the counts)
    ious = sorted(row["max_iou"] for row in scoring["per_ref"])
    assert ious[0] == pytest.approx(0.0, abs=1e-6)  # ref1 miss
    assert ious[1] == pytest.approx(1.0, abs=1e-6)
    assert ious[2] == pytest.approx(1.0, abs=1e-6)


def test_quantile_summary(tmp_path):
    bank, instances, refs = _core_world(tmp_path)
    scoring, _ = _counts(refs, instances, bank)
    quantiles = nof.quantile_summary(scoring["per_ref"])
    assert quantiles["n"] == 3
    assert quantiles["p10"] == pytest.approx(0.2, abs=1e-6)
    assert quantiles["p50"] == pytest.approx(1.0, abs=1e-6)
    assert quantiles["p90"] == pytest.approx(1.0, abs=1e-6)
    assert quantiles["mean"] == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_unique_object_dedup(tmp_path):
    """The same ann_id referenced twice counts once at the object level."""
    bank = _bank(tmp_path, {1: [[50, 50, 70, 70]]})
    instances = _instances((102, 1, [50, 50, 20, 20]))
    refs = [
        _ref(1, 1, "train", 102, n_sentences=1),
        _ref(2, 1, "train", 102, n_sentences=2),
    ]
    scoring, blob = _counts(refs, instances, bank)
    counts = blob["counts"]["0.5"]
    assert counts["train"]["expression"] == (3, 0)  # 1 + 2 sentences
    assert counts["train"]["unique_object"] == (1, 0)  # one distinct ann_id
    assert counts["train"]["image"] == (1, 0)
    assert blob["anomalies"] == []  # same ann_id judged consistently


# ---------------------------------------------------------------------------
# threshold sensitivity
# ---------------------------------------------------------------------------
def test_threshold_sensitivity_and_primary_flag(tmp_path):
    """IoU = 40/100 = 0.4: hit at 0.3, miss at 0.5 and 0.7."""
    bank = _bank(tmp_path, {1: [[2, 0, 6, 10]]})
    instances = _instances((201, 1, [0, 0, 10, 10]))
    refs = [_ref(1, 1, "train", 201, n_sentences=1)]
    scoring, blob = _counts(refs, instances, bank)
    assert scoring["per_ref"][0]["max_iou"] == pytest.approx(0.4, abs=1e-6)

    counts = blob["counts"]
    assert counts["0.3"]["train"]["expression"] == (1, 0)
    assert counts["0.5"]["train"]["expression"] == (1, 1)
    assert counts["0.7"]["train"]["expression"] == (1, 1)

    rows_low = nof.counts_to_rows(counts, 0.3, is_primary=False)
    row = next(r for r in rows_low if r["split"] == "train" and r["unit"] == "expression")
    assert row["num_total"] == 1 and row["num_miss"] == 0 and row["rate"] == 0.0
    assert row["is_primary"] is False

    rows_primary = nof.counts_to_rows(counts, 0.5, is_primary=True)
    row = next(r for r in rows_primary if r["split"] == "train" and r["unit"] == "expression")
    assert row["num_miss"] == 1 and row["rate"] == 1.0
    assert row["is_primary"] is True


# ---------------------------------------------------------------------------
# Wilson intervals / summary / artefacts
# ---------------------------------------------------------------------------
def test_wilson_ci_matches_audit(tmp_path):
    bank, instances, refs = _core_world(tmp_path)
    _, blob = _counts(refs, instances, bank)
    rows = nof.counts_to_rows(blob["counts"], 0.5)

    val_expression = next(r for r in rows if r["split"] == "val" and r["unit"] == "expression")
    assert val_expression["num_miss"] == 3 and val_expression["num_total"] == 4
    assert val_expression["rate"] == pytest.approx(0.75)
    low, high = wilson_ci(3, 4)
    assert val_expression["ci_low"] == pytest.approx(low)
    assert val_expression["ci_high"] == pytest.approx(high)

    val_object = next(r for r in rows if r["split"] == "val" and r["unit"] == "unique_object")
    assert (val_object["ci_low"], val_object["ci_high"]) == pytest.approx(wilson_ci(1, 2))

    # empty split: 0/0 keeps the audit convention (rate 0.0, CI [0, 1])
    empty = next(r for r in rows if r["split"] == "testB" and r["unit"] == "image")
    assert empty["num_total"] == 0 and empty["rate"] == 0.0
    assert (empty["ci_low"], empty["ci_high"]) == (0.0, 1.0)


def _summary_for(tmp_path, scoring, blob, n_refs_total: int) -> dict:
    args = argparse.Namespace(
        bank=tmp_path / "bank.h5",
        refs=tmp_path / "refs(unc).p",
        instances=tmp_path / "instances.json",
        out=tmp_path / "out",
    )
    return nof.build_summary(
        scoring,
        blob,
        args,
        bank_meta={"schema_version": "bank-v1", "top_n": 64},
        n_refs_total=n_refs_total,
        runtime_seconds=0.25,
    )


def test_build_summary_and_csv_contract(tmp_path):
    bank, instances, refs = _core_world(tmp_path)
    scoring, blob = _counts(refs, instances, bank)
    summary = _summary_for(tmp_path, scoring, blob, n_refs_total=len(refs))

    assert summary["primary_iou_threshold"] == 0.5
    assert summary["iou_thresholds"] == [0.3, 0.5, 0.7]
    assert summary["n_refs_total"] == 4
    assert summary["n_refs_scored"] == 3
    assert summary["n_refs_missing_bank_image"] == 1
    assert summary["counting_only"] is True
    assert summary["bank_attrs"]["top_n"] == 64

    # sensitivity table: 3 thresholds x 4 splits x 3 units, only 0.5 is primary
    sensitivity = summary["threshold_sensitivity"]
    assert len(sensitivity) == 3 * 4 * 3
    assert sum(1 for row in sensitivity if row["is_primary"]) == 4 * 3
    for row in sensitivity:
        assert (row["is_primary"] is True) == (row["iou_threshold"] == 0.5)

    # CSV artefacts: frozen column contracts
    written = nof.write_outputs(summary, tmp_path / "out")
    with written["by_split"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert list(rows[0].keys()) == list(nof.CSV_COLUMNS)
    assert len(rows) == 4 * 3  # 4 splits x 3 units

    with written["sensitivity"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert list(rows[0].keys()) == list(nof.SENSITIVITY_COLUMNS)
    assert len(rows) == 36

    payload = json.loads(written["summary"].read_text(encoding="utf-8"))
    assert payload["task"] == "natural_omission_full"
    assert len(payload["by_split"]) == 12


# ---------------------------------------------------------------------------
# accounting of breakage: corrupt groups / unjoinable ann_ids
# ---------------------------------------------------------------------------
def test_corrupt_group_recorded_as_unscorable(tmp_path):
    bank = _bank(tmp_path, {1: [[0, 0, 10, 10]]})
    with h5py.File(bank, "a") as handle:  # half-written group: no objectness
        group = handle.create_group("image_55")
        group.create_dataset("boxes", data=np.zeros((2, 4), dtype=np.float32))

    instances = _instances((301, 1, [0, 0, 10, 10]), (302, 55, [0, 0, 10, 10]))
    refs = [_ref(1, 1, "train", 301), _ref(2, 55, "train", 302)]
    scoring, blob = _counts(refs, instances, bank)

    assert scoring["missing_bank_images"] == []
    assert len(scoring["unscorable_bank_images"]) == 1
    record = scoring["unscorable_bank_images"][0]
    assert record["image_id"] == 55 and record["n_refs"] == 1
    assert "corrupt" in record["reason"]
    # the corrupt image contributes nothing to any denominator
    assert blob["counts"]["0.5"]["train"]["unique_object"] == (1, 0)

    summary = _summary_for(tmp_path, scoring, blob, n_refs_total=2)
    assert summary["n_refs_unscorable"] == 1
    assert summary["n_refs_scored"] == 1


def test_unresolved_ann_id_recorded(tmp_path):
    bank = _bank(tmp_path, {1: [[0, 0, 10, 10]]})
    instances = _instances((401, 1, [0, 0, 10, 10]))  # ann 402 missing
    refs = [_ref(1, 1, "val", 401), _ref(2, 1, "val", 402)]
    scoring, blob = _counts(refs, instances, bank)

    assert len(scoring["unresolved_refs"]) == 1
    unresolved = scoring["unresolved_refs"][0]
    assert unresolved["ann_id"] == 402 and "instances json" in unresolved["reason"]
    assert blob["counts"]["0.5"]["val"]["expression"] == (1, 0)  # only the joined ref


def test_other_splits_reported_not_swallowed(tmp_path):
    bank = _bank(tmp_path, {1: [[0, 0, 10, 10]]})
    instances = _instances((501, 1, [0, 0, 10, 10]))
    refs = [_ref(1, 1, "weird_split", 501)]
    _, blob = _counts(refs, instances, bank)
    assert blob["other_splits"] == {"weird_split": 1}
    for split in nof.SPLITS:
        assert blob["counts"]["0.5"][split]["expression"] == (0, 0)
