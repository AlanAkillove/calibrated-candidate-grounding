"""The common cohort (protocol section 29): identical ref_ids across primary Ks.

:func:`ccg.data.manifests.common_cohort` is *defined* as
``{entry : eligible[K] == True}`` with the frozen ``K = 50``.  Eligibility is
monotone in K (a larger candidate set needs a larger distractor pool:
``eligible[K] => eligible[K']`` for every ``K' <= K``), so the K=50 set is
exactly the intersection of the per-K eligible sets.  Restricting every
primary K's evaluation to this cohort is what makes "identical ref_ids" hold
by construction - this file pins that property down end to end (synthetic
entries plus one build on a synthetic bank) so it cannot regress silently.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")  # noqa: E402  (integration test writes a bank)

from ccg.data.manifests import (  # noqa: E402
    MANIFEST_SEED,
    PRIMARY_KS,
    ManifestEntry,
    ManifestFile,
    build_manifests,
    common_cohort,
)


# ---------------------------------------------------------------------------
# synthetic entries
# ---------------------------------------------------------------------------
def synthetic_entry(ref_id, n_valid, *, target_present=True, image_id=None, split="train"):
    """A manually computed :class:`ManifestEntry` with ``eligible`` derived by hand."""
    return ManifestEntry(
        ref_id=int(ref_id),
        image_id=int(image_id if image_id is not None else ref_id),
        split=split,
        target_index=0 if target_present else None,
        distractor_order=np.arange(1, int(n_valid) + 1, dtype=np.int32),
        n_valid_distractors=int(n_valid),
        n_target_equiv_removed=0,
        n_same_category_available=0,
        eligible={K: bool(target_present and n_valid >= K - 1) for K in PRIMARY_KS},
        n_same_used_by_K={K: 0 for K in PRIMARY_KS},
        hard_fraction_by_K={K: 0.0 for K in PRIMARY_KS},
        target_max_iou=0.9 if target_present else 0.1,
    )


def mixed_entries():
    """Hand-computed mix: full pools, boundary pool, short pool, and a miss.

    ============  ======  ========  ============  ==============
    ref_id        n_valid eligible5 eligible50    rationale
    ============  ======  ========  ============  ==============
    1             63      True      True          full bank
    2             49      True      True          exactly the K=50 boundary
    3             48      True      False         one short of K=50
    4             10      True      False         K=5 only
    5             63      False     False         natural miss (no target)
    6             0       False     False         target present but no pool
    ============  ======  ========  ==============
    """
    return [
        synthetic_entry(1, 63),
        synthetic_entry(2, 49),
        synthetic_entry(3, 48),
        synthetic_entry(4, 10),
        synthetic_entry(5, 63, target_present=False),
        synthetic_entry(6, 0),
    ]


def expected_eligible(entries, K):
    return sorted(int(entry.ref_id) for entry in entries if entry.eligible[K])


# ---------------------------------------------------------------------------
# strict equivalence with {eligible[50]}
# ---------------------------------------------------------------------------
def test_common_cohort_equals_eligible_k50():
    entries = mixed_entries()
    cohort = common_cohort(entries)

    assert cohort.dtype == np.int64
    assert cohort.tolist() == expected_eligible(entries, 50) == [1, 2]
    # strict equivalence in both directions
    assert set(cohort.tolist()) == {entry.ref_id for entry in entries if entry.eligible[50]}
    for entry in entries:
        assert (entry.ref_id in cohort) == bool(entry.eligible[50])


def test_common_cohort_drops_misses_and_short_pools():
    entries = mixed_entries()
    cohort = common_cohort(entries)

    # the miss (5) and the no-pool sample (6) are not eligible at any K
    assert 5 not in cohort and 6 not in cohort
    # short pools stay in what they can support but never reach K=50
    per_k = {K: expected_eligible(entries, K) for K in PRIMARY_KS}
    assert per_k[5] == [1, 2, 3, 4]
    assert per_k[10] == [1, 2, 3, 4]
    assert per_k[20] == [1, 2, 3]
    assert per_k[50] == [1, 2]
    # n_common matches the hand computation (2 of 6, the two pools >= 49)
    assert int(cohort.size) == 2 == sum(1 for entry in entries if entry.n_valid_distractors >= 49
                                        and entry.target_index is not None)


# ---------------------------------------------------------------------------
# section 29: identical ref_ids across the primary Ks
# ---------------------------------------------------------------------------
def test_identical_ref_ids_across_primary_ks():
    entries = mixed_entries()
    cohort = common_cohort(entries, K=50)
    cohort_set = set(cohort.tolist())

    per_k = {K: {entry.ref_id for entry in entries if entry.eligible[K]} for K in PRIMARY_KS}

    # eligibility monotonicity: larger K is strictly harder
    assert per_k[5] >= per_k[10] >= per_k[20] >= per_k[50]

    # the K=50 cohort is the intersection of the per-K eligible sets
    intersection = set.intersection(*per_k.values())
    assert intersection == cohort_set

    # section 29: evaluating every primary K *on the cohort* yields the very
    # same ref_id set for K = 5 / 10 / 20 / 50
    eval_sets = {
        K: {ref_id for ref_id in cohort_set if ref_id in per_k[K]} for K in PRIMARY_KS
    }
    for K in PRIMARY_KS:
        assert eval_sets[K] == cohort_set, f"K={K} does not cover the common cohort"


def test_common_cohort_is_monotone_in_k():
    entries = mixed_entries()
    cohorts = {K: set(common_cohort(entries, K=K).tolist()) for K in PRIMARY_KS}

    assert cohorts[50] <= cohorts[20] <= cohorts[10] <= cohorts[5]
    assert cohorts[50] == {1, 2}
    assert cohorts[20] == {1, 2, 3}
    assert cohorts[5] == {1, 2, 3, 4}


def test_common_cohort_accepts_a_manifest_file_and_kwarg():
    entries = mixed_entries()
    manifest = ManifestFile(regime="random", meta={"n_refs": len(entries)}, entries=entries)

    np.testing.assert_array_equal(common_cohort(manifest), common_cohort(entries))
    np.testing.assert_array_equal(common_cohort(manifest, K=5), common_cohort(entries, K=5))


def test_common_cohort_empty_input():
    cohort = common_cohort([])
    assert cohort.dtype == np.int64
    assert cohort.size == 0


# ---------------------------------------------------------------------------
# integration: build_manifests on a synthetic bank
# ---------------------------------------------------------------------------
def tiled_boxes(n, size=10.0, gap=10.0, per_row=8):
    step = size + gap
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        boxes[i] = [
            (i % per_row) * step,
            (i // per_row) * step,
            (i % per_row) * step + size,
            (i // per_row) * step + size,
        ]
    return boxes


def shrunken(box, factor):
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width, height = (x2 - x1) * factor, (y2 - y1) * factor
    return np.asarray([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
                      dtype=np.float32)


def test_common_cohort_after_build_manifests(tmp_path):
    """Full-pool image contributes, short-pool and miss images are excluded."""
    full, short_a, short_b, miss = 31, 32, 34, 33
    full_boxes, short_boxes = tiled_boxes(64), tiled_boxes(16)
    far = np.asarray([10_000.0, 10_000.0, 10_020.0, 10_020.0], dtype=np.float32)

    with h5py.File(tmp_path / "proposals.h5", "w") as handle:
        for image_id, boxes in (
            (full, full_boxes),
            (short_a, short_boxes),
            (short_b, short_boxes),
            (miss, full_boxes),
        ):
            group = handle.create_group(f"image_{image_id}")
            group.create_dataset("boxes", data=boxes.astype(np.float32))
            group.create_dataset(
                "objectness", data=np.linspace(1.0, 0.1, boxes.shape[0], dtype=np.float32)
            )

    records, join = [], {}
    for index, ref_id in enumerate((101, 102, 103)):  # three refs on the full image
        ann_id = 900 + ref_id
        records.append({"ref_id": ref_id, "image_id": full, "split": "train", "ann_id": ann_id})
        join[str(ann_id)] = {"image_id": full, "bbox": _xywh(shrunken(full_boxes[index], 0.9))}
    for index, ref_id in enumerate((201, 202)):  # two refs on the short image (N=16)
        ann_id = 900 + ref_id
        records.append({"ref_id": ref_id, "image_id": short_a, "split": "val", "ann_id": ann_id})
        join[str(ann_id)] = {"image_id": short_a, "bbox": _xywh(shrunken(short_boxes[index], 0.9))}
    ann_id = 1103
    records.append({"ref_id": 203, "image_id": short_b, "split": "val", "ann_id": ann_id})
    join[str(ann_id)] = {"image_id": short_b, "bbox": _xywh(shrunken(short_boxes[0], 0.9))}
    records.append({"ref_id": 301, "image_id": miss, "split": "testA", "ann_id": 1201})
    join["1201"] = {"image_id": miss, "bbox": _xywh(far)}  # natural miss

    manifest = build_manifests(tmp_path / "proposals.h5", records, join, "random",
                               seed=MANIFEST_SEED)
    assert len(manifest.entries) == 7

    cohort = common_cohort(manifest)
    assert cohort.tolist() == [101, 102, 103]

    by_id = {entry.ref_id: entry for entry in manifest.entries}
    assert by_id[101].eligible[50] and by_id[103].eligible[50]
    assert not by_id[201].eligible[50]  # N=16 -> 15 valid distractors < 49
    assert by_id[201].eligible[5]  # ... but K=5 is supported
    assert not by_id[203].eligible[50] and by_id[203].eligible[5]
    assert not any(by_id[301].eligible.values())  # miss: never eligible
    assert {by_id[201].eval_split, by_id[203].eval_split} == {"val_select", "val_calib"}

    # section 29 on real build output: the cohort is per-K interchangeable
    per_k = {
        K: {entry.ref_id for entry in manifest.entries if entry.eligible[K]}
        for K in PRIMARY_KS
    }
    intersection = set.intersection(*per_k.values())
    assert intersection == set(cohort.tolist())
    for K in PRIMARY_KS:
        assert {ref_id for ref_id in cohort.tolist() if ref_id in per_k[K]} == set(cohort.tolist())


def _xywh(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]
