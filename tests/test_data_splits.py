"""Image-level ``val -> val_select / val_calib`` split discipline."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.data.splits import (
    CALIBRATION_SPLITS,
    DEFAULT_CALIB_FRACTION,
    DEFAULT_SPLIT_SEED,
    SELECTION_SPLITS,
    SPLIT_NAMES,
    SPLIT_NATURAL_OMISSION,
    SPLIT_TEST_A,
    SPLIT_TEST_B,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_VAL_CALIB,
    SPLIT_VAL_SELECT,
    SplitPlan,
    apply_split_plan,
    assign_val_subsplits,
    find_image_leakage,
    image_ids_of,
    normalize_split_name,
    plan_val_split,
    split_examples,
    split_val_images,
)
from ccg.data.types import ReferringExample


def make_examples(image_ids, per_image: int = 3, split: str = SPLIT_VAL):
    """``per_image`` expressions on every image - the unit that must not be split."""
    out = []
    ref = 0
    for image_id in image_ids:
        for k in range(per_image):
            ref += 1
            out.append(
                ReferringExample(
                    ref_id=ref,
                    image_id=int(image_id),
                    text=f"expression {ref} on {image_id}",
                    gt_box=np.array([0.0, 0.0, 10.0 + k, 20.0 + k], dtype=np.float32),
                    split=split,
                )
            )
    return out


# ---------------------------------------------------------------------------
# name normalisation
# ---------------------------------------------------------------------------
def test_split_constants_are_consistent():
    assert SPLIT_VAL_SELECT in SPLIT_NAMES and SPLIT_VAL_CALIB in SPLIT_NAMES
    assert SELECTION_SPLITS == (SPLIT_VAL_SELECT,)
    assert CALIBRATION_SPLITS == (SPLIT_VAL_CALIB,)
    assert SPLIT_NATURAL_OMISSION == "natural_omission"
    assert (DEFAULT_SPLIT_SEED, DEFAULT_CALIB_FRACTION) == (0, 0.5)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("training", SPLIT_TRAIN),
        ("TR", SPLIT_TRAIN),
        ("validation", SPLIT_VAL),
        ("val", SPLIT_VAL),
        ("testA", SPLIT_TEST_A),
        ("test-a", SPLIT_TEST_A),
        ("testB", SPLIT_TEST_B),
        ("test_u", SPLIT_VAL),      # RefCOCOg test-u is a validation-style split
        ("val_u", SPLIT_VAL),
        ("  val  ", SPLIT_VAL),
        ("whatever", "whatever"),  # unknown labels survive, lower-cased
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_split_name(raw, expected):
    assert normalize_split_name(raw) == expected


# ---------------------------------------------------------------------------
# the image-level cut
# ---------------------------------------------------------------------------
def test_val_images_are_cut_at_image_level_without_overlap():
    images = list(range(100, 110))  # 10 images
    select, calib = split_val_images(images, seed=0, calib_fraction=0.5)

    assert set(select.tolist()).isdisjoint(set(calib.tolist())), "no image may be in both"
    assert sorted(select.tolist() + calib.tolist()) == images, "the cut must be exhaustive"
    assert select.size == 5 and calib.size == 5


def test_val_image_cut_is_deterministic_and_seed_dependent():
    images = list(range(1, 21))
    a_select, a_calib = split_val_images(images, seed=7)
    b_select, b_calib = split_val_images(images, seed=7)
    np.testing.assert_array_equal(a_calib, b_calib)
    np.testing.assert_array_equal(a_select, b_select)

    other_select, other_calib = split_val_images(images, seed=0)
    assert set(other_calib.tolist()) != set(a_calib.tolist())


def test_duplicate_val_images_and_fraction_rounding():
    select, calib = split_val_images([1, 1, 2, 3, 4, 5, 5], seed=0, calib_fraction=0.3)
    assert calib.size == 2 and select.size == 3
    # a fraction that would empty one side is clamped to keep both non-empty
    select, calib = split_val_images([1, 2, 3], seed=0, calib_fraction=0.01)
    assert calib.size == 1 and select.size == 2


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_bad_calib_fraction_rejected(bad):
    with pytest.raises(ValueError, match="calib_fraction"):
        split_val_images([1, 2, 3], calib_fraction=bad)


def test_too_few_val_images_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        split_val_images([1], seed=0)
    with pytest.raises(ValueError, match="at least 2"):
        split_val_images([], seed=0)


def test_split_plan_rejects_overlapping_images():
    with pytest.raises(ValueError, match="share"):
        SplitPlan(val_select_image_ids=[1, 2, 3], val_calib_image_ids=[3, 4])


def test_split_plan_helpers():
    plan = SplitPlan(val_select_image_ids=[5, 1], val_calib_image_ids=[3, 4], seed=1, calib_fraction=0.4)
    assert plan.num_images == 4
    mapping = plan.image_to_subsplit()
    assert mapping == {1: SPLIT_VAL_SELECT, 5: SPLIT_VAL_SELECT, 3: SPLIT_VAL_CALIB, 4: SPLIT_VAL_CALIB}
    payload = plan.to_dict()
    assert payload["seed"] == 1 and payload["calib_fraction"] == pytest.approx(0.4)
    assert payload["val_select_image_ids"] == [1, 5], "ids are sorted and de-duplicated"


# ---------------------------------------------------------------------------
# routing examples through the plan
# ---------------------------------------------------------------------------
def test_every_expression_of_an_image_lands_in_the_same_subsplit():
    images = list(range(200, 212))
    examples = make_examples(images, per_image=3)
    grouped = split_examples(examples, seed=3, calib_fraction=0.5)

    assert SPLIT_VAL not in grouped, "the ambiguous 'val' label must disappear"
    select_ids = {ex.image_id for ex in grouped[SPLIT_VAL_SELECT]}
    calib_ids = {ex.image_id for ex in grouped[SPLIT_VAL_CALIB]}
    assert select_ids.isdisjoint(calib_ids), "image-level leakage between the two calibration groups"

    # and per image: all three expressions share one sub-split
    per_image: dict = {}
    num_routed = 0
    for name in (SPLIT_VAL_SELECT, SPLIT_VAL_CALIB):
        for ex in grouped[name]:
            per_image.setdefault(ex.image_id, set()).add(name)
            num_routed += 1
    assert all(len(names) == 1 for names in per_image.values())
    assert len(per_image) == len(images)
    assert num_routed == len(images) * 3


def test_leakage_detector():
    images = list(range(1, 11))
    grouped = split_examples(make_examples(images, per_image=2), seed=0)
    assert find_image_leakage(grouped) == []

    # deliberately move one expression to a group it does not belong to
    moved = grouped[SPLIT_VAL_SELECT][0]
    contaminated = dict(grouped)
    contaminated[SPLIT_TRAIN] = list(grouped.get(SPLIT_TRAIN, [])) + [moved]
    violations = find_image_leakage(contaminated)
    assert violations and "share image" in violations[0]


def test_apply_split_plan_keeps_train_test_labels_and_original_objects():
    images = list(range(1, 7))
    val_examples = make_examples(images, per_image=2, split=SPLIT_VAL)
    train_examples = make_examples([50, 51], per_image=1, split="train")
    test_examples = make_examples([60], per_image=1, split="testA")
    everything = val_examples + train_examples + test_examples

    plan = plan_val_split(everything, seed=0, calib_fraction=0.5)
    grouped = apply_split_plan(everything, plan)

    assert len(grouped[SPLIT_TRAIN]) == 2
    assert len(grouped[SPLIT_TEST_A]) == 1
    assert len(grouped[SPLIT_VAL_SELECT]) + len(grouped[SPLIT_VAL_CALIB]) == len(val_examples)
    assert SPLIT_TEST_B not in grouped, "empty groups are omitted, not faked as size 0"
    # original examples are untouched (a shallow copy is relabelled instead)
    assert all(ex.split == SPLIT_VAL for ex in val_examples)


def test_apply_split_plan_refuses_unplanned_val_images():
    examples = make_examples([1, 2, 3, 4], per_image=1)
    plan = SplitPlan(val_select_image_ids=[1, 2], val_calib_image_ids=[3])
    with pytest.raises(ValueError, match="not covered by the val split plan"):
        apply_split_plan(examples, plan)


def test_apply_split_plan_refuses_missing_split_label():
    examples = make_examples([1, 2], per_image=1)
    broken = ReferringExample(ref_id=99, image_id=1, text="no split", gt_box=[0, 0, 5, 5], split="")
    plan = plan_val_split(examples, seed=0)
    with pytest.raises(ValueError, match="empty split"):
        apply_split_plan([broken], plan)


def test_plan_val_split_needs_val_examples():
    with pytest.raises(ValueError, match="split='val'"):
        plan_val_split(make_examples([1, 2], per_image=1, split="train"))


def test_assign_val_subsplits_labels_and_strictness():
    labels = assign_val_subsplits([1, 2, 3], [1, 3], [2])
    assert labels.tolist() == [SPLIT_VAL_SELECT, SPLIT_VAL_CALIB, SPLIT_VAL_SELECT]
    with pytest.raises(ValueError, match="neither val_select nor val_calib"):
        assign_val_subsplits([1, 9], [1], [2])


def test_image_ids_of():
    assert image_ids_of(make_examples([5, 1, 5, 3], per_image=1)).tolist() == [1, 3, 5]
