"""Data layer of :mod:`ccg`: records, splits, proposal banks, candidate sets.

Public surface (see the module docstrings for the full API):

* :mod:`ccg.data.types` - :class:`ReferringExample`, :class:`ProposalBank`,
  :class:`CandidateSet` (validated pure-data containers, numpy only).
* :mod:`ccg.data.splits` - split name constants and the image-level
  ``val -> val_select / val_calib`` cut.
* :mod:`ccg.data.refcoco` - RefCOCO+ annotation parsers (json primary, UNC mats
  skeleton).
* :mod:`ccg.data.candidate_sets` - nested random / same-category hard / CLIP
  hard candidate construction, synthetic target omission.
* :mod:`ccg.data.proposals` - IoU geometry, unique-target assignment, proposal
  caches (HDF5 / NPZ) and the proposal-quality audit.
"""

from __future__ import annotations

from .candidate_sets import (
    CandidateAvailability,
    CandidateShortage,
    NestedCandidateSets,
    assert_nested,
    build_clip_hard_sets,
    build_nested_random_sets,
    build_ranked_nested_sets,
    build_same_category_hard_sets,
    select_clip_hard_negatives,
    select_same_category_hard_negatives,
    synthetic_omit,
)
from .proposals import (
    TargetAssignment,
    assign_target,
    audit_examples,
    box_iou,
    iou_matrix,
    iter_banks,
    natural_miss_rate,
    proposal_recall,
    read_bank,
    read_features,
    remove_proposals,
    summarize_proposal_quality,
    write_bank,
    write_banks,
    xywh_to_xyxy,
    xyxy_to_xywh,
)
from .refcoco import AnnotationFormatError, ParseReport, parse_refs_json, refs_from_records
from .splits import (
    CALIBRATION_SPLITS,
    SELECTION_SPLITS,
    SPLIT_NATURAL_OMISSION,
    SPLIT_TEST_A,
    SPLIT_TEST_B,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_VAL_CALIB,
    SPLIT_VAL_SELECT,
    SplitPlan,
    split_examples,
    split_val_images,
)
from .types import CandidateSet, ProposalBank, ReferringExample

__all__ = [
    # types
    "ReferringExample",
    "ProposalBank",
    "CandidateSet",
    # splits
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_VAL_SELECT",
    "SPLIT_VAL_CALIB",
    "SPLIT_TEST_A",
    "SPLIT_TEST_B",
    "SPLIT_NATURAL_OMISSION",
    "SELECTION_SPLITS",
    "CALIBRATION_SPLITS",
    "SplitPlan",
    "split_val_images",
    "split_examples",
    # refcoco
    "parse_refs_json",
    "refs_from_records",
    "ParseReport",
    "AnnotationFormatError",
    # candidate sets
    "build_nested_random_sets",
    "build_ranked_nested_sets",
    "build_same_category_hard_sets",
    "build_clip_hard_sets",
    "select_same_category_hard_negatives",
    "select_clip_hard_negatives",
    "synthetic_omit",
    "assert_nested",
    "CandidateAvailability",
    "CandidateShortage",
    "NestedCandidateSets",
    # proposals
    "assign_target",
    "TargetAssignment",
    "remove_proposals",
    "proposal_recall",
    "natural_miss_rate",
    "summarize_proposal_quality",
    "audit_examples",
    "write_bank",
    "write_banks",
    "read_bank",
    "iter_banks",
    "read_features",
    "iou_matrix",
    "box_iou",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
]
