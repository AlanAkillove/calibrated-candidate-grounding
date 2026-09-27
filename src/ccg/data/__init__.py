"""Data layer of :mod:`ccg`: records, splits, proposal banks, candidate sets.

Public surface (see the module docstrings for the full API):

* :mod:`ccg.data.types` - :class:`ReferringExample`, :class:`ProposalBank`,
  :class:`CandidateSet` (validated pure-data containers, numpy only).
* :mod:`ccg.data.splits` - split name constants and the image-level
  ``val -> val_select / val_calib`` cut.
* :mod:`ccg.data.refcoco` - RefCOCO+ annotation parsers (json primary, UNC mats
  skeleton).
* :mod:`ccg.data.rpn` - class-agnostic RPN proposal extraction (torchvision
  Faster R-CNN; resized-space -> original-image coordinate mapping).
* :mod:`ccg.data.audit` - frozen proposal-audit statistics (GT-object recall,
  candidate availability, redundancy, same-category supply, natural omission,
  Wilson intervals); pure numpy, deterministic, no silent filtering.
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
# Audit statistics (frozen multi-agent brief).  ``ccg.data.audit.proposal_recall``
# is the GT-object-level variant and is intentionally NOT re-exported here: the
# name is already bound to the expression-level one from .proposals below.
from .audit import (
    aggregate_availability,
    assign_gt_category,
    iou_quantiles,
    natural_omission_rate,
    redundancy_stats,
    remaining_candidate_count,
    same_category_counts,
    target_availability,
    wilson_ci,
)
# Proposal bank v1 (frozen h5 layout; see the bank-v1 contract in bank.py).
# Names do not collide with the legacy .proposals helpers (write_bank /
# read_bank / iter_banks above): the v1 surface is *_bank_entry / iter_bank /
# read_bank_image / bank_* and is what the multi-agent pipeline reads.
from .bank import (
    BANK_SCHEMA_VERSION,
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
from .manifests import (
    FILE_SPLITS,
    MANIFEST_SEED,
    PRIMARY_KS,
    REGIMES,
    ManifestEntry,
    ManifestFile,
    build_manifests,
    common_cohort,
    filter_entries,
    manifest_path,
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
from .refcoco import (
    AnnotationFormatError,
    ParseReport,
    load_instances_json,
    load_refs_pickle,
    parse_refs_json,
    refs_by_image,
    refs_from_records,
)
from .rpn import (
    RPNProposals,
    build_rpn_model,
    extract_proposals,
    inverse_resized_boxes,
)
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
    "load_refs_pickle",
    "load_instances_json",
    "refs_by_image",
    "ParseReport",
    "AnnotationFormatError",
    # audit statistics (ccg.data.audit; its proposal_recall is the GT-object-level
    # one and stays importable from the submodule, see the import note above)
    "assign_gt_category",
    "target_availability",
    "aggregate_availability",
    "remaining_candidate_count",
    "redundancy_stats",
    "same_category_counts",
    "natural_omission_rate",
    "wilson_ci",
    "iou_quantiles",
    # proposal bank v1 (ccg.data.bank, frozen schema "bank-v1")
    "BANK_SCHEMA_VERSION",
    "write_bank_entry",
    "iter_bank",
    "read_bank_image",
    "bank_attrs",
    "image_ids",
    "bank_has_image",
    "finalize_bank",
    "find_corrupt_images",
    "repair_bank",
    "delete_bank_entry",
    "bank_group_name",
    # manifests (frozen candidate orderings + common cohort, "manifests-v1")
    "MANIFEST_SEED",
    "PRIMARY_KS",
    "REGIMES",
    "FILE_SPLITS",
    "ManifestEntry",
    "ManifestFile",
    "build_manifests",
    "common_cohort",
    "filter_entries",
    "manifest_path",
    # rpn (class-agnostic proposal extraction, torchvision Faster R-CNN)
    "RPNProposals",
    "build_rpn_model",
    "extract_proposals",
    "inverse_resized_boxes",
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
