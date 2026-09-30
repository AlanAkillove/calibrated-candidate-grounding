"""Audit helpers for the V2-M LCR stage.

Two families of pure functions:

* **grounding invariance** -- :func:`grounding_fingerprint` /
  :func:`assert_grounding_unchanged`.  The frozen grounding decision
  (``argmax`` index per row) and the raw score matrix must stay bit-identical
  through every reliability stage; any change is an IMPLEMENTATION FAILURE and
  raises immediately (protocol section "grounding_invariance_check").
* **zero-hard-exposure guards** -- :func:`assert_no_hard_rows` /
  :func:`m1_training_guard`.  M1 trains and tunes on ``val_calib`` random
  K5/K10 rows only: SameCategory rows, curriculum levels m=2/4/8, K20 and K50
  must never leak into a training or tuning mask.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

__all__ = [
    "assert_grounding_unchanged",
    "assert_no_hard_rows",
    "grounding_fingerprint",
    "m1_training_guard",
]


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def grounding_fingerprint(scores: Any) -> Mapping[str, Any]:
    """Fingerprint of one frozen score block: scores hash, argmax hash, stats.

    The fingerprint identifies a grounding decision bit-for-bit: the sha256 of
    the score matrix covers the values, the sha256 of the ``argmax`` index
    vector covers the decision, and ``accuracy`` is the derived top-1 accuracy
    (target is always local column 0 in this project's frozen layouts).
    """
    sc = np.asarray(scores)
    if sc.ndim != 2:
        raise ValueError(f"scores must be 2-D [n, K], got {sc.shape}")
    idx = np.argmax(sc, axis=1).astype(np.int64, copy=False)
    return {
        "n": int(sc.shape[0]),
        "k": int(sc.shape[1]),
        "dtype": str(sc.dtype),
        "scores_sha256": _sha256(sc),
        "argmax_sha256": _sha256(idx),
        "accuracy": float(np.mean(idx == 0)),
    }


def assert_grounding_unchanged(
    fingerprint_before: Mapping[str, Any],
    fingerprint_after: Mapping[str, Any],
    *,
    context: str = "",
) -> None:
    """Raise when any grounding-invariance field changed between two snapshots."""
    if dict(fingerprint_before) != dict(fingerprint_after):
        differing = [
            key
            for key in fingerprint_before
            if fingerprint_before.get(key) != fingerprint_after.get(key)
        ]
        raise AssertionError(
            "GROUNDING INVARIANCE VIOLATION"
            + (f" [{context}]" if context else "")
            + f": differing fields {differing} -- LCR must only change the "
            "reliability score (protocol: bit-identical grounding decision)"
        )


def assert_no_hard_rows(
    sentence_id: Any,
    hard_sentence_ids: Any,
    *,
    context: str = "",
) -> int:
    """Raise when any row belongs to the hard-competition sentence universe.

    Returns the number of overlapping rows (0 on success, always raising first
    when nonzero) so callers can record the assertion in metadata.
    """
    rows = np.asarray(sentence_id, dtype=np.int64).reshape(-1)
    hard = np.asarray(hard_sentence_ids, dtype=np.int64).reshape(-1)
    overlap = int(np.isin(rows, hard).sum())
    if overlap:
        raise AssertionError(
            "ZERO-HARD-EXPOSURE VIOLATION"
            + (f" [{context}]" if context else "")
            + f": {overlap} row(s) overlap the SameCategory universe; M1 "
            "training/tuning must never see hard competition"
        )
    return overlap


def m1_training_guard(
    eval_split: Any,
    sentence_id: Any,
    hard_sentence_ids: Any,
    *,
    train_ks: Any = (5, 10),
    seen_ks: Any = (5, 10),
    context: str = "",
) -> Mapping[str, Any]:
    """Full M1 zero-hard-exposure assertion for one train/tune mask.

    Checks (all must pass; any failure raises):

    1. every masked row has ``eval_split == "val_calib"`` (random-only regime);
    2. zero overlap with the frozen SameCategory sentence universe;
    3. the candidate sizes touched during training/tuning are a subset of
       ``train_ks`` (K20/K50 must never appear).
    """
    splits = np.asarray(eval_split)
    bad = int(np.sum(splits != "val_calib"))
    if bad:
        names = sorted(set(splits[splits != "val_calib"].tolist()))
        raise AssertionError(
            "ZERO-HARD-EXPOSURE VIOLATION"
            + (f" [{context}]" if context else "")
            + f": {bad} row(s) outside val_calib (splits {names})"
        )
    overlap = assert_no_hard_rows(sentence_id, hard_sentence_ids, context=context)
    unseen = sorted(set(int(k) for k in seen_ks) - set(int(k) for k in train_ks))
    if unseen:
        raise AssertionError(
            "ZERO-HARD-EXPOSURE VIOLATION"
            + (f" [{context}]" if context else "")
            + f": candidate sizes {unseen} outside the M1 training regimes"
        )
    return {
        "n_rows": int(splits.shape[0]),
        "splits": sorted(set(splits.tolist())),
        "hard_overlap": overlap,
        "ks": sorted(int(k) for k in seen_ks),
        "pass": True,
    }
