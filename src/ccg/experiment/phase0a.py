"""Phase 0A: fixed-CLIP cosine audit driver and artifact writer.

Evaluation unit (frozen; multi-agent brief, 2026-09-27)
-------------------------------------------------------
* **Sentence level.** A referring expression is one *sentence* of the UNC
  ``refs(unc).p`` archive (``ref['sentences'][i]['sent_id']``).  Every sentence
  of one ref shares that ref's frozen candidate manifest (same ``target_index``,
  same ``distractor_order``, same ``eligible`` flags): the candidate set
  ``C_K = [target] + distractor_order[:K-1]`` depends on the ref only, never on
  the sentence.
* **sentence_id ordering.** ``sentence_id`` is the archive-wide ``sent_id``:
  ids enumerate sentences by ascending ``ref_id`` and, within a ref, by
  ascending in-ref ``sent_id``.  Verified against ``refs(unc).p``: the ids are
  the contiguous range ``0 .. 141563``.  This is the exact key of the feature
  side (``cache/features/text_index.csv`` -> ``text_features.h5``); the loader
  re-validates uniqueness and (when the cache metadata is present) the
  ``n_sentences`` coverage.
* **Bootstrap clustering.** The paired cluster bootstrap resamples whole
  ``image_id`` groups: sentences sharing an image share the proposal bank,
  region features and candidate pool, so they are *not* independent draws.

Probability variants (protocol section 11)
------------------------------------------
Let ``s`` be the FP32 cosine scores of one sentence and ``alpha`` the model's
native logit scale (``metadata["native_logit_scale"]``).  Three variants are
reported side by side, never collapsed:

``native``
    ``P = softmax(alpha * s)`` - the model's own probability at deployment.
    **Primary reported variant**: it is the only one whose calibration
    statements transfer to the unmodified system; reliability bins, selective
    metrics and the calibration-map-shift table are computed on it.  ``T=1``
    and ``global-T`` are reported next to it as diagnostics, not as
    replacements.
``T1``
    ``P = softmax(s)`` - the unscaled-cosine baseline (what a naive
    ``softmax(cosine)`` readout gives).
``global_T``
    ``P = softmax(s / T*)`` with a single ``T*`` fitted on **val_calib only**,
    pooled over ``K in {5, 10}`` (one global temperature, not per-K).
    Protocol section 22.

Calibration isolation (hard requirement, protocol section 9)
------------------------------------------------------------
Anything that *fits* a parameter (temperature, or any future calibrator) may
see **val_calib rows only**.  ``fit_global_temperature`` and the oracle
diagnostic re-assert this on their inputs and raise
:class:`CalibrationIsolationError` on any other eval split; testA / testB
never participate in fitting.  The pairwise bootstrap never fits anything, it
only evaluates frozen rows.

Hard sanity checks and the STOP contract (protocol sections 17-19)
------------------------------------------------------------------
Three checks run on every audit before any artifact is written:

1. **score invariance** - the same candidate of the same sentence has
   bit-identical raw scores across every K (``atol=0``);
2. **rank monotonicity** - on the nested sets, ``rank(K') >= rank(K)`` for
   ``K' > K``;
3. **accuracy monotonicity** - ``Acc(K5) >= Acc(K10) >= Acc(K20) >= Acc(K50)``
   on the common cohort.

A violation means the driver is broken, not the data: the audit writes
``VALIDATION_FAILURE.json`` into the output directory and raises
``SystemExit(2)`` - it never silently continues.  Confidence monotonicity in
K is explicitly **not** checked (protocol section 20: calibration drift across
K is the *object of study*, not an invariant).

Artifacts (protocol section 24)
-------------------------------
Everything lands in ``results/phase0a_cosine/`` (see :func:`run_audit`):
``metadata.json``, ``cohort_summary.json``, ``candidate_manifests/``,
``raw_predictions/K{K}.npz`` (raw logits are never dropped),
``prediction_summary.csv``, ``ranking_metrics.csv``,
``calibration_metrics.csv``, ``selective_metrics.csv``,
``reliability_bins.csv``, ``calibration_map_shift.csv``,
``global_temperature.json``, ``oracle_temperature_diagnostic.json``,
``bootstrap_ci.csv``, ``selective_curves.npz`` and ``figures/``.

Only the ``random`` construction regime is run here (the same-category regime
is a separate experiment; CLIP-hard construction is forbidden for this audit).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccg.calibration import DEFAULT_TEMP_BOUNDS, fit_temperature
from ccg.data.manifests import (
    ManifestEntry,
    ManifestFile,
    common_cohort,
    filter_entries,
    manifest_path,
)
from ccg.data.refcoco import load_refs_pickle
from ccg.metrics import (
    aurc,
    brier_binary,
    confidence_accuracy_gap,
    confidence_from_scores,
    mrr,
    multiclass_brier,
    multiclass_nll,
    nll_binary,
    reliability_table,
    risk_at_coverage,
    risk_coverage_curve,
    softmax_scores,
    target_rank,
    top1_accuracy,
    topk_accuracy,
    top_label_ece,
    top_label_ece_adaptive,
)

__all__ = [
    "DEFAULT_KS",
    "PRIMARY_SPLITS",
    "REGIME_RANDOM",
    "REPORT_VARIANTS",
    "VARIANT_GLOBAL",
    "VARIANT_NATIVE",
    "VARIANT_T1",
    "LEGAL_CALIBRATION_KS",
    "RELIABILITY_BINS",
    "RISK_COVERAGE_LEVELS",
    "MIN_RELIABLE_SHIFT_N",
    "SHIFT_BIN_EDGES",
    "SPLIT_CODES",
    "FAILURE_FILENAME",
    "CalibrationIsolationError",
    "SampleStats",
    "ScoreBundle",
    "SentenceRecord",
    "TemperatureFit",
    "ValidationFailure",
    "assert_accuracy_monotonic",
    "assert_rank_monotonic",
    "assert_score_invariance",
    "calibration_map_shift",
    "calibration_metrics",
    "common_cohort_rows",
    "compute_candidate_scores",
    "fit_global_temperature",
    "load_cosine_inputs",
    "oracle_temperature_diagnostic",
    "paired_cluster_bootstrap",
    "probabilities_for_variant",
    "ranking_metrics",
    "reliability_bins",
    "run_audit",
    "score_sets",
    "selective_curves",
    "selective_metrics",
    "target_local_index",
    "top1_delta_vs_k5",
]

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------
#: The primary candidate-set sizes of the audit.
DEFAULT_KS: Tuple[int, ...] = (5, 10, 20, 50)
#: Evaluation splits, in reporting order (val cut by image, then the frozen tests).
PRIMARY_SPLITS: Tuple[str, ...] = ("val_select", "val_calib", "testA", "testB")
#: Only the random construction regime is executed by this audit.
REGIME_RANDOM = "random"
#: Variant names; ``REPORT_VARIANTS`` is the report column order.
VARIANT_NATIVE = "native"
VARIANT_T1 = "T1"
VARIANT_GLOBAL = "global_T"
REPORT_VARIANTS: Tuple[str, ...] = (VARIANT_NATIVE, VARIANT_T1, VARIANT_GLOBAL)
#: The only Ks a *legal* (non-oracle) global temperature may be fitted on.
LEGAL_CALIBRATION_KS: Tuple[int, ...] = (5, 10)
#: Equal-width reliability bins (protocol section 24 / metric contract).
RELIABILITY_BINS = 15
#: Coarse common confidence bins of the calibration-map-shift table.
SHIFT_BIN_EDGES: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#: Bins with fewer samples get an ``unreliable`` flag in the shift table.
MIN_RELIABLE_SHIFT_N = 30
#: Coverage levels reported by the selective metrics.
RISK_COVERAGE_LEVELS: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
#: |‖v‖_2 - 1| tolerated before a vector is counted as renormalised.
L2_TOLERANCE = 1e-2
#: Name of the STOP artifact written when a hard sanity check fails.
FAILURE_FILENAME = "VALIDATION_FAILURE.json"
#: Sub-directory names of the artifact set.
RAW_PREDICTIONS_DIRNAME = "raw_predictions"
CANDIDATE_MANIFESTS_DIRNAME = "candidate_manifests"
FIGURES_DIRNAME = "figures"

#: eval split -> manifest *file* split it lives in.  val_select / val_calib are
#: derived from the single ``val`` manifest by its ``eval_split`` column.
_FILE_SPLIT_OF: Mapping[str, str] = {
    "val_select": "val",
    "val_calib": "val",
    "testA": "testA",
    "testB": "testB",
}
#: split code stored in the npz artifacts (int8).
SPLIT_CODES: Mapping[str, int] = {name: pos for pos, name in enumerate(PRIMARY_SPLITS)}


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------
class CalibrationIsolationError(ValueError):
    """Raised when something other than val_calib data reaches a calibration fit.

    Protocol section 9: temperature (and every future calibrator) is fitted on
    ``val_calib`` only; val_select is for model selection and testA / testB are
    frozen.  This is deliberately an error, never a warning.
    """


class ValidationFailure(RuntimeError):
    """A hard sanity check (protocol sections 17-19) failed; the audit must STOP.

    Carries the check name and the (bounded) list of concrete violations so the
    caller can write ``VALIDATION_FAILURE.json`` before aborting.
    """

    def __init__(self, check: str, message: str, violations: Sequence[Mapping[str, Any]]):
        super().__init__(f"[{check}] {message}")
        self.check = str(check)
        self.message = str(message)
        self.violations = [dict(item) for item in violations]


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SentenceRecord:
    """One evaluated sentence and its (ref-shared) frozen candidate manifest.

    Attributes
    ----------
    sentence_id:
        Archive-wide ``sent_id`` (see module docstring for the frozen order).
    ref_id / image_id / eval_split:
        Identity of the expression and the split it is evaluated on.
    target_index:
        Bank row of the target, or ``None`` for a natural miss (such rows are
        skipped per K and counted in ``ScoreBundle.n_skipped_target_missing``).
    distractor_order:
        ``int32`` frozen ordering shared by every K of the ref.
    target_max_iou:
        ``max_i IoU(p_i, b*)`` of the expression (< 0.5 on a miss).
    eligible:
        ``K -> bool`` copied from the manifest (present only for the Ks the
        manifest was built for).
    """

    sentence_id: int
    ref_id: int
    image_id: int
    eval_split: str
    target_index: Optional[int]
    distractor_order: np.ndarray
    target_max_iou: float
    eligible: Mapping[int, bool] = field(default_factory=dict)


@dataclass
class ScoreBundle:
    """Per-K FP32 scores, candidate indices and target positions of an audit.

    Row ``i`` of every ``[n_K, K]`` array belongs to ``records_by_k[K][i]``
    (input order of :func:`score_sets`, i.e. sentence_id-ascending per split).
    """

    ks: Tuple[int, ...]
    records_by_k: Dict[int, List[SentenceRecord]]
    raw_scores: Dict[int, np.ndarray]
    candidate_indices: Dict[int, np.ndarray]
    target_local: Dict[int, np.ndarray]
    n_skipped_target_missing: Dict[int, int]
    n_skipped_insufficient: Dict[int, int]
    n_l2_renormalized: int = 0

    # -- small accessors (kept cheap; ``n`` is a few 10^4 rows) --------------
    def n_rows(self, k: int) -> int:
        """Number of scored rows for candidate size ``k``."""
        return int(self.raw_scores[int(k)].shape[0])

    def sentence_ids(self, k: int) -> np.ndarray:
        """``int64[n]`` sentence ids of K's rows, in row order."""
        return np.asarray([r.sentence_id for r in self.records_by_k[int(k)]], dtype=np.int64)

    def ref_ids(self, k: int) -> np.ndarray:
        """``int64[n]`` ref ids of K's rows (a ref repeats per sentence)."""
        return np.asarray([r.ref_id for r in self.records_by_k[int(k)]], dtype=np.int64)

    def image_ids(self, k: int) -> np.ndarray:
        """``int64[n]`` image ids of K's rows (bootstrap clusters)."""
        return np.asarray([r.image_id for r in self.records_by_k[int(k)]], dtype=np.int64)

    def splits(self, k: int) -> np.ndarray:
        """``<U[n]`` eval split labels of K's rows, in row order."""
        return np.asarray([r.eval_split for r in self.records_by_k[int(k)]], dtype="<U10")

    def rows_for_split(self, k: int, split: str) -> np.ndarray:
        """Row indices of K whose eval split equals ``split``."""
        return np.nonzero(self.splits(k) == str(split))[0]


@dataclass(frozen=True)
class TemperatureFit:
    """One fitted scalar temperature plus its provenance (protocol section 22)."""

    temperature: float
    ks: Tuple[int, ...]
    split: str
    n_sets: int
    n_valid_sets: int
    nll_before: float
    nll_after: float
    bounds: Tuple[float, float]
    kind: str = "global"  # "global" (legal) or "oracle" (diagnostic only)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "temperature": float(self.temperature),
            "ks": [int(k) for k in self.ks],
            "split": self.split,
            "n_sets": int(self.n_sets),
            "n_valid_sets": int(self.n_valid_sets),
            "nll_before": float(self.nll_before),
            "nll_after": float(self.nll_after),
            "bounds": [float(self.bounds[0]), float(self.bounds[1])],
        }


@dataclass(frozen=True, eq=False)
class SampleStats:
    """Aligned per-sample confidence / correctness vectors of one metric cell.

    ``confidence`` is the top-label probability (max softmax) and ``correct``
    the 0/1 indicator ``1[predicted == target]``.  Every paired-bootstrap
    ``metric_fn`` consumes a :class:`SampleStats` (resampled rows included).
    """

    confidence: np.ndarray
    correct: np.ndarray

    @classmethod
    def from_conf_correct(cls, confidence: Any, correct: Any) -> "SampleStats":
        conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
        corr = np.asarray(correct, dtype=np.float64).reshape(-1)
        if conf.shape[0] != corr.shape[0]:
            raise ValueError(
                f"confidence ({conf.shape[0]}) and correct ({corr.shape[0]}) differ in length"
            )
        return cls(confidence=conf, correct=corr)

    def take(self, rows: np.ndarray) -> "SampleStats":
        """A new stats object restricted to (possibly repeated) row indices."""
        return SampleStats(confidence=self.confidence[rows], correct=self.correct[rows])

    def __len__(self) -> int:
        return int(self.confidence.shape[0])


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    """JSON encoder fallback for numpy scalars / arrays."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value {value!r} is not JSON serialisable")


def _as_unit_rows(matrix: Any, name: str, *, tolerance: float = L2_TOLERANCE) -> Tuple[np.ndarray, int]:
    """FP16 (or any) -> float32, then L2-normalise every row/vector.

    Returns ``(float32 array, n_out_of_tolerance)`` where the second value
    counts vectors whose stored norm deviated from 1 by more than
    ``tolerance`` (diagnostic only - they are renormalised like everyone else).
    Zero-norm vectors and non-finite values raise: silently scoring them would
    corrupt every downstream metric.
    """
    arr = np.asarray(matrix)
    if arr.dtype != np.float32:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: contains non-finite (nan/inf) values")
    norms = np.linalg.norm(arr, axis=-1)
    zero = norms <= 0.0
    if np.any(zero):
        raise ValueError(f"{name}: contains zero-norm vector(s); cannot normalise")
    out_of_tol = int(np.count_nonzero(np.abs(norms - 1.0) > float(tolerance)))
    if arr.ndim == 1:
        return (arr / np.float32(norms)).astype(np.float32, copy=False), out_of_tol
    return arr / norms[..., None].astype(np.float32, copy=False), out_of_tol


def target_local_index(candidate_indices: Any, target_index: Any) -> Any:
    """Position of ``target_index`` inside ``candidate_indices`` (array search).

    ``C_K`` is *constructed* with the target first, but nothing may rely on that
    layout: this helper finds the position by value.  Supports one candidate
    list (``[K]`` + scalar -> ``int``) and a batch (``[n, K]`` + ``[n]`` or
    scalar -> ``int64[n]``); a missing target raises :class:`ValueError`.
    """
    cand = np.asarray(candidate_indices)
    if cand.ndim == 1:
        hits = np.nonzero(cand == int(target_index))[0]
        if hits.size == 0:
            raise ValueError(f"target {target_index} not found in candidate indices {cand.tolist()}")
        return int(hits[0])
    if cand.ndim == 2:
        targets = np.asarray(target_index, dtype=np.int64)
        if targets.ndim == 0:
            targets = np.full(cand.shape[0], int(targets), dtype=np.int64)
        if targets.shape[0] != cand.shape[0]:
            raise ValueError(
                f"candidate_indices rows ({cand.shape[0]}) and target_index ({targets.shape[0]}) differ"
            )
        found = np.full(cand.shape[0], -1, dtype=np.int64)
        for row in range(cand.shape[0]):
            hits = np.nonzero(cand[row] == targets[row])[0]
            if hits.size == 0:
                raise ValueError(f"row {row}: target {targets[row]} not found in its candidate indices")
            found[row] = int(hits[0])
        return found
    raise ValueError(f"candidate_indices must be 1-D or 2-D, got shape {cand.shape}")


def compute_candidate_scores(
    text_vec: np.ndarray,
    region_rows: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    K: Optional[int] = None,
) -> np.ndarray:
    """Score the requested candidate rows against one sentence query (FP32).

    Module-level hook: the whole audit calls this function through the module
    namespace, so a test can monkeypatch ``phase0a.compute_candidate_scores``
    to inject a deliberate inconsistency (the injection test does exactly
    that).

    Default implementation: **one** FP32 gemv over the *full* region matrix of
    the image, then a gather of ``candidate_indices``.  Scoring the full matrix
    in a single call - instead of slicing the rows per K - is what makes the
    per-K scores of shared candidates bit-identical (the hard score-invariance
    check runs with ``atol=0``; BLAS gemv results are *not* bit-stable across
    different row counts, so per-K slicing cannot be used).

    Parameters
    ----------
    text_vec:
        ``[D]`` float32 normalised query (the sentence's text feature).
    region_rows:
        ``[N, D]`` float32 normalised region features of one image.
    candidate_indices:
        ``[K]`` int bank rows of ``C_K``.
    K:
        Candidate-set size (informational; ``len(candidate_indices)`` is the
        authority).  Passed through so injected hooks can target one K.

    Returns
    -------
    np.ndarray
        ``float32[K]`` cosine scores, ``s[i] = <text_vec, region_rows[cand[i]]>``.
    """
    query = np.asarray(text_vec, dtype=np.float32)
    rows = np.asarray(region_rows, dtype=np.float32)
    idx = np.asarray(candidate_indices, dtype=np.int64)
    scores_full = rows @ query  # one call, one shape -> deterministic
    return scores_full[idx].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# manifest / refs loading
# ---------------------------------------------------------------------------
def _normalize_split_filter(split_filter: Any) -> List[str]:
    """Validate and de-duplicate an eval-split filter (order = PRIMARY_SPLITS)."""
    if isinstance(split_filter, str):
        names = [split_filter]
    else:
        names = [str(item) for item in split_filter]
    requested: List[str] = []
    for name in names:
        if name not in PRIMARY_SPLITS:
            raise ValueError(
                f"unknown eval split {name!r}; expected one of {PRIMARY_SPLITS} "
                "(val_select / val_calib are derived from the val manifest)"
            )
        if name not in requested:
            requested.append(name)
    if not requested:
        raise ValueError("split_filter is empty")
    return [name for name in PRIMARY_SPLITS if name in requested]


def _manifest_file_for(manifests_root: Any, regime: str, file_split: str) -> Path:
    """Locate ``{regime}_{file_split}.jsonl`` under ``manifests_root``.

    The frozen layout is ``<root>/manifests/{regime}_{split}.jsonl``
    (:func:`ccg.data.manifests.manifest_path`); callers that already point at
    the ``manifests/`` directory itself are accepted as well, then a bounded
    glob fallback.  Nothing is guessed: an absent file raises.
    """
    root = Path(manifests_root)
    name = f"{regime}_{file_split}.jsonl"
    candidates = [manifest_path(root, regime, file_split), root / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    hits: List[Path] = []
    for pattern in (f"**/manifests/{name}", f"**/{name}"):
        hits.extend(sorted(root.glob(pattern)))
        if hits:
            break
    for hit in hits:
        if hit.exists():
            return hit
    raise FileNotFoundError(
        f"manifest file {name!r} not found under {root} "
        f"(tried {candidates[0]} and {candidates[1]}); build the {regime!r} "
        f"manifests under {root}/manifests/ first"
    )


def _sentence_index_from_refs(refs: Any) -> Dict[int, List[Tuple[int, int]]]:
    """``ref_id -> [(sentence_id, image_id), ...]`` with global-id validation.

    Accepts the UNC pickle path or already-loaded records.  The archive-wide
    ``sent_id`` contract of the module docstring is *validated*, not assumed:
    every sentence id must be unique across the archive.
    """
    if isinstance(refs, (str, Path)):
        records: Iterable[Mapping[str, Any]] = load_refs_pickle(refs)
    else:
        records = refs
    index: Dict[int, List[Tuple[int, int]]] = {}
    seen: Dict[int, int] = {}
    for record in records:
        for required in ("ref_id", "image_id"):
            if record.get(required) is None:
                raise ValueError(f"ref record misses {required!r}: keys={sorted(record)}")
        ref_id = int(record["ref_id"])
        image_id = int(record["image_id"])
        if ref_id in index:
            raise ValueError(f"duplicate ref_id {ref_id} in refs records")
        sent_ids = record.get("sent_ids")
        if sent_ids is None:
            sentences = record.get("sentences") or []
            sent_ids = [sentence.get("sent_id") for sentence in sentences]
            if any(value is None for value in sent_ids):
                raise ValueError(f"ref {ref_id}: sentences carry no sent_id")
        ids = [int(value) for value in sent_ids]
        if not ids:
            raise ValueError(f"ref {ref_id}: no sentences")
        entries: List[Tuple[int, int]] = []
        for sent_id in ids:
            previous = seen.get(sent_id)
            if previous is not None:
                raise ValueError(
                    f"sent_id {sent_id} is not archive-unique: refs {previous} and {ref_id}"
                )
            seen[sent_id] = ref_id
            entries.append((sent_id, image_id))
        entries.sort(key=lambda pair: pair[0])  # in-ref sent_id ascending
        index[ref_id] = entries
    if not index:
        raise ValueError("refs is empty - nothing to evaluate")
    return index


def _cache_sentence_count(features_root: Any) -> Optional[int]:
    """``n_sentences`` from the feature cache metadata, or ``None`` when absent."""
    path = Path(features_root) / "metadata.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get("n_sentences")
    return None if value is None else int(value)


def load_cosine_inputs(
    features_root: Any,
    manifests_root: Any,
    refs: Any,
    split_filter: Any,
    *,
    regime: str = REGIME_RANDOM,
    ks: Sequence[int] = DEFAULT_KS,
) -> List[SentenceRecord]:
    """Sentence-level evaluation records for the requested eval splits.

    One :class:`SentenceRecord` per sentence of every manifest entry in
    ``split_filter``: the candidate manifest is shared by all sentences of a
    ref; ids follow the frozen archive order (module docstring).  val_select /
    val_calib rows come from the single ``val`` manifest filtered by its
    ``eval_split`` column.

    Parameters
    ----------
    features_root:
        Feature cache root (``cache/features``).  Used to re-validate that the
        cache covers every ``sentence_id`` (``metadata.json["n_sentences"]``);
        a cache without metadata is accepted silently.
    manifests_root:
        Root that contains ``manifests/{regime}_{split}.jsonl`` (or the
        ``manifests`` directory itself).
    refs:
        Path to ``refs(unc).p`` or the loaded records.
    split_filter:
        One eval split or an iterable, each in :data:`PRIMARY_SPLITS`.
    regime:
        Construction regime; frozen to ``"random"`` for this audit.
    ks:
        Primary Ks (used for a validation echo in the returned ordering of
        ``eligible``; the candidate sets themselves come from the manifest).

    Returns
    -------
    list[SentenceRecord]
        Sorted by (split filter order, sentence_id ascending).
    """
    if str(regime) != REGIME_RANDOM:
        raise ValueError(
            f"regime {regime!r} is not supported by the Phase 0A cosine audit; "
            f"only {REGIME_RANDOM!r} (same-category is a separate experiment, CLIP-hard is forbidden)"
        )
    split_names = _normalize_split_filter(split_filter)
    index = _sentence_index_from_refs(refs)
    n_sentences_cache = _cache_sentence_count(features_root)
    max_sentence_id = max(sid for entries in index.values() for sid, _ in entries)
    if n_sentences_cache is not None and max_sentence_id >= n_sentences_cache:
        raise ValueError(
            f"cache covers {n_sentences_cache} sentences but refs reach sentence_id "
            f"{max_sentence_id}; the feature cache and the refs archive are out of sync"
        )

    records: List[SentenceRecord] = []
    for split in split_names:
        file_split = _FILE_SPLIT_OF[split]
        path = _manifest_file_for(manifests_root, regime, file_split)
        manifest = ManifestFile.load(path)
        entries = filter_entries(manifest, split)
        if not entries:
            # an empty cohort that *should* exist is a bug, not a quiet corner
            raise ValueError(f"manifest {path} holds no entries for split {split!r}")
        for entry in entries:
            sentences = index.get(int(entry.ref_id))
            if sentences is None:
                raise ValueError(
                    f"manifest entry ref {entry.ref_id} ({path}) has no matching refs record"
                )
            for sentence_id, image_id in sentences:
                if int(image_id) != int(entry.image_id):
                    raise ValueError(
                        f"ref {entry.ref_id}: manifest says image {entry.image_id}, "
                        f"refs archive says {image_id}"
                    )
                records.append(
                    SentenceRecord(
                        sentence_id=int(sentence_id),
                        ref_id=int(entry.ref_id),
                        image_id=int(image_id),
                        eval_split=split,
                        target_index=entry.target_index,
                        distractor_order=np.asarray(entry.distractor_order, dtype=np.int32),
                        target_max_iou=float(entry.target_max_iou),
                        eligible={int(k): bool(v) for k, v in entry.eligible.items()},
                    )
                )
    order = {name: pos for pos, name in enumerate(split_names)}
    records.sort(key=lambda rec: (order[rec.eval_split], rec.sentence_id))
    return records


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_sets(
    cache: Any,
    records: Sequence[SentenceRecord],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    compute_fn: Optional[Callable[..., np.ndarray]] = None,
) -> ScoreBundle:
    """Assemble ``C_K`` per sentence and score every candidate set in FP32.

    For each sentence and every ``K`` the candidate set is
    ``C_K = [target] + distractor_order[:K-1]``; a sentence whose target is a
    natural miss is skipped for every K (counted), a sentence whose stored
    ordering is shorter than ``K-1`` is skipped for that K only (counted).
    Rows keep the input order of ``records`` in every per-K array.

    The cache is duck-typed: anything exposing ``region_features(image_id) ->
    [N, D]`` and ``text_features(sentence_id) -> [D]`` (float16 or float32)
    works - the real :class:`ccg.features.cache.FeatureCache` and the tiny
    synthetic caches used in tests alike.

    ``compute_fn`` defaults to the module-level :func:`compute_candidate_scores`
    (looked up at call time, so monkeypatching the module attribute works).

    Returns
    -------
    ScoreBundle
        FP32 ``raw_scores[K]``/``candidate_indices[K]``/``target_local[K]`` and
        the per-K skip counters; ``n_l2_renormalized`` counts vectors whose
        stored L2 norm deviated from 1 by more than :data:`L2_TOLERANCE`.
    """
    record_list = list(records)
    Ks = tuple(sorted({int(k) for k in ks}))
    if not Ks:
        raise ValueError("ks is empty")
    if any(k < 2 for k in Ks):
        raise ValueError(f"every K must be >= 2 (candidate set includes the target), got {Ks}")

    member_rows: Dict[int, List[int]] = {k: [] for k in Ks}
    row_of_record: Dict[int, np.ndarray] = {
        k: np.full(len(record_list), -1, dtype=np.int64) for k in Ks
    }
    skipped_target: Dict[int, int] = {k: 0 for k in Ks}
    skipped_insufficient: Dict[int, int] = {k: 0 for k in Ks}

    for pos, record in enumerate(record_list):
        order_size = int(np.asarray(record.distractor_order).size)
        for k in Ks:
            if record.target_index is None:
                skipped_target[k] += 1
                continue
            if order_size < k - 1:
                skipped_insufficient[k] += 1
                continue
            row_of_record[k][pos] = len(member_rows[k])
            member_rows[k].append(pos)

    raw_scores: Dict[int, np.ndarray] = {}
    candidate_indices: Dict[int, np.ndarray] = {}
    target_local: Dict[int, np.ndarray] = {}
    for k in Ks:
        n = len(member_rows[k])
        raw_scores[k] = np.empty((n, k), dtype=np.float32)
        candidate_indices[k] = np.empty((n, k), dtype=np.int32)
        target_local[k] = np.empty(n, dtype=np.int32)

    # group by image so the region matrix (and its normalisation) is read once
    by_image: Dict[int, List[int]] = {}
    for pos, record in enumerate(record_list):
        if any(row_of_record[k][pos] >= 0 for k in Ks):
            by_image.setdefault(int(record.image_id), []).append(pos)

    compute = compute_candidate_scores if compute_fn is None else compute_fn
    n_l2_bad = 0
    for image_id, positions in by_image.items():
        region = cache.region_features(int(image_id))
        region32, bad_rows = _as_unit_rows(region, f"region_features[{image_id}]")
        n_l2_bad += bad_rows
        for pos in positions:
            record = record_list[pos]
            text = cache.text_features(record.sentence_id)
            vec32, bad_vec = _as_unit_rows(text, f"text_features[{record.sentence_id}]")
            n_l2_bad += bad_vec
            order = np.asarray(record.distractor_order, dtype=np.int64)
            for k in Ks:
                row = int(row_of_record[k][pos])
                if row < 0:
                    continue
                cand = np.empty(k, dtype=np.int64)
                cand[0] = int(record.target_index)  # C_K = [target] + prefix
                cand[1:] = order[: k - 1]
                scores = np.asarray(
                    compute(vec32, region32, cand, K=k), dtype=np.float32
                ).reshape(-1)
                if scores.shape[0] != k:
                    raise ValueError(
                        f"compute hook returned {scores.shape[0]} scores for K={k} of "
                        f"sentence {record.sentence_id}"
                    )
                if not np.all(np.isfinite(scores)):
                    raise ValueError(
                        f"non-finite score for sentence {record.sentence_id}, K={k}"
                    )
                # the local position is *searched*, never assumed to be 0
                local = target_local_index(cand, int(record.target_index))
                raw_scores[k][row] = scores
                candidate_indices[k][row] = cand.astype(np.int32, copy=False)
                target_local[k][row] = int(local)

    records_by_k = {k: [record_list[pos] for pos in member_rows[k]] for k in Ks}
    return ScoreBundle(
        ks=Ks,
        records_by_k=records_by_k,
        raw_scores=raw_scores,
        candidate_indices=candidate_indices,
        target_local=target_local,
        n_skipped_target_missing=skipped_target,
        n_skipped_insufficient=skipped_insufficient,
        n_l2_renormalized=int(n_l2_bad),
    )


# ---------------------------------------------------------------------------
# common cohort
# ---------------------------------------------------------------------------
def common_cohort_rows(
    bundle: ScoreBundle, ks: Optional[Sequence[int]] = None
) -> Tuple[Tuple[int, ...], Dict[int, np.ndarray], np.ndarray]:
    """Rows of the sentences available for **every** K (common cohort).

    Returns ``(Ks, {K: int64 row indices}, common_sentence_ids)`` where the row
    arrays index ``bundle.raw_scores[K]`` and all share one denominator.  The
    intersection is taken on ``sentence_id`` (a ref is either eligible for
    every K or for none, but the sentence id is what the two arrays store).
    """
    Ks = tuple(sorted({int(k) for k in (bundle.ks if ks is None else ks)}))
    for k in Ks:
        if k not in bundle.raw_scores:
            raise KeyError(f"K={k} is not part of this bundle: {bundle.ks}")
    if not Ks:
        raise ValueError("ks is empty")

    sorted_ids: Dict[int, np.ndarray] = {}
    orders: Dict[int, np.ndarray] = {}
    for k in Ks:
        ids = bundle.sentence_ids(k)
        order = np.argsort(ids, kind="stable")
        sorted_ids[k] = ids[order]
        orders[k] = order
        if np.unique(sorted_ids[k]).size != sorted_ids[k].size:
            raise ValueError(f"K={k}: sentence_ids are not unique")

    common = sorted_ids[Ks[0]]
    for k in Ks[1:]:
        common = np.intersect1d(common, sorted_ids[k], assume_unique=True)
    rows: Dict[int, np.ndarray] = {}
    for k in Ks:
        positions = np.searchsorted(sorted_ids[k], common)
        if not np.array_equal(sorted_ids[k][positions], common):
            raise RuntimeError(f"K={k}: common-cohort alignment failed")
        rows[k] = orders[k][positions]
    return Ks, rows, common


# (section 2 continues below)


# ---------------------------------------------------------------------------
# probability variants (protocol section 11)
# ---------------------------------------------------------------------------
def _variant_temperature(
    variant: str,
    *,
    logit_scale: Optional[float] = None,
    temperature: Optional[float] = None,
) -> float:
    """Softmax temperature expressing each variant as ``softmax(s / T)``.

    ``native``  -> ``T = 1 / logit_scale`` (because ``softmax(alpha * s)``),
    ``T1``      -> ``T = 1``,
    ``global_T``-> ``T =`` the fitted global temperature.
    """
    name = str(variant)
    if name == VARIANT_NATIVE:
        if logit_scale is None:
            raise ValueError(
                "the native variant needs `logit_scale` (metadata['native_logit_scale'])"
            )
        scale = float(logit_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"native_logit_scale must be positive and finite, got {logit_scale}")
        return 1.0 / scale
    if name == VARIANT_T1:
        return 1.0
    if name == VARIANT_GLOBAL:
        if temperature is None:
            raise ValueError("the global_T variant needs a fitted `temperature`")
        value = float(temperature)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"global temperature must be positive and finite, got {temperature}")
        return value
    raise ValueError(f"unknown probability variant {variant!r}; expected one of {REPORT_VARIANTS}")


def probabilities_for_variant(
    raw_scores: Any,
    variant: str,
    *,
    logit_scale: Optional[float] = None,
    temperature: Optional[float] = None,
) -> np.ndarray:
    """Row-wise probabilities of one variant for an ``[n, K]`` raw-score block.

    All three variants are plain temperature rescalings of the same FP32 cosine
    scores (module docstring); this function is the single place they are
    materialised, so every table and the npz artifacts agree bit-for-bit.
    """
    temp = _variant_temperature(variant, logit_scale=logit_scale, temperature=temperature)
    return softmax_scores(raw_scores, temperature=temp).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# row selection helpers
# ---------------------------------------------------------------------------
def _pooled_name() -> str:
    return "__pooled__"


def _split_names_of(bundle: ScoreBundle, splits: Optional[Sequence[str]]) -> List[str]:
    """Eval splits to report on: ``splits`` validated against the bundle's rows."""
    present = set()
    for k in bundle.ks:
        present.update(str(item) for item in np.unique(bundle.splits(k)).tolist())
    if splits is None:
        names = [name for name in PRIMARY_SPLITS if name in present]
    else:
        names = []
        for name in splits:
            if name not in PRIMARY_SPLITS:
                raise ValueError(f"unknown eval split {name!r}; expected one of {PRIMARY_SPLITS}")
            if name in present:
                names.append(name)
    if not names:
        raise ValueError("no eval split of the bundle matches the requested filter")
    return names


def _ks_of(bundle: ScoreBundle, ks: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if ks is None:
        return tuple(bundle.ks)
    names = tuple(sorted({int(k) for k in ks}))
    for k in names:
        if k not in bundle.raw_scores:
            raise KeyError(f"K={k} is not part of this bundle: {bundle.ks}")
    if not names:
        raise ValueError("ks is empty")
    return names


def _correctness(bundle: ScoreBundle, k: int, rows: np.ndarray) -> np.ndarray:
    """``[m]`` float64 correctness indicators at raw-score argmax (tie -> lower index).

    The argmax is invariant under every monotone temperature rescaling, so one
    correctness vector serves all three probability variants (they differ in
    *confidence*, never in the predicted candidate).
    """
    predicted = np.argmax(bundle.raw_scores[int(k)][rows], axis=1)
    return (predicted == bundle.target_local[int(k)][rows]).astype(np.float64)


# ---------------------------------------------------------------------------
# temperature fitting (protocol sections 22-23)
# ---------------------------------------------------------------------------
def _fit_temperature_on_rows(
    bundle: ScoreBundle,
    k: int,
    rows: np.ndarray,
    *,
    bounds: Tuple[float, float],
    kind: str,
) -> TemperatureFit:
    """Fit one scalar temperature on the given rows of candidate size ``k``."""
    if rows.size == 0:
        raise ValueError(f"K={k}: no rows to fit a temperature on")
    logits = bundle.raw_scores[int(k)][rows]
    targets = bundle.target_local[int(k)][rows]
    sets = [row for row in logits]  # list of [k] float32 arrays (mixed Ks stay separate)
    label_list = [int(value) for value in targets.tolist()]
    temperature = fit_temperature(sets, label_list, bounds=bounds)
    nll_before = float(np.mean([_row_nll(row, label, 1.0) for row, label in zip(sets, label_list)]))
    nll_after = float(
        np.mean([_row_nll(row, label, temperature) for row, label in zip(sets, label_list)])
    )
    return TemperatureFit(
        temperature=float(temperature),
        ks=(int(k),),
        split="val_calib",
        n_sets=int(rows.size),
        n_valid_sets=int(rows.size),
        nll_before=nll_before,
        nll_after=nll_after,
        bounds=(float(bounds[0]), float(bounds[1])),
        kind=str(kind),
    )


def _row_nll(logits: np.ndarray, target: int, temperature: float) -> float:
    """``-log softmax(logits / T)[target]`` of one candidate set (float64)."""
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - scaled.max()
    log_sum = float(np.log(np.exp(shifted).sum()))
    return float(-(shifted[int(target)] - log_sum))


def fit_global_temperature(
    bundle: ScoreBundle,
    *,
    ks: Sequence[int] = LEGAL_CALIBRATION_KS,
    bounds: Tuple[float, float] = DEFAULT_TEMP_BOUNDS,
    rows_by_k: Optional[Mapping[int, np.ndarray]] = None,
) -> TemperatureFit:
    """Fit **one** global temperature on val_calib, pooled over ``K in {5, 10}``.

    Protocol section 22: a single ``T*`` is shared by every K - per-K
    temperatures exist only in the oracle diagnostic and are never reported as
    a legal model result.  The fit uses the list-of-sets form of
    :func:`ccg.calibration.fit_temperature` so that K=5 and K=10 sets (widths 5
    and 10) are pooled without being reshaped.

    Calibration isolation (hard requirement): every row that enters the fit
    must carry ``eval_split == "val_calib"``.  The default (``rows_by_k=None``)
    uses *all* rows of the involved Ks, so passing a bundle that contains any
    val_select / testA / testB rows raises :class:`CalibrationIsolationError` -
    the caller must hand over an explicitly val_calib-only view.  There is no
    switch that turns this check off.
    """
    ks_t = tuple(sorted({int(k) for k in ks}))
    if not set(ks_t) <= set(LEGAL_CALIBRATION_KS):
        raise ValueError(
            f"a legal global temperature is fitted on K in {LEGAL_CALIBRATION_KS} only; "
            f"got K={list(ks_t)} (per-K fitting belongs to the oracle diagnostic)"
        )
    if not ks_t:
        raise ValueError("ks is empty")
    for k in ks_t:
        if k not in bundle.raw_scores:
            raise KeyError(f"K={k} is not part of this bundle: {bundle.ks}")

    if rows_by_k is None:
        rows: Dict[int, np.ndarray] = {
            k: np.arange(bundle.n_rows(k), dtype=np.int64) for k in ks_t
        }
    else:
        rows = {}
        for k in ks_t:
            if k not in rows_by_k:
                raise KeyError(f"rows_by_k misses K={k}")
            rows[k] = np.asarray(rows_by_k[k], dtype=np.int64)

    # calibration isolation: assert on exactly the rows that enter the fit
    for k in ks_t:
        splits = bundle.splits(k)[rows[k]]
        offenders = sorted({str(item) for item in np.unique(splits).tolist() if item != "val_calib"})
        if offenders:
            raise CalibrationIsolationError(
                f"fit_global_temperature may only see val_calib rows, but K={k} rows carry "
                f"eval splits {offenders}; testA/testB never participate in fitting "
                "(protocol section 9). Restrict the input to val_calib first."
            )

    sets: List[np.ndarray] = []
    labels: List[int] = []
    for k in ks_t:
        for row in bundle.raw_scores[k][rows[k]]:
            sets.append(row)
        labels.extend(int(value) for value in bundle.target_local[k][rows[k]].tolist())
    n_sets = len(sets)
    if n_sets == 0:
        raise ValueError("no candidate sets to fit the global temperature on (empty val_calib)")
    temperature = fit_temperature(sets, labels, bounds=bounds)
    nll_before = float(np.mean([_row_nll(s_, t_, 1.0) for s_, t_ in zip(sets, labels)]))
    nll_after = float(np.mean([_row_nll(s_, t_, temperature) for s_, t_ in zip(sets, labels)]))
    return TemperatureFit(
        temperature=float(temperature),
        ks=ks_t,
        split="val_calib",
        n_sets=int(n_sets),
        n_valid_sets=int(n_sets),
        nll_before=nll_before,
        nll_after=nll_after,
        bounds=(float(bounds[0]), float(bounds[1])),
        kind="global",
    )


def oracle_temperature_diagnostic(
    bundle: ScoreBundle,
    global_fit: Optional[TemperatureFit],
    *,
    ks: Optional[Sequence[int]] = None,
    bounds: Tuple[float, float] = DEFAULT_TEMP_BOUNDS,
    splits: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Per-K oracle temperatures on val_calib - **DIAGNOSTIC ONLY** (section 23).

    For every K one temperature ``T_K`` is fitted on the val_calib rows of that
    K (never pooled across Ks, never fitted on val_select/test splits), and the
    adaptive ECE of ``softmax(s / T_K)`` is compared with the adaptive ECE of
    ``softmax(s / T*)`` (the legal global temperature) on every eval split of
    the common cohort.

    This is an **oracle / diagnostic** table: ``T_K`` is fitted on the same
    val_calib data it is scored on and it is *illegal* as a model result.  It
    exists to answer one question: would per-K temperatures visibly repair the
    calibration drift across K that the audit measures?  The returned dict
    carries that disclaimer verbatim in ``"status"``.
    """
    ks_t = _ks_of(bundle, ks)
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)

    oracle_fits: Dict[str, Any] = {}
    oracle_temperatures: Dict[str, float] = {}
    for k in ks_t:
        val_rows = bundle.rows_for_split(k, "val_calib")
        if val_rows.size == 0:
            oracle_fits[str(k)] = {"status": "not_fitted", "reason": "no val_calib rows"}
            continue
        fit = _fit_temperature_on_rows(bundle, k, val_rows, bounds=bounds, kind="oracle")
        oracle_fits[str(k)] = fit.to_dict()
        oracle_temperatures[str(k)] = float(fit.temperature)

    comparisons: List[Dict[str, Any]] = []
    deltas: List[float] = []
    for name in list(split_names) + [_pooled_name()]:
        for k in ks_t:
            if name == _pooled_name():
                rows = common_rows[k]
            else:
                mask = bundle.splits(k)[common_rows[k]] == name
                rows = common_rows[k][mask]
            if rows.size == 0:
                continue
            correct = _correctness(bundle, k, rows)
            entry: Dict[str, Any] = {
                "eval_split": name,
                "K": int(k),
                "n": int(rows.size),
                "ece_global_T": None,
                "ece_oracle_T_K": None,
                "delta_oracle_minus_global": None,
            }
            if global_fit is not None:
                prob = probabilities_for_variant(
                    bundle.raw_scores[k][rows],
                    VARIANT_GLOBAL,
                    temperature=global_fit.temperature,
                )
                entry["ece_global_T"] = float(
                    top_label_ece_adaptive(prob.max(axis=1), correct, n_bins=RELIABILITY_BINS)
                )
            if str(k) in oracle_temperatures:
                prob = probabilities_for_variant(
                    bundle.raw_scores[k][rows],
                    VARIANT_GLOBAL,
                    temperature=oracle_temperatures[str(k)],
                )
                entry["ece_oracle_T_K"] = float(
                    top_label_ece_adaptive(prob.max(axis=1), correct, n_bins=RELIABILITY_BINS)
                )
            if entry["ece_global_T"] is not None and entry["ece_oracle_T_K"] is not None:
                delta = float(entry["ece_oracle_T_K"] - entry["ece_global_T"])
                entry["delta_oracle_minus_global"] = delta
                deltas.append(delta)
            comparisons.append(entry)

    improve_threshold = -0.01  # 1 percentage point of ECE
    worse_threshold = 0.01
    n_improved = sum(1 for value in deltas if value <= improve_threshold)
    n_worse = sum(1 for value in deltas if value >= worse_threshold)
    if not deltas:
        verdict = "not_evaluable"
        verdict_text = "no cell had both a global and an oracle temperature"
    elif n_improved > 0 and n_worse == 0:
        verdict = "oracle_repairs_drift"
        verdict_text = (
            "per-K oracle temperatures reduce adaptive ECE by >=1pp on at least one cell "
            "without worsening any cell by >=1pp: the K-drift is (partially) a per-K "
            "sharpness issue - but these numbers are ORACLE and must not be reported as "
            "model performance"
        )
    elif n_improved == 0 and n_worse == 0:
        verdict = "no_meaningful_change"
        verdict_text = (
            "every cell moves by less than 1pp: a per-K temperature would not visibly "
            "repair the calibration drift"
        )
    else:
        verdict = "mixed_or_partial"
        verdict_text = (
            f"per-K temperatures improve {n_improved} cell(s) but worsen {n_worse}: "
            "the drift is not a simple per-K sharpness issue"
        )

    return {
        "status": (
            "ORACLE/DIAGNOSTIC - NOT A LEGAL MODEL RESULT: each T_K is fitted on the same "
            "val_calib rows it is evaluated on (protocol section 23); never quote these "
            "numbers as model performance"
        ),
        "oracle_temperatures": oracle_temperatures,
        "oracle_fits": oracle_fits,
        "global_temperature": None if global_fit is None else global_fit.to_dict(),
        "comparisons": comparisons,
        "verdict": {
            "rule": (
                "improved = delta <= -0.01, worse = delta >= +0.01 on adaptive ECE; "
                "oracle repairs drift when improved>0 and worse==0"
            ),
            "n_cells": len(deltas),
            "n_improved": int(n_improved),
            "n_worse": int(n_worse),
            "min_delta": None if not deltas else float(min(deltas)),
            "max_delta": None if not deltas else float(max(deltas)),
            "verdict": verdict,
            "text": verdict_text,
        },
    }


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------
def _ranking_cell(bundle: ScoreBundle, k: int, rows: np.ndarray) -> Dict[str, Any]:
    """Ranking numbers of one (K, row-set) cell (variant-invariant)."""
    scores = bundle.raw_scores[k][rows]
    targets = bundle.target_local[k][rows]
    ranks = target_rank(scores, targets)
    return {
        "n": int(rows.size),
        "n_correct": int(np.count_nonzero(ranks == 1)),
        "top1": float(top1_accuracy(scores, targets)),
        "top5": float(topk_accuracy(scores, targets, k=5)),
        "mean_target_rank": float(np.mean(ranks)),
        "mrr": float(mrr(scores, targets)),
    }


def ranking_metrics(
    bundle: ScoreBundle,
    *,
    ks: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """``ranking_metrics.csv`` rows: per split / K / denominator.

    Two denominators are always reported side by side (protocol section 29):
    ``available`` (every K on its own eligible rows) and ``common`` (the
    intersection cohort shared by all reported Ks, so that ``DeltaAcc(K;5)`` is
    a same-sample difference).  Variants are irrelevant here - softmax is
    monotone, so every variant picks the same candidate.
    """
    ks_t = _ks_of(bundle, ks)
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)

    names = list(split_names) + [_pooled_name()]
    cells: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for name in names:
        for k in ks_t:
            if name == _pooled_name():
                available = np.arange(bundle.n_rows(k), dtype=np.int64)
                common = common_rows[k]
            else:
                available = bundle.rows_for_split(k, name)
                common = common_rows[k][bundle.splits(k)[common_rows[k]] == name]
            if available.size:
                cells[(name, k, "available")] = _ranking_cell(bundle, k, available)
            if common.size:
                cells[(name, k, "common")] = _ranking_cell(bundle, k, common)

    rows: List[Dict[str, Any]] = []
    for name in names:
        base = cells.get((name, 5, "common"))
        for k in ks_t:
            for denominator in ("available", "common"):
                cell = cells.get((name, k, denominator))
                if cell is None:
                    continue
                delta = None
                if denominator == "common" and base is not None:
                    delta = float(cell["top1"] - base["top1"])
                rows.append(
                    {
                        "eval_split": name,
                        "K": int(k),
                        "denominator": denominator,
                        **cell,
                        "delta_acc_vs_k5": delta,
                    }
                )
    return rows


def top1_delta_vs_k5(
    bundle: ScoreBundle,
    *,
    ks: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[int, float]]:
    """``DeltaAcc(K; 5) = Acc(K) - Acc(5)`` on the common cohort, per split.

    Same-denominator by construction (common cohort); the pooled row summary is
    keyed ``"__pooled__"``.  ``K=5`` yields exactly ``0.0``.
    """
    ks_t = _ks_of(bundle, ks)
    if 5 not in ks_t:
        raise ValueError(f"DeltaAcc(K;5) needs K=5 among the reported Ks, got {ks_t}")
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)

    names = list(split_names) + [_pooled_name()]
    accuracies: Dict[str, Dict[int, float]] = {}
    for name in names:
        accuracies[name] = {}
        for k in ks_t:
            if name == _pooled_name():
                rows = common_rows[k]
            else:
                rows = common_rows[k][bundle.splits(k)[common_rows[k]] == name]
            if rows.size == 0:
                continue
            correct = _correctness(bundle, k, rows)
            accuracies[name][int(k)] = float(np.mean(correct))
    return {
        name: {
            k: float(value - values[5]) for k, value in values.items() if 5 in values
        }
        for name, values in accuracies.items()
    }


# ---------------------------------------------------------------------------
# calibration metrics
# ---------------------------------------------------------------------------
def _entropy_mean(prob: np.ndarray, eps: float = 1e-12) -> float:
    """Mean Shannon entropy of the predicted distributions."""
    clipped = np.clip(prob.astype(np.float64), eps, 1.0)
    return float(np.mean(-np.sum(clipped * np.log(clipped), axis=1)))


def calibration_metrics(
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float] = None,
    temperature: Optional[float] = None,
    ks: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """``calibration_metrics.csv`` rows: split x K x variant x denominator.

    Primary metric: adaptive top-label ECE (:func:`ccg.metrics.top_label_ece_adaptive`,
    15 equal-mass bins); equal-width ECE is the secondary reference.  Binary
    Brier / NLL / confidence-accuracy gap refer to the top-label *correctness*
    event; the multiclass NLL / Brier (secondary) are computed on the raw
    logits at the variant's temperature.  The ``global_T`` variant is emitted
    only when a fitted ``temperature`` is supplied.
    """
    ks_t = _ks_of(bundle, ks)
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)
    variants = [
        variant
        for variant in REPORT_VARIANTS
        if variant != VARIANT_GLOBAL or temperature is not None
    ]
    if VARIANT_GLOBAL in REPORT_VARIANTS and temperature is None:
        variants.append("__global_missing__")

    names = list(split_names) + [_pooled_name()]
    rows: List[Dict[str, Any]] = []
    for name in names:
        for k in ks_t:
            if name == _pooled_name():
                available = np.arange(bundle.n_rows(k), dtype=np.int64)
                common = common_rows[k]
            else:
                available = bundle.rows_for_split(k, name)
                common = common_rows[k][bundle.splits(k)[common_rows[k]] == name]
            for denominator, row_set in (("available", available), ("common", common)):
                if row_set.size == 0:
                    continue
                raw = bundle.raw_scores[k][row_set]
                targets = bundle.target_local[k][row_set]
                correct = _correctness(bundle, k, row_set)
                for variant in variants:
                    if variant == "__global_missing__":
                        # explicit row instead of a silent omission
                        rows.append(
                            {
                                "eval_split": name,
                                "K": int(k),
                                "variant": VARIANT_GLOBAL,
                                "denominator": denominator,
                                "n": int(row_set.size),
                                "status": "not_fitted",
                            }
                        )
                        continue
                    temp = _variant_temperature(
                        variant, logit_scale=logit_scale, temperature=temperature
                    )
                    prob = probabilities_for_variant(
                        raw, variant, logit_scale=logit_scale, temperature=temperature
                    )
                    confidence = prob.max(axis=1)
                    rows.append(
                        {
                            "eval_split": name,
                            "K": int(k),
                            "variant": variant,
                            "denominator": denominator,
                            "n": int(row_set.size),
                            "status": "ok",
                            "accuracy": float(np.mean(correct)),
                            "ece_adaptive": float(
                                top_label_ece_adaptive(
                                    confidence, correct, n_bins=RELIABILITY_BINS
                                )
                            ),
                            "ece_equal_width": float(
                                top_label_ece(confidence, correct, n_bins=RELIABILITY_BINS)
                            ),
                            "brier_binary": float(brier_binary(confidence, correct)),
                            "nll_binary": float(nll_binary(confidence, correct)),
                            "conf_acc_gap": float(confidence_accuracy_gap(confidence, correct)),
                            "multiclass_nll": float(
                                multiclass_nll(raw, targets, temperature=temp)
                            ),
                            "multiclass_brier": float(
                                multiclass_brier(raw, targets, temperature=temp)
                            ),
                            "mean_entropy": _entropy_mean(prob),
                        }
                    )
    return rows


# ---------------------------------------------------------------------------
# selective prediction (primary variant: native)
# ---------------------------------------------------------------------------
def selective_metrics(
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float] = None,
    ks: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """``selective_metrics.csv`` rows at the **native** max-softmax confidence.

    Risk is ``1 - accuracy`` of the accepted set (``ccg.metrics`` convention);
    ``risk@c`` accepts the most confident ``ceil(c * n)`` rows.  AURC is the
    trapezoidal area of the full risk-coverage curve.
    """
    if logit_scale is None:
        raise ValueError("selective metrics are reported on the native confidence; pass logit_scale")
    ks_t = _ks_of(bundle, ks)
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)

    names = list(split_names) + [_pooled_name()]
    rows: List[Dict[str, Any]] = []
    for name in names:
        for k in ks_t:
            if name == _pooled_name():
                available = np.arange(bundle.n_rows(k), dtype=np.int64)
                common = common_rows[k]
            else:
                available = bundle.rows_for_split(k, name)
                common = common_rows[k][bundle.splits(k)[common_rows[k]] == name]
            for denominator, row_set in (("available", available), ("common", common)):
                if row_set.size < 2:
                    continue
                prob = probabilities_for_variant(
                    bundle.raw_scores[k][row_set], VARIANT_NATIVE, logit_scale=logit_scale
                )
                confidence = prob.max(axis=1)
                correct = _correctness(bundle, k, row_set)
                coverage, risk = risk_coverage_curve(confidence, correct)
                entry: Dict[str, Any] = {
                    "eval_split": name,
                    "K": int(k),
                    "denominator": denominator,
                    "n": int(row_set.size),
                    "aurc": float(aurc(coverage, risk)),
                }
                for level in RISK_COVERAGE_LEVELS:
                    entry[f"risk_at_{int(round(level * 100))}"] = float(
                        risk_at_coverage(confidence, correct, level)
                    )
                rows.append(entry)
    return rows


def selective_curves(
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float] = None,
    ks: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
    max_points: int = 1000,
    denominator: str = "common",
) -> Dict[str, np.ndarray]:
    """Risk-coverage curve data (down-sampled) for the artifact npz.

    Keys are ``"{split}__K{k}__coverage"`` / ``"...__risk"`` for every reported
    split and the ``"__pooled__"`` summary; curves are thinned to at most
    ``max_points`` points (always keeping the full-coverage end point).
    """
    if denominator not in ("available", "common"):
        raise ValueError(f"denominator must be 'available' or 'common', got {denominator!r}")
    if logit_scale is None:
        raise ValueError("selective curves need the native logit_scale")
    ks_t = _ks_of(bundle, ks)
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)

    curves: Dict[str, np.ndarray] = {}
    for name in list(split_names) + [_pooled_name()]:
        for k in ks_t:
            if denominator == "common" or name == _pooled_name():
                base = common_rows[k]
                rows_k = base if name == _pooled_name() else base[bundle.splits(k)[base] == name]
            else:
                rows_k = bundle.rows_for_split(k, name)
            if rows_k.size < 2:
                continue
            prob = probabilities_for_variant(
                bundle.raw_scores[k][rows_k], VARIANT_NATIVE, logit_scale=logit_scale
            )
            coverage, risk = risk_coverage_curve(prob.max(axis=1), _correctness(bundle, k, rows_k))
            if coverage.size > max_points:
                picks = np.unique(
                    np.linspace(0, coverage.size - 1, max_points).astype(np.int64)
                )
                coverage, risk = coverage[picks], risk[picks]
            curves[f"{name}__K{k}__coverage"] = coverage
            curves[f"{name}__K{k}__risk"] = risk
    return curves


# ---------------------------------------------------------------------------
# reliability bins and calibration-map shift (protocol sections 21 / 24)
# ---------------------------------------------------------------------------
def reliability_bins(
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float] = None,
    variant: str = VARIANT_NATIVE,
    temperature: Optional[float] = None,
    n_bins: int = RELIABILITY_BINS,
    ks: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Per-K reliability tables with **equal-width** ``n_bins`` bins.

    One row per (split | pooled, K, bin) with ``bin_confidence`` /
    ``bin_accuracy`` / ``bin_count`` (empty bins are kept, count 0, so diagrams
    and tables do not silently disagree about the bin layout).
    """
    ks_t = _ks_of(bundle, ks)
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)

    rows: List[Dict[str, Any]] = []
    for name in list(split_names) + [_pooled_name()]:
        for k in ks_t:
            if name == _pooled_name():
                row_set = common_rows[k]
            else:
                row_set = common_rows[k][bundle.splits(k)[common_rows[k]] == name]
            if row_set.size == 0:
                continue
            prob = probabilities_for_variant(
                bundle.raw_scores[k][row_set],
                variant,
                logit_scale=logit_scale,
                temperature=temperature,
            )
            correct = _correctness(bundle, k, row_set)
            table = reliability_table(
                prob.max(axis=1), correct, n_bins=int(n_bins), adaptive=False
            )
            edges = np.asarray(table["bin_edges"], dtype=np.float64)
            for index in range(int(table["n_bins"])):
                rows.append(
                    {
                        "eval_split": name,
                        "K": int(k),
                        "variant": variant,
                        "bin_index": int(index),
                        "bin_low": float(edges[index]),
                        "bin_high": float(edges[index + 1]),
                        "bin_confidence": float(table["avg_confidence"][index]),
                        "bin_accuracy": float(table["accuracy"][index]),
                        "bin_count": int(table["counts"][index]),
                    }
                )
    return rows


def calibration_map_shift(
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float] = None,
    variant: str = VARIANT_NATIVE,
    temperature: Optional[float] = None,
    ks: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
    min_n: int = MIN_RELIABLE_SHIFT_N,
) -> List[Dict[str, Any]]:
    """Empirical ``P(correct | confidence in bin, K)`` on common confidence bins.

    The bins are the coarse ``0.0-0.1 .. 0.9-1.0`` bins of
    :data:`SHIFT_BIN_EDGES`, so the per-K rows are directly comparable: the
    extra column ``delta_acc_k50_vs_k5`` (filled on the ``K=5`` rows) shows how
    much the conditional accuracy moves from ``K=5`` to ``K=50`` inside the
    *same* confidence bin - the calibration-map drift of protocol section 21.
    Bins with ``n < min_n`` are flagged ``unreliable`` and never contribute to
    the delta.
    """
    ks_t = _ks_of(bundle, ks)
    split_names = _split_names_of(bundle, splits)
    _, common_rows, _ = common_cohort_rows(bundle, ks_t)

    edges = np.asarray(SHIFT_BIN_EDGES, dtype=np.float64)
    n_shift_bins = edges.size - 1

    per_cell: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    for name in list(split_names) + [_pooled_name()]:
        for k in ks_t:
            if name == _pooled_name():
                row_set = common_rows[k]
            else:
                row_set = common_rows[k][bundle.splits(k)[common_rows[k]] == name]
            if row_set.size == 0:
                continue
            prob = probabilities_for_variant(
                bundle.raw_scores[k][row_set],
                variant,
                logit_scale=logit_scale,
                temperature=temperature,
            )
            confidence = prob.max(axis=1)
            bin_index = np.clip(
                np.floor(confidence * n_shift_bins).astype(np.int64), 0, n_shift_bins - 1
            )
            correct = _correctness(bundle, k, row_set)
            per_cell[(name, k)] = (bin_index, correct)

    rows: List[Dict[str, Any]] = []
    for name in list(split_names) + [_pooled_name()]:
        for k in ks_t:
            if (name, k) not in per_cell:
                continue
            bin_index, correct = per_cell[(name, k)]
            k5 = per_cell.get((name, 5))
            k50 = per_cell.get((name, 50))
            for index in range(n_shift_bins):
                mask = bin_index == index
                n_bin = int(np.count_nonzero(mask))
                acc = float(np.mean(correct[mask])) if n_bin else None
                delta = None
                if k == 5 and k50 is not None:
                    mask50 = k50[0] == index
                    n50 = int(np.count_nonzero(mask50))
                    if n_bin >= min_n and n50 >= min_n:
                        delta = float(np.mean(k50[1][mask50]) - np.mean(correct[mask]))
                rows.append(
                    {
                        "eval_split": name,
                        "K": int(k),
                        "variant": variant,
                        "bin_index": int(index),
                        "bin_low": float(edges[index]),
                        "bin_high": float(edges[index + 1]),
                        "n": n_bin,
                        "empirical_acc": acc,
                        "unreliable": bool(n_bin < min_n),
                        "delta_acc_k50_vs_k5": delta,
                    }
                )
            # keep `k5` referenced for readability of the delta branch
            _ = k5
    return rows


# ---------------------------------------------------------------------------
# hard sanity checks (protocol sections 17-19; violations STOP the audit)
# ---------------------------------------------------------------------------
def assert_score_invariance(
    bundle: ScoreBundle,
    *,
    atol: float = 0.0,
    max_violations: int = 20,
) -> Dict[str, Any]:
    """Every shared candidate keeps bit-identical raw scores across Ks.

    The candidate sets are nested, so two Ks always share rows; with FP32
    storage and a single scoring call per record the score of a shared
    candidate must be *exactly* equal (``atol=0``).  Any violation means the
    scoring pipeline is broken -> :class:`ValidationFailure` (the caller writes
    ``VALIDATION_FAILURE.json`` and exits 2).
    """
    Ks = tuple(bundle.ks)
    if len(Ks) < 2:
        return {
            "check": "score_invariance",
            "status": "skipped",
            "reason": "fewer than two Ks in the bundle",
        }
    total_pairs = 0
    total_bad = 0
    max_diff = 0.0
    violations: List[Dict[str, Any]] = []
    chunk = 4096
    for i, k_lo in enumerate(Ks):
        for k_hi in Ks[i + 1 :]:
            _, rows, _ = common_cohort_rows(bundle, (k_lo, k_hi))
            c_lo = bundle.candidate_indices[k_lo][rows[k_lo]]
            c_hi = bundle.candidate_indices[k_hi][rows[k_hi]]
            s_lo = bundle.raw_scores[k_lo][rows[k_lo]]
            s_hi = bundle.raw_scores[k_hi][rows[k_hi]]
            sids = bundle.sentence_ids(k_lo)[rows[k_lo]]
            n = c_lo.shape[0]
            for start in range(0, n, chunk):
                sl = slice(start, start + chunk)
                match = c_hi[sl, None, :] == c_lo[sl, :, None]  # [m, k_lo, k_hi]
                found = match.any(axis=2)
                first = match.argmax(axis=2)
                vals = np.take_along_axis(s_hi[sl], first, axis=1)
                diff = np.abs(s_lo[sl] - vals)
                total_pairs += int(found.sum())
                bad = found & (diff > float(atol))
                total_bad += int(np.count_nonzero(bad))
                if diff.size:
                    max_diff = max(max_diff, float(diff.max()))
                if np.any(bad) and len(violations) < max_violations:
                    bad_rows, bad_cols = np.nonzero(bad)
                    for row, col in zip(bad_rows.tolist(), bad_cols.tolist()):
                        if len(violations) >= max_violations:
                            break
                        violations.append(
                            {
                                "sentence_id": int(sids[sl][row]),
                                "K_lo": int(k_lo),
                                "K_hi": int(k_hi),
                                "candidate": int(c_lo[sl][row, col]),
                                "score_lo": float(s_lo[sl][row, col]),
                                "score_hi": float(vals[row, col]),
                                "abs_diff": float(diff[row, col]),
                            }
                        )
    summary = {
        "check": "score_invariance",
        "status": "passed" if total_bad == 0 else "failed",
        "atol": float(atol),
        "n_candidate_pairs": int(total_pairs),
        "n_violations": int(total_bad),
        "max_abs_diff": float(max_diff),
    }
    if total_bad:
        raise ValidationFailure(
            "score_invariance",
            f"{total_bad} shared-candidate score(s) differ across Ks (atol={atol}); "
            "the scoring pipeline must score every K from one call per record",
            violations,
        )
    return summary


def assert_rank_monotonic(bundle: ScoreBundle, *, max_violations: int = 20) -> Dict[str, Any]:
    """On the common cohort ``rank(K') >= rank(K)`` for every ``K' > K``.

    With nested candidate sets and invariant shared scores, adding candidates
    can only keep the target's rank or push it down; observing the opposite is
    a STOP-level bug (protocol sections 17-18).
    """
    Ks = tuple(bundle.ks)
    if len(Ks) < 2:
        return {"check": "rank_monotonic", "status": "skipped", "reason": "fewer than two Ks"}
    _, rows, common_sids = common_cohort_rows(bundle, Ks)
    ranks: Dict[int, np.ndarray] = {
        k: target_rank(bundle.raw_scores[k][rows[k]], bundle.target_local[k][rows[k]])
        for k in Ks
    }
    pairs: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    total_bad = 0
    for i, k_lo in enumerate(Ks):
        for k_hi in Ks[i + 1 :]:
            delta = ranks[k_hi].astype(np.int64) - ranks[k_lo].astype(np.int64)  # >= 0 required
            bad = delta < 0
            n_bad = int(np.count_nonzero(bad))
            total_bad += n_bad
            pairs.append({"K_lo": int(k_lo), "K_hi": int(k_hi), "n_violations": n_bad})
            if n_bad and len(violations) < max_violations:
                for position in np.nonzero(bad)[0].tolist():
                    if len(violations) >= max_violations:
                        break
                    violations.append(
                        {
                            "sentence_id": int(common_sids[position]),
                            "K_lo": int(k_lo),
                            "K_hi": int(k_hi),
                            "rank_lo": int(ranks[k_lo][position]),
                            "rank_hi": int(ranks[k_hi][position]),
                        }
                    )
    summary = {
        "check": "rank_monotonic",
        "status": "passed" if total_bad == 0 else "failed",
        "n_sentences": int(common_sids.size),
        "pairs": pairs,
        "n_violations": int(total_bad),
    }
    if total_bad:
        raise ValidationFailure(
            "rank_monotonic",
            f"{total_bad} sentence(s) have a better target rank for a larger K; nested "
            "candidate sets cannot improve ranks - the scores or the sets are inconsistent",
            violations,
        )
    return summary


def assert_accuracy_monotonic(bundle: ScoreBundle, *, max_violations: int = 20) -> Dict[str, Any]:
    """On the common cohort ``Acc(K5) >= Acc(K10) >= Acc(K20) >= Acc(K50)``.

    A deterministic scorer over nested sets cannot gain accuracy when more
    distractors are added: every correct small-K choice stays correct in the
    larger set (protocol section 19).  A violation is a STOP-level bug.
    """
    Ks = tuple(bundle.ks)
    if len(Ks) < 2:
        return {"check": "accuracy_monotonic", "status": "skipped", "reason": "fewer than two Ks"}
    _, rows, _ = common_cohort_rows(bundle, Ks)
    accuracies: Dict[int, float] = {}
    for k in Ks:
        correct = _correctness(bundle, k, rows[k])
        accuracies[k] = float(np.mean(correct))
    violations: List[Dict[str, Any]] = []
    total_bad = 0
    for i, k_lo in enumerate(Ks):
        for k_hi in Ks[i + 1 :]:
            diff = accuracies[k_hi] - accuracies[k_lo]
            if diff > 1e-12:
                total_bad += 1
                if len(violations) < max_violations:
                    violations.append(
                        {
                            "K_lo": int(k_lo),
                            "K_hi": int(k_hi),
                            "acc_lo": float(accuracies[k_lo]),
                            "acc_hi": float(accuracies[k_hi]),
                            "diff": float(diff),
                        }
                    )
    summary = {
        "check": "accuracy_monotonic",
        "status": "passed" if total_bad == 0 else "failed",
        "accuracies": {str(k): float(v) for k, v in accuracies.items()},
        "n_violations": int(total_bad),
        "note": "confidence monotonicity in K is explicitly NOT checked (protocol section 20)",
    }
    if total_bad:
        raise ValidationFailure(
            "accuracy_monotonic",
            f"{total_bad} pair(s) violate Acc(K') <= Acc(K) on the common cohort; a "
            "deterministic scorer over nested sets cannot gain accuracy with more distractors",
            violations,
        )
    return summary


def run_hard_checks(bundle: ScoreBundle) -> Dict[str, Any]:
    """Run all three STOP-level checks; first failure raises :class:`ValidationFailure`."""
    return {
        "score_invariance": assert_score_invariance(bundle),
        "rank_monotonic": assert_rank_monotonic(bundle),
        "accuracy_monotonic": assert_accuracy_monotonic(bundle),
    }


# ---------------------------------------------------------------------------
# paired cluster bootstrap (protocol section 25)
# ---------------------------------------------------------------------------
class _ClusterSampler:
    """Pre-computed group layout for image-clustered row resampling."""

    def __init__(self, cluster_ids: np.ndarray):
        labels = np.asarray(cluster_ids).reshape(-1)
        uniq, inverse = np.unique(labels, return_inverse=True)
        self.n_clusters = int(uniq.size)
        self.counts = np.bincount(inverse, minlength=self.n_clusters).astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(np.int64)
        self.perm = np.argsort(inverse, kind="stable").astype(np.int64)  # flat pos -> row

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One cluster-bootstrap draw: row indices (with repetition across clusters)."""
        drawn = rng.integers(0, self.n_clusters, size=self.n_clusters)
        counts = self.counts[drawn]
        total = int(counts.sum())
        rep_starts = np.repeat(self.starts[drawn], counts)
        intra = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        flat = rep_starts + intra
        return self.perm[flat]


def paired_cluster_bootstrap(
    metric_fn: Callable[[SampleStats], float],
    pred_a: SampleStats,
    pred_b: SampleStats,
    cluster_ids: Any = None,
    *,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Image-clustered paired bootstrap of ``metric_fn(a) - metric_fn(b)``.

    The resample unit is the cluster (``image_id``): whole images are drawn
    with replacement and all rows of a drawn image enter the replicate, which
    is what makes the CI honest under within-image correlation.  Without
    ``cluster_ids`` the unit degenerates to the sample (i.i.d. bootstrap) -
    used by tests to show that clustering widens the interval.

    A and B are always resampled with the *same* draw (paired design), so the
    reported ``diff`` has the correlation removed.  ECE is notoriously noisy;
    never report it without this interval.

    Returns
    -------
    dict
        ``diff`` (point estimate on the full rows), ``mean_a`` / ``mean_b``,
        ``ci_low`` / ``ci_high`` / ``ci_level``, ``std_diff``,
        ``n`` / ``n_clusters`` / ``resample_unit`` / ``n_replicates`` and the
        raw ``replicates`` array.
    """
    a = pred_a if isinstance(pred_a, SampleStats) else SampleStats.from_conf_correct(*pred_a)
    b = pred_b if isinstance(pred_b, SampleStats) else SampleStats.from_conf_correct(*pred_b)
    if len(a) != len(b):
        raise ValueError(f"pred_a has {len(a)} rows but pred_b has {len(b)}")
    n = len(a)
    if n < 2:
        raise ValueError("paired bootstrap needs at least 2 rows")
    reps = int(n_replicates)
    if reps < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    level = float(ci)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    point_a = float(metric_fn(a))
    point_b = float(metric_fn(b))
    rng = np.random.default_rng(int(seed))
    if cluster_ids is None:
        resample_unit = "sample"
        sampler = None
        n_clusters = n
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.shape[0] != n:
            raise ValueError(
                f"cluster_ids has {clusters.shape[0]} entries but predictions have {n} rows"
            )
        sampler = _ClusterSampler(clusters)
        resample_unit = "image"
        n_clusters = sampler.n_clusters

    replicates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        if sampler is None:
            draw = rng.integers(0, n, size=n)
        else:
            draw = sampler.draw(rng)
        replicates[index] = float(metric_fn(a.take(draw))) - float(metric_fn(b.take(draw)))

    alpha = (1.0 - level) / 2.0
    ci_low = float(np.quantile(replicates, alpha))
    ci_high = float(np.quantile(replicates, 1.0 - alpha))
    return {
        "metric": metric_name or getattr(metric_fn, "__name__", "metric"),
        "diff": float(point_a - point_b),
        "mean_a": point_a,
        "mean_b": point_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": level,
        "std_diff": float(np.std(replicates)),
        "n": int(n),
        "n_clusters": int(n_clusters),
        "resample_unit": resample_unit,
        "n_replicates": reps,
        "replicates": replicates,
    }


# ---------------------------------------------------------------------------
# bootstrap metric registry (SampleStats -> float)
# ---------------------------------------------------------------------------
def accuracy_metric(stats: SampleStats) -> float:
    """Top-1 accuracy of a (resampled) stats block."""
    return float(np.mean(stats.correct))


def ece_metric(stats: SampleStats) -> float:
    """Adaptive top-label ECE of a (resampled) stats block."""
    return float(
        top_label_ece_adaptive(stats.confidence, stats.correct, n_bins=RELIABILITY_BINS)
    )


def brier_metric(stats: SampleStats) -> float:
    """Binary correctness Brier of a (resampled) stats block."""
    return float(brier_binary(stats.confidence, stats.correct))


def aurc_metric(stats: SampleStats) -> float:
    """AURC of a (resampled) stats block (risk = 1 - accuracy)."""
    coverage, risk = risk_coverage_curve(stats.confidence, stats.correct)
    return float(aurc(coverage, risk))


BOOTSTRAP_METRICS: Mapping[str, Callable[[SampleStats], float]] = {
    "accuracy": accuracy_metric,
    "ece_adaptive": ece_metric,
    "brier_binary": brier_metric,
    "aurc": aurc_metric,
}


# ---------------------------------------------------------------------------
# artifact writers (protocol section 24)
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> Path:
    """``json.dump`` to ``<path>.tmp`` then rename (never a half-written file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))
    return path


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    """CSV with a frozen column order; missing keys render as empty cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="raise", restval=""
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(tmp), str(path))
    return path


def _git_commit() -> Optional[str]:
    """HEAD commit hash as provenance, or ``None`` when not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


# ---------------------------------------------------------------------------
# cohort summary
# ---------------------------------------------------------------------------
def _cohort_summary(
    records: Sequence[SentenceRecord], bundle: ScoreBundle, ks: Tuple[int, ...]
) -> Dict[str, Any]:
    """``cohort_summary.json``: both denominators per split, with sample counts.

    ``available`` = per-K eligible sentences (own denominator);
    ``common``    = sentences eligible for *every* reported K (single shared
    denominator; the mandatory cohort for cross-K comparisons).  Target-missing
    and insufficient-ordering counts are always shown, so no table can hide
    dropped samples.
    """
    split_names = _split_names_of(bundle, None)
    Ks_c, common_rows, _ = common_cohort_rows(bundle, ks)
    reference_k = Ks_c[0]
    common_records = [bundle.records_by_k[reference_k][i] for i in common_rows[reference_k]]

    summary: Dict[str, Any] = {
        "denominators": {
            "common": (
                "sentences eligible for every reported K (target present and "
                "ordering >= max(K)-1); single shared denominator of cross-K comparisons"
            ),
            "available": "per-K eligible sentences; denominator varies with K",
        },
        "splits": {},
    }
    for name in list(split_names) + [_pooled_name()]:
        if name == _pooled_name():
            split_records = list(records)
        else:
            split_records = [record for record in records if record.eval_split == name]
        entry: Dict[str, Any] = {
            "n_sentences_total": int(len(split_records)),
            "n_refs_total": int(len({record.ref_id for record in split_records})),
            "n_target_missing_sentences": int(
                sum(1 for record in split_records if record.target_index is None)
            ),
            "per_K": {},
        }
        for k in ks:
            if name == _pooled_name():
                rows = np.arange(bundle.n_rows(k), dtype=np.int64)
            else:
                rows = bundle.rows_for_split(k, name)
            expected = sum(
                1
                for record in split_records
                if record.target_index is not None
                and int(np.asarray(record.distractor_order).size) >= int(k) - 1
            )
            if int(rows.size) != expected:
                # ScoreBundle rows and records disagree -> a construction bug
                raise RuntimeError(
                    f"cohort summary mismatch for split {name!r}, K={k}: bundle has "
                    f"{int(rows.size)} rows, eligibility filter says {expected}"
                )
            entry["per_K"][str(k)] = {
                "n_available_sentences": int(rows.size),
                "n_available_refs": int(
                    len({bundle.records_by_k[k][i].ref_id for i in rows.tolist()})
                ),
                "n_skipped_target_missing": int(
                    sum(1 for record in split_records if record.target_index is None)
                ),
                "n_skipped_insufficient": int(
                    sum(
                        1
                        for record in split_records
                        if record.target_index is not None
                        and int(np.asarray(record.distractor_order).size) < int(k) - 1
                    )
                ),
            }
        if name == _pooled_name():
            common_here = common_records
        else:
            common_here = [record for record in common_records if record.eval_split == name]
        entry["n_common_sentences"] = int(len(common_here))
        entry["n_common_refs"] = int(len({record.ref_id for record in common_here}))
        summary["splits"][name] = entry
    return summary


# ---------------------------------------------------------------------------
# prediction artifacts
# ---------------------------------------------------------------------------
def _split_codes_of(bundle: ScoreBundle, k: int) -> np.ndarray:
    """``int8[n]`` split codes of K's rows (``SPLIT_CODES`` order)."""
    return np.asarray(
        [SPLIT_CODES[record.eval_split] for record in bundle.records_by_k[k]], dtype=np.int8
    )


def _write_raw_predictions(
    out_dir: Path,
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float],
    global_temperature: Optional[float],
) -> List[str]:
    """``raw_predictions/K{K}.npz`` - raw logits are *never* dropped."""
    if logit_scale is None:
        raise ValueError("raw predictions need the native logit_scale")
    dest = Path(out_dir) / RAW_PREDICTIONS_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for k in bundle.ks:
        raw = bundle.raw_scores[k]
        payload: Dict[str, np.ndarray] = {
            "raw_scores": raw.astype(np.float32, copy=False),
            "probabilities_native": probabilities_for_variant(
                raw, VARIANT_NATIVE, logit_scale=logit_scale
            ),
            "probabilities_T1": probabilities_for_variant(raw, VARIANT_T1),
            "candidate_indices": bundle.candidate_indices[k].astype(np.int32, copy=False),
            "sentence_ids": bundle.sentence_ids(k).astype(np.int64, copy=False),
            "ref_ids": bundle.ref_ids(k).astype(np.int64, copy=False),
            "image_ids": bundle.image_ids(k).astype(np.int64, copy=False),
            "split_codes": _split_codes_of(bundle, k),
            "target_local": bundle.target_local[k].astype(np.int32, copy=False),
        }
        if global_temperature is not None:
            payload["probabilities_global_T"] = probabilities_for_variant(
                raw, VARIANT_GLOBAL, temperature=global_temperature
            )
        path = dest / f"K{k}.npz"
        np.savez_compressed(str(path), **payload)
        written.append(str(path))
    return written


def _write_prediction_summary(
    out_dir: Path,
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float],
    global_temperature: Optional[float],
) -> str:
    """Row-level ``prediction_summary.csv`` (raw logits live in the npz files)."""
    if logit_scale is None:
        raise ValueError("prediction summary needs the native logit_scale")
    fieldnames = [
        "sentence_id",
        "ref_id",
        "image_id",
        "eval_split",
        "K",
        "target_rank",
        "predicted_index",
        "confidence_native",
        "confidence_T1",
        "confidence_global",
        "correct",
    ]

    def rows() -> Iterable[Dict[str, Any]]:
        for k in bundle.ks:
            raw = bundle.raw_scores[k]
            local = bundle.target_local[k].reshape(-1)
            ranks = target_rank(raw, local)
            predicted = np.argmax(raw, axis=1)
            conf_native = probabilities_for_variant(
                raw, VARIANT_NATIVE, logit_scale=logit_scale
            ).max(axis=1)
            conf_t1 = probabilities_for_variant(raw, VARIANT_T1).max(axis=1)
            if global_temperature is not None:
                conf_global: Optional[np.ndarray] = probabilities_for_variant(
                    raw, VARIANT_GLOBAL, temperature=global_temperature
                ).max(axis=1)
            else:
                conf_global = None
            records = bundle.records_by_k[k]
            for index, record in enumerate(records):
                yield {
                    "sentence_id": int(record.sentence_id),
                    "ref_id": int(record.ref_id),
                    "image_id": int(record.image_id),
                    "eval_split": record.eval_split,
                    "K": int(k),
                    "target_rank": int(ranks[index]),
                    "predicted_index": int(predicted[index]),
                    "confidence_native": float(conf_native[index]),
                    "confidence_T1": float(conf_t1[index]),
                    "confidence_global": (
                        None if conf_global is None else float(conf_global[index])
                    ),
                    "correct": int(predicted[index] == local[index]),
                }

    path = _write_csv(Path(out_dir) / "prediction_summary.csv", fieldnames, rows())
    return str(path)


# ---------------------------------------------------------------------------
# candidate manifest copies
# ---------------------------------------------------------------------------
def _write_candidate_manifests(
    out_dir: Path, manifests_root: Any, regime: str, splits: Sequence[str]
) -> Dict[str, Any]:
    """Copy the eval-relevant manifest pairs and their summary into the artifact.

    ``candidate_manifests/`` holds byte-identical copies of
    ``{regime}_{file_split}.jsonl`` plus the sibling ``.meta.json``, and a
    ``manifest_summary.json`` recording source paths and entry counts - the
    audit is self-contained even if the cache directory later disappears.
    """
    dest = Path(out_dir) / CANDIDATE_MANIFESTS_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)
    file_splits = sorted({_FILE_SPLIT_OF[str(split)] for split in splits})
    summary: Dict[str, Any] = {"regime": str(regime), "files": {}}
    for file_split in file_splits:
        source = _manifest_file_for(manifests_root, regime, file_split)
        target = dest / source.name
        shutil.copy2(source, target)
        meta_source = source.with_suffix(".meta.json")
        meta_target = None
        if meta_source.exists():
            meta_target = dest / meta_source.name
            shutil.copy2(meta_source, meta_target)
        n_entries = 0
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    n_entries += 1
        summary["files"][file_split] = {
            "source": str(source),
            "copied_as": target.name,
            "meta_copied_as": None if meta_target is None else meta_target.name,
            "n_entries": int(n_entries),
        }
    _write_json(dest / "manifest_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# bootstrap comparison table
# ---------------------------------------------------------------------------
def _stats_cell(
    bundle: ScoreBundle,
    k: int,
    rows: np.ndarray,
    variant: str,
    *,
    logit_scale: Optional[float],
    global_temperature: Optional[float],
) -> SampleStats:
    """``SampleStats`` of one (K, row-set) cell under one probability variant."""
    prob = probabilities_for_variant(
        bundle.raw_scores[k][rows],
        variant,
        logit_scale=logit_scale,
        temperature=global_temperature,
    )
    return SampleStats.from_conf_correct(prob.max(axis=1), _correctness(bundle, k, rows))


def _bootstrap_comparisons(
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float],
    global_temperature: Optional[float],
    replicates: int,
    seed: int,
    ci: float,
    ks: Tuple[int, ...],
    splits: Optional[Sequence[str]] = None,
    on_comparison: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """Paired image-clustered bootstrap: K=5 vs K in {10, 20, 50} per split.

    Comparisons run on the common cohort of every (split | pooled) name, with
    image_id clusters.  ``accuracy`` and ``aurc`` use the native confidence (the
    argmax - and thus accuracy - is variant-invariant; AURC is defined on the
    primary native variant).  ``ece_adaptive`` and ``brier_binary`` are
    bootstrapped for **native / T1 / global-T** alike, so every probability
    variant carries a CI rather than a bare point estimate.

    ``on_comparison(done, planned, label)`` fires after every finished
    comparison (``planned`` counts exactly the cells that will run, skipped
    small cells excluded), so a CLI can mount a filling progress bar on the
    longest stage without duplicating its skip logic.
    """
    if 5 not in ks:
        raise ValueError(f"the K=5 baseline is required for the pairwise bootstrap, got {ks}")
    names = list(_split_names_of(bundle, splits)) + [_pooled_name()]
    _, common_rows, _ = common_cohort_rows(bundle, ks)

    variants = [
        variant
        for variant in REPORT_VARIANTS
        if variant != VARIANT_GLOBAL or global_temperature is not None
    ]
    metric_specs: List[Tuple[str, Callable[[SampleStats], float], Tuple[str, ...]]] = [
        ("accuracy", accuracy_metric, ("native",)),
        ("ece_adaptive", ece_metric, tuple(variants)),
        ("brier_binary", brier_metric, tuple(variants)),
        ("aurc", aurc_metric, ("native",)),
    ]

    cell_rows: Dict[Tuple[str, int], np.ndarray] = {}
    for name in names:
        for k in ks:
            if name == _pooled_name():
                cell_rows[(name, k)] = common_rows[k]
            else:
                cell_rows[(name, k)] = common_rows[k][bundle.splits(k)[common_rows[k]] == name]

    planned = sum(
        1
        for name in names
        for k_b in (k for k in ks if k != 5)
        if cell_rows[(name, 5)].size >= 2 and cell_rows[(name, k_b)].size >= 2
        for _metric_key, _metric_fn, metric_variants in metric_specs
        for _variant in metric_variants
    )

    stats_cache: Dict[Tuple[str, int, str], SampleStats] = {}

    def stats_for(name: str, k: int, variant: str) -> SampleStats:
        key = (name, k, variant)
        if key not in stats_cache:
            stats_cache[key] = _stats_cell(
                bundle,
                k,
                cell_rows[(name, k)],
                variant,
                logit_scale=logit_scale,
                global_temperature=global_temperature,
            )
        return stats_cache[key]

    rows: List[Dict[str, Any]] = []
    for name in names:
        for k_b in [k for k in ks if k != 5]:
            if cell_rows[(name, 5)].size < 2 or cell_rows[(name, k_b)].size < 2:
                continue
            clusters = bundle.image_ids(5)[cell_rows[(name, 5)]]
            for metric_key, metric_fn, metric_variants in metric_specs:
                for variant in metric_variants:
                    stats_a = stats_for(name, 5, variant)
                    stats_b = stats_for(name, k_b, variant)
                    result = paired_cluster_bootstrap(
                        metric_fn,
                        stats_a,
                        stats_b,
                        clusters,
                        n_replicates=int(replicates),
                        seed=int(seed),
                        ci=float(ci),
                        metric_name=metric_key,
                    )
                    rows.append(
                        {
                            "eval_split": name,
                            "metric": metric_key,
                            "variant": variant,
                            "K_a": 5,
                            "K_b": int(k_b),
                            "n": int(result["n"]),
                            "n_clusters": int(result["n_clusters"]),
                            "resample_unit": result["resample_unit"],
                            "diff": float(result["diff"]),
                            "ci_low": float(result["ci_low"]),
                            "ci_high": float(result["ci_high"]),
                            "ci_level": float(result["ci_level"]),
                            "n_replicates": int(result["n_replicates"]),
                            "std_diff": float(result["std_diff"]),
                        }
                    )
                    if on_comparison is not None:
                        on_comparison(
                            len(rows), planned, f"{name} {metric_key}/{variant} K5-vs-K{k_b}"
                        )
    return rows


# ---------------------------------------------------------------------------
# figures (plain diagnostics; no styling beyond labels)
# ---------------------------------------------------------------------------
def _write_figures(
    out_dir: Path,
    bundle: ScoreBundle,
    *,
    logit_scale: Optional[float],
    global_temperature: Optional[float],
    ks: Tuple[int, ...],
    splits: Optional[Sequence[str]] = None,
) -> List[str]:
    """Four diagnostic figures on the pooled common cohort (native if unsigned).

    1. ``reliability_per_K.png``      - per-K reliability curves + diagonal;
    2. ``risk_coverage_per_K.png``    - per-K risk-coverage curves;
    3. ``ece_brier_acc_vs_K.png``     - adaptive ECE / Brier / accuracy vs K by variant;
    4. ``confidence_hist_per_K.png``  - per-K native confidence histograms.
    """
    if logit_scale is None:
        raise ValueError("figures need the native logit_scale")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "matplotlib is required for the diagnostic figures; install it or call "
            "run_audit without the figure stage"
        ) from exc

    fig_dir = Path(out_dir) / FIGURES_DIRNAME
    fig_dir.mkdir(parents=True, exist_ok=True)
    _, common_rows, _ = common_cohort_rows(bundle, ks)
    variants = [
        variant
        for variant in REPORT_VARIANTS
        if variant != VARIANT_GLOBAL or global_temperature is not None
    ]
    written: List[str] = []

    def pooled_prob(k: int, variant: str) -> Tuple[np.ndarray, np.ndarray]:
        prob = probabilities_for_variant(
            bundle.raw_scores[k][common_rows[k]],
            variant,
            logit_scale=logit_scale,
            temperature=global_temperature,
        )
        return prob.max(axis=1), _correctness(bundle, k, common_rows[k])

    # 1. reliability per K (equal-width bins)
    fig, ax = plt.subplots()
    for k in ks:
        confidence, correct = pooled_prob(k, VARIANT_NATIVE)
        table = reliability_table(confidence, correct, n_bins=RELIABILITY_BINS, adaptive=False)
        edges = np.asarray(table["bin_edges"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.asarray(table["counts"])
        mask = counts > 0
        ax.plot(centers[mask], np.asarray(table["accuracy"])[mask], marker="o", label=f"K={k}")
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0)
    ax.set_xlabel("confidence (native)")
    ax.set_ylabel("accuracy")
    ax.set_title("reliability per K (common cohort, pooled eval)")
    ax.legend()
    path = fig_dir / "reliability_per_K.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    written.append(str(path))

    # 2. risk-coverage per K
    fig, ax = plt.subplots()
    for k in ks:
        confidence, correct = pooled_prob(k, VARIANT_NATIVE)
        coverage, risk = risk_coverage_curve(confidence, correct)
        ax.plot(coverage, risk, label=f"K={k}")
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (1 - accuracy)")
    ax.set_title("risk-coverage per K (native, common cohort, pooled eval)")
    ax.legend()
    path = fig_dir / "risk_coverage_per_K.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    written.append(str(path))

    # 3. ECE / Brier / accuracy vs K by variant
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric_key in zip(axes, ("ece_adaptive", "brier_binary", "accuracy")):
        for variant in variants:
            values = []
            for k in ks:
                confidence, correct = pooled_prob(k, variant)
                if metric_key == "ece_adaptive":
                    value = top_label_ece_adaptive(
                        confidence, correct, n_bins=RELIABILITY_BINS
                    )
                elif metric_key == "brier_binary":
                    value = brier_binary(confidence, correct)
                else:
                    value = float(np.mean(correct))
                values.append(value)
            axis.plot(ks, values, marker="o", label=variant)
        axis.set_xlabel("K")
        axis.set_title(metric_key)
        axis.legend()
    fig.tight_layout()
    path = fig_dir / "ece_brier_acc_vs_K.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    written.append(str(path))

    # 4. confidence histograms per K
    fig, ax = plt.subplots()
    for k in ks:
        confidence, _ = pooled_prob(k, VARIANT_NATIVE)
        ax.hist(confidence, bins=50, histtype="step", density=True, label=f"K={k}")
    ax.set_xlabel("confidence (native)")
    ax.set_ylabel("density")
    ax.set_title("confidence distribution per K (common cohort, pooled eval)")
    ax.legend()
    path = fig_dir / "confidence_hist_per_K.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# the audit driver
# ---------------------------------------------------------------------------
def run_audit(
    *,
    features_root: Any,
    manifests_root: Any,
    refs: Any,
    out_dir: Any,
    ks: Sequence[int] = DEFAULT_KS,
    splits: Sequence[str] = PRIMARY_SPLITS,
    regime: str = REGIME_RANDOM,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 0,
    bootstrap_ci: float = 0.95,
    device: str = "cpu",
    temperature_bounds: Tuple[float, float] = DEFAULT_TEMP_BOUNDS,
    on_bootstrap_pair: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the full Phase 0A cosine audit and write the artifact set.

    Stages (each timed and logged): input loading, sentence scoring, the three
    hard sanity checks, the val_calib global temperature, the metric tables, the
    paired image-clustered bootstrap, the oracle diagnostic, and the artifact
    writers.  A failed hard check writes ``VALIDATION_FAILURE.json`` into
    ``out_dir`` and raises ``SystemExit(2)`` *before* any result artifact is
    written - a broken audit never produces plottable numbers.

    Only ``regime="random"`` is supported (this audit); ``device`` is recorded
    for provenance - the whole pipeline is NumPy/CPU.  ``on_bootstrap_pair``
    (optional) forwards the per-comparison callback of
    :func:`_bootstrap_comparisons` so callers can show a filling bar on the
    bootstrap, the longest stage.
    """
    started = time.perf_counter()
    durations: Dict[str, float] = {}

    def stage_done(name: str, stage_started: float) -> None:
        elapsed = time.perf_counter() - stage_started
        durations[name] = float(elapsed)
        if log is not None:
            log(f"[phase0a] {name}: {elapsed:.2f}s")

    out_dir = Path(out_dir)
    ks_t = tuple(sorted({int(k) for k in ks}))
    splits_t = tuple(str(name) for name in splits)
    if str(regime) != REGIME_RANDOM:
        raise ValueError(f"only the {REGIME_RANDOM!r} regime is audited here, got {regime!r}")

    # -- stage 1: inputs ----------------------------------------------------
    stage_started = time.perf_counter()
    records = load_cosine_inputs(
        features_root, manifests_root, refs, splits_t, regime=regime, ks=ks_t
    )
    if log is not None:
        n_refs = len({record.ref_id for record in records})
        log(f"[phase0a] loaded {len(records)} sentences of {n_refs} refs")
    stage_done("load_inputs", stage_started)

    # -- stage 2: scoring ---------------------------------------------------
    stage_started = time.perf_counter()
    from ccg.features.cache import FeatureCache  # lazy: keeps light imports light

    with FeatureCache.open(features_root) as cache:
        cache_metadata = dict(getattr(cache, "metadata", {}) or {})
        logit_scale = cache_metadata.get("native_logit_scale")
        if logit_scale is None:
            raise ValueError(
                "the feature cache metadata carries no 'native_logit_scale'; the native "
                "probability variant (primary) cannot be formed without it"
            )
        logit_scale = float(logit_scale)
        bundle = score_sets(cache, records, ks_t)
    stage_done("score_sets", stage_started)

    # -- stage 3: hard sanity checks (violations STOP the audit) ------------
    stage_started = time.perf_counter()
    try:
        checks = run_hard_checks(bundle)
    except ValidationFailure as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            out_dir / FAILURE_FILENAME,
            {
                "status": "STOP",
                "check": exc.check,
                "message": exc.message,
                "violations": exc.violations,
                "created_utc": _utc_now(),
                "note": (
                    "a hard sanity check of protocol sections 17-19 failed; the audit "
                    "aborted before any result artifact was written"
                ),
            },
        )
        if log is not None:
            log(
                f"[phase0a] HARD CHECK FAILED [{exc.check}]: {exc.message} -> "
                f"{FAILURE_FILENAME}; exiting with code 2"
            )
        raise SystemExit(2) from exc
    stage_done("hard_checks", stage_started)

    # -- stage 4: legal global temperature (val_calib, K in {5,10}) ---------
    stage_started = time.perf_counter()
    global_fit: Optional[TemperatureFit] = None
    legal_ks = tuple(k for k in ks_t if k in LEGAL_CALIBRATION_KS)
    if legal_ks:
        val_rows = {k: bundle.rows_for_split(k, "val_calib") for k in legal_ks}
        val_rows = {k: rows for k, rows in val_rows.items() if rows.size > 0}
        if val_rows:
            global_fit = fit_global_temperature(
                bundle,
                ks=tuple(val_rows.keys()),
                bounds=temperature_bounds,
                rows_by_k=val_rows,
            )
        elif log is not None:
            log("[phase0a] no val_calib rows for K in {5,10}; the global-T variant is disabled")
    global_temperature = None if global_fit is None else float(global_fit.temperature)
    stage_done("fit_global_temperature", stage_started)

    # -- stage 5: metric tables --------------------------------------------
    stage_started = time.perf_counter()
    ranking_rows = ranking_metrics(bundle, ks=ks_t)
    calibration_rows = calibration_metrics(
        bundle, logit_scale=logit_scale, temperature=global_temperature, ks=ks_t
    )
    selective_rows = selective_metrics(bundle, logit_scale=logit_scale, ks=ks_t)
    reliability_rows = reliability_bins(bundle, logit_scale=logit_scale, ks=ks_t)
    shift_rows = calibration_map_shift(bundle, logit_scale=logit_scale, ks=ks_t)
    stage_done("metric_tables", stage_started)

    # -- stage 6: paired image-clustered bootstrap --------------------------
    stage_started = time.perf_counter()
    bootstrap_rows = _bootstrap_comparisons(
        bundle,
        logit_scale=logit_scale,
        global_temperature=global_temperature,
        replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed),
        ci=float(bootstrap_ci),
        ks=ks_t,
        on_comparison=on_bootstrap_pair,
    )
    stage_done("bootstrap", stage_started)

    # -- stage 7: oracle diagnostic ----------------------------------------
    stage_started = time.perf_counter()
    oracle = oracle_temperature_diagnostic(
        bundle, global_fit, ks=ks_t, bounds=temperature_bounds
    )
    stage_done("oracle_diagnostic", stage_started)

    # -- stage 8: artifacts -------------------------------------------------
    stage_started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = _cohort_summary(records, bundle, ks_t)
    _write_json(out_dir / "cohort_summary.json", cohort)
    raw_written = _write_raw_predictions(
        out_dir, bundle, logit_scale=logit_scale, global_temperature=global_temperature
    )
    prediction_summary_path = _write_prediction_summary(
        out_dir, bundle, logit_scale=logit_scale, global_temperature=global_temperature
    )
    _write_csv(
        out_dir / "ranking_metrics.csv",
        [
            "eval_split",
            "K",
            "denominator",
            "n",
            "n_correct",
            "top1",
            "top5",
            "mean_target_rank",
            "mrr",
            "delta_acc_vs_k5",
        ],
        ranking_rows,
    )
    _write_csv(
        out_dir / "calibration_metrics.csv",
        [
            "eval_split",
            "K",
            "variant",
            "denominator",
            "n",
            "status",
            "accuracy",
            "ece_adaptive",
            "ece_equal_width",
            "brier_binary",
            "nll_binary",
            "conf_acc_gap",
            "multiclass_nll",
            "multiclass_brier",
            "mean_entropy",
        ],
        calibration_rows,
    )
    _write_csv(
        out_dir / "selective_metrics.csv",
        [
            "eval_split",
            "K",
            "denominator",
            "n",
            "aurc",
            "risk_at_50",
            "risk_at_80",
            "risk_at_90",
            "risk_at_95",
        ],
        selective_rows,
    )
    _write_csv(
        out_dir / "reliability_bins.csv",
        [
            "eval_split",
            "K",
            "variant",
            "bin_index",
            "bin_low",
            "bin_high",
            "bin_confidence",
            "bin_accuracy",
            "bin_count",
        ],
        reliability_rows,
    )
    _write_csv(
        out_dir / "calibration_map_shift.csv",
        [
            "eval_split",
            "K",
            "variant",
            "bin_index",
            "bin_low",
            "bin_high",
            "n",
            "empirical_acc",
            "unreliable",
            "delta_acc_k50_vs_k5",
        ],
        shift_rows,
    )
    _write_csv(
        out_dir / "bootstrap_ci.csv",
        [
            "eval_split",
            "metric",
            "variant",
            "K_a",
            "K_b",
            "n",
            "n_clusters",
            "resample_unit",
            "diff",
            "ci_low",
            "ci_high",
            "ci_level",
            "n_replicates",
            "std_diff",
        ],
        bootstrap_rows,
    )
    if global_fit is not None:
        global_payload: Dict[str, Any] = global_fit.to_dict()
        global_payload["objective"] = (
            "candidate-set NLL (ccg.calibration.mean_nll), pooled over K in "
            f"{list(global_fit.ks)} on val_calib"
        )
    else:
        global_payload = {
            "status": "not_fitted",
            "reason": "no val_calib rows for K in {5, 10} in this audit",
        }
    _write_json(out_dir / "global_temperature.json", global_payload)
    _write_json(out_dir / "oracle_temperature_diagnostic.json", oracle)

    curves = selective_curves(bundle, logit_scale=logit_scale, ks=ks_t)
    if curves:
        np.savez_compressed(str(out_dir / "selective_curves.npz"), **curves)
    manifest_summary = _write_candidate_manifests(out_dir, manifests_root, regime, splits_t)
    figures = _write_figures(
        out_dir,
        bundle,
        logit_scale=logit_scale,
        global_temperature=global_temperature,
        ks=ks_t,
        splits=splits_t,
    )
    stage_done("write_artifacts", stage_started)

    total_seconds = time.perf_counter() - started
    metadata = {
        "artifact": "phase0a_cosine",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "config": {
            "features_root": str(features_root),
            "manifests_root": str(manifests_root),
            "refs": str(refs) if isinstance(refs, (str, Path)) else "<records>",
            "out_dir": str(out_dir),
            "regime": str(regime),
            "ks": [int(k) for k in ks_t],
            "splits": [str(name) for name in splits_t],
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_ci": float(bootstrap_ci),
            "device": str(device),
            "temperature_bounds": [float(temperature_bounds[0]), float(temperature_bounds[1])],
        },
        "feature_cache_metadata": cache_metadata,
        "native_logit_scale": float(logit_scale),
        "primary_variant": VARIANT_NATIVE,
        "variant_note": (
            "native is the primary reported variant (deployment probability); T1 and "
            "global_T are diagnostics reported next to it, never as replacements; the "
            "bootstrap covers native, T1 and global_T"
        ),
        "durations_seconds": durations,
        "total_seconds": float(total_seconds),
        "n_records": int(len(records)),
        "n_l2_renormalized": int(bundle.n_l2_renormalized),
        "hard_checks": checks,
        "cohort_summary_file": "cohort_summary.json",
        "candidate_manifests": manifest_summary,
        "artifacts": [
            "metadata.json",
            "cohort_summary.json",
            FAILURE_FILENAME + " (written only on a failed hard check)",
            RAW_PREDICTIONS_DIRNAME + f"/K{{K}}.npz ({len(raw_written)} files)",
            "prediction_summary.csv",
            "ranking_metrics.csv",
            "calibration_metrics.csv",
            "selective_metrics.csv",
            "reliability_bins.csv",
            "calibration_map_shift.csv",
            "global_temperature.json",
            "oracle_temperature_diagnostic.json",
            "bootstrap_ci.csv",
            "selective_curves.npz",
            FIGURES_DIRNAME + f"/ ({len(figures)} files)",
            CANDIDATE_MANIFESTS_DIRNAME + "/",
        ],
    }
    _write_json(out_dir / "metadata.json", metadata)
    if log is not None:
        log(f"[phase0a] total: {total_seconds:.2f}s -> {out_dir}")

    return {
        "out_dir": str(out_dir),
        "n_records": int(len(records)),
        "ks": [int(k) for k in ks_t],
        "splits": [str(name) for name in splits_t],
        "native_logit_scale": float(logit_scale),
        "global_temperature": global_payload,
        "hard_checks": checks,
        "durations_seconds": durations,
        "total_seconds": float(total_seconds),
        "n_bootstrap_comparisons": int(len(bootstrap_rows)),
        "cohort_summary": cohort,
        "figures": figures,
        "prediction_summary": prediction_summary_path,
    }
