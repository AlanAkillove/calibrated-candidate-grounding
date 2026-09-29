"""Phase 1F hard-competition cohorts and candidate construction (protocol A8).

This module owns the *data* half of the Phase 1F confirmatory stress test:
it loads the frozen canonical cohort restricted to the pooled test rows
(``testA`` + ``testB``) and the two frozen ref-level manifests (``random`` /
``same_category``, manifests-v1, seed 20260927), asserts that the two regimes
are strictly matched per ref (same target / valid pool / same-category supply)
and exposes

* the A8.2 cohort masks (``base`` / ``same4`` / ``same8`` / ``same9``);
* the A8.3 candidate builders ``C_K = [target] + order[:K-1]`` and the
  hard-fraction level orderings of Experiment B.

Nothing here scores, trains or resamples: every ordering is read from the
frozen manifests and the rows are the frozen Phase 0.5 canonical cohort rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data.manifests import ManifestEntry, ManifestFile
from . import data as sdata

__all__ = [
    "POOLED",
    "REPORT_SPLITS",
    "REGIMES",
    "HARD_FRACTION_LEVELS",
    "COHORT_CUTOFFS",
    "DEFAULT_MANIFESTS_ROOT",
    "FrozenSample",
    "HardCohort",
    "load_hard_cohort",
    "cohort_summary",
]

#: Pooled-test label used across the phase 0.5 / 1 / 1F artifacts.
POOLED = "__pooled_test__"
#: The two evaluation splits of the Phase 1F base cohort.
REPORT_SPLITS: Tuple[str, ...] = ("testA", "testB")
#: The frozen construction regimes (manifests-v1).
REGIMES: Tuple[str, ...] = ("random", "same_category")
#: Experiment B hard-fraction levels (A8.2) - never adjusted after the fact.
HARD_FRACTION_LEVELS: Tuple[int, ...] = (0, 2, 4, 8)
#: Cohort cutoffs of the A8.2 masks: rows keep ``n_same >= cutoff``.
COHORT_CUTOFFS: Dict[str, int] = {"base": 0, "same4": 4, "same8": 8, "same9": 9}
#: Manifest directory holding ``{regime}_{split}.jsonl`` pairs.
DEFAULT_MANIFESTS_ROOT = Path("cache/manifests")


@dataclass(frozen=True)
class FrozenSample:
    """Duck-typed sample for :meth:`ccg.models.b3_data.B3Corpus.example_for`.

    Exposes exactly the attributes the corpus reads (``sentence_id`` /
    ``ref_id`` / ``image_id`` / ``target_index`` / ``distractor_order``); the
    order is the frozen regime ordering, so ``example_for(sample, K)`` yields
    ``C_K = [target] + order[:K-1]`` of that regime.
    """

    sentence_id: int
    ref_id: int
    image_id: int
    target_index: int
    distractor_order: np.ndarray


@dataclass
class HardCohort:
    """Pooled-test rows plus the frozen per-regime manifest entries.

    Attributes
    ----------
    sentence_id / ref_id / image_id / eval_split:
        Row identities of the pooled-test subset of the Phase 0.5 canonical
        cohort (``sentence_id`` ascending, same row universe as Phase 1's
        pooled rows).
    nsame:
        Per-row ``n_same_category_available`` of the ref (ref-level manifest
        attribute broadcast to its sentences).
    masks:
        ``{"base", "same4", "same8", "same9"}`` -> ``[n]`` boolean row masks
        (``sameX`` = ``nsame >= X``; ``base`` is all rows).
    entries:
        ``{regime: {ref_id: ManifestEntry}}`` for ``random`` / ``same_category``.
    """

    sentence_id: np.ndarray
    ref_id: np.ndarray
    image_id: np.ndarray
    eval_split: np.ndarray
    nsame: np.ndarray
    masks: Dict[str, np.ndarray]
    entries: Dict[str, Dict[int, ManifestEntry]]

    def __len__(self) -> int:
        return int(self.sentence_id.shape[0])

    # -- candidate builders --------------------------------------------------
    def _entry(self, row: int, regime: str) -> ManifestEntry:
        ref = int(self.ref_id[row])
        try:
            return self.entries[str(regime)][ref]
        except KeyError as exc:  # pragma: no cover - guarded by load_hard_cohort
            raise KeyError(f"ref {ref} has no {regime!r} manifest entry") from exc

    def candidate_indices(self, row: int, regime: str, k: int) -> np.ndarray:
        """``C_K = [target] + order_regime[:K-1]`` of one row (int64 ``[k]``)."""
        k = int(k)
        entry = self._entry(row, regime)
        if entry.target_index is None:
            raise ValueError(f"row {row}: target absent in the {regime!r} manifest")
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        if order.size < k - 1:
            raise ValueError(
                f"row {row}: {regime!r} ordering has {order.size} distractors, need {k - 1}"
            )
        out = np.empty(k, dtype=np.int64)
        out[0] = int(entry.target_index)
        out[1:] = order[: k - 1]
        return out

    def level_indices(self, row: int, m: int, k: int = 10) -> np.ndarray:
        """Experiment B level ``m`` of one row (int64 ``[k]``).

        ``[target] + same_order[:m] + same_order[n_avail : n_avail + (k-1-m)]``:
        the m lowest-index same-category distractors plus the first
        ``k-1-m`` entries of the frozen rest shuffle of the same manifest.
        """
        m = int(m)
        k = int(k)
        if m < 0 or m > k - 1:
            raise ValueError(f"level m={m} outside [0, {k - 1}]")
        entry = self._entry(row, "same_category")
        if entry.target_index is None:
            raise ValueError(f"row {row}: target absent in the same_category manifest")
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        n_avail = int(entry.n_same_category_available)
        if n_avail < m:
            raise ValueError(f"row {row}: only {n_avail} same-category distractors for level {m}")
        need_fill = k - 1 - m
        start = n_avail
        stop = start + need_fill
        if order.size < stop:
            raise ValueError(
                f"row {row}: rest pool has {order.size - n_avail} entries, need {need_fill}"
            )
        out = np.empty(k, dtype=np.int64)
        out[0] = int(entry.target_index)
        out[1 : 1 + m] = order[:m]
        out[1 + m :] = order[start:stop]
        return out

    # -- corpus samples ------------------------------------------------------
    def sample(self, row: int, regime: str) -> FrozenSample:
        """A :class:`FrozenSample` carrying the ref's full regime ordering."""
        entry = self._entry(row, regime)
        return FrozenSample(
            sentence_id=int(self.sentence_id[row]),
            ref_id=int(self.ref_id[row]),
            image_id=int(self.image_id[row]),
            target_index=int(entry.target_index),
            distractor_order=np.asarray(entry.distractor_order, dtype=np.int64),
        )

    def level_sample(self, row: int, m: int, k: int = 10) -> FrozenSample:
        """A :class:`FrozenSample` whose ordering is the level-``m`` construction."""
        m = int(m)
        k = int(k)
        entry = self._entry(row, "same_category")
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        n_avail = int(entry.n_same_category_available)
        seq = np.concatenate([order[:m], order[n_avail : n_avail + (k - 1 - m)]])
        return FrozenSample(
            sentence_id=int(self.sentence_id[row]),
            ref_id=int(self.ref_id[row]),
            image_id=int(self.image_id[row]),
            target_index=int(entry.target_index),
            distractor_order=seq,
        )

    def subsample(self, mask: np.ndarray) -> "HardCohort":
        """A new cohort restricted to ``mask`` (entries shared, arrays copied)."""
        rows = np.asarray(mask, dtype=bool)
        if rows.shape != (len(self),):
            raise ValueError(f"mask shape {rows.shape} does not match n={len(self)}")
        sub_masks = {name: flags[rows] for name, flags in self.masks.items()}
        return HardCohort(
            sentence_id=self.sentence_id[rows],
            ref_id=self.ref_id[rows],
            image_id=self.image_id[rows],
            eval_split=self.eval_split[rows],
            nsame=self.nsame[rows],
            masks=sub_masks,
            entries=self.entries,
        )


# ---------------------------------------------------------------------------
# loading + matched-manifest assertions
# ---------------------------------------------------------------------------
def load_hard_cohort(
    *,
    features_dir: Path = sdata.PHASE05_FEATURES_DIR,
    manifests_root: Path = DEFAULT_MANIFESTS_ROOT,
    log: Optional[Callable[[str], None]] = None,
) -> HardCohort:
    """Load the pooled-test hard-competition cohort of Phase 1F (A8.2/A8.3).

    Loads the frozen Phase 0.5 canonical cohort (``K=5`` row universe), keeps
    the ``testA`` / ``testB`` rows, joins the two frozen ref-level manifests and
    asserts the strict matched-control property per ref; violations raise
    :class:`AssertionError` (a mismatched control invalidates the stress test).
    """
    cohort = sdata.load_phase05_cohort(5, features_dir=Path(features_dir))
    pooled = np.isin(cohort["eval_split"], np.asarray(REPORT_SPLITS))
    if not np.any(pooled):
        raise ValueError("no pooled-test rows in the canonical cohort")

    entries: Dict[str, Dict[int, ManifestEntry]] = {}
    for regime in REGIMES:
        table: Dict[int, ManifestEntry] = {}
        for split in REPORT_SPLITS:
            path = Path(manifests_root) / f"{regime}_{split}.jsonl"
            manifest = ManifestFile.load(path)
            for entry in manifest.entries:
                table[int(entry.ref_id)] = entry
        entries[regime] = table

    ref_id = np.asarray(cohort["ref_id"], dtype=np.int64)[pooled]
    rand_table = entries["random"]
    hard_table = entries["same_category"]

    nsame = np.empty(ref_id.size, dtype=np.int64)
    for pos, ref in enumerate(ref_id.tolist()):
        hard = hard_table.get(int(ref))
        rand = rand_table.get(int(ref))
        if rand is None or hard is None:
            raise AssertionError(f"ref {ref}: missing one of the matched manifest entries")
        if rand.target_index != hard.target_index:
            raise AssertionError(
                f"ref {ref}: target_index mismatch random={rand.target_index} "
                f"same_category={hard.target_index}"
            )
        if not np.array_equal(
            np.sort(np.asarray(rand.distractor_order)), np.sort(np.asarray(hard.distractor_order))
        ):
            raise AssertionError(f"ref {ref}: valid distractor pool differs across regimes")
        if int(rand.n_same_category_available) != int(hard.n_same_category_available):
            raise AssertionError(f"ref {ref}: n_same_category_available differs across regimes")
        for k in (5, 10):
            if not (bool(rand.eligible.get(k, False)) and bool(hard.eligible.get(k, False))):
                raise AssertionError(f"ref {ref}: not eligible at K={k} in both regimes")
        nsame[pos] = int(hard.n_same_category_available)

    masks: Dict[str, np.ndarray] = {}
    for name, cutoff in COHORT_CUTOFFS.items():
        masks[name] = np.ones(ref_id.size, dtype=bool) if cutoff == 0 else nsame >= cutoff

    result = HardCohort(
        sentence_id=np.asarray(cohort["sentence_id"], dtype=np.int64)[pooled],
        ref_id=ref_id,
        image_id=np.asarray(cohort["image_id"], dtype=np.int64)[pooled],
        eval_split=np.asarray(cohort["eval_split"])[pooled],
        nsame=nsame,
        masks=masks,
        entries=entries,
    )

    # A8.3: the K5/K10 prefixes must be *fully* same-category on the A8.2 cohorts.
    for mask_name, regime_k in (("same4", 5), ("same9", 10)):
        rows = np.flatnonzero(masks[mask_name])
        for row in rows.tolist():
            fraction = float(
                hard_table[int(result.ref_id[row])].hard_fraction_by_K[regime_k]
            )
            if abs(fraction - 1.0) > 1e-12:
                raise AssertionError(
                    f"row {row}: hard_fraction[K={regime_k}] = {fraction} != 1.0"
                )

    if log is not None:
        log(
            f"[hard] pooled rows={len(result)} images={int(np.unique(result.image_id).size)} "
            f"refs={int(np.unique(result.ref_id).size)}"
        )
        for name in ("same4", "same8", "same9"):
            flags = masks[name]
            log(
                f"[hard]   {name}: rows={int(flags.sum())} "
                f"images={int(np.unique(result.image_id[flags]).size)} "
                f"refs={int(np.unique(result.ref_id[flags]).size)}"
            )
    return result


def cohort_summary(cohort: HardCohort) -> Dict[str, Any]:
    """Serialisable cohort statistics for ``cohort_summary.json``."""
    out: Dict[str, Any] = {
        "n_rows": len(cohort),
        "n_images": int(np.unique(cohort.image_id).size),
        "n_refs": int(np.unique(cohort.ref_id).size),
        "splits": {
            str(split): int((cohort.eval_split == split).sum())
            for split in REPORT_SPLITS
        },
        "cohorts": {},
    }
    for name, flags in cohort.masks.items():
        out["cohorts"][name] = {
            "cutoff": int(COHORT_CUTOFFS[name]),
            "n_rows": int(flags.sum()),
            "n_images": int(np.unique(cohort.image_id[flags]).size),
            "n_refs": int(np.unique(cohort.ref_id[flags]).size),
        }
    nsame = np.asarray(cohort.nsame, dtype=np.float64)
    out["n_same_category_available"] = {
        "min": int(nsame.min()),
        "mean": float(nsame.mean()),
        "median": float(np.median(nsame)),
        "max": int(nsame.max()),
    }
    return out
