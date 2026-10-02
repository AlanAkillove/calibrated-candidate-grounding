"""RQ4-M1: exact transition-confidence decomposition of cardinality-induced degradation.

The question (frozen in
``results/v2_rq4_mechanism/m1_transition_confidence/protocol_freeze.json``)::

    Decompose the K5 -> K50 AUROC degradation into (1) correctness-state
    transition and (2) confidence-ranking change, and determine which component
    accounts for the cross-proposal-family amplification.

This module is **pure algebra on already-frozen arrays**.  It reads no files, fits
nothing, imports no model and never touches a GPU: every input is a ``(n,)`` vector
of correctness labels or frozen ``global_T_corrected`` confidences handed in by the
driver.  Classification is ``DESCRIPTIVE_MECHANISM_DECOMPOSITION`` - an exact
accounting identity, not a causal test and not a new method.

Notation (one family, one B3 seed, one common-K50 cohort; rows are aligned so that
``row i`` is the same expression at K5 and at K50)::

    r5  = correctness at K5            p5  = frozen confidence at K5
    r50 = correctness at K50           p50 = frozen confidence at K50

    A00 = AUROC(r5,  p5)     A10 = AUROC(r50, p5)
    A01 = AUROC(r5,  p50)    A11 = AUROC(r50, p50)

    delta_total = A11 - A00                  (the observed change, negative here)
    D_total     = A00 - A11                  (the degradation, positive here)

    L = 0.5 * ((A10 - A00) + (A11 - A01))    label-transition contribution
    C = 0.5 * ((A01 - A00) + (A11 - A10))    confidence-change contribution

    L + C == A11 - A00                       (two-factor Shapley, exact)
    D_label = -L,  D_conf = -C,  D_label + D_conf == D_total

The two contributions are **signed**: either may be negative (partially offsetting)
or exceed ``D_total``.  Nothing is clipped, renormalised or forced to sum to 100%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ccg.metrics.discrimination import auroc_correct

__all__ = [
    "Decomposition",
    "IDENTITY_TOLERANCE",
    "SHARE_DENOM_FLOOR",
    "StructuralInvariantError",
    "TransitionGroups",
    "component_ordering",
    "corner_aurocs",
    "cross_family_gap",
    "decompose",
    "group_weight_reconstruction",
    "image_cluster_bootstrap",
    "pairwise_auc",
    "selective_error_sources",
    "shapley_decomposition",
    "signed_share",
    "transition_groups",
]

#: The nested-K pair analysed by RQ4-M1 (identical to the frozen C1 comparison).
K_BASELINE = 5
K_PRIMARY = 50

#: ``share = component / D_total`` is reported only when ``|D_total|`` clears this floor.
SHARE_DENOM_FLOOR = 1e-3

#: Machine-precision budget for the two frozen identities (Shapley sum, group-weight
#: AUROC reconstruction).  Both are exact in real arithmetic; the residual is float64
#: summation noise on cohorts of ~10^4 rows.
IDENTITY_TOLERANCE = 1e-12


class StructuralInvariantError(RuntimeError):
    """Raised when ``G`` (gained-correct) is non-empty: the nested-K design broke."""


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def _vector(values: Any, name: str, n: Optional[int] = None) -> NDArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    return arr


def _binary(arr: NDArray, name: str) -> NDArray:
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError(f"{name} must be binary 0/1")
    return arr


# ---------------------------------------------------------------------------
# correctness-transition groups (S / F / E / G)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransitionGroups:
    """Boolean row masks of the four transition groups, exhaustive and disjoint."""

    stable_correct: NDArray  #: ``S``: r5 == 1 and r50 == 1
    lost: NDArray            #: ``F``: r5 == 1 and r50 == 0 (flipped to error)
    stable_error: NDArray    #: ``E``: r5 == 0 and r50 == 0
    gained: NDArray          #: ``G``: r5 == 0 and r50 == 1 (structurally expected empty)

    @property
    def n(self) -> int:
        return int(self.stable_correct.shape[0])

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "S": int(self.stable_correct.sum()),
            "F": int(self.lost.sum()),
            "E": int(self.stable_error.sum()),
            "G": int(self.gained.sum()),
        }

    @property
    def rates(self) -> Dict[str, float]:
        n = float(self.n)
        return {k: int(v) / n for k, v in self.counts.items()}


def transition_groups(correct_k5: Any, correct_k50: Any) -> TransitionGroups:
    """Partition the cohort into S / F / E / G by its (r5, r50) correctness pair."""
    r5 = _binary(_vector(correct_k5, "correct_k5"), "correct_k5")
    r50 = _binary(_vector(correct_k50, "correct_k50", r5.shape[0]), "correct_k50")
    c1, c0 = r5 == 1.0, r5 == 0.0
    d1, d0 = r50 == 1.0, r50 == 0.0
    return TransitionGroups(
        stable_correct=c1 & d1, lost=c1 & d0, stable_error=c0 & d0, gained=c0 & d1
    )


def check_structural_invariant(groups: TransitionGroups, *, family: str, seed: str) -> Dict[str, Any]:
    """Assert ``G == 0`` (nested candidates + per-candidate independent scorer).

    ``G`` must be empty for every family and seed: ``C5`` is a subset of ``C50`` and a
    target that lost its top-1 place inside ``C5`` cannot regain it when candidates are
    only added.  A non-empty ``G`` means the frozen prediction files do not satisfy the
    nested-K design, so the decomposition is meaningless and the driver must STOP.
    """
    counts = groups.counts
    if counts["G"] > 0:
        raise StructuralInvariantError(
            f"STRUCTURAL_INVARIANT_FAILURE: {family}/{seed} has G={counts['G']} "
            "gained-correct rows > 0 on a nested C5 subset-of C50 cohort; STOP"
        )
    if counts["S"] + counts["F"] + counts["E"] + counts["G"] != groups.n:
        raise AssertionError("transition groups are not exhaustive")
    return {
        "family": family,
        "seed": seed,
        "n": groups.n,
        **counts,
        "gained_zero": True,
        "partition_exhaustive": True,
    }


# ---------------------------------------------------------------------------
# AUROC primitives
# ---------------------------------------------------------------------------
def _auroc(scores: NDArray, labels: NDArray) -> float:
    return float(auroc_correct(scores, labels))


def pairwise_auc(scores_positive: Any, scores_negative: Any) -> float:
    """``P(pos > neg) + 0.5 * P(tie)`` - the Mann-Whitney pair statistic of two groups.

    Uses the same tie-correct average-rank estimator as :func:`auroc_correct`, i.e. the
    scikit-learn 0.5 tie convention, so group-level components sum back into the pooled
    AUROC exactly.  An empty group is a validation error rather than a silent ``nan``,
    because every caller of this statistic needs both groups populated.
    """
    pos = _vector(scores_positive, "scores_positive")
    neg = _vector(scores_negative, "scores_negative", None)
    if pos.shape[0] == 0 or neg.shape[0] == 0:      # pragma: no cover - _vector guards first
        return float("nan")
    scores = np.concatenate((pos, neg))
    labels = np.concatenate((np.ones(pos.shape[0]), np.zeros(neg.shape[0])))
    return _auroc(scores, labels)


@dataclass(frozen=True)
class Decomposition:
    """The four corners, the Shapley split and the signed degradation components."""

    a00: float
    a10: float
    a01: float
    a11: float
    delta_total: float
    d_total: float
    d_label: float
    d_conf: float
    shapley_residual: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "A00": self.a00, "A10": self.a10, "A01": self.a01, "A11": self.a11,
            "delta_total": self.delta_total, "D_total": self.d_total,
            "D_label": self.d_label, "D_conf": self.d_conf,
            "shapley_residual": self.shapley_residual,
        }

    def shares(self, *, floor: float = SHARE_DENOM_FLOOR) -> Dict[str, Optional[float]]:
        """Signed descriptive shares; ``None`` when ``|D_total|`` is below ``floor``."""
        if abs(self.d_total) < float(floor):
            return {"share_label": None, "share_conf": None, "share_denominator_ok": None}
        return {
            "share_label": self.d_label / self.d_total,
            "share_conf": self.d_conf / self.d_total,
            "share_denominator_ok": True,
        }


def corner_aurocs(conf_k5: Any, correct_k5: Any, conf_k50: Any, correct_k50: Any) -> Tuple[float, float, float, float]:
    """``(A00, A10, A01, A11)`` on one aligned cohort (see the module docstring)."""
    r5 = _binary(_vector(correct_k5, "correct_k5"), "correct_k5")
    r50 = _binary(_vector(correct_k50, "correct_k50", r5.shape[0]), "correct_k50")
    p5 = _vector(conf_k5, "conf_k5", r5.shape[0])
    p50 = _vector(conf_k50, "conf_k50", r5.shape[0])
    return _auroc(p5, r5), _auroc(p5, r50), _auroc(p50, r5), _auroc(p50, r50)


def shapley_decomposition(a00: float, a10: float, a01: float, a11: float) -> Decomposition:
    """Symmetric two-factor Shapley split of ``A11 - A00`` into label / confidence."""
    corner = (float(a00), float(a10), float(a01), float(a11))
    if not all(math.isfinite(v) for v in corner):
        raise ValueError(f"decomposition needs four finite corners, got {corner}")
    label = 0.5 * ((a10 - a00) + (a11 - a01))
    conf = 0.5 * ((a01 - a00) + (a11 - a10))
    delta_total = a11 - a00
    return Decomposition(
        a00=a00, a10=a10, a01=a01, a11=a11,
        delta_total=delta_total, d_total=-delta_total,
        d_label=-label, d_conf=-conf,
        shapley_residual=(label + conf) - delta_total,
    )


def decompose(conf_k5: Any, correct_k5: Any, conf_k50: Any, correct_k50: Any) -> Decomposition:
    """Corner AUROCs + Shapley decomposition in one call."""
    return shapley_decomposition(*corner_aurocs(conf_k5, correct_k5, conf_k50, correct_k50))


# ---------------------------------------------------------------------------
# group-weight AUROC reconstruction (exact identity, algebra check)
# ---------------------------------------------------------------------------
def group_weight_reconstruction(
    conf_k5: Any, correct_k5: Any, conf_k50: Any, correct_k50: Any,
    groups: Optional[TransitionGroups] = None,
) -> Dict[str, Any]:
    """Rebuild each corner from pairwise group AUCs weighted by cross-group pair counts.

    With ``G`` empty the four corners are::

        A00 = (nS * AUC(S,E;p5)  + nF * AUC(F,E;p5))  / (nS + nF)      [pos S+F, neg E]
        A01 = (nS * AUC(S,E;p50) + nF * AUC(F,E;p50)) / (nS + nF)
        A10 = (nF * AUC(S,F;p5)  + nE * AUC(S,E;p5))  / (nF + nE)      [pos S, neg F+E]
        A11 = (nF * AUC(S,F;p50) + nE * AUC(S,E;p50)) / (nF + nE)

    which is exact because the Mann-Whitney pair set of a corner partitions into the two
    cross-group pair families shown, each normalised by its own pair count.
    """
    r5 = _binary(_vector(correct_k5, "correct_k5"), "correct_k5")
    r50 = _binary(_vector(correct_k50, "correct_k50", r5.shape[0]), "correct_k50")
    p5 = _vector(conf_k5, "conf_k5", r5.shape[0])
    p50 = _vector(conf_k50, "conf_k50", r5.shape[0])
    tg = groups if groups is not None else transition_groups(r5, r50)
    counts = tg.counts
    if counts["G"] > 0:
        raise StructuralInvariantError(
            f"group reconstruction needs G == 0, got G={counts['G']}"
        )
    nS, nF, nE = counts["S"], counts["F"], counts["E"]
    if min(nS, nF, nE) == 0:
        raise ValueError(f"degenerate transition groups (need S,F,E > 0), got {counts}")

    parts = {
        "AUC_SE_p5": pairwise_auc(p5[tg.stable_correct], p5[tg.stable_error]),
        "AUC_FE_p5": pairwise_auc(p5[tg.lost], p5[tg.stable_error]),
        "AUC_SF_p5": pairwise_auc(p5[tg.stable_correct], p5[tg.lost]),
        "AUC_SE_p50": pairwise_auc(p50[tg.stable_correct], p50[tg.stable_error]),
        "AUC_FE_p50": pairwise_auc(p50[tg.lost], p50[tg.stable_error]),
        "AUC_SF_p50": pairwise_auc(p50[tg.stable_correct], p50[tg.lost]),
    }
    reconstructed = {
        "A00": (nS * parts["AUC_SE_p5"] + nF * parts["AUC_FE_p5"]) / (nS + nF),
        "A01": (nS * parts["AUC_SE_p50"] + nF * parts["AUC_FE_p50"]) / (nS + nF),
        "A10": (nF * parts["AUC_SF_p5"] + nE * parts["AUC_SE_p5"]) / (nF + nE),
        "A11": (nF * parts["AUC_SF_p50"] + nE * parts["AUC_SE_p50"]) / (nF + nE),
    }
    direct = dict(zip(("A00", "A10", "A01", "A11"), corner_aurocs(p5, r5, p50, r50)))
    residuals = {k: direct[k] - reconstructed[k] for k in direct}
    return {
        "counts": counts,
        "pairwise": parts,
        "direct": direct,
        "reconstructed": reconstructed,
        "residual": residuals,
        "max_abs_residual": float(max(abs(v) for v in residuals.values())),
    }


# ---------------------------------------------------------------------------
# signed share helper
# ---------------------------------------------------------------------------
def signed_share(component: float, total: float, *, floor: float = SHARE_DENOM_FLOOR) -> Optional[float]:
    """``component / total`` as a signed descriptive share, or ``None`` if unsafe.

    Deliberately unclipped: a share below 0 or above 1 means one component worsens
    reliability while the other partly offsets it, which is a finding, not an error.
    """
    total = float(total)
    if not math.isfinite(total) or abs(total) < float(floor):
        return None
    return float(component) / total


# ---------------------------------------------------------------------------
# selective-reliability error sources
# ---------------------------------------------------------------------------
def selective_error_sources(
    conf_k50: Any, groups: TransitionGroups, *, coverage_levels: Sequence[float] = (0.5, 0.8)
) -> List[Dict[str, Any]]:
    """Where the accepted errors at K50 come from: newly introduced (F) or persistent (E).

    Acceptance follows the frozen selective convention exactly: ``n_keep =
    max(1, ceil(coverage * n))`` rows taken in descending-confidence order with a stable
    sort, so ``1 - accuracy`` of the accepted set reproduces the published RER family.
    """
    p50 = _vector(conf_k50, "conf_k50", groups.n)
    n = p50.shape[0]
    order = np.argsort(-p50, kind="stable")
    rows: List[Dict[str, Any]] = []
    for level in coverage_levels:
        level = float(level)
        if not 0.0 < level <= 1.0:
            raise ValueError(f"coverage must be in (0, 1], got {level}")
        n_keep = max(1, int(math.ceil(level * n)))
        accepted = order[:n_keep]
        acc_S = int(np.count_nonzero(groups.stable_correct[accepted]))
        acc_F = int(np.count_nonzero(groups.lost[accepted]))
        acc_E = int(np.count_nonzero(groups.stable_error[accepted]))
        acc_G = int(np.count_nonzero(groups.gained[accepted]))
        n_err = acc_F + acc_E
        rows.append({
            "coverage": level,
            "n_accepted": int(n_keep),
            "accepted_S": acc_S,
            "accepted_F_new_errors": acc_F,
            "accepted_E_persistent_errors": acc_E,
            "accepted_G": acc_G,
            "accepted_error_count": int(n_err),
            "risk": 1.0 - (acc_S + acc_G) / float(n_keep),
            "fraction_of_accepted_errors_from_F": (acc_F / n_err) if n_err else None,
            "fraction_of_accepted_errors_from_E": (acc_E / n_err) if n_err else None,
        })
    return rows


# ---------------------------------------------------------------------------
# cross-family gap decomposition
# ---------------------------------------------------------------------------
def cross_family_gap(
    family_a: str, comps_a: Mapping[str, float], family_b: str, comps_b: Mapping[str, float],
    *, floor: float = SHARE_DENOM_FLOOR,
) -> Dict[str, Any]:
    """``D_total(a) - D_total(b)`` split into its label and confidence parts.

    ``comps_a`` / ``comps_b`` need only the three keys ``D_total``, ``D_label``,
    ``D_conf`` (a :class:`Decomposition` exposes them through :meth:`Decomposition.as_dict`).
    The gap identity is exact because the Shapley identity holds per family and
    subtraction of two exact identities is exact.
    """
    a_total, a_label, a_conf = (float(comps_a["D_total"]), float(comps_a["D_label"]),
                                float(comps_a["D_conf"]))
    b_total, b_label, b_conf = (float(comps_b["D_total"]), float(comps_b["D_label"]),
                                float(comps_b["D_conf"]))
    gap_total = a_total - b_total
    gap_label = a_label - b_label
    gap_conf = a_conf - b_conf
    return {
        "comparison": f"{family_a}_minus_{family_b}",
        "gap_total_degradation": gap_total,
        "gap_label_component": gap_label,
        "gap_conf_component": gap_conf,
        "gap_residual": gap_total - (gap_label + gap_conf),
        "share_of_gap_from_label": signed_share(gap_label, gap_total, floor=floor),
        "share_of_gap_from_confidence": signed_share(gap_conf, gap_total, floor=floor),
        "heavier_component": _heavier(gap_label, gap_conf),
    }


def _heavier(label_part: float, conf_part: float) -> str:
    """Descriptive ordering only - no tolerance, no threshold, no success label."""
    if label_part > conf_part:
        return "TRANSITION_HEAVIER"
    if conf_part > label_part:
        return "CONFIDENCE_CHANGE_HEAVIER"
    return "EXACT_TIE"


def component_ordering(dec: Decomposition) -> str:
    """Descriptive ordering of the two components of one family's degradation."""
    return _heavier(dec.d_label, dec.d_conf)


# ---------------------------------------------------------------------------
# image-cluster bootstrap of an arbitrary vector-valued statistic
# ---------------------------------------------------------------------------
def image_cluster_bootstrap(
    statistic: Callable[[NDArray], Mapping[str, float]],
    clusters: Any,
    *,
    n_replicates: int,
    seed: int,
    ci: float = 0.95,
    resample_unit: str = "image",
    keep_replicates: bool = False,
) -> Dict[str, Any]:
    """Percentile CI for every key of ``statistic``, drawn on whole clusters.

    ``statistic(draw)`` receives row indices (one shared draw, so all inputs of one
    family move together and every identity stays valid inside a replicate) and returns
    a mapping of named quantities.  The point estimate is the **same** estimator on the
    full row set, so point and interval describe one functional.  The sampler is the
    frozen ``phase0a._ClusterSampler`` design used by every C1 CI in this project.

    With ``keep_replicates`` the raw per-key replicate series are returned under
    ``"_replicates"`` so a caller can build replicate-index-aligned cross-family
    contrasts from the same draws (no second resampling loop).
    """
    from ccg.experiment.phase0a import _ClusterSampler  # local import: keeps algebra pure

    labels = np.asarray(clusters, dtype=np.int64).reshape(-1)
    n_rows = labels.shape[0]
    if n_rows < 2:
        raise ValueError(f"bootstrap needs at least 2 rows, got {n_rows}")
    reps = int(n_replicates)
    if reps < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    level = float(ci)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    point = dict(statistic(np.arange(n_rows, dtype=np.int64)))
    keys = sorted(point)
    values = np.full((reps, len(keys)), np.nan, dtype=np.float64)
    sampler = _ClusterSampler(labels)
    rng = np.random.default_rng(int(seed))
    for index in range(reps):
        draw = sampler.draw(rng)
        row = statistic(draw)
        for pos, key in enumerate(keys):
            value = row.get(key, float("nan"))
            values[index, pos] = float("nan") if value is None else float(value)

    alpha = (1.0 - level) / 2.0
    out: Dict[str, Any] = {}
    for pos, key in enumerate(keys):
        series = values[:, pos]
        finite = series[np.isfinite(series)]
        out[key] = {
            "point": float(point[key]),
            "boot_mean": float(finite.mean()) if finite.size else float("nan"),
            "ci_low": float(np.quantile(finite, alpha)) if finite.size else float("nan"),
            "ci_high": float(np.quantile(finite, 1.0 - alpha)) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "n_valid_replicates": int(finite.size),
            "n_replicates": reps,
            "ci_level": level,
            "resample_unit": resample_unit,
            "n_clusters": int(sampler.n_clusters),
            "n": int(n_rows),
            "seed": int(seed),
        }
    if keep_replicates:
        out["_replicates"] = {key: values[:, pos] for pos, key in enumerate(keys)}
    return out


def aligned_cross_family_gap_ci(
    reps_a: Mapping[str, NDArray], reps_b: Mapping[str, NDArray], *, key: str,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Replicate-index-aligned CI of ``reps_a[key] - reps_b[key]``.

    Descriptive only: the two families were resampled **independently** on their own
    cohorts (their row sets differ), so this interval is not a paired-expression test.
    """
    a = np.asarray(reps_a[key], dtype=np.float64)
    b = np.asarray(reps_b[key], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"replicate shapes differ: {a.shape} vs {b.shape}")
    diff = a - b
    alpha = (1.0 - float(ci)) / 2.0
    return {
        "gap_mean": float(diff.mean()),
        "gap_ci_low": float(np.quantile(diff, alpha)),
        "gap_ci_high": float(np.quantile(diff, 1.0 - alpha)),
        "n_replicates": int(diff.shape[0]),
        "ci_level": float(ci),
    }
