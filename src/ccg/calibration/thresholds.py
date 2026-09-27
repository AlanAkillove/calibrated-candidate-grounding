"""Abstention thresholds fitted on the calibration split (N0 / N1).

Sections 13 and 20: selective prediction and target-absence detection are
driven by a single scalar decision - *accept* the top candidate when its
confidence (N0) or its top1-top2 margin (N1) exceeds a threshold chosen on
``val_calib``.  Pure numpy, no torch, and deliberately independent of
``ccg.metrics``: the risk-coverage *curve reporting* belongs there, while this
module only answers "which threshold satisfies the target".

Definitions used here (kept identical to the protocol):

``coverage(tau) = mean(score >= tau)``
``risk(tau) = 1 - accuracy`` over the accepted examples, with
``accuracy(tau) = mean(correctness[score >= tau])``.

``correctness`` is the binary top-label indicator ``1[argmax p_i == c*]`` for
target-present evaluation, or ``1[target present and chosen correctly]`` for
Phase 2 abstention - the maths is the same, the semantic must be stated in the
experiment record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ThresholdSelection",
    "max_confidences",
    "top1_top2_margins",
    "coverage_risk_curve",
    "fit_max_conf_threshold",
    "fit_margin_threshold",
    "accept_mask",
]

#: Fine threshold grid: confidences are continuous, so a dense grid lets the
#: achieved coverage land within ~1/M of the requested coverage instead of
#: snapping to the next observed value.
DEFAULT_GRID_POINTS = 2000


@dataclass
class ThresholdSelection:
    """A threshold plus the calibration-split operating point it produces."""

    threshold: float
    coverage: float
    risk: float
    accuracy: float
    criterion: str
    target_value: Optional[float] = None
    satisfied: bool = True
    num_examples: int = 0
    num_accepted: int = 0
    score_kind: str = "max_confidence"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "threshold": float(self.threshold),
            "coverage": float(self.coverage),
            "risk": float(self.risk),
            "accuracy": float(self.accuracy),
            "criterion": self.criterion,
            "target_value": None if self.target_value is None else float(self.target_value),
            "satisfied": bool(self.satisfied),
            "num_examples": int(self.num_examples),
            "num_accepted": int(self.num_accepted),
            "score_kind": self.score_kind,
            "notes": list(self.notes),
        }


def accept_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of examples whose score reaches ``threshold`` (accept side)."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    return values >= float(threshold)


def max_confidences(logits: Sequence[np.ndarray]) -> np.ndarray:
    """``max_i softmax(logits)_i`` per candidate set (N0 score)."""
    out = np.empty(len(logits), dtype=np.float64)
    for index, item in enumerate(logits):
        values = np.asarray(item, dtype=np.float64).reshape(-1)
        exp = np.exp(values - values.max())
        probs = exp / exp.sum()
        out[index] = probs.max()
    return out


def top1_top2_margins(logits: Sequence[np.ndarray], kind: str = "prob") -> np.ndarray:
    """Top1-top2 margin per candidate set (N1 score).

    ``kind="prob"`` uses softmax probabilities (``p1 - p2``), ``kind="score"``
    the raw logit difference.  The choice must be fixed in the protocol and
    recorded; both are provided because they induce different abstention
    patterns when ``K`` changes (the softmax denominator shrinks every
    probability as ``K`` grows, while logit margins do not).
    """
    if kind not in ("prob", "score"):
        raise ValueError(f"kind must be 'prob' or 'score', got {kind!r}")
    out = np.empty(len(logits), dtype=np.float64)
    for index, item in enumerate(logits):
        values = np.sort(np.asarray(item, dtype=np.float64).reshape(-1))[::-1]
        if values.size < 2:
            raise ValueError("margin needs at least two candidates")
        if kind == "score":
            out[index] = values[0] - values[1]
        else:
            exp = np.exp(values - values.max())
            probs = np.sort(exp / exp.sum())[::-1]
            out[index] = probs[0] - probs[1]
    return out


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Descending candidate thresholds: observed values plus a fine grid."""
    observed = np.unique(scores)
    grid = np.linspace(observed.min(), observed.max(), DEFAULT_GRID_POINTS)
    return np.unique(np.concatenate([observed, grid]))[::-1]


def coverage_risk_curve(
    scores: Sequence[float] | np.ndarray,
    correctness: Sequence[float] | np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
    *,
    score_kind: str = "max_confidence",
) -> Dict[str, np.ndarray]:
    """Evaluate coverage / risk / accuracy on a threshold grid."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if values.shape != labels.shape:
        raise ValueError(f"scores {values.shape} and correctness {labels.shape} must align")
    if values.size == 0:
        raise ValueError("coverage_risk_curve() needs at least one example")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contain non-finite values")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("correctness must be a 0/1 indicator")

    grid = _candidate_thresholds(values) if thresholds is None else np.asarray(
        list(thresholds), dtype=np.float64
    )
    coverage = np.empty(grid.size, dtype=np.float64)
    risk = np.full(grid.size, np.nan, dtype=np.float64)
    accuracy = np.full(grid.size, np.nan, dtype=np.float64)
    for index, tau in enumerate(grid):
        mask = values >= tau
        coverage[index] = float(mask.mean())
        if mask.any():
            accuracy[index] = float(labels[mask].mean())
            risk[index] = 1.0 - accuracy[index]
    return {
        "thresholds": grid,
        "coverage": coverage,
        "risk": risk,
        "accuracy": accuracy,
        "num_examples": np.full(grid.size, values.size),
        "score_kind": np.full(grid.size, score_kind, dtype="<U16"),
    }


def _select_threshold(
    scores: np.ndarray,
    correctness: np.ndarray,
    target_risk: Optional[float],
    target_coverage: Optional[float],
    min_coverage: float,
    score_kind: str,
) -> ThresholdSelection:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if (target_risk is None) == (target_coverage is None):
        raise ValueError(
            "exactly one of target_risk / target_coverage must be given "
            f"(got target_risk={target_risk}, target_coverage={target_coverage})"
        )
    if not 0.0 < float(min_coverage) <= 1.0:
        raise ValueError(f"min_coverage must be in (0,1], got {min_coverage}")

    curve = coverage_risk_curve(values, labels, score_kind=score_kind)
    grid, coverage, risk = curve["thresholds"], curve["coverage"], curve["risk"]
    valid = (coverage >= float(min_coverage)) & np.isfinite(risk)
    if not np.any(valid):
        raise ValueError(
            f"no threshold keeps coverage >= min_coverage={min_coverage}; the calibration "
            "split is too small or too concentrated to fit an abstention rule"
        )

    notes: List[str] = []
    if target_coverage is not None:
        desired = float(target_coverage)
        if not 0.0 < desired <= 1.0:
            raise ValueError(f"target_coverage must be in (0,1], got {desired}")
        distance = np.abs(coverage - desired)
        distance = np.where(valid, distance, np.inf)
        best = float(np.min(distance))
        candidates = np.nonzero(np.isclose(distance, best, rtol=0, atol=1e-12))[0]
        # tie-break: lowest risk among the equally-close operating points
        chosen = candidates[int(np.argmin(np.where(np.isfinite(risk[candidates]), risk[candidates], np.inf)))]
        satisfied = best <= 0.02 + 1e-9
        criterion = "target_coverage"
        target_value = desired
        if not satisfied:
            notes.append(
                f"nearest achievable coverage differs from the target by {best:.4f} (> 0.02); "
                "the calibration score distribution is too coarse for this target"
            )
    else:
        allowed = float(target_risk)
        if not 0.0 <= allowed < 1.0:
            raise ValueError(f"target_risk must be in [0,1), got {allowed}")
        feasible = valid & (risk <= allowed)
        criterion = "target_risk"
        target_value = allowed
        if np.any(feasible):
            chosen = int(np.argmax(np.where(feasible, coverage, -np.inf)))
            satisfied = True
        else:
            # Nothing satisfies the risk budget: report the minimum-risk point and
            # flag it, rather than pretending the target was met.
            chosen = int(np.argmin(np.where(valid, risk, np.inf)))
            satisfied = False
            notes.append(
                f"no threshold achieves risk <= {allowed} at coverage >= {min_coverage}; "
                "returning the minimum-risk operating point (satisfied=False)"
            )

    index = int(chosen)
    return ThresholdSelection(
        threshold=float(grid[index]),
        coverage=float(coverage[index]),
        risk=float(risk[index]),
        accuracy=float(curve["accuracy"][index]),
        criterion=criterion,
        target_value=target_value,
        satisfied=bool(satisfied),
        num_examples=int(values.size),
        num_accepted=int(round(float(coverage[index]) * values.size)),
        score_kind=score_kind,
        notes=notes,
    )


def fit_max_conf_threshold(
    confidence: Sequence[float] | np.ndarray,
    correctness: Sequence[float] | np.ndarray,
    *,
    target_risk: Optional[float] = None,
    target_coverage: Optional[float] = None,
    min_coverage: float = 0.05,
) -> ThresholdSelection:
    """N0: threshold on ``max_i P(c_i)`` fitted on the calibration split.

    Give exactly one target:

    ``target_coverage``
        operate as close as possible to that coverage (the achieved coverage is
        reported so a +/-2% tolerance can be checked).
    ``target_risk``
        accept the largest coverage whose risk stays <= ``target_risk``; when
        impossible, the minimum-risk point is returned with ``satisfied=False``
        - an unreachable risk budget is a *finding*, not something to hide.
    """
    return _select_threshold(
        confidence,
        correctness,
        target_risk,
        target_coverage,
        min_coverage=min_coverage,
        score_kind="max_confidence",
    )


def fit_margin_threshold(
    margins: Sequence[float] | np.ndarray,
    correctness: Sequence[float] | np.ndarray,
    *,
    target_risk: Optional[float] = None,
    target_coverage: Optional[float] = None,
    min_coverage: float = 0.05,
    score_kind: str = "margin",
) -> ThresholdSelection:
    """N1: same selection on top1-top2 margins instead of max confidence."""
    return _select_threshold(
        margins,
        correctness,
        target_risk,
        target_coverage,
        min_coverage=min_coverage,
        score_kind=score_kind,
    )
