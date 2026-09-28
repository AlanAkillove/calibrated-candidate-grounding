"""Corrected global-temperature fitting for the Phase 0A.1 metric audit.

Phase 0A fitted the global temperature by a golden-section search over
``T in [0.05, 100]`` and landed on the *lower bound* (``T* = 0.05``): the true
optimum was below the search interval, so the reported temperature was an
artefact of the grid rather than a fit.  This module redoes the fit so that
such a bound collision is (a) detected, (b) mitigated once by widening the
interval, and (c) reported honestly when it cannot be repaired.

Objective
---------
The objective is the *mean candidate-set NLL* of the temperature-rescaled
softmax, i.e. for candidate sets :math:`s^{(j)}` (row :math:`j`, ``K_j``
candidates) and targets :math:`t_j`:

.. math::

    \\mathrm{NLL}(T) = \\frac{1}{N} \\sum_j
        -\\log \\mathrm{softmax}(s^{(j)} / T)[t_j]

The optimisation runs in **log space** (``u = ln T``) so positivity is exact:
``scipy.optimize.minimize_scalar(bounds=(ln t_min, ln t_max), method="bounded")``.
No coarse grid is used anywhere - the bound collision of Phase 0A is exactly
the failure mode a grid invites.

Interior criterion and one-shot expansion
-----------------------------------------
A fit is *interior* when the optimum is at least one decade away from **both**
bounds: ``T* / t_min >= 10`` and ``t_max / T* >= 10``.  When the first fit is
not interior and ``expand_once`` is set, the interval is widened **once** to
``(t_min / expansion, t_max * expansion)`` and the fit is repeated; the
expansion is recorded (``bounds_expanded`` / ``n_expansions``).  If the fit is
still not interior afterwards the result carries
:data:`TEMPERATURE_OPTIMIZATION_WARNING` and the colliding side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

__all__ = [
    "DEFAULT_EXPANSION",
    "DEFAULT_T_MAX",
    "DEFAULT_T_MIN",
    "METHOD",
    "TEMPERATURE_OPTIMIZATION_WARNING",
    "TemperatureFit",
    "fit_temperature_log_space",
    "nll_of_temperature",
]

#: Warning attached to a fit whose optimum is still not interior after the
#: optional single expansion.
TEMPERATURE_OPTIMIZATION_WARNING = "TEMPERATURE_OPTIMIZATION_WARNING"
#: Default (initial) search interval for the temperature.
DEFAULT_T_MIN = 1e-3
DEFAULT_T_MAX = 10.0
#: Default widening factor of the single expansion (two extra decades per side).
DEFAULT_EXPANSION = 100.0
#: Human-readable description of the optimiser, recorded for provenance.
METHOD = "scipy.optimize.minimize_scalar(bounded) on u=lnT"
#: Number of decades required on each side for an optimum to count as interior.
_DECADES = 10.0


@dataclass(frozen=True)
class TemperatureFit:
    """One fitted scalar temperature plus its bound/expansion provenance.

    All floats are FP64.  ``temperature = exp(u_opt)``; ``bounds`` is the
    interval actually searched (post-expansion when applicable) and
    ``bounds_initial`` the pre-expansion interval.  ``decades_to_low`` /
    ``decades_to_high`` are ``log10(T* / bound)`` on each side - both must be
    ``>= 1`` for ``interior`` to hold.
    """

    temperature: float
    u_opt: float
    bounds: Tuple[float, float]
    bounds_initial: Tuple[float, float]
    bounds_expanded: bool
    n_expansions: int
    interior: bool
    decades_to_low: float
    decades_to_high: float
    warning: Optional[str]
    objective: float
    n_sets: int
    nll_before: float
    nll_after: float
    method: str = METHOD

    @property
    def warning_detail(self) -> Optional[str]:
        """The warning with the colliding side spelled out (``None`` if interior)."""
        if self.warning is None:
            return None
        low_ratio = self.temperature / self.bounds[0]
        high_ratio = self.bounds[1] / self.temperature
        if low_ratio < _DECADES and high_ratio < _DECADES:  # pragma: no cover - degenerate
            side = "optimum collides with BOTH bounds"
        elif low_ratio < _DECADES:
            side = (
                f"optimum pinned at the LOWER bound "
                f"(T*={self.temperature:.6g}, t_min={self.bounds[0]:.6g})"
            )
        else:
            side = (
                f"optimum pinned at the UPPER bound "
                f"(T*={self.temperature:.6g}, t_max={self.bounds[1]:.6g})"
            )
        return f"{self.warning}: {side}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view used by the audit artifacts."""
        return {
            "method": self.method,
            "temperature": float(self.temperature),
            "u_opt": float(self.u_opt),
            "bounds": [float(self.bounds[0]), float(self.bounds[1])],
            "bounds_initial": [float(self.bounds_initial[0]), float(self.bounds_initial[1])],
            "bounds_expanded": bool(self.bounds_expanded),
            "n_expansions": int(self.n_expansions),
            "interior": bool(self.interior),
            "decades_to_low": float(self.decades_to_low),
            "decades_to_high": float(self.decades_to_high),
            "warning": self.warning,
            "warning_detail": self.warning_detail,
            "objective": float(self.objective),
            "n_sets": int(self.n_sets),
            "nll_before": float(self.nll_before),
            "nll_after": float(self.nll_after),
        }


def _coerce_sets(
    scores_per_set: Sequence[NDArray],
    targets: Sequence[int],
) -> Tuple[List[NDArray], NDArray]:
    """Validate ``(candidate sets, targets)`` and return FP64 copies + labels.

    Raises :class:`ValueError` on empty input, a length mismatch, an empty set,
    non-finite scores, or a target index outside its set.
    """
    if scores_per_set is None:
        raise ValueError("scores_per_set must not be None")
    sets = [np.asarray(item, dtype=np.float64).reshape(-1) for item in scores_per_set]
    if len(sets) == 0:
        raise ValueError("scores_per_set is empty: nothing to fit / evaluate")
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    if labels.shape[0] != len(sets):
        raise ValueError(f"got {len(sets)} candidate sets but {labels.shape[0]} targets")
    for index, (item, label) in enumerate(zip(sets, labels.tolist())):
        if item.size == 0:
            raise ValueError(f"set #{index} is empty")
        if not np.all(np.isfinite(item)):
            raise ValueError(f"set #{index} contains non-finite (nan/inf) scores")
        if not 0 <= int(label) < item.size:
            raise ValueError(
                f"set #{index}: target index {label} outside [0, {item.size})"
            )
    return sets, labels


def _mean_nll(sets: Sequence[NDArray], labels: NDArray, temperature: float) -> float:
    """Mean candidate-set NLL over already-validated FP64 sets (max-shifted)."""
    temp = float(temperature)
    total = 0.0
    for item, label in zip(sets, labels.tolist()):
        scaled = item / temp
        shifted = scaled - scaled.max()
        log_sum = math.log(float(np.exp(shifted).sum()))
        total += -(float(shifted[int(label)]) - log_sum)
    return total / len(sets)


def nll_of_temperature(
    scores_per_set: Sequence[NDArray],
    targets: Sequence[int],
    temperature: float,
) -> float:
    """Mean candidate-set NLL of ``softmax(scores / temperature)`` (FP64).

    Parameters
    ----------
    scores_per_set:
        Sequence of ``[K_j]`` score arrays (one per candidate set; the ``K_j``
        may differ).  Converted to FP64 and evaluated with max-subtraction, so
        the value is numerically stable for any temperature / score scale.
    targets:
        Sequence of int target indices, aligned with ``scores_per_set``.
    temperature:
        Strictly positive finite temperature ``T``.

    Raises
    ------
    ValueError
        On empty input, a length mismatch, an empty set, non-finite scores, a
        target index outside its set, or a non-positive / non-finite
        ``temperature``.
    """
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    sets, labels = _coerce_sets(scores_per_set, targets)
    return float(_mean_nll(sets, labels, temp))


def _is_interior(temperature: float, low: float, high: float) -> bool:
    """``T*`` at least one decade from *both* bounds (``>= 10x`` each side)."""
    return (temperature / low >= _DECADES) and (high / temperature >= _DECADES)


def fit_temperature_log_space(
    scores_per_set: Sequence[NDArray],
    targets: Sequence[int],
    *,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    expand_once: bool = True,
    expansion: float = DEFAULT_EXPANSION,
) -> TemperatureFit:
    """Fit one scalar temperature by bounded 1-D minimisation of the mean NLL.

    Optimises ``u = ln T`` in ``[ln t_min, ln t_max]`` with
    :func:`scipy.optimize.minimize_scalar` (``method="bounded"``); no grid is
    used.  When the optimum is not interior (see module docstring) and
    ``expand_once`` is true, the interval is widened once to
    ``(t_min / expansion, t_max * expansion)`` and the fit is repeated.

    Parameters
    ----------
    scores_per_set, targets:
        Candidate sets and targets (see :func:`nll_of_temperature`).
    t_min, t_max:
        Initial search interval (``0 < t_min < t_max``).
    expand_once:
        Whether to widen the interval a single time on a bound collision.
    expansion:
        Widening factor (``> 1``); ``100`` adds two decades per side.

    Returns
    -------
    TemperatureFit
        Fitted temperature plus bound / expansion / warning provenance.
    """
    t_lo0 = float(t_min)
    t_hi0 = float(t_max)
    if not (np.isfinite(t_lo0) and np.isfinite(t_hi0)) or t_lo0 <= 0.0 or t_hi0 <= t_lo0:
        raise ValueError(f"invalid temperature bounds ({t_min}, {t_max}); need 0 < t_min < t_max")
    factor = float(expansion)
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError(f"expansion must be a finite factor > 1, got {expansion}")

    sets, labels = _coerce_sets(scores_per_set, targets)
    n_sets = len(sets)

    def objective(u: float) -> float:
        return _mean_nll(sets, labels, math.exp(float(u)))

    def run_fit(low: float, high: float) -> Tuple[float, float]:
        result = minimize_scalar(
            objective,
            bounds=(math.log(low), math.log(high)),
            method="bounded",
        )
        return float(result.x), float(result.fun)

    bounds_initial = (t_lo0, t_hi0)
    low, high = t_lo0, t_hi0
    u_opt, obj = run_fit(low, high)
    temperature = math.exp(u_opt)
    bounds_expanded = False
    n_expansions = 0
    if not _is_interior(temperature, low, high) and expand_once:
        low, high = t_lo0 / factor, t_hi0 * factor
        u_opt, obj = run_fit(low, high)
        temperature = math.exp(u_opt)
        bounds_expanded = True
        n_expansions = 1

    interior = _is_interior(temperature, low, high)
    warning = None if interior else TEMPERATURE_OPTIMIZATION_WARNING
    nll_before = float(_mean_nll(sets, labels, 1.0))
    nll_after = float(_mean_nll(sets, labels, temperature))

    return TemperatureFit(
        temperature=float(temperature),
        u_opt=float(u_opt),
        bounds=(float(low), float(high)),
        bounds_initial=bounds_initial,
        bounds_expanded=bounds_expanded,
        n_expansions=n_expansions,
        interior=bool(interior),
        decades_to_low=float(math.log10(temperature / low)),
        decades_to_high=float(math.log10(high / temperature)),
        warning=warning,
        objective=float(obj),
        n_sets=int(n_sets),
        nll_before=nll_before,
        nll_after=nll_after,
    )
