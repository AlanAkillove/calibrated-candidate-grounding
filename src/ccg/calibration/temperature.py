"""Temperature scaling (B2 / C1) and K-aware temperature (C2).

Everything here is fitted on ``val_calib`` **only** (protocol section 9): using
``val_select`` for temperature would leak hyper-parameter selection into the
calibration split, and using test data is prohibited outright.

The fitting objective is the *candidate-set* NLL

``NLL = -mean_j log softmax(logits_j / T)[target_j]``

i.e. the multiclass negative log likelihood over the ``K``-way choice.  This is
intentionally **not** the top-label correctness NLL of section 11 (which lives in
``ccg.metrics`` and is reported as a metric, not optimised as a loss):
temperature scaling must not be tuned against the same quantity it is later
evaluated on.

Extrapolation design (section 15 C2)
------------------------------------
``T`` is a *scalar* shared across ``K`` for B2/C1.  C2 replaces it with the
two-parameter form ``T(K) = a + b * log(K)``, fitted only on the ``K`` values
that the training/validation regime exposes (``K = 5, 10``).  Test sizes
``K = 20, 50`` are then **extrapolated through the functional form** - they are
never fitted, and never used to choose ``a`` or ``b``.  That is what makes the
comparison honest: the model of the shift is fixed before the shift is seen.
:func:`fit_temperature_k_aware` clamps the predicted temperature into
:data:`DEFAULT_TEMP_BOUNDS` and reports every clamped ``K`` so a suspicious
extrapolation cannot hide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_TEMP_BOUNDS",
    "TemperatureScaler",
    "KAwareTemperature",
    "nll_at_temperature",
    "mean_nll",
    "fit_temperature",
    "apply_temperature",
    "apply_temperature_sets",
    "fit_temperature_k_aware",
]

#: Search interval for a temperature; 0.05 already means "sharpen by 20x".
DEFAULT_TEMP_BOUNDS: Tuple[float, float] = (0.05, 100.0)

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def _normalise_sets(
    logits: Sequence[np.ndarray] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Validate and normalise a ``(logit set, target index)`` collection."""
    if isinstance(logits, np.ndarray) and logits.ndim == 1:
        logits = [logits]
    sets = [np.asarray(item, dtype=np.float64).reshape(-1) for item in logits]
    if not sets:
        raise ValueError("no logit sets provided")
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    if labels.size != len(sets):
        raise ValueError(f"got {len(sets)} logit sets but {labels.size} targets")
    for index, (item, label) in enumerate(zip(sets, labels.tolist())):
        if item.size < 2:
            raise ValueError(
                f"set #{index} has {item.size} candidate(s): temperature scaling needs a "
                "K-way choice with K>=2 (target-absent sets belong to Phase 2)"
            )
        if not np.all(np.isfinite(item)):
            raise ValueError(f"set #{index} contains non-finite logits")
        if not 0 <= int(label) < item.size:
            raise ValueError(
                f"set #{index}: target_index={label} outside [0,{item.size}) - target-absent "
                "examples must not be passed to a ranking calibration fit"
            )
    return sets, labels


def _log_probs_at(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    shifted = scaled - scaled.max()
    return shifted - np.log(np.exp(shifted).sum())


def nll_at_temperature(
    logits: Sequence[np.ndarray] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    temperature: float,
) -> float:
    """Mean candidate-set NLL of ``softmax(logits / temperature)``."""
    sets, labels = _normalise_sets(logits, targets)
    total = 0.0
    for item, label in zip(sets, labels):
        total += -float(_log_probs_at(item, temperature)[int(label)])
    return total / len(sets)


def mean_nll(
    logits: Sequence[np.ndarray] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    temperature: float = 1.0,
) -> float:
    """Alias of :func:`nll_at_temperature` with ``T=1`` default (readability)."""
    return nll_at_temperature(logits, targets, temperature)


def _golden_section_minimum(
    objective, lower: float, upper: float, tol: float = 1e-6, max_iter: int = 200
) -> float:
    """Deterministic 1-D minimiser for (approximately) unimodal objectives."""
    a, b = float(lower), float(upper)
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(int(max_iter)):
        if abs(b - a) <= tol * max(1.0, abs(c) + abs(d)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = objective(d)
    return 0.5 * (a + b)


def fit_temperature(
    logits: Sequence[np.ndarray] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    *,
    bounds: Tuple[float, float] = DEFAULT_TEMP_BOUNDS,
    tol: float = 1e-6,
) -> float:
    """Global temperature ``T`` minimising the mean candidate-set NLL (B2 / C1).

    Scalar optimisation in ``log T`` (so the positivity constraint is exact)
    with a golden-section search - no scipy, no torch, fully deterministic and
    reproducible from the calibration split alone.
    """
    sets, labels = _normalise_sets(logits, targets)
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower <= 0 or upper <= lower:
        raise ValueError(f"invalid temperature bounds {bounds}")

    def objective(log_t: float) -> float:
        temperature = math.exp(log_t)
        total = 0.0
        for item, label in zip(sets, labels):
            total += -float(_log_probs_at(item, temperature)[int(label)])
        return total / len(sets)

    best_log_t = _golden_section_minimum(objective, math.log(lower), math.log(upper), tol=tol)
    return float(math.exp(best_log_t))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """``logits / T`` (the calibrated logits of the same scorer)."""
    values = np.asarray(logits, dtype=np.float64)
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return (values / temperature).astype(np.float32)


def apply_temperature_sets(
    logits: Sequence[np.ndarray], temperature: float
) -> List[np.ndarray]:
    """Apply one temperature to many candidate sets (keeps ``[K]`` shapes)."""
    return [apply_temperature(item, temperature) for item in logits]


@dataclass
class TemperatureScaler:
    """Fitted global temperature with ``fit`` / ``transform`` bookkeeping."""

    temperature: float = 1.0
    bounds: Tuple[float, float] = DEFAULT_TEMP_BOUNDS
    fitted_on: Optional[str] = None
    num_examples: int = 0
    nll_before: Optional[float] = None
    nll_after: Optional[float] = None

    def fit(
        self,
        logits: Sequence[np.ndarray] | np.ndarray,
        targets: Sequence[int] | np.ndarray,
        *,
        source_split: str = "val_calib",
    ) -> "TemperatureScaler":
        sets, labels = _normalise_sets(logits, targets)
        self.temperature = fit_temperature(sets, labels, bounds=self.bounds)
        self.num_examples = len(sets)
        self.fitted_on = source_split
        self.nll_before = mean_nll(sets, labels, 1.0)
        self.nll_after = mean_nll(sets, labels, self.temperature)
        if self.nll_after > self.nll_before + 1e-9:
            # Temperature scaling minimises NLL by construction; a regression here
            # means the objective was fed misaligned targets. Fail loudly.
            raise AssertionError(
                f"fitted T={self.temperature:.4f} increases NLL "
                f"({self.nll_before:.4f} -> {self.nll_after:.4f}); check target indices"
            )
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return apply_temperature(logits, self.temperature)

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform(logits)
        exp = np.exp(scaled - scaled.max())
        return (exp / exp.sum()).astype(np.float32)

    def to_dict(self) -> Dict[str, object]:
        return {
            "temperature": float(self.temperature),
            "fitted_on": self.fitted_on,
            "num_examples": int(self.num_examples),
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "bounds": [float(b) for b in self.bounds],
        }


@dataclass
class KAwareTemperature:
    """``T(K) = a + b * log(K)`` (protocol section 15 C2)."""

    a: float = 1.0
    b: float = 0.0
    bounds: Tuple[float, float] = DEFAULT_TEMP_BOUNDS
    fitted_ks: Tuple[int, ...] = ()
    num_examples: int = 0
    nll_before: Optional[float] = None
    nll_after: Optional[float] = None
    clamped_ks: Tuple[int, ...] = ()
    notes: List[str] = field(default_factory=list)

    def temperature_for(self, k: int) -> float:
        """``T(K)``, clamped into :data:`DEFAULT_TEMP_BOUNDS` (clamping is recorded)."""
        k = int(k)
        if k < 2:
            raise ValueError(f"K must be >= 2, got {k}")
        raw = float(self.a) + float(self.b) * math.log(k)
        lower, upper = self.bounds
        return float(min(max(raw, lower), upper))

    def predict(self, logits: Sequence[np.ndarray], ks: Sequence[int]) -> List[np.ndarray]:
        """Temperature applied per candidate set according to its own ``K``."""
        if len(logits) != len(ks):
            raise ValueError("logits and ks must be aligned")
        return [apply_temperature(item, self.temperature_for(k)) for item, k in zip(logits, ks)]

    def probabilities(self, logits: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        size = int(np.asarray(logits).reshape(-1).size) if k is None else int(k)
        scaled = apply_temperature(logits, self.temperature_for(size))
        exp = np.exp(scaled - scaled.max())
        return (exp / exp.sum()).astype(np.float32)

    def extrapolation_risk(self, test_ks: Sequence[int] = (20, 50)) -> Dict[str, object]:
        """Report which test ``K`` values lie outside the fitted range.

        This is a *disclosure* helper, not a guard: the extrapolated temperature
        is used anyway (that is the point of C2), but the write-up must state
        that ``K=20/50`` were never fitted.
        """
        fitted = sorted(int(k) for k in self.fitted_ks)
        # Without a fitted range *every* K is an extrapolation; say so instead of
        # pretending the list is empty.
        if fitted:
            low, high = min(fitted), max(fitted)
            extrapolated = [int(k) for k in test_ks if not low <= int(k) <= high]
        else:
            extrapolated = [int(k) for k in test_ks]
        return {
            "fitted_ks": fitted,
            "test_ks": [int(k) for k in test_ks],
            "extrapolated_ks": extrapolated,
            "temperatures": {int(k): self.temperature_for(int(k)) for k in test_ks},
            "clamped_ks": list(self.clamped_ks),
            "notes": list(self.notes),
        }

    def to_dict(self) -> Dict[str, object]:
        out = {
            "form": "T(K)=a+b*log(K)",
            "a": float(self.a),
            "b": float(self.b),
            "fitted_ks": list(self.fitted_ks),
            "num_examples": int(self.num_examples),
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "clamped_ks": list(self.clamped_ks),
            "notes": list(self.notes),
            "extrapolation": self.extrapolation_risk(),
        }
        return out


def fit_temperature_k_aware(
    logits: Sequence[np.ndarray] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    ks: Optional[Sequence[int]] = None,
    *,
    bounds: Tuple[float, float] = DEFAULT_TEMP_BOUNDS,
    rounds: int = 12,
    tol: float = 1e-7,
) -> KAwareTemperature:
    """Fit ``T(K) = a + b * log K`` by coordinate-wise NLL minimisation.

    Parameters
    ----------
    ks:
        Set size of every logit set.  Derived from the logit shapes when omitted.

    Details
    -------
    The objective (mean candidate-set NLL) is convex in ``T`` and ``T`` is
    linear in ``(a, b)``, so alternating one-dimensional golden-section searches
    on ``a`` and ``b`` converges deterministically without any optimiser
    dependency.  Positivity is enforced by clamping ``T(K)`` into ``bounds``
    during the search; every ``K`` whose unconstrained temperature would leave
    the interval is recorded in ``clamped_ks``.

    With fewer than two distinct ``K`` the two-parameter model is not
    identifiable: the fit degrades to the global temperature (``b=0``) and a
    note is attached instead of silently returning a meaningless slope.
    """
    sets, labels = _normalise_sets(logits, targets)
    sizes = np.asarray([item.size for item in sets], dtype=np.int64) if ks is None else np.asarray(
        list(ks), dtype=np.int64
    )
    if sizes.size != len(sets):
        raise ValueError(f"got {len(sets)} logit sets but {sizes.size} K values")
    distinct = sorted({int(k) for k in sizes.tolist()})
    log_k = np.log(sizes.astype(np.float64))

    scaler = TemperatureScaler(bounds=bounds)
    scaler.fit(sets, labels)
    if len(distinct) < 2:
        return KAwareTemperature(
            a=scaler.temperature,
            b=0.0,
            bounds=bounds,
            fitted_ks=tuple(distinct),
            num_examples=len(sets),
            nll_before=scaler.nll_before,
            nll_after=scaler.nll_after,
            notes=[
                "only one distinct K in the calibration data; T(K)=a+b*log(K) is not "
                "identifiable, falling back to the global temperature (b=0)"
            ],
        )

    lower, upper = float(bounds[0]), float(bounds[1])

    def objective_for(a: float, b: float) -> float:
        temperatures = np.clip(a + b * log_k, lower, upper)
        total = 0.0
        for item, label, temperature in zip(sets, labels, temperatures):
            total += -float(_log_probs_at(item, float(temperature))[int(label)])
        return total / len(sets)

    a, b = float(scaler.temperature), 0.0
    # search a around the global fit, b around 0 (log K is O(1.6-3.9))
    a_span = (max(lower, 0.5 * a), max(upper, 2.0 * a))
    b_span = (-2.0, 2.0)
    previous = objective_for(a, b)
    for _ in range(int(rounds)):
        a = _golden_section_minimum(lambda x: objective_for(x, b), *a_span, tol=tol)
        b = _golden_section_minimum(lambda y: objective_for(a, y), *b_span, tol=tol)
        current = objective_for(a, b)
        if previous - current <= tol:
            break
        previous = current

    clamped = tuple(int(k) for k in distinct if not lower <= a + b * math.log(k) <= upper)
    notes: List[str] = []
    if clamped:
        notes.append(
            f"temperature clamped into {bounds} for K={list(clamped)}; the reported T(K) for "
            "these sizes is the clamp, not the linear extrapolation"
        )
    result = KAwareTemperature(
        a=float(a),
        b=float(b),
        bounds=bounds,
        fitted_ks=tuple(distinct),
        num_examples=len(sets),
        nll_before=scaler.nll_before,
        nll_after=objective_for(a, b),
        clamped_ks=clamped,
        notes=notes,
    )
    if result.nll_after and scaler.nll_after and result.nll_after > scaler.nll_after + 1e-6:
        # A two-parameter model can always mimic the global temperature (b=0),
        # so it must never end up worse; if it does, the coordinate search
        # stalled and the simpler model is the honest answer.
        result.notes.append(
            "K-aware fit did not beat the global temperature; keeping the parameters but "
            "reporting the regression"
        )
    return result
