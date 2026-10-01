"""V2-M3 competition-adaptive reliability mixture: frozen formulas.

Amendment V2-M3 (``results/v2_local_competition/m3_mixture/protocol_m3.json``)
mixes two *frozen* reliability experts

    p_mix = (1 - alpha(A)) p_R + alpha(A) p_C

where the competition index ``A`` is built from three frozen cross-backbone
competition features through per-backbone train-only empirical CDFs

    u1 = F_train(winner_competitor_max_cos)
    u2 = F_train(winner_top2_cos)
    u3 = 1 - F_train(q_margin12)
    A  = (u1 + u2 + u3) / 3            in [0, 1]

and the adaptive weight is

    alpha(A) = sigmoid(beta (A - tau)),   beta = softplus(theta) >= 0.

Only ``theta`` and ``tau`` are trainable (2 scalars).  ``StaticMix`` (a single
fitted ``c``) and ``EqualMix`` (``c = 0.5``) are the frozen controls.  Nothing
here touches an expert: predictions are plain arrays and every function is a
pure arithmetic transform (amendment sections 3 / 6 / 9 / 27).

This module never trains, scores or evaluates a grounding model; mixture inputs
(``p_R``, ``p_C``, ``A``) are outputs of frozen experts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

__all__ = [
    "COMPETITION_FEATURES",
    "COMPETITION_DIRECTIONS",
    "MIXER_LEVELS",
    "softplus",
    "CompetitionIndex",
    "competition_features_from_sem14",
    "EqualMix",
    "StaticMix",
    "AdaptiveMix",
]

#: Frozen competition features of the index (order fixed; never searched).
COMPETITION_FEATURES: Tuple[str, ...] = (
    "winner_competitor_max_cos",
    "winner_top2_cos",
    "q_margin12",
)

#: Direction of every feature: +1 -- competition grows with the value;
#: -1 -- competition grows as the value shrinks.  Frozen (amendment section 4).
COMPETITION_DIRECTIONS: Tuple[int, ...] = (+1, +1, -1)

#: The mixer may only ever be fitted on these levels (amendment section 8).
MIXER_LEVELS: Tuple[int, ...] = (0, 2, 4)

_NLL_CLIP = 1e-12


def softplus(x: Any) -> np.ndarray:
    """Numerically stable ``log(1 + exp(x))``; strictly positive and >= 0."""
    values = np.asarray(x, dtype=np.float64)
    magnitude = np.log1p(np.exp(-np.abs(values)))
    return np.where(values > 0.0, values + magnitude, magnitude)


def _binary_nll(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), _NLL_CLIP, 1.0 - _NLL_CLIP)
    y = np.asarray(target, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _as_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_probability(values: np.ndarray, name: str) -> None:
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")


def _validate_groups(
    groups: Mapping[int, Mapping[str, np.ndarray]], *, require_a: bool
) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    """Validate the frozen fitting contract: levels {0,2,4} only, rows shared per level.

    Severity cells have different sizes in the real cohorts, so row counts may differ
    *between* levels; within one level ``p_r`` / ``p_c`` / ``y`` (and ``a`` when
    required) must all describe exactly the same rows.
    """
    if not groups:
        raise ValueError("no fitting groups supplied")
    levels = sorted(int(level) for level in groups)
    extra = sorted(set(levels) - set(MIXER_LEVELS))
    if extra:
        raise ValueError(
            f"mixer fitting touched levels {extra}: m=8 is never used for fitting "
            "(amendment V2-M3 section 8)"
        )
    blocks: List[Tuple[int, Dict[str, np.ndarray]]] = []
    for level in levels:
        block = groups[level]
        p_r = _as_vector(block["p_r"], f"m{level}.p_r")
        p_c = _as_vector(block["p_c"], f"m{level}.p_c")
        _validate_probability(p_r, f"m{level}.p_r")
        _validate_probability(p_c, f"m{level}.p_c")
        y = _as_vector(block["y"], f"m{level}.y")
        if not np.all(np.isin(np.unique(y), (0.0, 1.0))):
            raise ValueError(f"m{level}.y must be binary")
        if not (p_r.size == p_c.size == y.size):
            raise ValueError(
                f"m{level} blocks disagree on the row count "
                f"({p_r.size} / {p_c.size} / {y.size})"
            )
        entry: Dict[str, np.ndarray] = {"p_r": p_r, "p_c": p_c, "y": y}
        if require_a:
            a = _as_vector(block["a"], f"m{level}.a")
            if np.any(a < 0.0) or np.any(a > 1.0):
                raise ValueError(f"m{level}.a must lie in [0, 1]")
            if a.size != p_r.size:
                raise ValueError(f"m{level}.a disagrees on the row count")
            entry["a"] = a
        blocks.append((level, entry))
    return blocks


def _balanced_nll(blocks: Sequence[Tuple[int, Dict[str, np.ndarray]]], predict: Any) -> float:
    """``L = (1/3)(L_m0 + L_m2 + L_m4)``: equal weight per regime."""
    parts = []
    for _, block in blocks:
        probability = predict(block)
        parts.append(float(np.mean(_binary_nll(probability, block["y"]))))
    return float(np.mean(parts))


# ---------------------------------------------------------------------------
# competition index (per-backbone train-only empirical CDF transform)
# ---------------------------------------------------------------------------
class CompetitionIndex:
    """Empirical-CDF competition index of the three frozen features.

    ``fit`` may only be called once per backbone with that backbone's
    ``reliability_train`` m={0,2,4} rows; ``transform`` is a pure read-only
    map from raw feature values to ``A in [0, 1]`` (amendment section 5).
    """

    def __init__(self) -> None:
        self._sorted: Dict[str, np.ndarray] = {}

    @property
    def fitted(self) -> bool:
        return set(self._sorted) == set(COMPETITION_FEATURES)

    def fit(self, features: Mapping[str, np.ndarray]) -> "CompetitionIndex":
        keys = set(features)
        if keys != set(COMPETITION_FEATURES):
            raise ValueError(
                f"competition index features must be exactly {COMPETITION_FEATURES}, got {sorted(keys)}"
            )
        fitted: Dict[str, np.ndarray] = {}
        for name in COMPETITION_FEATURES:
            values = _as_vector(features[name], name)
            fitted[name] = np.sort(values)
        self._sorted = fitted
        return self

    def transform(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CompetitionIndex.transform called before fit")
        u: List[np.ndarray] = []
        for name, direction in zip(COMPETITION_FEATURES, COMPETITION_DIRECTIONS):
            values = np.asarray(features[name], dtype=np.float64).reshape(-1)
            sorted_values = self._sorted[name]
            rank = np.searchsorted(sorted_values, values, side="right").astype(np.float64)
            u.append(rank / float(sorted_values.size) if direction > 0 else 1.0 - rank / float(sorted_values.size))
        index = np.mean(np.stack(u, axis=0), axis=0)
        return np.clip(index, 0.0, 1.0)

    def summary(self) -> Dict[str, Any]:
        """Small serialisable digest of the fitted CDFs (for artifacts)."""
        return {
            name: {
                "n": int(self._sorted[name].size),
                "min": float(self._sorted[name][0]),
                "median": float(np.median(self._sorted[name])),
                "max": float(self._sorted[name][-1]),
            }
            for name in COMPETITION_FEATURES
        }


def competition_features_from_sem14(
    sem14: np.ndarray, sem14_names: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Extract the three frozen competition features from a sem14 feature block."""
    names = list(sem14_names)
    block = np.asarray(sem14, dtype=np.float64)
    missing = [name for name in COMPETITION_FEATURES if name not in names]
    if missing:
        raise ValueError(f"sem14 block misses competition features {missing}")
    return {name: block[:, names.index(name)] for name in COMPETITION_FEATURES}


# ---------------------------------------------------------------------------
# mixers
# ---------------------------------------------------------------------------
class EqualMix:
    """``p = 0.5 p_R + 0.5 p_C`` (no trainable parameter)."""

    name = "EqualMix"

    def n_parameters(self) -> int:
        return 0

    def predict_proba(self, p_r: Any, p_c: Any, a: Any = None) -> np.ndarray:
        p_r = np.asarray(p_r, dtype=np.float64)
        p_c = np.asarray(p_c, dtype=np.float64)
        if p_r.shape != p_c.shape:
            raise ValueError("p_r and p_c shapes differ")
        return 0.5 * (p_r + p_c)


@dataclass
class StaticMix:
    """``p = (1 - c) p_R + c p_C`` with a single fitted ``c in [0, 1]``."""

    c: float = 0.5
    fitted: bool = False
    nll: Optional[float] = None
    success: Optional[bool] = None
    name: str = field(default="StaticMix", init=False)

    def n_parameters(self) -> int:
        return 1

    def predict_proba(self, p_r: Any, p_c: Any, a: Any = None) -> np.ndarray:
        p_r = np.asarray(p_r, dtype=np.float64)
        p_c = np.asarray(p_c, dtype=np.float64)
        if p_r.shape != p_c.shape:
            raise ValueError("p_r and p_c shapes differ")
        c = float(self.c)
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"c must lie in [0, 1], got {c}")
        return (1.0 - c) * p_r + c * p_c

    def fit(self, groups: Mapping[int, Mapping[str, np.ndarray]]) -> "StaticMix":
        blocks = _validate_groups(groups, require_a=False)

        def objective(phi: np.ndarray) -> float:
            c = float(expit(phi[0]))
            return _balanced_nll(blocks, lambda block: (1.0 - c) * block["p_r"] + c * block["p_c"])

        result = minimize(objective, x0=np.array([0.0]), method="L-BFGS-B")
        self.c = float(expit(float(result.x[0])))
        self.fitted = True
        self.nll = float(result.fun)
        self.success = bool(result.success)
        return self

    def alpha(self, a: Any) -> np.ndarray:
        """Constant mixing weight (the static control has no A dependence)."""
        values = np.asarray(a, dtype=np.float64).reshape(-1)
        return np.full(values.shape, float(self.c), dtype=np.float64)


@dataclass
class AdaptiveMix:
    """``alpha(A) = sigmoid(softplus(theta) (A - tau))``; two trainable scalars."""

    theta: float = 0.0
    tau: float = 0.5
    fitted: bool = False
    nll: Optional[float] = None
    success: Optional[bool] = None
    name: str = field(default="AdaptiveMix", init=False)

    def n_parameters(self) -> int:
        return 2

    def beta(self) -> float:
        return float(softplus(self.theta))

    def alpha(self, a: Any) -> np.ndarray:
        values = np.asarray(a, dtype=np.float64).reshape(-1)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("competition index A must lie in [0, 1]")
        return expit(self.beta() * (values - float(self.tau)))

    def predict_proba(self, p_r: Any, p_c: Any, a: Any) -> np.ndarray:
        if a is None:
            raise ValueError("AdaptiveMix requires the competition index A")
        p_r = np.asarray(p_r, dtype=np.float64)
        p_c = np.asarray(p_c, dtype=np.float64)
        if p_r.shape != p_c.shape:
            raise ValueError("p_r and p_c shapes differ")
        weight = self.alpha(a)
        if weight.size != p_r.size:
            raise ValueError("A disagrees with the prediction row count")
        return (1.0 - weight) * p_r + weight * p_c

    def fit(self, groups: Mapping[int, Mapping[str, np.ndarray]]) -> "AdaptiveMix":
        blocks = _validate_groups(groups, require_a=True)

        def objective(params: np.ndarray) -> float:
            theta = float(params[0])
            tau = float(params[1])
            beta = float(softplus(theta))

            def predict(block: Dict[str, np.ndarray]) -> np.ndarray:
                weight = expit(beta * (block["a"] - tau))
                return (1.0 - weight) * block["p_r"] + weight * block["p_c"]

            return _balanced_nll(blocks, predict)

        result = minimize(objective, x0=np.array([0.0, 0.5]), method="L-BFGS-B")
        self.theta = float(result.x[0])
        self.tau = float(result.x[1])
        self.fitted = True
        self.nll = float(result.fun)
        self.success = bool(result.success)
        return self
