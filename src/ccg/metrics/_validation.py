"""Shared input-validation helpers for the :mod:`ccg.metrics` sub-package.

Every public metric in :mod:`ccg.metrics` funnels its inputs through the helpers
below so that error messages stay consistent, torch tensors are accepted without
importing torch, and the "target absent" (``target_index == -1``) convention is
handled the same way everywhere.

Target-absent convention
------------------------
A row of ``target_index`` equal to ``-1`` means the ground-truth candidate is not
part of the candidate set (synthetic omission or natural proposal miss). Such
rows have no well-defined ranking metric, so:

* ``strict=True`` (default): a :class:`ValueError` is raised, listing how many
  absent rows were received. Callers that *do* want to evaluate on them should
  first route them to :mod:`ccg.metrics.absence`.
* ``strict=False``: absent rows are silently dropped from numerator and
  denominator (i.e. metrics are computed over target-present rows only).
* ``mask``: an explicit boolean row selector. It is AND-ed with the
  target-present selector, so it can also drop malformed rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "as_correctness",
    "as_int_array",
    "require_same_length",
    "resolve_rows",
    "to_numpy_1d",
    "to_numpy_2d",
    "to_probability_1d",
    "trapz",
]


def _unwrap(values: Any, name: str) -> Any:
    """Convert tensors / array-likes to something ``np.asarray`` understands."""
    if values is None:
        raise ValueError(f"`{name}` must not be None")
    detach = getattr(values, "detach", None)
    if callable(detach):  # torch.Tensor and friends, without importing torch
        values = detach()
        cpu = getattr(values, "cpu", None)
        if callable(cpu):
            values = cpu()
        as_numpy = getattr(values, "numpy", None)
        if callable(as_numpy):
            values = as_numpy()
    return values


def to_numpy_1d(values: Any, name: str, dtype: Any = np.float64) -> NDArray:
    """Return ``values`` as a non-empty 1-D ``np.ndarray`` of ``dtype``.

    Scalars are promoted to length-1 arrays; a 2-D column/row vector is flattened.
    """
    arr = np.asarray(_unwrap(values, name), dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            raise ValueError(f"`{name}` must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"`{name}` is empty")
    return arr


def to_numpy_2d(values: Any, name: str, dtype: Any = np.float64) -> NDArray:
    """Return ``values`` as a finite ``[M, K]`` array with ``M >= 1`` and ``K >= 1``."""
    arr = np.asarray(_unwrap(values, name), dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D [M, K], got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"`{name}` must have at least one row and one column, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` contains NaN or inf values")
    return arr


def to_probability_1d(values: Any, name: str, tol: float = 1e-9) -> NDArray:
    """Validate that ``values`` are probabilities inside ``[0, 1]`` (clipped on return)."""
    arr = to_numpy_1d(values, name, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` contains NaN or inf values")
    if np.any(arr < -tol) or np.any(arr > 1.0 + tol):
        raise ValueError(f"`{name}` must lie in [0, 1]; got range [{arr.min()}, {arr.max()}]")
    return np.clip(arr, 0.0, 1.0)


def as_int_array(values: Any, name: str) -> NDArray:
    """Return ``values`` as an ``int64`` 1-D array, rejecting fractional floats."""
    arr = to_numpy_1d(values, name, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` contains NaN or inf values")
    if np.any(np.abs(arr - np.round(arr)) > 1e-9):
        raise ValueError(f"`{name}` must contain integer indices, got fractional values")
    return np.round(arr).astype(np.int64)


def require_same_length(a: NDArray, b: NDArray, name_a: str, name_b: str) -> None:
    """Raise ``ValueError`` when two 1-D arrays disagree on length."""
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"`{name_a}` ({a.shape[0]}) and `{name_b}` ({b.shape[0]}) differ in length"
        )


def resolve_rows(
    target_index: NDArray,
    n_rows: int,
    mask: Any = None,
    strict: bool = True,
) -> NDArray:
    """Return a boolean ``[n_rows]`` array of rows to keep.

    See the module docstring for the semantics of ``strict`` and ``mask``.
    """
    if target_index.shape[0] != n_rows:
        raise ValueError(
            f"`target_index` ({target_index.shape[0]}) and `scores` rows ({n_rows}) "
            "differ in length"
        )
    keep = np.ones(n_rows, dtype=bool)
    if mask is not None:
        m = np.asarray(_unwrap(mask, "mask"))
        if m.dtype != np.dtype("bool"):
            m = m.astype(np.float64).astype(bool)
        m = m.reshape(-1)
        if m.shape[0] != n_rows:
            raise ValueError(f"`mask` ({m.shape[0]}) and `scores` rows ({n_rows}) differ in length")
        keep &= m
    absent = target_index < 0
    if bool(np.any(absent & keep)):
        if strict:
            raise ValueError(
                f"{int(np.sum(absent & keep))} row(s) have target_index < 0 (target absent). "
                "Ranking/calibration metrics are undefined there: pass `mask=` to select "
                "target-present rows, `strict=False` to drop absent rows automatically, or use "
                "ccg.metrics.absence for the target-absent evaluation."
            )
        keep &= ~absent
    if not bool(np.any(keep)):
        raise ValueError("No rows left to evaluate after applying `mask` / target-absent filtering")
    return keep


def as_correctness(values: Any, name: str = "correctness") -> NDArray:
    """Return ``values`` as float 0/1 correctness labels."""
    arr = to_numpy_1d(values, name, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` contains NaN or inf values")
    if np.any((arr != 0.0) & (arr != 1.0)):
        raise ValueError(f"`{name}` must be binary (0/1 or bool)")
    return arr


def trapz(y: NDArray, x: NDArray) -> float:
    """Trapezoidal integral of ``y`` over ``x`` (numpy 1.x / 2.x compatible)."""
    func = getattr(np, "trapezoid", None)
    if func is None:  # pragma: no cover - numpy < 2.0
        func = np.trapz  # type: ignore[attr-defined]
    return float(func(y, x))
