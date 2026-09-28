"""Score-only reliability features for the Phase 0.5 sufficiency audit (Amendment A6).

This module answers a single, deliberately narrow question: *how much reliability
information is already present in a candidate set's score vector*
``{s_1, ..., s_K}`` produced by a **frozen** grounding scorer?

Everything here therefore consumes **score matrices only** - a ``[n, K]`` float
array, one row per candidate set.  It never reads candidate embeddings, query
features, box geometry, objectness or any ground-truth metadata, and it never
touches a feature store.  That restriction is what makes a "stats-only" model a
fair, interpretable baseline: if simple score statistics already explain the
reliability shift, no candidate-aware representation is warranted.

Two normalisation layers are kept strictly separate (Amendment A6, section A6.3):

* **per-set** statistics computed inside one row (e.g. ``z_top1``); and
* **cross-row** standardisation whose ``mean``/``std`` are fitted on the
  ``reliability_train`` rows only and then applied unchanged to every ``K``
  (:class:`NormalizationFit`, :func:`normalize_fit`, :func:`normalize_apply`).

All public functions are pure numpy, numerically stable (softmax / log-sum-exp
subtract the row maximum) and return ``float64`` feature matrices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SCALAR_NAMES",
    "scalar_confidence",
    "stat_feature_names",
    "stat_features",
    "top_scores_feature_names",
    "top_scores_features",
    "top1_column",
    "entry_moments",
    "sds_pack",
    "NormalizationFit",
    "normalize_fit",
    "normalize_apply",
]

#: Guard added to per-set standard deviations so a constant row never divides by 0.
_EPS = 1e-8

#: The five L0 scalar confidences (``scalar_confidence`` return order).
SCALAR_NAMES: tuple[str, ...] = ("msp", "top1_score", "margin", "neg_entropy", "norm_entropy")


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _as_score_matrix(scores: np.ndarray) -> np.ndarray:
    """Coerce ``scores`` to a finite ``[n, K]`` float64 matrix (1-D -> one row)."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be a [n, K] matrix, got shape {matrix.shape}")
    if matrix.shape[1] < 1:
        raise ValueError("scores must contain at least one candidate per row (K >= 1)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scores must be finite (no NaN/inf)")
    return matrix


def _as_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    """Coerce a feature matrix to float64 ``[n, d]`` *without* rejecting NaN."""
    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"feature matrix must be 2-D [n, d], got shape {data.shape}")
    return data


def _require_temperature(temperature: float) -> float:
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
    return value


def _softmax_max(matrix: np.ndarray, temperature: float) -> np.ndarray:
    """Row-wise softmax of ``matrix / temperature`` (stable via max subtraction)."""
    scaled = matrix / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _logsumexp(matrix: np.ndarray) -> np.ndarray:
    """Stable ``ln sum_i exp(s_i)`` per row."""
    row_max = np.max(matrix, axis=1, keepdims=True)
    return np.squeeze(row_max, axis=1) + np.log(np.sum(np.exp(matrix - row_max), axis=1))


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy ``-sum p ln p`` (nats), safe when a probability underflows to 0."""
    log_p = np.zeros_like(probs)
    np.log(probs, out=log_p, where=probs > 0.0)
    return -np.sum(probs * log_p, axis=1)


def _descending(matrix: np.ndarray) -> np.ndarray:
    """Each row sorted in descending order of score."""
    return np.sort(matrix, axis=1)[:, ::-1]


# ---------------------------------------------------------------------------
# L0: scalar confidences
# ---------------------------------------------------------------------------
def scalar_confidence(scores: np.ndarray, *, temperature: float) -> dict[str, np.ndarray]:
    """The five score-only scalar confidences of every candidate set.

    Parameters
    ----------
    scores
        ``[n, K]`` float score matrix (one row per candidate set).  ``K`` is read
        from ``scores.shape[1]``.
    temperature
        Positive scorer temperature ``T`` (the frozen corrected global ``T``);
        the probabilities always use ``softmax(scores / T)``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping :data:`SCALAR_NAMES` -> ``[n]`` float64 arrays:

        ``msp``
            ``max_i softmax(scores / T)[i]`` (numerically stable).
        ``top1_score``
            the largest raw score ``s_(1)`` (temperature-independent).
        ``margin``
            ``s_(1) - s_(2)`` - requires ``K >= 2`` (raises for ``K == 1``).
        ``neg_entropy``
            ``-H(P) = sum_i P_i ln P_i`` with ``P = softmax(scores / T)``
            (nats, ``<= 0``; higher = more confident, instruction section 7).
        ``norm_entropy``
            ``1 - H(P) / ln K`` (``1`` on a point mass, ``0`` when uniform).

    Notes
    -----
    Score-only: this function never touches candidate embeddings, geometry,
    objectness or GT.
    """
    matrix = _as_score_matrix(scores)
    temp = _require_temperature(temperature)
    n, k = matrix.shape
    if k < 2:
        raise ValueError(f"margin requires K >= 2, got K = {k}")

    probs = _softmax_max(matrix, temp)
    ordered = _descending(matrix)
    top1 = ordered[:, 0]
    margin = ordered[:, 0] - ordered[:, 1]
    entropy = _entropy(probs)
    neg_entropy = -entropy  # section 7: -H(P), higher = more confident
    norm_entropy = 1.0 - entropy / np.log(k)

    return {
        "msp": np.asarray(np.max(probs, axis=1), dtype=np.float64),
        "top1_score": np.asarray(top1, dtype=np.float64),
        "margin": np.asarray(margin, dtype=np.float64),
        "neg_entropy": np.asarray(neg_entropy, dtype=np.float64),
        "norm_entropy": np.asarray(np.broadcast_to(norm_entropy, (n,)), dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# L1: handcrafted set statistics
# ---------------------------------------------------------------------------
def stat_feature_names(
    *,
    include_logk: bool = True,
    include_entropy: bool = True,
    include_quantiles: bool = True,
) -> tuple[str, ...]:
    """Column names of :func:`stat_features`, in the exact output order.

    The optional blocks are ``log_k`` (``include_logk``), the entropy pair
    ``entropy`` / ``entropy_over_logk`` (``include_entropy``) and the quantile
    triple ``q25`` / ``q50`` / ``q75`` (``include_quantiles``); their position is
    fixed regardless of which blocks are enabled.
    """
    names: list[str] = []
    if include_logk:
        names.append("log_k")
    names += ["top1", "top2", "top3", "margin_12", "margin_23", "mean", "std", "msp"]
    if include_entropy:
        names += ["entropy", "entropy_over_logk"]
    names += ["logsumexp", "z_top1", "z_margin"]
    if include_quantiles:
        names += ["q25", "q50", "q75"]
    return tuple(names)


def stat_features(
    scores: np.ndarray,
    *,
    temperature: float,
    include_logk: bool = True,
    include_entropy: bool = True,
    include_quantiles: bool = True,
) -> np.ndarray:
    """Handcrafted per-set score statistics -> ``[n, d]`` float64 (``d <= 17``).

    Column order (and ``d``) always equals :func:`stat_feature_names` for the same
    flags.  Definitions per row (raw scores ``s``, ``P = softmax(s / T)``, ``K``
    the row width):

    ``log_k``
        ``ln K``.
    ``top1`` / ``top2`` / ``top3``
        ``s_(1)`` / ``s_(2)`` / ``s_(3)`` - requires ``K >= 3`` (raises otherwise).
    ``margin_12`` / ``margin_23``
        ``top1 - top2`` and ``top2 - top3``.
    ``mean`` / ``std``
        per-set mean and population std (``ddof=0``) of the raw scores.
    ``msp``
        ``max_i P_i``.
    ``entropy`` / ``entropy_over_logk``
        ``H(P)`` (nats) and ``H(P) / ln K``.
    ``logsumexp``
        ``ln sum_i exp(s_i)`` (stable).
    ``z_top1`` / ``z_margin``
        per-set standardised ``(top1 - mean) / (std + 1e-8)`` and
        ``(top1 - top2) / (std + 1e-8)`` (section 9 per-set statistics).
    ``q25`` / ``q50`` / ``q75``
        row quantiles with numpy's default ``linear`` interpolation.

    Score-only: only the score matrix is read.
    """
    matrix = _as_score_matrix(scores)
    temp = _require_temperature(temperature)
    n, k = matrix.shape
    if k < 3:
        raise ValueError(f"stat_features needs K >= 3 (top3 / margin_23), got K = {k}")

    ordered = _descending(matrix)
    top1 = ordered[:, 0]
    top2 = ordered[:, 1]
    top3 = ordered[:, 2]
    margin_12 = top1 - top2
    margin_23 = top2 - top3
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1, ddof=0)

    probs = _softmax_max(matrix, temp)
    msp = np.max(probs, axis=1)
    entropy = _entropy(probs)
    entropy_over_logk = entropy / np.log(k)
    lse = _logsumexp(matrix)

    denom = std + _EPS
    z_top1 = (top1 - mean) / denom
    z_margin = margin_12 / denom
    log_k = np.full(n, np.log(k), dtype=np.float64)

    columns: list[np.ndarray] = []
    if include_logk:
        columns.append(log_k)
    columns += [top1, top2, top3, margin_12, margin_23, mean, std, msp]
    if include_entropy:
        columns += [entropy, entropy_over_logk]
    columns += [lse, z_top1, z_margin]
    if include_quantiles:
        quantiles = np.quantile(matrix, [0.25, 0.5, 0.75], axis=1)
        columns += [quantiles[0], quantiles[1], quantiles[2]]

    return np.column_stack(columns).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# L1 variant: TopScores-M (section 14)
# ---------------------------------------------------------------------------
def top_scores_feature_names(*, top_m: int = 5) -> tuple[str, ...]:
    """Column names of :func:`top_scores_features` (``2 * top_m + 3`` columns)."""
    if int(top_m) < 1:
        raise ValueError(f"top_m must be >= 1, got {top_m!r}")
    m = int(top_m)
    names = [f"ts_z{i}" for i in range(1, m + 1)]
    names += [f"ts_g{i}{i + 1}" for i in range(1, m)]
    names.append(f"ts_g1{m}")
    names += ["ts_logk", "ts_mean", "ts_std"]
    return tuple(names)


def top_scores_features(scores: np.ndarray, *, top_m: int = 5) -> np.ndarray:
    """TopScores-M block -> ``[n, 2 * top_m + 3]`` float64 (section 14).

    Columns (see :func:`top_scores_feature_names`), computed per row after
    standardising the scores to ``z = (s - mean) / (std + 1e-8)`` where
    ``mean`` / ``std`` are the **per-set** moments of the full ``K`` scores:

    ``ts_z1 .. ts_z{top_m}``
        the ``top_m`` largest scores, standardised.
    ``ts_g12 .. ts_g{top_m-1}{top_m}``
        consecutive gaps ``z_i - z_{i+1}``.
    ``ts_g1{top_m}``
        the overall spread ``z_1 - z_{top_m}``.
    ``ts_logk`` / ``ts_mean`` / ``ts_std``
        ``ln K`` and the raw per-set mean / std (``ddof=0``).

    Requires ``K >= top_m`` (raises :class:`ValueError` otherwise).
    Score-only: only the score matrix is read.
    """
    m = int(top_m)
    if m < 1:
        raise ValueError(f"top_m must be >= 1, got {top_m!r}")
    matrix = _as_score_matrix(scores)
    n, k = matrix.shape
    if k < m:
        raise ValueError(f"top_scores_features needs K >= top_m = {m}, got K = {k}")

    top = _descending(matrix)[:, :m]
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1, ddof=0)
    z = (top - mean[:, None]) / (std[:, None] + _EPS)

    columns: list[np.ndarray] = [z[:, i] for i in range(m)]
    columns += [z[:, i] - z[:, i + 1] for i in range(m - 1)]
    columns.append(z[:, 0] - z[:, m - 1])
    columns += [np.full(n, np.log(k), dtype=np.float64), mean, std]

    return np.column_stack(columns).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# L2: ScoreDeepSets packing helpers
# ---------------------------------------------------------------------------
def top1_column(scores: np.ndarray) -> np.ndarray:
    """Per-set top-1 raw score ``s_(1)`` as a ``[n]`` float64 array."""
    matrix = _as_score_matrix(scores)
    return np.asarray(np.max(matrix, axis=1), dtype=np.float64)


def entry_moments(mats: Sequence[np.ndarray]) -> tuple[float, float]:
    """Pooled ``mean`` / ``std`` (``ddof=0``) of **all** entries across ``mats``.

    Used once on the ``reliability_train`` score matrices to fit the SDS score
    standardisation; every entry of every matrix contributes equally.  Raises on
    an empty input (the pools must be non-degenerate).
    """
    arrays = [np.asarray(m, dtype=np.float64).reshape(-1) for m in mats]
    if not arrays:
        raise ValueError("entry_moments() needs at least one matrix")
    pooled = np.concatenate(arrays)
    if pooled.size == 0:
        raise ValueError("entry_moments() received matrices with no entries")
    return float(np.mean(pooled)), float(np.std(pooled, ddof=0))


def sds_pack(mats: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad/vstack varying-width score sets into ``(padded [N, Kmax] float32, mask)``.

    ``mats`` is a sequence of ``[n_j, K_j]`` matrices (``K_j`` may differ, e.g.
    ``K=5`` mixed with ``K=10``).  Rows are stacked in order; all rows are padded
    to the widest ``Kmax`` with ``0.0``.  The boolean mask has ``True`` exactly on
    the real (non-padding) entries, so a downstream set model can ignore padding.
    """
    arrays: list[np.ndarray] = []
    for mat in mats:
        arr = np.asarray(mat, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"sds_pack() expects 2-D matrices, got shape {arr.shape}")
        arrays.append(arr)
    if not arrays:
        raise ValueError("sds_pack() needs at least one matrix")

    k_max = max(int(arr.shape[1]) for arr in arrays)
    n_total = int(sum(arr.shape[0] for arr in arrays))
    padded = np.zeros((n_total, k_max), dtype=np.float32)
    mask = np.zeros((n_total, k_max), dtype=bool)

    start = 0
    for arr in arrays:
        n, k = arr.shape
        padded[start : start + n, :k] = arr
        mask[start : start + n, :k] = True
        start += n
    return padded, mask


# ---------------------------------------------------------------------------
# cross-row standardisation (train-fitted, then frozen)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizationFit:
    """Frozen per-column ``mean`` / ``std`` (``ddof=0``) plus the column ``keys``."""

    mean: np.ndarray
    std: np.ndarray
    keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialisable payload (lists, not arrays) for experiment logs."""
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "keys": list(self.keys),
            "dim": int(mean.size),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NormalizationFit":
        """Inverse of :meth:`to_dict`."""
        mean = np.asarray(payload["mean"], dtype=np.float64)
        std = np.asarray(payload["std"], dtype=np.float64)
        keys = tuple(str(key) for key in payload["keys"])
        return cls(mean=mean, std=std, keys=keys)


def normalize_fit(
    matrix: np.ndarray,
    *,
    fit_rows: np.ndarray,
    keys: Sequence[str] | None = None,
) -> NormalizationFit:
    """Fit per-column ``mean`` / ``std`` (``ddof=0``) using **only** ``fit_rows``.

    ``matrix`` is a ``[n, d]`` feature matrix (e.g. the stacked L1 features); only
    the rows selected by ``fit_rows`` (an index array or boolean mask) are read.
    All other rows - including any containing ``NaN`` or junk - are never touched,
    so the fit is exactly the one computed on the clean ``reliability_train`` rows.

    ``keys`` stores the column names (e.g. :func:`stat_feature_names` or the
    top-scores names); when omitted, generic ``f0 .. f{d-1}`` names are used.
    """
    data = _as_feature_matrix(matrix)
    rows = _resolve_rows(fit_rows, data.shape[0])
    subset = data[rows]
    mean = np.mean(subset, axis=0)
    std = np.std(subset, axis=0, ddof=0)

    dim = data.shape[1]
    if keys is None:
        key_tuple = tuple(f"f{i}" for i in range(dim))
    else:
        key_tuple = tuple(str(key) for key in keys)
        if len(key_tuple) != dim:
            raise ValueError(
                f"keys has {len(key_tuple)} entries but the matrix has {dim} columns"
            )
    return NormalizationFit(mean=mean, std=std, keys=key_tuple)


def normalize_apply(matrix: np.ndarray, fit: NormalizationFit) -> np.ndarray:
    """Apply a frozen :class:`NormalizationFit`: ``(x - mean) / (std + 1e-8)``."""
    data = _as_feature_matrix(matrix)
    mean = np.asarray(fit.mean, dtype=np.float64).reshape(-1)
    std = np.asarray(fit.std, dtype=np.float64).reshape(-1)
    if mean.size != data.shape[1] or std.size != data.shape[1]:
        raise ValueError(
            f"fit dimension {mean.size} does not match matrix width {data.shape[1]}"
        )
    return (data - mean[None, :]) / (std[None, :] + _EPS)


def _resolve_rows(fit_rows: np.ndarray, n_rows: int) -> np.ndarray:
    """Normalise ``fit_rows`` (indices or boolean mask) to a 1-D int index array."""
    rows = np.asarray(fit_rows)
    if rows.ndim == 0:
        rows = rows.reshape(1)
    rows = rows.reshape(-1)
    if rows.dtype == np.bool_:
        if rows.size != n_rows:
            raise ValueError(
                f"boolean fit_rows has length {rows.size} but the matrix has {n_rows} rows"
            )
        rows = np.flatnonzero(rows)
    else:
        rows = rows.astype(np.int64, copy=False)
    if rows.size == 0:
        raise ValueError("normalize_fit() needs at least one fit row")
    if np.any(rows < 0) or np.any(rows >= n_rows):
        raise ValueError(f"fit_rows out of range [0, {n_rows})")
    return rows
