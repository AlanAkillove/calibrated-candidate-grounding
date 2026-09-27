"""Paired bootstrap confidence intervals (protocol section 23).

Contract functions: :func:`paired_bootstrap`, :func:`bootstrap_ci`.

Why paired and image-clustered
------------------------------
Every model is evaluated on *exactly the same* test candidate sets, so the
per-sample statistics ``sample_stat_a`` and ``sample_stat_b`` are compared with a
*paired* bootstrap: the same resampled index set is applied to both arrays and
the difference of means is recorded. Because several referring expressions share
one image and are therefore strongly correlated, the recommended resampling unit
is the **image**: pass ``cluster_ids`` (= ``image_id`` per sample) and the
bootstrap resamples clusters instead of individual samples. Omitting it falls
back to the (anti-conservative) expression-level bootstrap; the returned dict
records which unit was used so that result tables can state it.

Reproducibility
---------------
All randomness goes through ``numpy.random.default_rng(seed)``. Two calls with
the same inputs and the same ``seed`` return byte-identical dicts, independently
of any global numpy seed or of the calling order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ccg.metrics._validation import _unwrap, to_numpy_1d

__all__ = ["bootstrap_ci", "paired_bootstrap"]

# cap on how many index rows are materialised at once (memory guard)
_MAX_INDEX_CELLS = 4_000_000


def _check_ci(ci: float) -> float:
    level = float(ci)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError(f"`ci` must be a confidence level in (0, 1), got {ci}")
    return level


def _check_replicates(n_replicates: int) -> int:
    n_rep = int(n_replicates)
    if n_rep < 1:
        raise ValueError(f"`n_replicates` must be >= 1, got {n_replicates}")
    return n_rep


def _cluster_groups(cluster_ids: Any, n: int) -> tuple[NDArray, list[NDArray]]:
    """Return ``(unique_clusters, member_index_lists)`` for a fixed ordering."""
    ids = np.asarray(_unwrap(cluster_ids, "cluster_ids"))
    if ids.ndim == 0:
        ids = ids.reshape(1)
    elif ids.ndim > 1:
        if 1 not in ids.shape:
            raise ValueError(f"`cluster_ids` must be 1-D, got shape {ids.shape}")
        ids = ids.reshape(-1)
    if ids.shape[0] != n:
        raise ValueError(f"`cluster_ids` ({ids.shape[0]}) and sample stats ({n}) differ in length")
    clusters, inverse = np.unique(ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    sizes = np.bincount(inverse, minlength=clusters.shape[0])
    groups = np.split(order, np.cumsum(sizes)[:-1])
    return clusters, [np.asarray(g) for g in groups]


def _draw_indices(
    rng: np.random.Generator,
    n: int,
    groups: Sequence[NDArray] | None,
) -> NDArray:
    """One bootstrap resample: cluster-level when ``groups`` is given, else sample-level."""
    if groups is None:
        return rng.integers(0, n, size=n)
    drawn = rng.integers(0, len(groups), size=len(groups))
    picked = [groups[int(g)] for g in drawn]
    return np.concatenate(picked)


def _replicate_stats(
    values: NDArray,
    statistic: Callable[[NDArray], float],
    n_rep: int,
    rng: np.random.Generator,
    groups: Sequence[NDArray] | None,
) -> NDArray:
    """Bootstrap distribution of ``statistic`` over ``n_rep`` resamples."""
    n = values.shape[0]
    out = np.empty(n_rep, dtype=np.float64)
    if groups is not None or n > _MAX_INDEX_CELLS:
        for i in range(n_rep):
            out[i] = float(statistic(values[_draw_indices(rng, n, groups)]))
        return out
    # vectorised path: build the index matrix in memory-bounded chunks
    chunk = max(1, int(_MAX_INDEX_CELLS // n))
    done = 0
    while done < n_rep:
        size = min(chunk, n_rep - done)
        idx = rng.integers(0, n, size=(size, n))
        for j in range(size):
            out[done + j] = float(statistic(values[idx[j]]))
        done += size
    return out


def _percentile_ci(samples: NDArray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(samples, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(low), float(high)


def _two_sided_hint(replicates: NDArray) -> float:
    """Bootstrap two-sided "p-value hint" for the null hypothesis ``diff == 0``.

    ``2 * min(P(rep <= 0), P(rep >= 0))`` over the percentile bootstrap
    distribution, clipped to ``[0, 1]``. It is the exact inversion of the
    percentile CI (hint < 1 - ci whenever 0 falls outside the interval), but it
    is still only a *hint*: it has no small-sample calibration, so the CI stays
    the primary evidence in the Gate Q1 / Q3 tables.
    """
    n = float(replicates.shape[0])
    lower = float(np.sum(replicates <= 0.0)) / n
    upper = float(np.sum(replicates >= 0.0)) / n
    return float(min(1.0, 2.0 * min(lower, upper)))


def paired_bootstrap(
    sample_stat_a: Any,
    sample_stat_b: Any,
    cluster_ids: Any = None,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    statistic: Callable[[NDArray], float] = np.mean,
) -> dict[str, Any]:
    """Paired bootstrap CI for ``statistic(a - b)`` of two models' per-sample stats.

    Parameters
    ----------
    sample_stat_a, sample_stat_b
        ``[N]`` per-sample statistics of model A and model B on the *same* N test
        samples (e.g. per-sample correctness ``1[pred == target]``, per-sample
        Brier contribution or per-sample ``|conf - correct|``).
    cluster_ids
        Optional ``[N]`` cluster labels (use the ``image_id``). When given, whole
        clusters are resampled with replacement, which removes the
        same-image-correlation bias.
    n_replicates, seed, ci
        number of bootstrap resamples, RNG seed (reproducibility) and nominal
        confidence level.
    statistic
        aggregation applied to each resample; ``np.mean`` (default) turns the
        per-sample stats into accuracy / Brier / ECE-like means.

    Returns
    -------
    dict
        ``"diff"`` (the observed ``statistic(a - b)``), ``"ci_low"`` /
        ``"ci_high"`` (percentile bootstrap interval at level ``ci``) and
        ``"p_two_sided_hint"`` (see :func:`_two_sided_hint`). Extras:
        ``"mean_a"``, ``"mean_b"``, ``"std"``, ``"n"``, ``"n_replicates"``,
        ``"ci_level"``, ``"resample_unit"`` (``"sample"`` or ``"cluster"``),
        ``"n_clusters"``, ``"replicates"`` (the full bootstrap distribution).
    """
    a = to_numpy_1d(sample_stat_a, "sample_stat_a")
    b = to_numpy_1d(sample_stat_b, "sample_stat_b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"`sample_stat_a` ({a.shape[0]}) and `sample_stat_b` ({b.shape[0]}) differ in length"
        )
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("sample statistics contain NaN or inf values")
    n_rep = _check_replicates(n_replicates)
    level = _check_ci(ci)
    diff = a - b
    observed = float(statistic(diff))
    groups: Sequence[NDArray] | None = None
    n_clusters = None
    if cluster_ids is not None:
        clusters, groups = _cluster_groups(cluster_ids, diff.shape[0])
        n_clusters = int(clusters.shape[0])
        if n_clusters < 2:
            raise ValueError("cluster bootstrap needs at least two clusters")
    rng = np.random.default_rng(seed)
    replicates = _replicate_stats(diff, statistic, n_rep, rng, groups)
    low, high = _percentile_ci(replicates, level)
    return {
        "diff": observed,
        "ci_low": low,
        "ci_high": high,
        "p_two_sided_hint": _two_sided_hint(replicates),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std": float(np.std(replicates, ddof=1)) if n_rep > 1 else float("nan"),
        "n": int(diff.shape[0]),
        "n_replicates": n_rep,
        "ci_level": level,
        "resample_unit": "cluster" if groups is not None else "sample",
        "n_clusters": n_clusters,
        "replicates": replicates,
    }


def bootstrap_ci(
    values: Any,
    stat_fn: Callable[[NDArray], float] = np.mean,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    cluster_ids: Any = None,
) -> dict[str, Any]:
    """Percentile bootstrap CI of ``stat_fn(values)`` for a single quantity.

    Same resampling machinery and reproducibility guarantees as
    :func:`paired_bootstrap`; ``cluster_ids`` is an optional extra keyword
    (``None`` = i.i.d. sample-level resampling).

    Returns ``"point"``, ``"ci_low"``, ``"ci_high"``, ``"std"``, ``"n"``,
    ``"n_replicates"``, ``"ci_level"``, ``"resample_unit"``, ``"replicates"``.
    """
    arr = to_numpy_1d(values, "values")
    if not np.all(np.isfinite(arr)):
        raise ValueError("`values` contain NaN or inf")
    n_rep = _check_replicates(n_replicates)
    level = _check_ci(ci)
    point = float(stat_fn(arr))
    groups: Sequence[NDArray] | None = None
    if cluster_ids is not None:
        _clusters, groups = _cluster_groups(cluster_ids, arr.shape[0])
        if len(groups) < 2:
            raise ValueError("cluster bootstrap needs at least two clusters")
    rng = np.random.default_rng(seed)
    replicates = _replicate_stats(arr, stat_fn, n_rep, rng, groups)
    if not np.all(np.isfinite(replicates)):
        raise ValueError("`stat_fn` produced non-finite values on the bootstrap resamples")
    low, high = _percentile_ci(replicates, level)
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "std": float(np.std(replicates, ddof=1)) if n_rep > 1 else float("nan"),
        "n": int(arr.shape[0]),
        "n_replicates": n_rep,
        "ci_level": level,
        "resample_unit": "cluster" if groups is not None else "sample",
        "replicates": replicates,
    }
