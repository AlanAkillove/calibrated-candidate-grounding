"""Macro paired image-cluster bootstrap with shared draws across severity (V2-M3).

The confirmatory primary statistic of amendment V2-M3 is the macro difference

    delta_macro = MacroAUROC(Adaptive) - MacroAUROC(Static)
    MacroAUROC  = (1/4) sum_{m in {0,2,4,8}} AUROC_m

so every bootstrap draw must resample the *same* images for all four severity
levels of the shared K=10 cohort and average the per-severity paired
differences inside one replicate (amendment section 18: "same shared
image-cluster bootstrap draws across severity").

The draw mechanism is exactly :class:`ccg.experiment.phase0a._ClusterSampler`
with ``np.random.default_rng(seed)``, so with a single severity level the
replicates equal :func:`ccg.experiment.phase0a.paired_cluster_bootstrap`
bit-for-bit (verified by tests).  Nothing here trains or fits anything.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ccg.experiment.phase0a import SampleStats, _ClusterSampler

__all__ = ["macro_paired_cluster_bootstrap"]


def macro_paired_cluster_bootstrap(
    metric_fn: Callable[[SampleStats], float],
    severity_pairs: Sequence[Tuple[Any, Any]],
    cluster_ids: Any = None,
    *,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Paired macro bootstrap: one shared draw feeds every severity level.

    Parameters
    ----------
    metric_fn:
        ``SampleStats -> float`` (e.g. ``auroc_correct``).
    severity_pairs:
        Non-empty sequence of ``(pred_a, pred_b)`` pairs, one per severity
        level, where each prediction is a :class:`SampleStats` or a
        ``(confidence, correct)`` tuple;  e.g.
        ``[((conf_a0, corr0), (conf_b0, corr0)), ...]``.  Every pair must use
        the same rows in the same order (shared cohort).
    cluster_ids:
        Image ids of the shared cohort rows.  ``None`` degenerates to the
        i.i.d. sample bootstrap, mirroring ``paired_cluster_bootstrap``.

    Returns
    -------
    dict
        ``diff`` (mean over severities of the point delta), ``ci_low`` /
        ``ci_high`` / ``ci_level``, per-severity point deltas, ``n`` /
        ``n_clusters`` / ``n_replicates`` and the raw ``replicates`` array.
    """
    if not severity_pairs:
        raise ValueError("severity_pairs must contain at least one pair")

    def _to_stats(entry: Any, side: str) -> SampleStats:
        if isinstance(entry, SampleStats):
            return entry
        if isinstance(entry, tuple) and len(entry) == 2:
            return SampleStats.from_conf_correct(*entry)
        raise ValueError(
            f"{side} must be a SampleStats or a (confidence, correct) tuple; "
            "each severity entry is ((conf_a, corr_a), (conf_b, corr_b))"
        )

    pairs: List[Tuple[SampleStats, SampleStats]] = []
    for entry in severity_pairs:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise ValueError("each severity entry must be a (pred_a, pred_b) pair")
        a, b = entry
        a_stats = _to_stats(a, "pred_a")
        b_stats = _to_stats(b, "pred_b")
        if len(a_stats) != len(b_stats):
            raise ValueError(f"pair rows disagree: {len(a_stats)} vs {len(b_stats)}")
        pairs.append((a_stats, b_stats))
    n = len(pairs[0][0])
    if n < 2:
        raise ValueError("macro bootstrap needs at least 2 rows")
    for a_stats, _ in pairs:
        if len(a_stats) != n:
            raise ValueError("severity levels disagree on the row count (rows must be shared)")

    reps = int(n_replicates)
    if reps < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    level = float(ci)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    point_deltas = [float(metric_fn(a)) - float(metric_fn(b)) for a, b in pairs]
    means_a = [float(metric_fn(a)) for a, _ in pairs]
    means_b = [float(metric_fn(b)) for _, b in pairs]

    rng = np.random.default_rng(int(seed))
    if cluster_ids is None:
        sampler = None
        n_clusters = n
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.shape[0] != n:
            raise ValueError(
                f"cluster_ids has {clusters.shape[0]} entries but predictions have {n} rows"
            )
        sampler = _ClusterSampler(clusters)
        n_clusters = sampler.n_clusters

    replicates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        draw = rng.integers(0, n, size=n) if sampler is None else sampler.draw(rng)
        per_severity = [
            float(metric_fn(a.take(draw))) - float(metric_fn(b.take(draw)))
            for a, b in pairs
        ]
        replicates[index] = float(np.mean(per_severity))

    alpha = (1.0 - level) / 2.0
    return {
        "metric": metric_name or getattr(metric_fn, "__name__", "metric"),
        "diff": float(np.mean(point_deltas)),
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "ci_level": level,
        "mean_a": float(np.mean(means_a)),
        "mean_b": float(np.mean(means_b)),
        "per_severity_diff": point_deltas,
        "n": int(n),
        "n_clusters": int(n_clusters),
        "n_replicates": reps,
        "n_levels": len(pairs),
        "replicates": replicates,
    }
