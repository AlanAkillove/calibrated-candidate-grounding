"""Ranking metrics for candidate-set grounding.

Contract functions: :func:`top1_accuracy`, :func:`topk_accuracy`, :func:`mrr`,
:func:`accuracy_by_k`.  A helper :func:`target_rank` is exported as well because
every other ranking metric is a function of it.

Input shapes
------------
``scores``
    ``[M, K]`` array. Row ``m`` holds the score / logit of each of the ``K``
    candidates of candidate set ``m`` (e.g. the output of a cosine or MLP
    scorer). Larger is better.
``target_index``
    ``[M]`` integer array. ``target_index[m]`` is the column index (``0`` based)
    of the ground-truth candidate inside row ``m``. ``-1`` encodes *target
    absent* (the correct region is not in the candidate set).

Target-absent handling
----------------------
Ranking metrics are undefined when the target is absent, therefore the default
behaviour (``strict=True``) is to raise :class:`ValueError`. Two escape hatches:

* pass ``mask=target_present`` (a ``[M]`` boolean array) to restrict the metric
  to an explicit subset of rows, or
* pass ``strict=False`` to silently drop absent rows from both numerator and
  denominator.

Absent rows are never counted as errors, so ``strict=False`` yields
"accuracy over target-present rows" — report which convention was used.

Tie-breaking
------------
Ranks are derived from ``np.argsort(-scores, kind="stable")``, i.e. equal scores
are ordered by ascending candidate index. A target that ties with a distractor
therefore only wins Top-1 if it has the smaller index; the same rule is used
consistently by Top-1, Top-k and MRR, so ``top1_accuracy`` always equals
``topk_accuracy(k=1)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ccg.metrics._validation import as_int_array, resolve_rows, to_numpy_1d, to_numpy_2d

__all__ = ["accuracy_by_k", "mrr", "top1_accuracy", "topk_accuracy", "target_rank"]


def _prepare(
    scores: Any,
    target_index: Any,
    mask: Any = None,
    strict: bool = True,
) -> tuple[NDArray, NDArray, NDArray]:
    """Validate inputs and return ``(scores, target_index, keep_rows)``."""
    s = to_numpy_2d(scores, "scores")
    t = as_int_array(target_index, "target_index")
    keep = resolve_rows(t, s.shape[0], mask=mask, strict=strict)
    out_of_range = (t >= 0) & (t >= s.shape[1])
    if bool(np.any(out_of_range & keep)):
        raise ValueError(
            f"{int(np.sum(out_of_range & keep))} target index/indexes are outside [0, K="
            f"{s.shape[1]})"
        )
    return s, t, keep


def target_rank(
    scores: Any,
    target_index: Any,
    mask: Any = None,
    strict: bool = True,
) -> NDArray:
    """Return the 1-based rank of the target inside every kept candidate set.

    ``1`` means the target is ranked first. Rows are returned in input order for
    the kept rows only, so ``len(rank) == mask.sum()``.
    """
    s, t, keep = _prepare(scores, target_index, mask=mask, strict=strict)
    order = np.argsort(-s[keep], axis=1, kind="stable")
    hit = order == t[keep][:, None]
    # every row of `order` is a permutation containing exactly one match
    rank = np.argmax(hit, axis=1) + 1
    return rank.astype(np.int64)


def _mean_over_rows(values: NDArray) -> float:
    return float(np.mean(values))


def top1_accuracy(scores: Any, target_index: Any, mask: Any = None, strict: bool = True) -> float:
    """Fraction of candidate sets where the target is ranked first.

    See the module docstring for shapes, target-absent handling and ties.
    """
    s, t, keep = _prepare(scores, target_index, mask=mask, strict=strict)
    rank = target_rank(s, t, mask=keep, strict=False)
    return _mean_over_rows(rank == 1)


def topk_accuracy(
    scores: Any,
    target_index: Any,
    k: int,
    mask: Any = None,
    strict: bool = True,
) -> float:
    """Fraction of candidate sets where the target appears inside the Top-``k``.

    ``k`` is clamped to the candidate-set size ``K`` (so ``topk_accuracy(k=100)``
    on ``K=10`` sets degenerates to ``1.0``).
    """
    k_int = int(k)
    if k_int < 1:
        raise ValueError(f"`k` must be >= 1, got {k}")
    s, t, keep = _prepare(scores, target_index, mask=mask, strict=strict)
    k_eff = min(k_int, s.shape[1])
    rank = target_rank(s, t, mask=keep, strict=False)
    return _mean_over_rows(rank <= k_eff)


def mrr(scores: Any, target_index: Any, mask: Any = None, strict: bool = True) -> float:
    """Mean Reciprocal Rank of the target over the kept candidate sets."""
    s, t, keep = _prepare(scores, target_index, mask=mask, strict=strict)
    rank = target_rank(s, t, mask=keep, strict=False)
    return _mean_over_rows(1.0 / rank.astype(np.float64))


def accuracy_by_k(
    scores: Any,
    target_index: Any,
    ks: Sequence[int] | NDArray = (1, 5),
    mask: Any = None,
    strict: bool = True,
) -> dict[int, float]:
    """Return ``{k: top-k accuracy}`` for every ``k`` in ``ks``.

    ``ks`` values above ``K`` are evaluated with ``k`` clamped to ``K`` (the
    returned key is still the requested ``k``), which keeps the dict directly
    comparable across the ``K = 5/10/20/50`` test cells.
    """
    s, t, keep = _prepare(scores, target_index, mask=mask, strict=strict)
    ranks = target_rank(s, t, mask=keep, strict=False)
    k_arr = to_numpy_1d(list(ks), "ks", dtype=np.float64).astype(np.int64)
    out: dict[int, float] = {}
    for k in k_arr:
        if k < 1:
            raise ValueError(f"`ks` must contain integers >= 1, got {k}")
        out[int(k)] = _mean_over_rows(ranks <= min(int(k), s.shape[1]))
    return out
