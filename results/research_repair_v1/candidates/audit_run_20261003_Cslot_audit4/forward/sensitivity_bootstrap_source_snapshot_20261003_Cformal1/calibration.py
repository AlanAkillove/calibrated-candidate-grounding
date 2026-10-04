"""Calibration metrics for candidate-set grounding (Top-label ECE & friends).

Contract functions: :func:`confidence_from_scores`, :func:`top_label_ece`,
:func:`top_label_ece_adaptive`, :func:`brier_binary`, :func:`nll_binary`,
:func:`multiclass_nll`, :func:`multiclass_brier`, :func:`confidence_accuracy_gap`,
:func:`reliability_table`.

Definitions (project research protocol, section "Calibration Metrics")
----------------------------------------------------------------------
With :math:`\\hat p = \\mathrm{softmax}(s/T)`, :math:`\\hat c = \\arg\\max_i \\hat p_i`,
:math:`p = \\max_i \\hat p_i` and :math:`r = \\mathbf 1[\\hat c = c^*]`:

* **Top-label ECE** :math:`= \\sum_b (n_b / N)\\,|\\mathrm{acc}(b) - \\mathrm{conf}(b)|`
  with either equal-width (``top_label_ece``) or equal-mass / adaptive
  (``top_label_ece_adaptive``) bins. Adaptive bins are the project's primary
  reliability measure because max-softmax confidence concentrates in a narrow
  band when ``K`` grows, leaving many equal-width bins empty.
* **Binary correctness Brier** :math:`= \\mathrm{mean}\\,(p - r)^2`
* **Top-label correctness NLL** :math:`= -[r \\log p + (1-r)\\log(1-p)]`
* **Confidence-accuracy gap** :math:`= \\mathrm{mean}\\,p - \\mathrm{mean}\\,r` (signed:
  positive = overconfidence, negative = underconfidence)

Multiclass NLL / Brier are reported as *secondary* metrics only.

Target-absent rows (``target_index == -1``) raise :class:`ValueError` by default
for the multiclass functions; pass ``mask=`` or ``strict=False`` (see
:mod:`ccg.metrics.ranking` for the identical convention).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ccg.metrics._validation import (
    as_correctness,
    as_int_array,
    require_same_length,
    resolve_rows,
    to_numpy_1d,
    to_numpy_2d,
    to_probability_1d,
)

__all__ = [
    "brier_binary",
    "confidence_accuracy_gap",
    "confidence_from_scores",
    "multiclass_brier",
    "multiclass_nll",
    "nll_binary",
    "reliability_table",
    "softmax_scores",
    "top_label_ece",
    "top_label_ece_adaptive",
]


def softmax_scores(scores: Any, temperature: float = 1.0) -> NDArray:
    """Row-wise :math:`\\mathrm{softmax}(s/T)` with max-subtraction for stability.

    ``temperature`` must be strictly positive; ``T > 1`` flattens the
    distribution, ``T < 1`` sharpens it.
    """
    s = to_numpy_2d(scores, "scores")
    t = float(temperature)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"`temperature` must be a finite positive number, got {temperature}")
    logits = s / t
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def confidence_from_scores(
    scores: Any,
    temperature: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """Return ``(max_prob, predicted_index)`` of :func:`softmax_scores`.

    ``max_prob`` is the top-label confidence ``p`` used by every binary
    calibration metric; ``predicted_index`` is ``argmax`` over each row
    (ties resolved towards the smallest candidate index, as in ``np.argmax``).
    """
    prob = softmax_scores(scores, temperature=temperature)
    max_prob = prob.max(axis=1)
    predicted_index = prob.argmax(axis=1).astype(np.int64)
    return max_prob, predicted_index


def _validate_pairs(confidence: Any, correctness: Any) -> tuple[NDArray, NDArray]:
    conf = to_probability_1d(confidence, "confidence")
    corr = as_correctness(correctness, "correctness")
    require_same_length(conf, corr, "confidence", "correctness")
    return conf, corr


def _equal_width_bins(conf: NDArray, n_bins: int) -> NDArray:
    """Bin index per sample for equal-width bins over ``[0, 1]`` (right edge inclusive)."""
    nb = _check_n_bins(n_bins)
    idx = np.floor(conf * nb).astype(np.int64)
    return np.minimum(idx, nb - 1)  # conf == 1.0 falls into the last bin


def _check_n_bins(n_bins: int) -> int:
    nb = int(n_bins)
    if nb < 1:
        raise ValueError(f"`n_bins` must be >= 1, got {n_bins}")
    return nb


def _equal_mass_bins(conf: NDArray, n_bins: int) -> NDArray:
    """Bin index per sample for equal-mass (adaptive) bins.

    Samples are sorted by confidence and cut into ``n_bins`` contiguous groups of
    (as close as possible to) equal size. Cuts are **snapped to the nearest tie
    boundary**, so a block of identical confidences is never split between two
    bins: this keeps the statistic invariant to the input order of tied samples
    (essential when max-softmax confidence is constant, e.g. a flat model). If
    snapping collapses bins, fewer than ``n_bins`` bins remain - every returned
    bin index in ``0 .. bin_index.max()`` is non-empty.
    """
    n = conf.shape[0]
    nb = min(_check_n_bins(n_bins), n)
    order = np.argsort(conf, kind="stable")
    sorted_conf = conf[order]
    cuts: list[int] = [0]
    for nominal in np.linspace(0, n, nb + 1)[1:-1]:
        pos = int(nominal)
        value = sorted_conf[pos]
        left = int(np.searchsorted(sorted_conf, value, side="left"))
        right = int(np.searchsorted(sorted_conf, value, side="right"))
        cuts.append(left if pos - left <= right - pos else right)
    cuts.append(n)
    ordered: list[int] = []
    for c in sorted(set(cuts)):
        if not ordered or c > ordered[-1]:
            ordered.append(c)
    bin_index_sorted = np.concatenate(
        [np.full(ordered[i + 1] - ordered[i], i, dtype=np.int64) for i in range(len(ordered) - 1)]
    )
    bin_index = np.empty(n, dtype=np.int64)
    bin_index[order] = bin_index_sorted
    return bin_index


def _weighted_abs_gap(bin_index: NDArray, conf: NDArray, corr: NDArray, n_bins: int) -> float:
    """``sum_b (n_b / N) * |acc_b - conf_b|`` over non-empty bins."""
    counts = np.bincount(bin_index, minlength=n_bins).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return float("nan")
    acc = np.bincount(bin_index, weights=corr, minlength=n_bins)
    avg_conf = np.bincount(bin_index, weights=conf, minlength=n_bins)
    nonempty = counts > 0
    gaps = np.abs(acc[nonempty] / counts[nonempty] - avg_conf[nonempty] / counts[nonempty])
    return float(np.sum(counts[nonempty] * gaps) / total)


def top_label_ece(confidence: Any, correctness: Any, n_bins: int = 15) -> float:
    """Top-label ECE with **equal-width** confidence bins over ``[0, 1]``.

    ``confidence`` are the max-softmax probabilities ``p`` and ``correctness``
    the binary indicators ``r`` (bool or 0/1). Empty bins contribute nothing.
    """
    conf, corr = _validate_pairs(confidence, correctness)
    nb = _check_n_bins(n_bins)
    return _weighted_abs_gap(_equal_width_bins(conf, nb), conf, corr, nb)


def top_label_ece_adaptive(confidence: Any, correctness: Any, n_bins: int = 15) -> float:
    """Top-label ECE with **equal-mass / adaptive** bins (primary project metric).

    Each bin holds the same number of samples (differing by at most one, and ties
    are never split across bins), which makes the estimate stable when confidence
    is concentrated in a narrow band, e.g. for large candidate sets. On data whose
    confidence is concentrated in few distinct values this can differ
    substantially from :func:`top_label_ece`.
    """
    conf, corr = _validate_pairs(confidence, correctness)
    nb = min(_check_n_bins(n_bins), conf.shape[0])
    return _weighted_abs_gap(_equal_mass_bins(conf, nb), conf, corr, nb)


def brier_binary(confidence: Any, correctness: Any) -> float:
    """Mean binary-correctness Brier score :math:`\\mathrm{mean}\\,(p - r)^2` (lower better)."""
    conf, corr = _validate_pairs(confidence, correctness)
    return float(np.mean((conf - corr) ** 2))


def nll_binary(confidence: Any, correctness: Any, eps: float = 1e-12) -> float:
    """Mean top-label correctness NLL :math:`-[r\\log p + (1-r)\\log(1-p)]`.

    ``p`` is clipped to ``[eps, 1 - eps]`` so the value stays finite for
    perfectly confident wrong / right predictions (no ``inf``).
    """
    conf, corr = _validate_pairs(confidence, correctness)
    e = float(eps)
    if not np.isfinite(e) or e <= 0.0 or e >= 0.5:
        raise ValueError(f"`eps` must be in (0, 0.5), got {eps}")
    p = np.clip(conf, e, 1.0 - e)
    return float(-np.mean(corr * np.log(p) + (1.0 - corr) * np.log(1.0 - p)))


def confidence_accuracy_gap(confidence: Any, correctness: Any) -> float:
    """``mean(confidence) - accuracy``; positive = overconfident, negative = underconfident."""
    conf, corr = _validate_pairs(confidence, correctness)
    return float(np.mean(conf) - np.mean(corr))


def multiclass_nll(
    scores: Any,
    target_index: Any,
    temperature: float = 1.0,
    mask: Any = None,
    strict: bool = True,
) -> float:
    """Mean negative log-likelihood of the target under :math:`\\mathrm{softmax}(s/T)`.

    Target-absent rows raise by default (see module docstring). The log-prob is
    floored at ``-700`` so an exactly-zero target probability yields a large but
    finite value instead of ``inf``.
    """
    s, t, keep = _prepare_scores(scores, target_index, mask, strict)
    log_prob = np.log(np.clip(softmax_scores(s, temperature=temperature)[keep], 1e-304, 1.0))
    rows = np.arange(s.shape[0])[keep]
    return float(-np.mean(log_prob[rows, t[keep]]))


def multiclass_brier(
    scores: Any,
    target_index: Any,
    temperature: float = 1.0,
    mask: Any = None,
    strict: bool = True,
) -> float:
    """Mean multiclass Brier score :math:`\\sum_k (p_k - y_k)^2` (secondary metric)."""
    s, t, keep = _prepare_scores(scores, target_index, mask, strict)
    prob = softmax_scores(s, temperature=temperature)[keep]
    onehot = np.zeros_like(prob)
    onehot[np.arange(prob.shape[0]), t[keep]] = 1.0
    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))


def _prepare_scores(
    scores: Any,
    target_index: Any,
    mask: Any = None,
    strict: bool = True,
) -> tuple[NDArray, NDArray, NDArray]:
    s = to_numpy_2d(scores, "scores")
    t = as_int_array(target_index, "target_index")
    keep = resolve_rows(t, s.shape[0], mask=mask, strict=strict)
    bad = (t >= 0) & (t >= s.shape[1])
    if bool(np.any(bad & keep)):
        raise ValueError(f"{int(np.sum(bad & keep))} target index/indexes are outside [0, K)")
    return s, t, keep


def reliability_table(
    confidence: Any,
    correctness: Any,
    n_bins: int = 15,
    adaptive: bool = False,
) -> dict[str, Any]:
    """Per-bin summary for reliability diagrams.

    Returns a dict with:

    ``"n_bins"``, ``"adaptive"``
        the configuration actually used.
    ``"bin_edges"``
        ``[n_bins + 1]`` float array. Equal-width bins use ``linspace(0, 1)``;
        adaptive bins use the midpoints between neighbouring bin contents
        (first/last edge pinned to ``0`` / ``1``), so ``n_bins`` may shrink to the
        number of non-degenerate adaptive bins actually used.
    ``"counts"``
        ``[n_bins]`` integer array, samples per bin (adaptive bins are
        (almost) equal sized, equal-width bins may be empty).
    ``"avg_confidence"`` / ``"accuracy"``
        ``[n_bins]`` float arrays; ``0.0`` for empty bins - mask them with
        ``counts > 0`` before plotting.
    ``"ece"``
        the ECE recomputed from these bins, so table and scalar always agree.
    ``"n_total"``
        number of evaluated samples.
    """
    conf, corr = _validate_pairs(confidence, correctness)
    nb = _check_n_bins(n_bins)
    if adaptive:
        bin_index = _equal_mass_bins(conf, min(nb, conf.shape[0]))
        nb_eff = int(bin_index.max()) + 1
        sorted_conf = conf[np.argsort(conf, kind="stable")]
        sizes = np.bincount(bin_index, minlength=nb_eff)
        cuts = np.cumsum(sizes)[:-1]  # first index of bins 1..nb_eff-1
        edges = np.zeros(nb_eff + 1, dtype=np.float64)
        edges[0], edges[-1] = 0.0, 1.0
        edges[1:nb_eff] = 0.5 * (sorted_conf[cuts - 1] + sorted_conf[cuts])
    else:
        bin_index = _equal_width_bins(conf, nb)
        nb_eff = nb
        edges = np.linspace(0.0, 1.0, nb_eff + 1)
    counts = np.bincount(bin_index, minlength=nb_eff).astype(np.int64)
    acc = np.zeros(nb_eff, dtype=np.float64)
    avg_conf = np.zeros(nb_eff, dtype=np.float64)
    nonempty = counts > 0
    acc[nonempty] = (
        np.bincount(bin_index, weights=corr, minlength=nb_eff)[nonempty] / counts[nonempty]
    )
    avg_conf[nonempty] = (
        np.bincount(bin_index, weights=conf, minlength=nb_eff)[nonempty] / counts[nonempty]
    )
    return {
        "n_bins": nb_eff,
        "adaptive": bool(adaptive),
        "bin_edges": edges,
        "counts": counts,
        "avg_confidence": avg_conf,
        "accuracy": acc,
        "ece": _weighted_abs_gap(bin_index, conf, corr, nb_eff),
        "n_total": int(conf.shape[0]),
    }
