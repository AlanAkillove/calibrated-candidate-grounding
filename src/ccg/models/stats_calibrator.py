"""Score-statistic extraction and the Phase 0.5 stats-only calibrator (C3).

RQ2 asks whether *scalar* information about a candidate set already explains
the reliability degradation.  To answer that we need the statistic vector to be
defined exactly once, in numpy, independently of any model:

``[K, max_score, top1_top2_margin, entropy, mean_score, std_score, max_softmax]``

plus an optional top-3 block.  Candidate embeddings are *not* allowed here -
that is what makes C3 a fair "simple" baseline against the candidate-aware
models of Phase 1.

:class:`StatsOnlyCalibrator` provides the module *structure* (a tiny MLP on the
vector above) and the pure-numpy feature function.  Training is deliberately
not enabled: Phase 0.5 may only start after Gate Q1 passes, and the fitting
protocol (which split, which loss) is frozen in ``docs/research_protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .base import softmax_np

__all__ = [
    "STATS_FIELDS",
    "TOP3_FIELDS",
    "stats_dim",
    "extract_score_stats",
    "extract_score_stats_batch",
    "StatsOnlyCalibrator",
]

#: Order of the statistics produced by :func:`extract_score_stats` (section 15 C3).
STATS_FIELDS: tuple[str, ...] = (
    "K",
    "max_score",
    "top1_top2_margin",
    "entropy",
    "mean_score",
    "std_score",
    "max_softmax",
)
#: Optional extension, appended when ``include_top3=True``.
TOP3_FIELDS: tuple[str, ...] = ("top3_score", "top3_softmax", "top3_gap")


def stats_dim(include_top3: bool = False) -> int:
    return len(STATS_FIELDS) + (len(TOP3_FIELDS) if include_top3 else 0)


def extract_score_stats(
    scores: np.ndarray,
    *,
    include_top3: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    """Summarise one candidate set's logits into the C3 statistic vector.

    Definitions (all on the raw logits ``s`` of a single ``K``-way set; ``p`` is
    the softmax over the set, computed at ``s / temperature``):

    ``K``
        number of candidates (float, so the cardinality is explicitly visible -
        this is the variable whose effect RQ1 measures).
    ``max_score``
        ``max_i s_i``.
    ``top1_top2_margin``
        ``largest(s) - second largest(s)``; ``0.0`` when ``K == 1``.
    ``entropy``
        ``-sum_i p_i log p_i`` with the natural logarithm, i.e. normalised by
        nothing (``log K`` is the ceiling, and ``K`` is reported separately).
    ``mean_score``, ``std_score``
        mean and *population* standard deviation (``ddof=0``) of the logits.
    ``max_softmax``
        ``max_i p_i`` - the confidence a top-label ECE bin uses.
    ``top3_score``, ``top3_softmax``, ``top3_gap`` (optional)
        third-largest logit, its softmax probability and ``max_softmax`` minus
        it.  For ``K < 3`` the *smallest* available value is reused (padding by
        duplication keeps the vector comparable across ``K`` without inventing
        mass); the true ``K`` is always in slot 0 so the padding is detectable.

    Returns ``float32`` array of length :func:`stats_dim`.
    """
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("extract_score_stats() needs at least one candidate score")
    if not np.all(np.isfinite(values)):
        raise ValueError("extract_score_stats() received non-finite logits")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    probs = softmax_np(values / temperature)
    ordered = np.sort(values)[::-1]
    max_score = float(ordered[0])
    margin = float(ordered[0] - ordered[1]) if values.size > 1 else 0.0
    safe_probs = np.clip(probs, 1e-12, None)
    entropy = float(-np.sum(safe_probs * np.log(safe_probs)))
    stats: List[float] = [
        float(values.size),
        max_score,
        margin,
        entropy,
        float(values.mean()),
        float(values.std(ddof=0)),
        float(probs.max()),
    ]
    if include_top3:
        # softmax is monotone in the logits, so the ranked probability order is
        # the ranked logit order; for K < 3 the weakest entry is duplicated.
        ranked_probs = np.sort(probs)[::-1]
        slot = 2 if values.size >= 3 else -1
        stats.extend(
            [float(ordered[slot]), float(ranked_probs[slot]), float(probs.max()) - float(ranked_probs[slot])]
        )
    return np.asarray(stats, dtype=np.float32)


def extract_score_stats_batch(
    score_sets: Sequence[np.ndarray],
    *,
    include_top3: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    """``[M, stats_dim]`` stack of :func:`extract_score_stats` over several sets."""
    rows = [
        extract_score_stats(scores, include_top3=include_top3, temperature=temperature)
        for scores in score_sets
    ]
    if not rows:
        raise ValueError("extract_score_stats_batch() called with an empty sequence")
    return np.stack(rows, axis=0)


@dataclass
class StatsOnlyCalibratorConfig:
    """Structure description recorded in experiment logs (section 28)."""

    hidden_dim: int = 64
    include_top3: bool = False
    output_mode: str = "p_correct"

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": "c3_stats_only_calibrator",
            "hidden_dim": int(self.hidden_dim),
            "include_top3": bool(self.include_top3),
            "output_mode": self.output_mode,
            "stats_fields": list(STATS_FIELDS) + (list(TOP3_FIELDS) if self.include_top3 else []),
        }


class StatsOnlyCalibrator:
    """Tiny MLP on the score statistics -> ``P(correct)`` (or a temperature).

    ``stats -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU -> Linear(1)``
    with a sigmoid (``output_mode="p_correct"``) or a positive softplus
    (``output_mode="temperature"``) read-out.  Implemented as *structure only*:
    forward passes work (so shapes can be unit-tested and the module can be
    inspected), while :meth:`fit` refuses until Phase 0.5 is unlocked by
    Gate Q1.
    """

    name = "c3_stats_only_calibrator"

    def __init__(
        self,
        hidden_dim: int = 64,
        include_top3: bool = False,
        output_mode: str = "p_correct",
        temperature_floor: float = 0.05,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if output_mode not in ("p_correct", "temperature"):
            raise ValueError(
                f"output_mode must be 'p_correct' or 'temperature', got {output_mode!r}"
            )
        torch = _require_torch()
        self.config = StatsOnlyCalibratorConfig(
            hidden_dim=int(hidden_dim), include_top3=bool(include_top3), output_mode=output_mode
        )
        self.input_dim = stats_dim(include_top3)
        self.temperature_floor = float(temperature_floor)
        self.device = torch.device(device)
        torch.manual_seed(int(seed))
        h = int(hidden_dim)
        layers = [
            torch.nn.Linear(self.input_dim, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, 1),
        ]
        self.net = torch.nn.Sequential(*layers).to(self.device)
        self.output_mode = output_mode
        self.seed = int(seed)

    # -- inference -----------------------------------------------------------
    def forward_raw(self, stats: np.ndarray):
        """Un-activated network output for a ``[M, stats_dim]`` (or ``[stats_dim]``) batch."""
        torch = _require_torch()
        matrix = np.asarray(stats, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != self.input_dim:
            raise ValueError(f"stats must have {self.input_dim} columns, got shape {matrix.shape}")
        with torch.no_grad():
            return self.net(torch.as_tensor(matrix, device=self.device)).reshape(-1)

    def predict(self, stats: np.ndarray) -> np.ndarray:
        """``P(correct)`` in ``[0,1]`` or a positive temperature per set."""
        raw = self.forward_raw(stats)
        torch = _require_torch()
        if self.output_mode == "p_correct":
            return torch.sigmoid(raw).detach().cpu().numpy().astype(np.float32)
        softplus = torch.nn.functional.softplus(raw) + float(self.temperature_floor)
        return softplus.detach().cpu().numpy().astype(np.float32)

    def predict_from_scores(self, scores: Sequence[np.ndarray]) -> np.ndarray:
        """Convenience: raw logits per set -> :meth:`predict`."""
        stats = extract_score_stats_batch(
            scores, include_top3=self.config.include_top3, temperature=1.0
        )
        return self.predict(stats)

    # -- capacity / training -------------------------------------------------
    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def fit(self, *args, **kwargs):
        """Refuses to train: Phase 0.5 (C3) is gated behind Gate Q1."""
        raise NotImplementedError(
            "StatsOnlyCalibrator training belongs to Phase 0.5 and may only run after Gate Q1 "
            "passes on the frozen protocol. The structure and the numpy statistic extraction are "
            "complete and unit-tested; the fit loop is intentionally not enabled."
        )

    @property
    def description(self) -> Dict[str, object]:
        info = dict(self.config.to_dict())
        info["input_dim"] = self.input_dim
        info["num_parameters"] = self.num_parameters()
        return info


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "StatsOnlyCalibrator needs torch; extract_score_stats() is pure numpy and works "
            "without it."
        ) from exc
