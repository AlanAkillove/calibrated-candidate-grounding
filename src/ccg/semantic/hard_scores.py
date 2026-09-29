"""Frozen B3 scoring on the Phase 1F candidate constructions (protocol A8.4).

This module re-runs the *frozen* B3 checkpoints (``results/phase0b_independent/
seed_{s}/model.npz`` + ``training.json``) over the Phase 1F candidate sets:

* :func:`materialise_examples` gathers per-row ``(z_q, z_i, geometry)`` through
  the frozen :class:`~ccg.models.b3_data.B3Corpus` (region/text feature cache +
  proposal bank) with exactly the Phase 0B input assembly;
* :func:`score_examples` pushes them through one checkpoint with the same
  batched row-wise forward Phase 0B used (``phase0b._forward_superset``);
* :func:`verify_against_raw_scores` reproduces the frozen ``raw_scores`` of the
  random-regime ``C_K`` and raises when the reconstruction drifts beyond the
  A8.4 STOP tolerance.

Nothing here trains: the checkpoints are loaded read-only and the candidate
sets come from the frozen manifests only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..experiment import phase0b
from ..models.b3_data import FEATURE_DIM, GEO_DIM, B3Corpus, geometry_features_centered
from ..models.independent import IndependentMLPScorer
from ..reliability import data as rdata

__all__ = [
    "DEFAULT_B3_ROOT",
    "DEFAULT_FEATURES_ROOT",
    "DEFAULT_KS",
    "B3ExampleBatch",
    "load_frozen_scorers",
    "materialise_examples",
    "score_examples",
    "verify_against_raw_scores",
]

#: Frozen Phase 0B artifact root.
DEFAULT_B3_ROOT = rdata.B3_ROOT
#: Frozen Phase 0A feature cache root.
DEFAULT_FEATURES_ROOT = Path("cache/features")
#: Candidate-set sizes of the Phase 1F stress test.
DEFAULT_KS: Tuple[int, ...] = (5, 10)


@dataclass(frozen=True)
class B3ExampleBatch:
    """Materialised per-row B3 inputs of one candidate construction.

    All arrays are float32 and locally ordered (target at local index 0):
    ``z_q`` ``[n, 512]``, ``z_i`` ``[n, k, 512]``, ``geometry`` ``[n, k, 6]``.
    """

    z_q: np.ndarray
    z_i: np.ndarray
    geometry: np.ndarray

    @property
    def n(self) -> int:
        return int(self.z_q.shape[0])

    @property
    def k(self) -> int:
        return int(self.z_i.shape[1])


def load_frozen_scorers(
    *,
    b3_root: Path = DEFAULT_B3_ROOT,
    seeds: Sequence[int] = (1, 2, 3),
    device: str = "cpu",
) -> Dict[str, IndependentMLPScorer]:
    """Rebuild the frozen B3 checkpoints (read-only) keyed ``b3_seed{s}``."""
    models: Dict[str, IndependentMLPScorer] = {}
    for seed in seeds:
        model, _payload = phase0b.load_b3_model(Path(b3_root) / f"seed_{int(seed)}", device=device)
        models[f"b3_seed{int(seed)}"] = model
    return models


def materialise_examples(
    corpus: B3Corpus,
    samples: Sequence[object],
    k: int,
    *,
    text_cache: Optional[Dict[int, np.ndarray]] = None,
) -> B3ExampleBatch:
    """Gather ``(z_q, z_i, geometry)`` of ``samples`` at width ``k``.

    Mirrors :meth:`B3Corpus.example_for` step by step (``region[cand]`` cast to
    float32, the same stored text row, ``geometry_features_centered`` on the
    bank boxes / objectness) so the caller can verify bit-level reproduction
    against the frozen Phase 0B raw scores.  ``text_cache`` may be shared
    across calls to avoid re-reading repeated sentences from the h5 cache.
    """
    k = int(k)
    n = len(samples)
    z_q = np.empty((n, FEATURE_DIM), dtype=np.float32)
    z_i = np.empty((n, k, FEATURE_DIM), dtype=np.float32)
    geometry = np.empty((n, k, GEO_DIM), dtype=np.float32)
    cache = {} if text_cache is None else text_cache
    for i, sample in enumerate(samples):
        cand = B3Corpus.candidate_bank_indices(sample, k)
        image_id = int(getattr(sample, "image_id"))
        sentence_id = int(getattr(sample, "sentence_id"))
        region = corpus._region(image_id)
        z_i[i] = np.asarray(region[cand], dtype=np.float32)
        cached = cache.get(sentence_id)
        if cached is None:
            cached = np.asarray(corpus._text(sentence_id), dtype=np.float32)
            cache[sentence_id] = cached
        z_q[i] = cached
        boxes, objectness = corpus._bank(image_id)
        geometry[i] = geometry_features_centered(
            boxes[cand], corpus.image_size(image_id), objectness[cand]
        )
    return B3ExampleBatch(z_q=z_q, z_i=z_i, geometry=geometry)


def score_examples(
    model: IndependentMLPScorer,
    batch: B3ExampleBatch,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Frozen forward of one checkpoint: float32 ``[n, k]`` logits."""
    n, k = batch.n, batch.k
    out = np.empty((n, k), dtype=np.float32)
    size = max(1, int(batch_size))
    for start in range(0, n, size):
        stop = min(start + size, n)
        chunk = phase0b._forward_superset(
            model, batch.z_q[start:stop], batch.z_i[start:stop], batch.geometry[start:stop]
        )
        if chunk.shape != (stop - start, k):
            raise ValueError(
                f"frozen forward returned {chunk.shape}, expected ({stop - start}, {k})"
            )
        out[start:stop] = chunk
    return out


def verify_against_raw_scores(
    recomputed: Mapping[str, np.ndarray],
    frozen: Mapping[str, np.ndarray],
    *,
    k: int,
    atol: float = 1e-4,
) -> Dict[str, float]:
    """A8.4 STOP check: ``max|recomputed - raw_scores|`` per scorer.

    Raises :class:`AssertionError` when any scorer drifts beyond ``atol`` (the
    frozen checkpoints must reproduce the stored Phase 0B logits; a failure
    invalidates every Phase 1F number and must stop the stage).
    """
    deltas: Dict[str, float] = {}
    for scorer, values in recomputed.items():
        reference = np.asarray(frozen[scorer], dtype=np.float32)
        values32 = np.asarray(values, dtype=np.float32)
        if values32.shape != reference.shape:
            raise ValueError(
                f"{scorer} K={k}: recomputed {values32.shape} vs frozen {reference.shape}"
            )
        delta = float(
            np.max(np.abs(values32.astype(np.float64) - reference.astype(np.float64)))
        )
        if delta > float(atol):
            raise AssertionError(
                f"{scorer} K={k}: raw-score reproduction max|delta|={delta:.3e} exceeds "
                f"{atol:.1e} - the frozen scoring pipeline drifted; STOP"
            )
        deltas[scorer] = delta
    return deltas
