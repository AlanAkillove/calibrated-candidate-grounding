"""Deterministic seeding (protocol section 23: all learned models run 3 seeds).

``torch`` is imported lazily so the utility works in torch-free environments
(the data/metric unit tests must stay importable without it).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_deterministic", "default_rng", "SEEDS"]

#: The three seeds every learned model is run with (section 23).
SEEDS: tuple[int, ...] = (0, 1, 2)

#: Also fixed via ``PYTHONHASHSEED`` so set/dict iteration order does not drift.
_DETERMINISTIC_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def set_deterministic(seed: int = 0, *, cuda_deterministic: bool = True, env_threads: bool = False) -> int:
    """Seed ``random``, ``numpy`` and (when importable) ``torch``.

    Parameters
    ----------
    seed:
        Non-negative integer; returned unchanged so call sites can log it.
    cuda_deterministic:
        Also set ``torch.backends.cudnn.deterministic = True`` and disable the
        benchmark autotuner.  Slower, but a re-run must reproduce bit-for-bit.
    env_threads:
        Pin BLAS thread counts to 1.  Off by default because it changes global
        process state; enable it for exact-reproduction runs.

    Notes
    -----
    Candidate sets are *pre-generated and frozen* (section 23): a seed here is
    for the decision model and the reporting bootstrap, never a licence to
    re-sample candidate sets at evaluation time.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    if env_threads:
        for key, value in _DETERMINISTIC_ENV.items():
            os.environ.setdefault(key, value)

    random.seed(seed)
    np.random.seed(seed % (2**32))

    torch = _try_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(cuda_deterministic)
        torch.backends.cudnn.benchmark = not bool(cuda_deterministic)
    return seed


def default_rng(seed: Optional[int] = 0) -> np.random.Generator:
    """A fresh ``numpy`` generator (preferred over the legacy global RNG)."""
    return np.random.default_rng(seed)


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:  # pragma: no cover - torch absent
        return None
