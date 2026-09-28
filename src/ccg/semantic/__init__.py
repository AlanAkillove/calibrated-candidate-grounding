"""Phase 1 — candidate semantic information sufficiency audit (protocol A7).

This package implements the *frozen* Phase 1 model zoo and evaluation helpers:

* :mod:`ccg.semantic.data`     — frozen-cohort embedding store + score loading;
* :mod:`ccg.semantic.features` — E1 handcrafted semantic-geometry statistics;
* :mod:`ccg.semantic.models`   — E2 top-competitor model + E3 semantic DeepSets;
* :mod:`ccg.semantic.evaluate` — A7 gate semantics + paired-bootstrap rows.

Nothing here may touch grounding scores or rerank candidates: every model maps
``(query, candidate semantics, frozen scores) -> P(correct)`` only.
"""

from __future__ import annotations

__all__: list[str] = []
