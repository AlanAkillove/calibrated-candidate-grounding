"""Local Competition Reliability (V2-M): frozen relation features, LCR models, audits.

This package implements the first version of the LCR reliability module exactly
as frozen in ``results/v2_local_competition/protocol.json``:

* :mod:`ccg.lcr.features` -- the 7-d per-competitor relation vector (M = 4) and
  the 3-d ambiguity-gate input, computed only from ``(scores, z_q, z_i, T)``;
* :mod:`ccg.lcr.models` -- ``LCR`` (``z = b + g * delta``), ``LCR-noGate``
  (``z = b + delta``) and the ``Aggregate-MLP`` capacity control (31d -> 32 ->
  16 -> 1), all with the shared reliability training protocol;
* :mod:`ccg.lcr.audit` -- grounding-invariance fingerprints and the
  zero-hard-exposure training guards.

Nothing in this package trains, modifies or even holds a grounding scorer: the
frozen B3 decisions are inputs that must remain bit-identical.
"""

from __future__ import annotations

__all__ = [
    "M_COMPETITORS",
    "GATE_INPUT_NAMES",
    "RELATION_FEATURE_NAMES",
    "RelationBatch",
    "relation_features",
    "LCR",
    "LCRNoGate",
    "AggregateMLP",
]

from .features import (
    GATE_INPUT_NAMES,
    M_COMPETITORS,
    RELATION_FEATURE_NAMES,
    RelationBatch,
    relation_features,
)
from .models import LCR, AggregateMLP, LCRNoGate
