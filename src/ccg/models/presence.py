"""Phase 2 target-presence / NONE models - PLACEHOLDERS ONLY.

Protocol sections 6 and 20 distinguish *scene-negative* (the object is not in
the image) from *candidate omission* (the object is in the image but the
proposal bank did not supply it); this project cares about the latter, and it
must be evaluated with ``synthetic_omit`` and ``natural_omission`` reported
separately - never merged into one number.

What already exists (not gated):
    * N0 max-confidence threshold -> :func:`ccg.calibration.thresholds.fit_max_conf_threshold`
    * N1 margin threshold         -> :func:`ccg.calibration.thresholds.fit_margin_threshold`
    * synthetic omission          -> :func:`ccg.data.candidate_sets.synthetic_omit`

What is gated behind Phase 0/0.5 (implemented as structure only here):
    * N2 flat ``(K+1)``-way NONE classifier
    * N3 stats-only presence model
    * N4 set-aware presence model
    * N5 factorised ``p_present * P(c_i | present)`` model
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "GATE_MESSAGE",
    "FlatNoneClassifier",
    "StatsPresenceModel",
    "SetAwarePresenceModel",
    "FactorizedPresenceModel",
    "factorized_probabilities",
]

GATE_MESSAGE = (
    "Target-presence models belong to Phase 2 and may only be trained after Phase 0 / 0.5 "
    "gates (Q1, Q2) are resolved with the frozen protocol. Structure only - no training loop."
)

#: N5 factorisation, written down once so the maths cannot drift later:
#:
#:   P(NONE)   = 1 - p_present
#:   P(c_i)    = p_present * softmax(s / T)_i
#:
#: where ``p_present = P(y in C | I, q, C)`` comes from a presence head (N3/N4)
#: and ``softmax(s/T)`` is the *conditional* ranking distribution of the
#: target-present model.
def factorized_probabilities(
    conditional_probs: np.ndarray, p_present: float
) -> tuple[np.ndarray, float]:
    """Apply the N5 factorisation to already-computed conditional probabilities.

    Pure numpy and *deliberately* allowed before Phase 2: it introduces no
    learnable parameters, it only composes an existing ranking distribution
    with a presence estimate.  ``conditional_probs`` must sum to 1 over the
    candidates; the returned ``NONE`` probability is ``1 - p_present``.
    """
    conditional = np.asarray(conditional_probs, dtype=np.float32).reshape(-1)
    if conditional.size == 0:
        raise ValueError("conditional_probs must contain at least one candidate")
    if conditional.sum() <= 0:
        raise ValueError("conditional_probs must be a probability vector")
    conditional = conditional / conditional.sum()
    p_present = float(p_present)
    if not 0.0 <= p_present <= 1.0:
        raise ValueError(f"p_present must be in [0,1], got {p_present}")
    return (p_present * conditional).astype(np.float32, copy=False), 1.0 - p_present


class FlatNoneClassifier:
    """N2: a ``(K+1)``-way classifier whose last class is NONE - not implemented."""

    name = "n2_flat_none"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - gated stub
        raise NotImplementedError(GATE_MESSAGE)


class StatsPresenceModel:
    """N3: presence head on the C3 score statistics - not implemented."""

    name = "n3_stats_presence"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - gated stub
        raise NotImplementedError(GATE_MESSAGE)


class SetAwarePresenceModel:
    """N4: presence head on a pooled candidate representation - not implemented."""

    name = "n4_set_aware_presence"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - gated stub
        raise NotImplementedError(GATE_MESSAGE)


class FactorizedPresenceModel:
    """N5: ``P(c_i) = p_present * P(c_i | present)`` wrapper around a ranker.

    Only the composition helper :func:`factorized_probabilities` is provided;
    the presence head it would call is N3/N4 and stays gated.
    """

    name = "n5_factorized_presence"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - gated stub
        raise NotImplementedError(GATE_MESSAGE)

    @staticmethod
    def combine(conditional_probs: np.ndarray, p_present: float):
        return factorized_probabilities(conditional_probs, p_present)
