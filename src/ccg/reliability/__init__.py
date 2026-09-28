"""Phase 0.5 score-information sufficiency audit (score-only reliability models).

Amendment A6 (`docs/research_protocol.md`) freezes the protocol: the reliability
models may only see candidate *scores* of the frozen grounding scorers (B3
independent MLP seeds 1-3, frozen cosine) - never candidate embeddings, query
features, geometry, objectness or GT metadata.  The training rows come from the
image-level reliability_train/reliability_tune split of ``val_calib`` at
``K in {5, 10}``; ``testA``/``testB``/``K20``/``K50`` never enter training,
early stopping or model selection.

Submodules
----------
- :mod:`ccg.reliability.data` - load frozen score matrices and build the
  image-level split (deterministic, seeded, frozen before results).
- :mod:`ccg.reliability.features` - L0 scalar confidences, the L1 handcrafted
  score statistics and the packed inputs of the L2 ScoreDeepSets model.
- :mod:`ccg.reliability.models` - the frozen reliability-model zoo
  (``LogisticModel`` / ``TinyMLP`` / ``ScoreDeepSets``).
- :mod:`ccg.reliability.evaluate` - reliability metrics, image-clustered paired
  bootstrap rows and the sufficiency-gate verdict.
"""

from __future__ import annotations

__all__ = ["data", "evaluate", "features", "models"]
