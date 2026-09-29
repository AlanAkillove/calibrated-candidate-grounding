"""Phase 1E — FineCops-Ref external semantic confirmation (protocol A9).

This package holds the *external* side of the study only:

* :mod:`ccg.external.finecops`    — test-split annotation parsing, GQA scene-graph
  join, official difficulty/tuple-type distributions;
* :mod:`ccg.external.gqa_images`  — on-demand retrieval of the GQA JPEGs that the
  test split needs, straight out of the 20 GB official zip via HTTP range
  requests, plus a decode/dimension check;
* :mod:`ccg.external.feasibility` — the F2/F3 proposal-and-candidate audit and the
  frozen engineering criteria that decide the external protocol branch.

Nothing in here trains, calibrates or scores a reliability model: Phase 1E
re-uses the RefCOCO+-frozen pipeline unchanged and this package is restricted
to loading external data and measuring whether that pipeline *can* be applied
to it at all.
"""

from __future__ import annotations

__all__: list[str] = []
