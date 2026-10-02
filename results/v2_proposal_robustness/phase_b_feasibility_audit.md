# Phase B — External Validation Feasibility Audit (V2-P2-B0)

Date: 2026-10-02. Branch `v2-proposal-robustness`, after P1-F4/F5
(`CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST = YES`, commits `feb40d8` / `6496119` / `7441ab1`).
This is an **audit only** — no bank extraction, no scoring, no experiment was run here
(protocol: "先执行 Phase B feasibility audit，不要直接运行大型实验").

## 0. Frozen context this audit must respect

- **P1-A0**: the primary Random regime is the exact V1 seeded-random distractor
  construction (target slot 0, seed 20260927, nested prefixes). Detector-confidence
  ordering may only be bank *provenance*, never a candidate-selection rule.
- Single-variable design: only the **proposal family** changes; candidate-construction
  regime, frozen B3 + per-seed temperature, frozen R1/E1b heads, image-cluster paired
  bootstrap (5000 / seed 0 / CI 0.95) and the V2-G G3/G4 gates stay identical.
- Zero new training parameters; the entire `p1_replication_core` / F4 / F5 stack is
  already family-parameterized (`FAMILIES` dict: features_root / manifests_root / bank).

## 1. Environment findings (measured, not assumed)

| Item | Status | Evidence |
|---|---|---|
| HF connectivity | **reachable** | `HEAD https://huggingface.co` -> 200 |
| `transformers` | 4.57.6 installed; `GroundingDinoForObjectDetection` + `AutoProcessor` import OK | deepminer env |
| GDINO weights | tiny ~1.4 GB, base ~1.9 GB (downloadable) | HF API `?blobs=true` |
| DETR-R101 weights | ~486 MB (downloadable) | HF API |
| COCO train2014 images | 19,992 present locally (`data/raw/mscoco/train2014`) | counted |
| Bank writer | family-agnostic `bank-v1` layout (`scripts/p1_extract_detr_proposals.py` mirrors `extract_proposals`) | read |
| Existing banks | RPN + DETR-R50 complete; DETR probe: 6.8 img/s, 0.47 GB peak (RTX 4060 8 GB) | P1-F0 report |
| RefCOCOg external | already run (A11): Q1 semantic transfer **YES** (d_hard 0.0178), Q2 **INVALID_MANIPULATION** -> `EXTERNAL INCONCLUSIVE` | `a11_report.json` |
| FineCops external | already audited: engineering gate `EXTERNAL STOP` (RPN recall@0.5 0.758 < 0.90; same-name K5 supply 0.186 < 0.90) | `external_branch_decision.json` |

## 2. Candidate targets

### E1. Grounding DINO **class-prompt, query-independent bank** — RECOMMENDED (primary)

A fixed COCO-80 class prompt per image ("person . bicycle . ...") -> a per-image box
bank (top-64 by detector score as *provenance truncation only*), each box carrying an
open-vocabulary class label. Assign boxes to highest-IoU COCO GT (IoU >= 0.5) exactly
as in V1; `same_category` distractors then follow the unchanged V1 rule.

**Protocol compatibility (the decisive argument):** DETR-R50 is itself a class-
conditioned, expression-independent detector; GDINO in class-prompt mode has *exactly
the same conditioning form* (prompt set fixed, never the referring expression). The
only changed variable remains the proposal family. P1-A0 random ordering, nested-K,
common-K50 cohort, matched hard/random controls, frozen heads and gates all reuse the
P1 code path by adding one `FAMILIES` entry.

Cost estimate: weight download ~1.4–1.9 GB; bank extraction 19,992 images at ~2–4
img/s (Swin-B fp32, 8 GB VRAM budget feasible) ~1.5–3 h; SigLIP region-feature cache
~2–4 h (same as DETR-F1). Total: one unattended night. New code: one generator module
`ccg/data/gdino.py` + one extractor script (clone of the DETR extractor) + audit
manifests builder reuse.

Risks: (a) extreme box redundancy could inflate same-category supply (mitigated — it
helps C4 eligibility, and C1 is composition-controlled by the intersection design);
(b) recall@0.5 gate may fail for small referents — probe first; (c) variable box count
per image requires the documented top-64 truncation (provenance, not selection).

### E2. Grounding DINO **query-conditioned (expression-as-prompt) top-K** — SECONDARY ONLY

Feeding the referring expression as the prompt is a *different construct*: the bank
then depends on the query, distractor ordering is inherently confidence-ordered, and
nested-K prefixes over a seeded-random bank no longer exist. Running it as "C1
replication" would change proposal family **and** candidate regime simultaneously —
the exact confound P1-A0 was written to prevent. Optionally reportable as a
descriptive deployment-style diagnostic (does reliability degrade as a production
open-vocab pipeline's top-K grows?), never used for a C1/C4 gate.

### E3. DINO-DETR (or DETR-R101) query-independent bank — cheap robustness check

Same conditioning as both existing families; strongest compatibility, lowest cost
(weights 486 MB–1.2 GB; extraction ~1–2 h; features ~2 h). Scientific increment is
modest (same DETR lineage). Reasonable as an E1 fallback if E1 fails a probe gate,
or as a third family if E1 passes and reviewers ask for breadth.

### E4. RefCOCOg A11 re-interpretation — ZERO cost, feed Phase C instead of re-running

Already provides dataset-level external evidence: semantic transfer confirmed
(Q1 YES, 3/3 seeds), hard amplification **not assessable due to invalid manipulation
construction** (not a null effect). Phase C should reuse this: it separates "the
amplification phenomenon" from "our specific same-category manipulation". No re-run.

### E5. FineCops — REJECT

Already gated `EXTERNAL STOP` on frozen-artifact grounds (proposal recall 0.758,
same-name supply 0.186). Re-running under a new proposal family would need a new
feasibility justification and would not strengthen the core conclusion. Not proposed.

### E6. Random / grid proposal bank — sanity control, belongs to Phase C

Trivially cheap and nested-K-clean, but it is not a "mainstream pipeline"; its value
is mechanistic (does cardinality degradation survive a *degenerate* proposal family?
-> would show the phenomenon is candidate-space structural, not detector-specific).
Suggest listing it under the mechanism section, not as external validation.

## 3. Recommended decision

**GO on E1 (GDINO class-prompt bank), gated by a small engineering probe first.**
Probe (~100 stratified train2014 images + ~100 test images, cost < 30 min including
download), evaluated against frozen thresholds before any full-night extraction:

| Probe gate | Threshold | Rationale |
|---|---|---|
| boxes per image after truncation | >= 64 for >= 98% of probed images | nested-K needs 49 valid distractors |
| ref-target recall@0.5 (N=64) | >= 0.95 | GDINO is COCO-pretrained+; DETR was 0.997, RPN 0.986 |
| same-category supply (>=4, N=64) | >= 0.85 | above DETR's 0.7817; C4 cohort health |
| K50 availability (pooled-test rows) | >= 0.90 | matches the P1-F2 gate DETR passed (0.9224) |
| peak VRAM | <= 6 GB | RTX 4060 8 GB with margin |
| checkpoint identity | sha256 recorded, revision pinned | same regime as DETR-R50 / B3 |

If a gate fails: fall back to E3 (DINO-DETR bank) rather than weakening the protocol.
Do **not** substitute E2 as the primary replication.

If E1 passes: run the unchanged pipeline — manifests (random + same_category), feature
cache, `p1_frozen_inference --family GDINO`, then re-apply `p1_f4_cardinality` /
`p1_f5_hard_semantic` verbatim with the same frozen config (a new simultaneous config
freeze, committed before results, exactly as P1-A0/F4/F5 did).

## 4. What this audit does NOT authorize

No scorer changes, no new reliability model, no temperature refit, no re-running of
FineCops/RefCOCOg, no E2-primary framing, no paper writing (Phase A stays queued after
B/C). This document is the only artifact produced by this round.

## 5. Probe outcome (recorded 2026-10-02, after this audit was frozen)

The probe ran on the first 100 images of the frozen `data/audit_subset.csv` using the
unchanged V1 audit primitives (`scripts/p2_gdino_probe.py`; thresholds read from
`phase_b_feasibility_audit.json`, never re-specified). Full record:
`results/v2_proposal_robustness/p2_gdino_probe/probe_report.json`.

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| boxes >= 64 share | 0.98 | **1.0000** | PASS |
| ref-target recall @ 0.5 | 0.95 | **0.9880** | PASS |
| same-category supply (>= 4) | 0.85 | **0.6892** | **FAIL** |
| K50 availability | 0.90 | **0.9920** | PASS |
| peak VRAM (GB) | <= 6.0 | **2.63** | PASS |

Mechanism of the failure (descriptive): under the frozen top-64 truncation with no
NMS, GDINO's score-ordered bank is dominated by same-class duplicate detections
(`person` alone claims 20.3% of kept labels) and hallucinated small-object classes
(book / apple / cup / bowl), crowding out distinct same-category objects relative to
DETR's set-prediction bank (supply 0.7817 -> 0.6892). Cardinality-side gates (K50
availability 0.992, recall 0.988) are excellent; the deficit is specific to the
category-diversity requirement of the C4 same-category cohort.

Per the frozen `on_failure` rule this closes E1 as a **full-participation** external
validation target; the fallback named in the audit is E3 (DINO-DETR / DETR-R101).
Any alternative that admits GDINO for C1 only (its gates all pass) while sourcing the
cross-family C4 evidence elsewhere is a NEW protocol decision outside this audit and
requires explicit authorization before any experiment runs.
