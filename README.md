# Calibrated Candidate Grounding

Reliable visual grounding under dynamic candidate-set shift: calibration, abstention, hard negatives, and target omission with frozen vision-language features.

## Motivation

Visual grounding systems that rank a provided candidate set of regions are typically
evaluated under a fixed candidate protocol. In deployment (e.g., embodied vision agents),
the candidate set produced by an upstream proposal system changes constantly: its size
\(K\), its composition (random vs. hard distractors), and even whether the true target is
present at all. It is poorly understood whether the ranking scores and probabilities
emitted by grounding models remain **reliable** under this candidate-set shift, and
whether simple scalar/score-level statistics are sufficient to explain and correct any
resulting miscalibration.

## Research Questions

- **RQ1 — Reliability shift.** How do candidate-set cardinality and composition affect
  ranking accuracy and calibration differently?
- **RQ2 — Sufficiency of simple uncertainty statistics.** Can score-level statistics
  (max softmax, margin, entropy, \(T(K)\), stats-only calibrators) explain and correct
  the reliability degradation?
- **RQ3 — Set information.** Can candidate-level set information (DeepSets-style,
  permutation-invariant models) predict grounding reliability better than scalar or
  score-distribution statistics?
- **RQ4 — Target omission.** When the target is missing from the candidate set
  (synthetic omission or natural proposal miss), how should a system abstain
  (NONE / reject), and do factorized presence/ranking formulations beat flat
  \((K+1)\)-way or threshold-based approaches?

## Why Candidate-Set Shift?

Candidate-based grounding, region–text matching, and NONE-style rejection are not new per
se. The open question is whether **reliability** (calibration, selective risk, abstention
quality) degrades in a structured way as the candidate distribution shifts — and whether
that degradation can be characterized and fixed without heavy models. This distinction
matters because softmax denominators, hard-negative composition, and target-availability
all change the meaning of a "probability" over candidates.

## Research Protocol

The full protocol — dataset, backbone, proposal bank, candidate-set construction,
metrics, splits, calibration procedure, and pre-registered GO/NO-GO gates — is frozen in
[`docs/research_protocol.md`](docs/research_protocol.md) **before any Phase-0 results**.
Any later change requires a dated amendment that preserves the original criteria.

## Phase 0: Failure Audit

Phase 0 audits the failure mode using only simple baselines: random selection, frozen
OpenCLIP cosine scoring, global temperature scaling, and an independent (candidate-blind)
MLP scorer — evaluated on a \(4 \times 2\) test grid over
\(K \in \{5, 10, 20, 50\} \times \{\text{random}, \text{hard}\}\), with top-label ECE,
binary-correctness Brier/NLL, reliability diagrams, and risk–coverage/AURC. Complex
candidate-aware models are explicitly forbidden until Gate Q1 and Gate Q2 rules demand them.

## Project Status

```text
Current stage: main experimental study complete — results frozen
```

The candidate-cardinality / semantic-reliability mainline is **complete and frozen** after
the RefCOCOg external confirmation (Amendment A11). No further external benchmarks,
backbones, architectures, candidate-aware reranking, semantic-feature redesign, or
hard-negative redefinition are added. The frozen evidence set lives in
[`results/final_registry/`](results/final_registry/) and
[`docs/final_result_summary.md`](docs/final_result_summary.md).

This is a **controlled reliability study**, not a state-of-the-art or novel-architecture
claim. Target omission / abstention (RQ4) is deliberately **out of scope** for this
mainline: no target-absence result is reported here.

## Key Findings (frozen)

See [`results/final_registry/tables.md`](results/final_registry/tables.md) and
[`results/final_registry/claims.csv`](results/final_registry/claims.csv) for the full tables.

- **Candidate cardinality degrades reliability** (B3, random regime, K5→K50): E-AURC
  relative worsening ≈ +226%, RER@50 0.821→0.365, AUROC_correct 0.843→0.790. The raw
  accuracy decline is partly structural under nested candidate sets; the emphasis is on
  E-AURC / AUROC_correct / RER@coverage.
- **Temperature / score scaling is not the explanation**: the corrected global temperature
  is an interior optimum and per-K oracle temperatures shift only ~10%.
- **Score information is insufficient**: no score-only representation (MSP, margin, entropy,
  handcrafted statistics, log K, full score-set DeepSets) removes the degradation.
- **Candidate semantics add a modest signal** (Phase 1, gray zone → A7 INCONCLUSIVE).
- **Semantic value is amplified under controlled hard competition**: RefCOCO+
  SameCategory-K5 ΔAUROC +0.0318 vs +0.0012 under matched random, growing monotonically
  with the number of same-category competitors (m = 0/2/4/8).
- **Cross-dataset semantic transfer confirmed** on a strict image-disjoint RefCOCOg subset
  (same-category ΔAUROC +0.0178, 20.4% E-AURC reduction) — *under a shared COCO visual
  domain*, not cross-domain visual generalization.
- **External hard-amplification replication is not assessable** (RefCOCOg manipulation
  invalid), so the hard-minus-random gap is not claimed as replication.

## Experimental Stages

```text
Proposal audit → Phase 0A (cosine) → 0A.1 (metric/temp correction) → 0B (B3)
→ Phase 0.5 (score sufficiency) → Phase 1 (semantic sufficiency)
→ Phase 1F (hard-competition confirmation) → FineCops feasibility (STOP)
→ RefCOCOg feasibility → RefCOCOg A11 (external confirmation)
```

The full pre-registered protocol and its dated, append-only amendments (A1–A11, with
pre-result / post-result / staged / external-feasibility classification) are in
[`docs/research_protocol.md`](docs/research_protocol.md) and
[`results/final_registry/protocol_history.csv`](results/final_registry/protocol_history.csv).

## Negative Results

Negative results are retained as part of the logic chain, not hidden. See
[`results/final_registry/negative_results.csv`](results/final_registry/negative_results.csv):
global calibration does not explain the degradation; score statistics and full score-set
DeepSets fail; learned E2 and full-set E3 fail; the FineCops external route is stopped due
to proposal-domain mismatch; the RefCOCOg hard manipulation is invalid for the amplification
test.

## Reproducibility

Every frozen headline number is derived from existing phase artifacts — never hand-typed —
by a read-only aggregator:

```bash
python scripts/build_final_registry.py
```

The script re-derives Tables 1–4 and Figures 1–4 into `results/final_registry/`, and runs a
consistency audit against `docs/experiment_log.md` before writing anything: any discrepancy
is a hard STOP (exit non-zero, no outputs written). It performs no training, fitting,
resampling, model change, or new statistical test.


## Installation

```bash
conda create -n ccg python=3.10 -y
conda activate ccg
pip install -e ".[dev]"
```

(Any Python >= 3.10 environment with PyTorch works; all experiments target a single
8 GB-VRAM laptop GPU.)

## Data

Primary dataset: **RefCOCO+** (UNC image-level splits), with **FineCops-Ref** reserved as
an external stress test. Backbone: frozen **OpenCLIP ViT-B/32** with fully offline feature
caching. Candidates come from a frozen COCO-pretrained proposal bank (N ≈ 64 per image).
Download and organization instructions are in [`data/README.md`](data/README.md) and
[`docs/dataset_protocol.md`](docs/dataset_protocol.md). Please note the licenses and
registration requirements of COCO and RefCOCO before downloading.

## Reproduction

The study is complete; the pipeline below was run in this order (each phase writes its
artifact directory under `results/`, which the final aggregator then reads):

```bash
# Data + frozen resources
python scripts/prepare_refcoco.py                 # parse annotations, build UNC splits
python scripts/extract_proposals.py               # frozen proposal bank
python scripts/audit_proposals.py                 # recall@N, IoU distributions
python scripts/extract_features.py                # offline CLIP region/text embeddings
python scripts/build_candidate_sets.py            # nested, frozen candidate sets
# Staged experiments
python scripts/run_proposal_audit.py              # proposal audit
python scripts/run_phase0a.py                     # Phase 0A cosine
python scripts/run_phase0a_corrected.py           # Phase 0A.1 metric/temperature correction
python scripts/train_b3.py ; python scripts/run_phase0b.py    # Phase 0B candidate-blind scorer
python scripts/run_phase05.py                     # Phase 0.5 score sufficiency
python scripts/run_phase1.py                       # Phase 1 semantic sufficiency
python scripts/run_phase1f.py                      # Phase 1F hard-competition confirmation
python scripts/run_phase1e_feasibility.py          # FineCops feasibility (STOP)
python scripts/run_phase1e_refcocog_feasibility.py # RefCOCOg feasibility
python scripts/a11_external_inference.py ; python scripts/a11_external_analysis.py  # A11 external
```

All learned components run with 3 seeds, image-level paired bootstrap (95% CI, >= 5,000
replicates), and complete experiment logging (`docs/experiment_log.md`). The frozen final
result set is regenerated from the phase artifacts by `scripts/build_final_registry.py`
(see **Reproducibility** above).

## Citation

If you find this research useful, please cite:

```bibtex
@misc{ccg2026,
  title  = {Calibrated Candidate Grounding: Reliable Visual Grounding under Dynamic
            Candidate-Set Shift},
  author = {TBD},
  year   = {2026},
  url    = {https://github.com/AlanAkillove/calibrated-candidate-grounding}
}
```
