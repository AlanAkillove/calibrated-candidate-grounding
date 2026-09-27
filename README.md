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
Current stage: Phase 0 — Candidate-Set Failure Audit
```

The project is explicitly designed to allow an early NO-GO decision if candidate-set
shift does not produce a stable reliability failure or if simple score-level calibration
is sufficient. No results are claimed at this stage.

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

Planned end-to-end pipeline (see [`docs/phase0_plan.md`](docs/phase0_plan.md)):

```bash
python scripts/prepare_refcoco.py        # parse annotations, build UNC splits
python scripts/extract_proposals.py      # frozen proposal bank + audit
python scripts/extract_features.py       # offline CLIP region/text embeddings
python scripts/build_candidate_sets.py   # nested, frozen candidate sets
python scripts/audit_proposals.py        # recall@N, IoU distributions
python scripts/run_phase0.py             # baselines, calibration, gates
```

All learned components run with 3 seeds, image-level paired bootstrap (95% CI, >= 5,000
replicates), and complete experiment logging (`docs/experiment_log.md`).

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
