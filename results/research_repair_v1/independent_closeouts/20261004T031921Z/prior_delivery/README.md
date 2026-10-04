# Confidence Reliability under Candidate-Set Expansion

A controlled empirical study of candidate-based visual grounding: how do accuracy,
confidence discrimination, calibration, and selective utility change when the same
image and referring expression receive an expanding candidate set?

The evaluation uses GT assistance: it forces in a target proposal, removes other
valid target boxes, and conditions the common cohort on maximum-K feasibility.
Conclusions concern this target-present interface. RefCOCO+, RefCOCO, and RefCOCOg
share the COCO visual domain.

## Research questions

1. How do accuracy, correctness AUROC, calibration, and selective utility change along
   the specified nested expansion paths?
2. Can small-K score models generalize to large K, and which query–crop or
   candidate–candidate feature blocks add reliability information?
3. How do four corners, two paths, and their interaction account for AUROC changes,
   and what remains unexplained?

With fixed independent scoring and tie-breaking, adding distractors cannot correct
an already wrong choice. Accuracy decline is partly structural; AUROC decline requires
data. E-AURC and RER retain accuracy dependence. Logits temperature preserves argmax
but can change cross-expression maximum-softmax-confidence ordering.

## Current evidence and repair

Research Repair v1 is complete, executed by three specified `gpt-6-luna/max` agents.
All 9,955 stored estimates passed numerical verification; the full suite passed
with 1,224 tests and two original opt-in skips. Frozen-input verification and the
overall acceptance check passed. M3's three exact-recovery jobs remain explicitly
UNVERIFIABLE, with their original tolerances and failure evidence preserved. State:
[STATUS.json](results/research_repair_v1/STATUS.json).

The repaired study retains B3 discrimination degradation along the specified K
expansion and conditional extra signal from frozen embedding features. Incremental
candidate–candidate signal beyond query scoring varies by K and composition; the
paired intervals do not support a general interaction claim. Corrected mean
intervals and operational gate comparisons are preserved in the current export.
Score-model failure alone cannot establish an information limit. Exact Shapley
accounting does not identify a causal mechanism; small net components can contain
opposing path effects.

- [Current summary](docs/final_result_summary.md)
- [Repair protocol](reviews/repair_protocol.md)
- [Repair evidence index](results/research_repair_v1/registry/index.md)
- [Artifact-derived tables and figures](results/research_repair_v1/publication/statistics.md)
- [Historical V1 registry](results/final_registry/)
- [Pre-repair material bank](docs/appendix/final_result_summary_pre_repair.md)
- [Question-based paper blueprint](docs/paper_blueprint_v1.md)
- [Research review](reviews/strict_research_review_2026-10-03.md)

V2-G trained scorer/reliability heads for two additional frozen encoders. V2-P uses a
shared frozen scorer on different proposal families. V2-M/M3 train or fit reliability
models. These separate axes do not form a full backbone × proposal factorial study.
Strict RefCOCOg is image-disjoint within COCO; its invalid hard manipulation does not
confirm external amplification. Target absence and natural detector top-K evaluation
are outside this repair.

## Reproduction and provenance

This is a staged study. The append-only [protocol](docs/research_protocol.md) and
[log](docs/experiment_log.md) distinguish full pre-result configuration freezes, input
identity freezes, post-result corrections, and target redefinitions. Gates are
operational continuation rules. Repair experiments are result-driven supplementary
analyses on previously observed test distributions, not independently preregistered
confirmation on an unseen test set.

Old predictions, models, candidates, annotations, and caches are read-only; new outputs
live in `results/research_repair_v1/`. In this checkout use:

```powershell
& 'E:/conda/envs/deepminer/python.exe' -B scripts/prepare_research_repair.py --verify
```

Do not regenerate the historical `final_registry` during repair. The repair index
provides current artifact sources. Render stored evidence without training or
resampling, then verify the baseline and supplemental input identities:

```powershell
& 'E:/conda/envs/deepminer/python.exe' -B scripts/build_repair_report.py
& 'E:/conda/envs/deepminer/python.exe' -B scripts/build_repair_registry.py
& 'E:/conda/envs/deepminer/python.exe' -B scripts/verify_repair_inputs.py
```

Fresh environment installation: `pip install -e ".[dev]"`.

Data setup and attribution: [data/README.md](data/README.md) and
[dataset protocol](docs/dataset_protocol.md). Verified core references:
[literature notes](docs/literature_notes.md). This work makes no universal reliability
repair, architecture innovation, or first-benchmark claim.
