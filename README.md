# Correctness Reliability under Candidate-Set Changes

A controlled empirical study of candidate-based visual grounding: how do accuracy,
confidence discrimination, calibration, and selective utility change when the same
image and referring expression receive an expanding candidate set?

The controlled evaluation uses GT assistance: it forces in a target proposal, removes other
valid target boxes, and conditions the common cohort on maximum-K feasibility.
Controlled conclusions concern this target-present interface. RefCOCO+, RefCOCO, and RefCOCOg
share the COCO visual domain.

V3 is complete: an exposure-audited FineCops positive-val confirmation,
OpenCLIP B/16 feature-source replication, and natural detector top-K evaluation.
The natural interface retains proposal misses and multiple valid target boxes.
Source-split identity,
research independence, and cross-visual-domain evidence are assessed separately;
GQA/Visual Genome provenance does not by itself prove a non-COCO image source.
See the [prospective protocol](reviews/v3_protocol.md) and
[actual execution status](results/v3_final_validation/STATUS.json).
The first frozen confirmation, 5,000-draw shared-image bootstrap and saved-output
acceptance passed. The full suite passed 1,298 tests with two original opt-in skips;
834 historical frozen files remained unchanged. The [completion review](reviews/v3_completion.md)
records actual evidence and remaining limits. The [V3 paper blueprint](docs/paper_blueprint_v3.md) follows the
questions through controlled findings, information sources, explanation audits
and external boundaries. [Replay instructions](docs/v3_reproduction.md) distinguish
saved-evidence verification from full image/model reconstruction.

The confirmation qualifies the central claims:

- FineCops did **not confirm** B0 MSP AUROC degradation from K5 to K50; the
  interval crosses zero. Accuracy and selective risk still deteriorate.
- Controlled K50 supports Full−S+Q in both configurations and large-K-trained
  versus small-K-trained ScoreDeepSets. Full is not universally best, and the
  large-K ScoreDeepSets remains below simpler baseline point performances.
- Natural K5 reverses the Full−S+Q gain. At K20, overall gains mostly account for
  ranking proposal-miss errors. At K50, B0 gains are supported, while B/16 is
  inconclusive. Natural secondary intervals are marginal, not additional primary confirmations.

See [artifact-generated main results](results/v3_final_validation/publication/key_findings.md),
[all denominators and intervals](results/v3_final_validation/evidence_index.md),
and the [paper argument draft](docs/paper_draft_v3.md).

## Research questions

1. How do accuracy, correctness AUROC, calibration, and selective utility change along
   the specified nested expansion paths?
2. Can small-K score models generalize to large K, and which query–crop or
   candidate–candidate feature blocks add reliability information?
3. How do four corners, two paths, and their interaction account for AUROC changes,
   and what remains unexplained?
4. Which controlled findings transfer to project-unexposed FineCops images and
   natural detector candidates, and where do they fail?

With fixed independent scoring and tie-breaking, adding distractors cannot correct
an already wrong choice. Accuracy decline is partly structural; AUROC decline requires
data. E-AURC and RER retain accuracy dependence. The finite-sample E-AURC
definition subtracts a continuous oracle reference from trapezoidal AURC and
can be negative even under ideal ranking; it is not clipped. Logits temperature preserves argmax
but can change cross-expression maximum-softmax-confidence ordering.

## Current evidence and repair

The historical repair's three independent-review corrections are complete and regression-verified,
executed by the specified `gpt-6-luna/max` agents. Its closeout full suite passed
1,238 tests with two original opt-in skips; all 263 source/config files remained
unchanged. Five input manifests and 834 files passed verification; all 9,955
stored estimates and 77 raw archive identities remain unchanged.
The new versioned execution acceptance is PASS. The original independent
CONDITIONAL_ACCEPTANCE verdict, prior PASS and tests remain preserved, with
scientific and provenance limits retained. State:
[STATUS.json](results/research_repair_v1/STATUS.json); details:
[review closeout](results/research_repair_v1/independent_closeouts/20261004T031921Z/closeout.md).

The following command rechecks the historical repair source state against its dated evidence;
the current V3 source is validated by the separate V3 test and acceptance records:
`python scripts/check_repair_acceptance.py --evidence-dir results/research_repair_v1/independent_closeouts/20261004T031921Z/logs`.
The original `logs/` records describe the earlier delivery.

M3 remains UNVERIFIABLE. Its original frozen-store anchor uses 1e-9 and dose
rescore anchor uses 1e-4; the committed-point replay check at 1e-9 was imposed by
the repair wrapper. It is not an original M3 protocol point cutoff. No approximate
recovery interval substitutes for the failed exact replay.

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
confirm external amplification. Target absence remains outside the study. Natural
detector top-K is measured in V3, separately from the historical repair.

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
[dataset protocol](docs/dataset_protocol.md). Primary-source analysis references:
[literature review](reviews/analysis_literature_review_2026-10-04.md).
The [historical notes](docs/literature_notes.md) contain entries still requiring verification.
This work makes no universal reliability
repair, architecture innovation, or first-benchmark claim.

## Paper assessment and public text evidence

[Current scientific assessment and writing plan](reviews/paper_assessment_2026-10-04.md) evaluates the conditional findings, remaining logic gaps and contribution boundaries.

Large JSON/CSV evidence is published as byte-identical gzip with SHA-256 metadata. After cloning, run `python scripts/restore_repair_text_artifacts.py` to restore these text files; `--verify-only` checks archive identities without restoring missing files. See [text archive manifest](results/research_repair_v1/text_artifacts/manifest.json). Frozen data/features, model weights and raw bootstrap arrays remain local prerequisites; the public text export is not a cache-free reproduction of the entire study.

## Theory and pairwise analysis

[Theory draft](docs/theory_analysis_v1.md) gives assumptions, proofs and counterexamples
for structural accuracy monotonicity, AUROC pair accounting, label-path bounds,
interaction and the limits of causal interpretation.
[Saved-prediction analysis](results/research_repair_v1/theory_analysis/v1/analysis_report.md)
connects path cancellation to changes in S/F/E pair rankings, with 5,000 shared image
draws, per-seed intervals and raw distributions. This is a result-driven analytical
extension; the existing publication registry and historical gates retain their identities.

With the frozen prediction/cache prerequisites available, reproduce in a new directory:

```powershell
$env:PYTHONPATH = 'src'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
& 'E:/conda/envs/deepminer/python.exe' scripts/analyze_repair_pairwise.py --output results/research_repair_v1/theory_analysis/replay_v1
```

Existing output directories are retained. `scripts/export_repair_pairwise_theory.py`
derives per-seed interval tables and the theory draft's numerical section from the
published `v1` distributions. This stage requires no model training or feature extraction.
