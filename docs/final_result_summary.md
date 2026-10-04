# Current result summary — Repair v1 and V3

V3 completed its [prospective protocol](../reviews/v3_protocol.md): exposure audit,
repaired-definition B/16 replication, natural interface, a first frozen FineCops
confirmation, 5,000 shared-image draws, and saved-output acceptance. The main
numerical table is [artifact-generated](../results/v3_final_validation/publication/key_findings.md);
all endpoints, denominators, per-seed uncertainty and raw-distribution identities
are in the [V3 evidence index](../results/v3_final_validation/evidence_index.md).

FineCops controlled C1 did not confirm MSP AUROC degradation, even though accuracy
and selective utility deteriorated. C2 and C3 separately support the K50 Full−S+Q
increment; C4 supports large-K-trained versus small-K-trained ScoreDeepSets, whose
absolute point performance remains below MSP/Stats. These results establish
conditional signals and training-support effects, not universal architecture advantages.

Natural K5 gives negative Full−S+Q increments in both configurations. K20 overall
gains mostly account for ranking proposal-miss errors; covered-only increments are
inconclusive. At K50 B0 overall and covered AUROC gains have marginal support,
while B/16 is inconclusive. B0 Risk@50 gain is also inconclusive; Risk@80 improves.
These are predeclared secondary analyses, not extra family-wise primary confirmations.
Do not infer a configuration interaction from one supported and one inconclusive interval.

[Completion review](../reviews/v3_completion.md) and [paper draft](paper_draft_v3.md)
preserve these boundaries. New GQA/VG split identity alone does not establish
independence or a non-COCO visual domain; foundation pretraining exposure is unknown.
The V3 full suite passed 1,298 tests with two original opt-in skips; all previously
tested source bytes and 834 frozen historical files passed final verification.

Independent review conditionally accepted the core evidence and required three
implementation/report corrections. All three are now implemented and verified.
The repair-closeout full suite had 1,238 passed and two original opt-in skips; 263 source/config
files remained unchanged. Versioned execution acceptance is PASS; the earlier
PASS and original independent conditional verdict remain preserved. See the
[closeout](../results/research_repair_v1/independent_closeouts/20261004T031921Z/closeout.md).
M3 stays UNVERIFIABLE, with no new effect CI. The original M3.1 frozen-store and
dose-rescore anchors are 1e-9 and 1e-4 respectively. The committed-point 1e-9
replay criterion belongs to the repair wrapper, not the original protocol.
See `results/research_repair_v1/STATUS.json` and
`results/research_repair_v1/independent_closeouts/20261004T031921Z/`.

The pre-repair material bank is preserved in
`docs/appendix/final_result_summary_pre_repair.md`. Its historical gate labels and
intervals are records of the prior analysis, not current scientific conclusions.
The historical `results/final_registry/` has not been regenerated for V2 or repair;
original mechanism sources remain in `results/v2_rq4_mechanism/m1_transition_confidence/`.

## Scientific scope and questions

The historical core is GT-assisted target-present candidate grounding within the
shared COCO visual domain. V3 adds audited FineCops positive expressions and natural
top-K evaluation; it does not establish a pure visual-domain effect or a NONE task.
RQ1: expansion effects on accuracy, discrimination, calibration and selective utility.
RQ2: score extrapolation limits and query–crop versus candidate–candidate information.
RQ3: four-corner accounting, paths, interaction and explanation limits.

E-AURC and RER remain accuracy-dependent selective utility measures. E-AURC
retains trapezoidal finite-sample AURC minus the continuous oracle reference;
it can be negative under ideal ranking. With confidence [0.8, 0.3] and correctness
[1, 0], AURC is 0.125 and E-AURC is -0.028426409720027357. This is a difference
between the discrete and continuous definitions; values are not clipped.
Adaptive ECE percentile intervals do not guarantee nominal 95% coverage;
adaptive bins and absolute errors are non-smooth, especially near zero. A positive
lower endpoint alone does not establish population miscalibration. Temperature
preserves within-expression argmax but can change cross-expression MSP ranking.
Failure of tested score models does not establish insufficient score information.
Small symmetric Shapley label components may conceal opposing path effects.
Exact statistical accounting decomposition, not a causal identification of confidence
ranking changes, is the mechanism interpretation boundary. H1c measures accuracy
harm; it cannot stand in for AUROC-gap explanation. Threshold-unmatched proposals
are unmatched to the annotations used, not necessarily empty background.

## Training and provenance

V2-G freezes encoders but trains corresponding scorer/reliability heads. V2-P and
historical mechanism accounting reuse a frozen scorer. V2-M/M3 train or fit models.
RefCOCO language transfer shares images; strict RefCOCOg is image-disjoint within
COCO and its invalid hard manipulation cannot confirm amplification.
Full pre-result configuration freezes, input identity freezes (D1/D2), retrospective
corrections and D2 target redefinition have different evidential status. A staged
protocol does not erase adaptivity after observing prior test results. Repair analyses
are result-driven supplements, not independently preregistered confirmation.

<!-- artifact-derived-repair-evidence -->

## Artifact-derived evidence

Machine status: **COMPLETE_WITH_RETAINED_LIMITS**; completed=true.
The current formal export contains 6646 stored estimates from 59 jobs.
Only stored 5000-draw, seed-0, 95% image-cluster jobs enter that export.

- [Formal statistical tables](../results/research_repair_v1/publication/statistics.md)
- [Main metrics and proposition tables](../results/research_repair_v1/publication/main_tables.md)
- [Complete estimate export](../results/research_repair_v1/publication/formal_statistics.csv)
- [Required scope coverage](../results/research_repair_v1/publication/coverage.md)
- [Historical and repaired operational gates](../results/research_repair_v1/publication/gates.md)
- [Full minus S+Q paired evidence](../results/research_repair_v1/publication/information.md)
- [Candidate category, object and availability audit](../results/research_repair_v1/publication/candidates.md)
- [Four corners, paths and interaction](../results/research_repair_v1/publication/mechanism.md)
- [Source identity and rendering record](../results/research_repair_v1/publication/evidence.json)
- [All experimental artifacts and acceptance state](../results/research_repair_v1/registry/index.md)

Incremental V signal is not uniformly resolved across reported cells. The central claim is limited to the measured cellwise effects; no general candidate-interaction conclusion follows.

On pooled testA+testB, stored paired estimates show condition-specific incremental correctness-AUROC gains for V: random K=50 ΔAUROC=0.013638 (95% paired CI [0.011202, 0.016070]); matched hard K=5 ΔAUROC=0.027089 (95% paired CI [0.024338, 0.029744]); dose m=8 (K=10) ΔAUROC=0.025464 (95% paired CI [0.022670, 0.028348]). Intervals include zero for random K=5 ΔAUROC=0.000679 (95% paired CI [-0.001656, 0.002922]); dose m=0 (K=10) ΔAUROC=0.001379 (95% paired CI [-0.001006, 0.003688]). These fixed-model, reused-test-set contrasts support conditional cell-specific gains where resolved; they do not establish a universal V gain or identify a causal candidate-interaction mechanism.

Verified candidate audit: 138 mismatching rows among 21,026 family–expression audit records (RPN 86/10,425; DETR 52/10,601; 10,607 distinct sentence_id values; mismatches involve 47 family–target-object pairs). Sensitivity stage: **FORMAL_COMPLETE**; independent raw numerical QA: **PASS**.

Observed label path means have opposite signs in RPN, DETR, GDINO. The signed Shapley mean can conceal this cancellation; the accounting does not identify a cause.

## Post-repair theoretical analysis

The [theory chapter draft](theory_analysis_v1.md) now gives explicit assumptions,
proofs, counterexamples and label-path bounds. It retains the historical notation
S=stable correct, F=flipped to error, E=persistent error.
The [frozen-prediction pair analysis](../results/research_repair_v1/theory_analysis/v1/analysis_report.md)
adds shared-draw intervals for pair ranking changes and signed path terms.
RPN/DETR improve S-versus-F ranking while worsening S-versus-E and F-versus-E
ranking; GDINO worsens all three. Positive I can reflect different rates of ranking
decline, rather than semantic synergy. These are descriptive, fixed-model,
reused-test analyses; they do not identify an internal causal mechanism.
This supplementary export is separate from the existing 59-job publication registry;
it does not retroactively alter its estimate count, gates or completion status.
Latest theory-stage test and numerical verification is recorded in
[verification.json](../results/research_repair_v1/theory_analysis/v1/verification.json).
