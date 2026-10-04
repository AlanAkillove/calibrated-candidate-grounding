# Paper blueprint — Research Repair v1

Suggested title: *Confidence Reliability in Visual Grounding under Candidate-Set Expansion*.
The former plan is preserved in `docs/appendix/paper_blueprint_v1_pre_repair.md`.
All current numerical claims and intervals must come from the repair registry.
Pending or UNVERIFIABLE endpoints cannot become confirmed evidence.

## 1. Introduction

Start with a changing candidate interface and distinguish accuracy, correctness
AUROC, calibration, and selective utility. State the GT-assisted target-present
conditions immediately. Ask three questions: reliability under expansion; score
extrapolation and extra information sources; four-corner accounting and its limits.
Target absence is future work. Contributions are controlled observations and evidence
integrity, not a new architecture, universal repair, information necessity, or causal
mechanism. Repairs are result-driven supplements on previously observed test sets.

## 2. Related work and definitions

Use verified primary references for candidate grounding, CLIP REC, distribution-shift
selective prediction, metric limitations, and set representations. Unverified entries
in `literature_notes.md` cannot support factual claims. Explain structural accuracy
monotonicity under independent scoring and fixed tie-breaking, without implying
AUROC monotonicity. E-AURC and RER retain accuracy dependence. E-AURC uses
trapezoidal finite-sample AURC minus a continuous oracle reference, so it can be
negative even under ideal ranking; do not clip it or call the reference a finite
sample lower bound. Adaptive ECE percentile intervals do not establish nominal
coverage or population miscalibration merely from a positive lower endpoint. Monotonic scalar
confidence transforms preserve AUROC; logits temperature need not preserve MSP order.

## 3. Controlled evaluation design

Describe target inclusion, uniqueness filtering, maximum-K feasibility, common
cohorts, expansion identity, same-category construction, image splits, and fixed
grounding. Count proposals separately from distinct annotated objects. Accompany
composition claims with category audit and versioned paired sensitivity analysis.
Each bootstrap replicate shares image draws across fixed seeds, computes per-seed
effects, and then averages effects. Relative effects, DoD, and severity macros are
computed within each replicate. Report invalid draws, per-seed CIs and seed SD;
three seeds do not fully characterize training randomness.
Distinguish full protocol freezes, input freezes, retrospective corrections, and
target redefinitions. Full gates, chronology, and hashes belong in the appendix.

## 4. Reliability under candidate expansion

Show absolute accuracy, AUROC, calibration and selective utility, corrected CIs,
and cohort sizes. Discrimination claims require AUROC evidence. Direct temperature
diagnostics establish the effects of fitted temperatures; small temperature drift
cannot mathematically exclude scale effects. Integrate robustness beside the tested
proposition: two new frozen encoders have trained heads; B0 is a different historical
reference. Proposal axes measure shared-scorer migration rather than an intrinsic
ranking of generators. RefCOCO shares images; strict RefCOCOg is image-disjoint
within COCO. Separate axes do not form a factorial study.

## 5. When extra information helps

Compare original Stats Logistic and ScoreDeepSets trained at small versus large K,
with identical image splits and row budgets. Include train/tune curves, 128-row
fitting diagnostics, and deterministic score-to-MSP computation. Test data never
select models. A training or extrapolation failure does not prove information absence.
Then compare S, S+Q, S+V, and Full under fixed grounding. Central comparison:
Full minus S+Q, with absolute metrics, paired CIs, selective utility, random,
matched-hard, and fixed-K dose conditions. Without stable incremental V signal,
write complementary frozen scoring views rather than candidate interaction as the
central contribution. Show category mismatch, object counts, candidate distributions,
and intersection attrition beside strongest same-category results. Put external
transfer and invalid external amplification manipulation beside those claims.

## 6. Accounting and discussion

Display A00/A10/A01/A11, label-first and confidence-first paths, signed Shapley
components, and I=A11−A10−A01+A00 with shared CIs. Opposing paths can cancel in
small net label components; mean confidence cannot determine pairwise AUROC.
Ratios above 100% are signed accounting ratios, not exclusive causal shares.
H1c is accuracy harm and cannot explain an AUROC gap by identity. The tested
same-class redundancy channel did not explain between-family amplification in the
[historical P2-M analysis](../results/v2_proposal_robustness/p2_m_mechanism/mechanism_report.md);
that result does not exclude other semantic or redundancy mechanisms.
IoU-unmatched proposals are unmatched to the annotations used, not
necessarily empty background. Exact statistical accounting is not causal identification.

## 7. Conclusions and limitations

Answer the three questions in order with artifacts. State GT conditioning, shared
visual domain, proposal/scorer adaptation, conditional feature gains, staged test-set
reuse, and unresolved causal explanations. Completion does not require every gate
passing. Do not have separate robustness, negative-results, or chronology chapters:
place each result beside the proposition it tests.

## Main visuals, tables, and appendix

1. Controlled interface showing trained and frozen components.
2. Accuracy/AUROC/calibration/selective utility panels with corrected intervals.
3. Feature-source gains under matched composition/dose and ID/OOD contrast.
4. Four corners, both paths, signed contributions and interaction.

Rendered design schematic: [controlled interface](../results/research_repair_v1/publication/figure_controlled_interface.svg).
The [artifact-derived publication export](../results/research_repair_v1/publication/evidence.json)
records the source identities and formal status for numerical figures. A diagram
does not establish completion of the corresponding experiment.

Table 1: absolute metrics, effects, intervals and sample sizes. Table 2: propositions,
evidence conditions and limitations. Appendix: full model grids, operational gates,
amendments, hashes, anchors, detailed axis evidence, and negative-route history.
