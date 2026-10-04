# Research Repair v1 — completion audit (independent-review closeout)

Status: **COMPLETE WITH RETAINED LIMITS**, 2026-10-04 (Asia/Shanghai).
The three independent-review corrections are implemented and regression-verified.
The new versioned execution acceptance is PASS with zero failed checks; this is
an execution closeout, not a replacement independent scientific verdict.
The original CONDITIONAL_ACCEPTANCE review, earlier PASS and tests remain preserved.
Evidence: `results/research_repair_v1/independent_closeouts/20261004T031921Z/`.
Machine state: `results/research_repair_v1/STATUS.json`.

| Review issue | Correction | Formal artifact | Conclusion change | Acceptance / limits |
|---|---|---|---|---|
| E-AURC/RER accuracy dependence | Population and finite-sample counterexamples, continuous oracle reference, corrected interpretation | statistics math tests; publication/main_tables.md and formal_statistics.csv | selective utility distinguished from discrimination | full suite and numerical QA PASS |
| Logits temperature does not preserve MSP order | Argmax/order counterexample; native/global/per-K temperature comparison | phase0a formal jobs; publication/statistics.md and figure_temperature_gap.svg | scale-exclusion claim withdrawn; historical B0 temperature alias identified as the cosine scorer | diagnostics, tests and source checks PASS; validation-selected per-K remains diagnostic |
| Mean CI endpoints | Shared image bootstrap of mean per-seed effects | statistics/summary.json: 59 jobs / 6646 estimates; count_reconciliation.json | corrected effect CIs and old/new operational decisions | exact ledger and raw QA PASS; 23 terminal gates, no PENDING |
| Score information versus learning/extrapolation | Small/large-K training, fixed budgets, train-only normalization, fitting diagnostic | information/summary.json; information_bootstrap.json and .csv; models/; train_tune_curves.csv | limited model evidence cannot establish information absence; ScoreDeepSets extrapolation differs by training regime | formal B axis, full tests and source checks PASS; tested models only |
| Semantic source conflation | Fixed S/Q/V/Full feature groups | information/information_bootstrap.csv: 1860 estimates / nine calls; reference_check.csv: 108 anchors | Full−(S+Q) evidence depends on K and composition; K5/random and dose m0 intervals cross zero | numerical QA PASS; no multiplicity-adjusted global interaction claim |
| Same-category category identity | True target ann category audit and versioned sensitivity | candidates/current_audit.json; audit_run_20261003_Cslot_audit4/summary.json; forward/sensitivity_bootstrap/summary.json and estimates.csv | 138 mismatches among 21,026 family-expression audit records (10,607 unique sentence_id; 47 mismatched family-target-object pairs; RPN 86/10,425 and DETR 52/10,601); effects use paired old/new availability intersection, with head identity and seed SD | 15 groups / 24 mappings / 1404 estimates; metadata and raw QA PASS; CI crossing zero does not establish equivalence |
| Proposal count versus independent objects | Geometry/objectness/IoU/duplicate and distinct-object audit | candidates/audit_run_20261003_Cslot_audit4/ geometry and object tables | proposal dose is not distinct-object causal dose | formal audit and evidence consistency PASS; object counts use available annotations |
| Shapley net component cancellation | Four corners, two signed paths, interaction and CIs | mechanism/summary.json; bootstrap_ci.csv: 45 estimates; four_corners_paths_interaction.csv | opposing label paths occur in all three families; exact accounting does not identify a cause | formal mechanism, raw QA and full tests PASS; conditional accounting only |
| H1c and unmatched-box overinterpretation | Accuracy harm distinguished from AUROC gap; unmatched to used annotations | current summary/protocol/log | measured explanation unsupported, not broad mechanism falsification | evidence consistency PASS |
| Scope, RQ, training state, freeze levels | Three questions, per-axis training facts, staged/post-result provenance | current README/summary/blueprint; historical appendix | controlled target-present study within COCO | data-driven summary/index; test-set reuse explicitly retained |
| Wrong citations | Four core entries verified against official sources | docs/literature_notes.md | corrected fields/method descriptions | unverified background excluded |
| P2 numeric comparison errors | Means re-derived from frozen point CSV | registry/historical_corrections.json and .md | RPN highest RER50; GDINO lower than DETR accuracy | final index consistency PASS |
| Frozen identity and recovery | 794-file initial baseline, five final manifests, unchanged anchor tolerances | input_manifest.json; logs/final_input_verification.json | baseline and supplemental inputs unchanged | 834 unique files PASS; two Git document prefixes preserved |
| M3 exact historical recovery | Original deterministic fits and exact confidence replay; three independently inventoried backbones and nine seed attempts | statistics/recovery/runs/20261003T095000Z/failures/m3_b0_severity_macro.json; runs/20261003T130700Z/failures/m3_b1_severity_macro.json and m3_b2_severity_macro.json | B0/B1/B2 committed points fail the repair wrapper replay criterion of 1e-9; original M3.1 frozen-store 1e-9 and dose-rescore 1e-4 anchors are separate; no saved mixer checkpoints or per-sample confidence arrays permit exact replay. This is a recovery limit and supplies no new effect CI | evidence and acceptance PASS; M3 remains UNVERIFIABLE |
| Generic prediction column model identity | Phase1 per-seed CSV stores MSP/e2_score; global gate uses MSP/E1b | original metadata and writer; information/reference_check.csv; corrected statistics | column names cannot establish semantic information; fixed model anchors kept separate | model anchors and complete main gate coverage PASS |
| Test discipline | Historical invariants retained in appendix; current evidence renderer checks | registry/test_corrections.md; independent_closeouts/20261004T031921Z/logs/pytest_full_final.xml and final_test_sources.json | narrative tests no longer enforce obsolete primary text | 1240 tests: 1238 passed / two original opt-in skips; stable 263-source/config snapshot; original 1226-test record preserved |

Three specified gpt-6-luna/max agents were actually spawned and acknowledged startup.
Resource slots were granted after baseline verification. Development/slot allocation is
not evidence of completed experiments. This result-driven repair uses previously
observed test distributions and is not independent preregistered confirmation.

Retained provenance limitation: an early A batch did not capture its launch-time
runner fingerprint while adapters were being added. A later live source hash is
not attributed to that executed version. Stored-array checks can verify arithmetic
consistency but cannot reconstruct the missing launch snapshot. The later 31-job
batch preserves its source snapshots, and its ordinary-spec count is reconciled
separately with the 18 auxiliary estimates already present in the frozen code.

The complete collection of pre-derive raw archive hashes was not captured. The
review therefore does not assert an empirical before/after SHA match for every
repair-owned raw archive. The metadata updater does not write those arrays;
current arithmetic and archive fingerprints are verified for all 9955 estimates.
M2 additionally records its source SHA at derivation and checks it after reading.
These limits are recorded in statistics/final_chain_review.json.

Final numerical QA: logs/stored_estimate_verification.json. Independent M2 QA:
logs/m2_auxiliary_verification_final_20261004.json. Final input and full-test evidence:
logs/final_input_verification.json, logs/pytest_full_final.xml and
logs/final_test_sources.json. Original delivery decision: logs/acceptance_check.json, PASS; this historical
checker did not cover the independent review omissions. New closeout acceptance
is PASS in the versioned directory and does not rewrite the independent verdict. All paths above are within results/research_repair_v1/.


Independent review corrections preserve the existing numeric E-AURC definition:
finite-sample trapezoidal AURC minus the continuous oracle reference can be negative.
For [0.8,0.3] / [1,0], it equals -0.028426409720027357; no clipping or old result
rewrite is permitted. The risk-only metric request now avoids the RER branch.
The candidate counting correction changes presentation only; rows, image-cluster
bootstrap, confidence arrays, points, and intervals retain their identities.

M3's committed-point tolerance was added by the repair wrapper, not sourced from
an original protocol cutoff. Original store/rescore checks remain distinct and
all parent/seed failure records remain unchanged. Adaptive ECE intervals are
descriptive percentile intervals with unestablished nominal coverage near zero.
Positive Full−(S+Q) gains at high random K, matched hard K5 and dose m8 coexist
with unresolved random K5/dose m0; no universal or causal V conclusion follows.


The versioned full suite passed 1238 tests with two original opt-in skips, zero
failures/errors, and unchanged 263-source/config snapshots. Direct before/after
comparison covers 396 B3/M2 prediction settings and 6732 public metric values,
all identical; a docstring-only supplement binds the final source. Fresh input
verification covers 834 unique files/five manifests. All 9955 stored estimates
pass arithmetic verification and all 77 raw archive identities match the prior
delivery. This closeout before/after SHA comparison does not replace the missing
earlier pre-derive snapshot. All 53 preserved copies were checked, including the
original independent verdict and canonical test/acceptance records. Details:
[closeout](../results/research_repair_v1/independent_closeouts/20261004T031921Z/closeout.md).
