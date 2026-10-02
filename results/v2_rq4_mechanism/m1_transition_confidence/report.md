# RQ4-M1 — Transition–Confidence Decomposition

Classification: `DESCRIPTIVE_MECHANISM_DECOMPOSITION` — descriptive mechanism decomposition of an already-settled confirmatory effect (V2-P C1). No new existence claim, no intervention test, no new method.

Protocol: `RQ4-M1` (frozen before any RQ4-M1 result: `protocol_freeze.json`).

## 1. Identity checks (all frozen tolerances)

- max |Shapley residual| over families: `0.000e+00`
- max |group-weight reconstruction residual|: `1.110e-16`
- tolerance: `1.0e-12`, within tolerance: `True`
- A00 / A11 reproduce the published C1 K5 / K50 AUROC per (family, seed) (checked at computation time; a drift would have stopped the run)
- structural invariant G == 0 held for every family and seed (checked in `transition_groups.csv`)
- primary verdict: `DECOMPOSITION_REPORTED` — a constant, because an exact accounting decomposition cannot succeed or fail

## 2. Per-family decomposition of the K5→K50 AUROC degradation

`D_total = D_label + D_conf`, both components **signed** (a negative component means that factor improved reliability while the other worsened it more). Shares are signed descriptive shares, never clipped.

| family | cohort n | D_total | D_label | D_conf | share_label | share_conf | D_label CI | D_conf CI | D_total CI | ordering |
|---|---|---|---|---|---|---|---|---|---|
| RPN | 10286 | +0.051698 | +0.002605 | +0.049093 | 0.0504 | 0.9496 | [-0.008428, 0.013571] | [0.040877, 0.057612] | [0.039978, 0.063555] | `CONFIDENCE_CHANGE_HEAVIER` |
| DETR | 9665 | +0.085157 | +0.006166 | +0.078991 | 0.0724 | 0.9276 | [-0.008998, 0.021173] | [0.056062, 0.101692] | [0.061427, 0.108942] | `CONFIDENCE_CHANGE_HEAVIER` |
| GDINO | 10402 | +0.236599 | -0.009145 | +0.245744 | -0.0387 | 1.0387 | [-0.022816, 0.004701] | [0.227082, 0.265223] | [0.217215, 0.256076] | `CONFIDENCE_CHANGE_HEAVIER` |

Cohorts: RPN = 10286 rows (per-seed n = 10286), DETR = 9665 rows (per-seed n = 9665), GDINO = 10402 rows (per-seed n = 10402).

Per-seed values are in `point_decomposition.csv` (rows `seed1/seed2/seed3`), bootstrap details in `bootstrap_decomposition.csv`.

## 3. Cross-family exact gap decomposition

`D_total(a) - D_total(b) = (D_label(a) - D_label(b)) + (D_conf(a) - D_conf(b))`, exact by subtraction of two exact identities. The gap CIs are **descriptive under independent-family resampling** on each family's own cohort and are **not** paired-expression CIs.

| comparison | gap D_total | gap CI | gap D_label | label-gap CI | gap D_conf | conf-gap CI | heavier component | share from label | share from confidence |
|---|---|---|---|---|---|---|---|---|
| GDINO_minus_RPN | +0.184902 | [0.161441, 0.207721] | -0.011750 | [-0.029831, 0.005643] | +0.196651 | [0.176015, 0.218443] | `CONFIDENCE_CHANGE_HEAVIER` | -0.0635 | 1.0635 |
| GDINO_minus_DETR | +0.151442 | [0.121390, 0.183711] | -0.015311 | [-0.035627, 0.004672] | +0.166753 | [0.137418, 0.197490] | `CONFIDENCE_CHANGE_HEAVIER` | -0.1011 | 1.1011 |
| DETR_minus_RPN | +0.033459 | [0.006457, 0.059571] | +0.003561 | [-0.014829, 0.022773] | +0.029898 | [0.005804, 0.054084] | `CONFIDENCE_CHANGE_HEAVIER` | 0.1064 | 0.8936 |

## 4. Transition-group composition (S / F / E, G == 0 everywhere)

Counts and confidence summaries per family × seed are in `transition_groups.csv`; the key ranking quantity `AUC(S,F;p50)` (newly introduced errors keep high confidence at K50?) is reported next to `AUC(S,E;p50)` (persistent errors) in `pairwise_auc_components.csv`.

| family | seed | S | F | E | G | AUC(S,F;p50) | AUC(S,E;p50) |
|---|---|---|---|---|---|---|---|
| RPN | seed1 | 4373 | 3727 | 2186 | 0 | 0.763143 | 0.837490 |
| RPN | seed2 | 4416 | 3700 | 2170 | 0 | 0.762127 | 0.825152 |
| RPN | seed3 | 4449 | 3698 | 2139 | 0 | 0.765769 | 0.834655 |
| DETR | seed1 | 5647 | 2873 | 1145 | 0 | 0.736833 | 0.658004 |
| DETR | seed2 | 5655 | 2858 | 1152 | 0 | 0.731654 | 0.657410 |
| DETR | seed3 | 5658 | 2842 | 1165 | 0 | 0.740247 | 0.663396 |
| GDINO | seed1 | 4635 | 4166 | 1601 | 0 | 0.592638 | 0.580586 |
| GDINO | seed2 | 4603 | 4208 | 1591 | 0 | 0.593124 | 0.584085 |
| GDINO | seed3 | 4642 | 4157 | 1603 | 0 | 0.608761 | 0.586398 |

## 5. Accepted-error sources behind the published RER@50 / RER@80 degradation

No new gate: this only describes the composition of the accepted set under the frozen selective convention (`n_keep = max(1, ceil(coverage * n))`, descending `conf_msp`@K50, stable sort). Full table in `selective_error_sources.csv`.

| family | seed | coverage | accepted F (new) | accepted E (persistent) | fraction of accepted errors from F | risk |
|---|---|---|---|---|---|---|
| RPN | seed1 | 0.5 | 1422 | 484 | 0.7461 | 0.3706 |
| RPN | seed1 | 0.8 | 2823 | 1333 | 0.6793 | 0.5050 |
| RPN | seed2 | 0.5 | 1366 | 521 | 0.7239 | 0.3669 |
| RPN | seed2 | 0.8 | 2803 | 1345 | 0.6757 | 0.5041 |
| RPN | seed3 | 0.5 | 1360 | 496 | 0.7328 | 0.3609 |
| RPN | seed3 | 0.8 | 2810 | 1283 | 0.6865 | 0.4974 |
| DETR | seed1 | 0.5 | 777 | 456 | 0.6302 | 0.2551 |
| DETR | seed1 | 0.8 | 1896 | 816 | 0.6991 | 0.3508 |
| DETR | seed2 | 0.5 | 796 | 451 | 0.6383 | 0.2580 |
| DETR | seed2 | 0.8 | 1894 | 829 | 0.6956 | 0.3522 |
| DETR | seed3 | 0.5 | 762 | 452 | 0.6277 | 0.2512 |
| DETR | seed3 | 0.8 | 1884 | 807 | 0.7001 | 0.3480 |
| GDINO | seed1 | 0.5 | 1775 | 716 | 0.7126 | 0.4789 |
| GDINO | seed1 | 0.8 | 3132 | 1218 | 0.7200 | 0.5227 |
| GDINO | seed2 | 0.5 | 1810 | 712 | 0.7177 | 0.4849 |
| GDINO | seed2 | 0.8 | 3177 | 1201 | 0.7257 | 0.5261 |
| GDINO | seed3 | 0.5 | 1712 | 737 | 0.6991 | 0.4709 |
| GDINO | seed3 | 0.8 | 3085 | 1210 | 0.7183 | 0.5161 |

## 6. Matched-expression secondary (paired, SECONDARY_MATCHED_DIAGNOSTIC)

On each intersection both families are restricted to identical expression rows and resampled with **shared** image-cluster draws, so the gap there is genuinely paired-expression. It never overrides the primary own-cohort result; it reports whether the same component ordering survives cohort matching.

| pair | n expressions | family | D_total | D_label | D_conf | ordering |
|---|---|---|---|---|---|---|
| GDINO_intersection_RPN | 10125 | GDINO | +0.231230 | -0.007622 | +0.238852 | `CONFIDENCE_CHANGE_HEAVIER` |
| GDINO_intersection_RPN | 10125 | RPN | +0.053406 | +0.003975 | +0.049431 | `CONFIDENCE_CHANGE_HEAVIER` |
| GDINO_intersection_DETR | 9589 | GDINO | +0.221207 | -0.005407 | +0.226614 | `CONFIDENCE_CHANGE_HEAVIER` |
| GDINO_intersection_DETR | 9589 | DETR | +0.086257 | +0.006763 | +0.079493 | `CONFIDENCE_CHANGE_HEAVIER` |
| DETR_intersection_RPN | 9418 | DETR | +0.081947 | +0.005103 | +0.076844 | `CONFIDENCE_CHANGE_HEAVIER` |
| DETR_intersection_RPN | 9418 | RPN | +0.059251 | +0.009819 | +0.049432 | `CONFIDENCE_CHANGE_HEAVIER` |

Paired gap on the same intersection (shared image-cluster draws, so this CI is the genuine paired-expression contrast):

| comparison | gap D_total | CI | gap D_label | CI | gap D_conf | CI |
|---|---|---|---|---|---|---|
| GDINO_intersection_RPN | +0.177824 | [0.155407, 0.200415] | -0.011598 | [-0.028824, 0.005678] | +0.189421 | [0.169174, 0.210470] |
| GDINO_intersection_DETR | +0.134950 | [0.107595, 0.162878] | -0.012170 | [-0.031241, 0.006974] | +0.147120 | [0.120779, 0.173093] |
| DETR_intersection_RPN | +0.022697 | [-0.004215, 0.048774] | -0.004716 | [-0.024363, 0.014189] | +0.027413 | [0.002957, 0.051526] |

## 7. Figures (exactly three, per the frozen figure policy)

- `results\v2_rq4_mechanism\m1_transition_confidence\figures\fig1_signed_decomposition.png`
- `results\v2_rq4_mechanism\m1_transition_confidence\figures\fig2_transition_group_confidence.png`
- `results\v2_rq4_mechanism\m1_transition_confidence\figures\fig3_cross_family_gap_components.png`

## 8. Interpretation boundary (frozen wording)

Allowed:

- The observed AUROC degradation can be exactly decomposed into a correctness-transition component and a confidence-change component.
- The larger GDINO degradation is statistically accounted for more by X than by Y in this decomposition.

Forbidden:

- X causes the degradation.
- We discovered the true mechanism.
- unmatched boxes / H2a / semantic ambiguity is the cause
- any causal or intervention wording without a separate pre-result frozen intervention protocol

Phrase rule: use 'accounts for in the exact statistical decomposition', never 'causes' or 'is the causal mechanism'

The two research questions are answered only as far as this decomposition can answer them: Q1 by the reported components per family, Q2 by the exact cross-family gap split. `D_conf` is a confidence-**ranking**-change contribution, not a temperature or calibration-scale change (AUROC is invariant to monotone score transforms).

Out of scope and not reported: no R1/R2/R4/R5/H2a quantity enters the primary decomposition; no added-candidate tail-pressure metric is reported (freeze NOT USED: per-candidate raw scores are absent from the frozen prediction artifacts and recovering them would need a forbidden new model forward); no new threshold, gate, model, dataset, seed or parameter; no V2 or V2-P confirmatory number, gate or threshold was touched.

## 9. Provenance

- inputs: the 18 hashed prediction artifacts plus the published C1 reference listed in `input_artifact_manifest.csv`; every real analysis consumes exactly those hashed files
- no model forward pass, no training, no feature extraction, no GPU, no temperature refit, no new cohort / seed / family / threshold
- full machine-readable detail in `verdict.json` and `metadata.json`

