# V2-P2-C1 commit D - C1 replicated on the GDINO class-prompt family

Artifact: `p2_c1_verdict.json` (classification **CONFIRMATORY**, protocol
`V2-P2-C1`, config freeze `../p2_c1_gdino_config_freeze.json`, committed before
any P2 number was computed). Generated 2026-10-02T07:44:16Z by
`scripts/p2_c1_cardinality.py`.

## 0. Headline

C1 - "growing the candidate set from K=5 to K=50 damages confidence quality" -
replicates on a **third, structurally different proposal family** (Grounding DINO
class-prompt detection), and it does so **amplified**, not merely reproduced:

| family | dAUROC(K5 -> K50) | 95% CI (image-cluster paired, 5000 reps) |
|---|---|---|
| RPN | +0.0517 | seed1 [.0382,.0664] seed2 [.0392,.0661] seed3 [.0367,.0642] |
| DETR | +0.0852 | seed1 [.0548,.1047] seed2 [.0651,.1145] seed3 [.0618,.1103] |
| **GDINO** | **+0.2366** | seed1 [.2180,.2593] seed2 [.2201,.2609] seed3 [.2104,.2504] |

All nine CIs exclude 0. GDINO's **smallest** CI lower bound (0.2104) sits above
DETR's **largest** CI upper bound (0.1145), so the amplification is not an
overlap-of-noise artifact. `C1_GDINO_PROPOSAL_FAMILY_REPLICATED = YES`.

## 1. What was reused, and the anti-drift check

`run_c1_for`, `common_k50_ids`, `secondary_raw_auroc`, `_effect_summary`,
`_intersect_effects`, `c1_gate`, `matched_expression_ids`, `C1_METRIC_SPECS`,
KS=(5,10,20,50), K_BASELINE=5, PRIMARY_KB=50, B3_SEEDS=(1,2,3), bootstrap
5000/seed0/0.95, POOLED_TEST=(testA,testB) - all imported from the frozen F4
stack, zero numeric code re-implemented. `new_training_parameters = 0`.

Because RPN and DETR are re-run through the same call path, their recomputed
point rows were re-asserted against the committed `p1_f4_c1/c1_point.csv`:
**`rpn_detr_reuse_verbatim_check = PASSED (all frozen F4 point rows matched to
1e-12)`**. This is what licenses reading the GDINO column as an apples-to-apples
addition rather than a re-implementation.

## 2. Cohorts and the frozen stop condition

Common-K50 cohorts: RPN 10,286 / DETR 9,665 / **GDINO 10,402**. The frozen stop
condition ("if the GDINO common-K50 cohort is < 8,000, STOP and report as
underpowered") is satisfied with margin, no rescue parameters used.
K50 availability: RPN 0.9867 / DETR 0.9117 / GDINO **0.9863** - GDINO matches
RPN, i.e. its fixed-count bank (exactly 64 valid proposals for all 19,992
images) removes the DETR-style K50 starvation.

## 3. Point metrics (mean over the 3 frozen seeds)

| family | K | accuracy | AUROC | E-AURC | RER@50 | RER@80 |
|---|---|---|---|---|---|---|
| RPN | 5 | .7895 | .8407 | .0381 | .8248 | .3859 |
| | 10 | .6756 | .8103 | .0710 | .6515 | .2593 |
| | 20 | .5627 | .7995 | .0969 | .5058 | .1870 |
| | 50 | .4290 | .7890 | .1241 | .3588 | .1206 |
| DETR | 5 | .8806 | .7994 | .0321 | .6805 | .4749 |
| | 10 | .8103 | .7560 | .0671 | .5779 | .3524 |
| | 20 | .7258 | .7300 | .1142 | .4840 | .2676 |
| | 50 | .5849 | .7142 | .1783 | .3862 | .1560 |
| GDINO | 5 | .8463 | **.8308** | **.0313** | **.8065** | .4435 |
| | 10 | .7439 | .7762 | .0709 | .5976 | .2778 |
| | 20 | .6227 | .7040 | .1424 | .3603 | .1505 |
| | 50 | .4448 | **.5942** | **.3018** | **.1387** | .0605 |

GDINO starts at the **best** K=5 calibration of the three (lowest E-AURC .0313,
highest RER@50 .8065, AUROC .8308 within .010 of RPN while .057 more accurate
than DETR) and ends at the **worst** K=50 state (AUROC .5942 - barely above
chance-level ranking, E-AURC .3018, RER@50 .1387). Its curve is monotone and
convex in K; RPN's and DETR's are monotone and much flatter.

Relative degradation summary (K5 -> K50): dAUROC +0.0517 / +0.0852 / **+0.2366**;
E-AURC worsening x2.26 / x4.55 / **x8.65**; RER@50 drop .4660 / .2943 / **.6678**;
RER@80 drop .2653 / .3189 / **.3830**.

## 4. Gate arithmetic against the frozen thresholds

Frozen thresholds: route A requires dAUROC >= 0.03 with every seed CI excluding 0;
route B requires E-AURC worsening >= 0.2 **and** RER@50 drop >= 0.1.

GDINO: dAUROC 0.23660 (7.9x the route-A floor), ci_low_mean 0.21619,
all-seeds CI exclusion TRUE; E-AURC worsening 8.65414 (43x floor); RER@50 drop
0.66781 (6.7x floor); route A PASS, route B PASS, replicated TRUE. Same outcome
for RPN and DETR, as in P1/F4 - so the *direction* of C1 is family-independent
and the *magnitude* is not.

## 5. Secondary: the raw (no-rejection) profile

Mean over seeds, `c1_secondary_raw_auroc.csv` (detector-side signals, K-dependent
only through the pool):

| family | raw top-1 AUROC K5 -> K50 | raw top1-top2 margin AUROC K5 -> K50 |
|---|---|---|
| RPN | .4442 -> .4163 | .8333 -> .7716 |
| DETR | .6674 -> .4545 | .7984 -> .7009 |
| GDINO | .6713 -> **.5719** | .8307 -> **.5870** |

This split is the mechanistic core of the result. GDINO's *ranking* degrades
**least** of the two deep families (raw top-1 .671 -> .572 where DETR falls to
.455), yet its *corrected* AUROC degrades **most** (.831 -> .594). What collapses
is the discriminability of its confidence signal: the top1-top2 margin predictor
drops to .587, essentially chance, while RPN keeps .77.

Reading: the class-prompt bank fills its 64 slots with semantically redundant,
same-class boxes (the Phase-B probe already measured person at 20.3% of retained
labels, plus hallucinated small book/apple/cup/bowl boxes). Neighbouring
candidates are then nearly interchangeable in CLIP space, so the *confidence gap*
between the target and its distractors - the quantity the frozen temperature
transform is calibrated on - evaporates as K grows. The downstream harm is
therefore a **candidate-composition** effect, not a candidate-**count** effect:
all three banks carry exactly 64 proposals per image (`top_n=64`, verified in the
bank attrs), and every family is evaluated at the same presented K, so GDINO is
not handed more candidates than RPN or DETR anywhere in this comparison.

That is the strongest evidence in the program so far for the semantic-competition
mechanism: a detector tuned *by construction* to produce class-clustered
redundancy amplifies C1 by ~2.8x over DETR without any change to the scorer, the
temperature, the cohort rule, or the metric set.

> **Superseded in part by V2-P2-M (see section 9 below).** The *count* exclusion in
> this section stands (it is arithmetic). The mechanistic reading above - that
> GDINO's slots are filled with *same-class* redundant boxes, and that this is what
> amplifies C1 - was tested directly and did **not** survive: GDINO has the fewest
> same-class distractors of the three banks and matching on that variable does not
> shrink its excess harm. The confirmatory numbers in sections 1-5 are unaffected.

## 6. Like-for-like intersections (secondary, descriptive)

Expression-level intersections (identical expressions in both families), from
`intersections_secondary`:

| intersection | n | dAUROC family A | dAUROC GDINO | gap | RER@50 drop gap |
|---|---|---|---|---|---|
| GDINO u RPN | 10,125 | RPN .0534 | .2312 | **+0.1778** | +0.196 |
| GDINO u DETR | 9,589 | DETR .0863 | .2212 | **+0.1350** | +0.364 |

The ordering survives restricting to shared expressions, so it is not driven by
cohort differences. These are descriptive: no CI is computed on the gap and no
gate is applied to it (the freeze defines them as secondary).

## 7. What this does *not* claim

* No C4 / hard-regime claim. GDINO has no `same_category` manifest, no `hard_k5`
  job and no C4 verdict anywhere on disk; the driver records
  `same_category: "out_of_scope (V2-P2-C1 freeze: C1 only)"` and
  `out_of_scope_confirmed` lists the prohibition. The probe's failed
  same-category supply gate (0.6892 vs 0.85) means a GDINO C4 would have been
  structurally uninterpretable anyway.
* Not a detector leaderboard. Accuracy levels differ across families; C1 is about
  the *shape in K*, and the K=5 columns are the shared starting point.
* Not a new statistical test: same seeds, same bootstrap, same thresholds, all
  fixed before the data existed.

## 8. Reproduction

```
E:\conda\envs\deepminer\python.exe -u scripts/p2_extract_gdino_proposals.py --resume   # commit B
E:\conda\envs\deepminer\python.exe    scripts/build_candidate_sets.py --bank cache/proposals_gdino.h5 --out cache/gdino_stage --regime random --seed 20260927
E:\conda\envs\deepminer\python.exe -u scripts/extract_features.py --bank cache/proposals_gdino.h5 --out cache/features_gdino --batch-size 128 --precision fp16 --skip-global --resume
E:\conda\envs\deepminer\python.exe -u scripts/p1_frozen_inference.py --family GDINO     # commit C
E:\conda\envs\deepminer\python.exe -u scripts/p2_c1_cardinality.py                      # commit D (~50 min, bootstrap-bound)
```

Outputs: `c1_point.csv`, `c1_bootstrap.csv`, `c1_secondary_raw_auroc.csv`,
`p2_c1_verdict.json`, `figures/fig1_k_vs_auroc_three_families.png`,
`figures/fig2_k_vs_eaurc_three_families.png`.

## 9. Addendum (V2-P2-M, added after this report was committed at `7905aad`)

Nothing above is rewritten; the only change outside this section is the marked
pointer added under the section-5 reading. This addendum records what a later,
independently frozen measurement (`V2-P2-M`, freeze commit `87009ac`, verdict
`results/v2_proposal_robustness/p2_m_mechanism/p2_m_verdict.json`) showed about the
*interpretation* in section 5.

The P2-M diagnostic measured, per expression and per family, four pool-composition
variables built only from bank boxes and COCO GT (`R1` same-class distractors,
`R2` share of pool members matching no non-crowd GT object, `R4` geometric pair
overlap at IoU > .7 / > .9) plus one Class-I CLIP descriptor
(`query_cos_spread`), and related them to the per-expression pool-growth harm.

* **Within family, composition is a real but weak channel** in the two deep families:
  `rho(H1c, R1)` = +0.089 [.061, .119] (DETR) and +0.053 [.024, .081] (GDINO), CI
  excluding 0 in 2 of 3 families; RPN shows no such association
  (-0.014 [-.038, +.010]).
* **The 2.8x amplification is not carried by that channel.** Stratifying expressions
  into pooled R1 quintiles (0 bins dropped) leaves the GDINO-minus-RPN and
  GDINO-minus-DETR harm gaps completely unshrunk - shrinkage -0.18 and -0.07 against
  the frozen 0.50 bar. GDINO has the *fewest* same-class distractors
  (mean 6.23 vs DETR 8.71 and RPN 11.51) and less geometric redundancy than DETR,
  yet the largest harm. Frozen verdict: **MECHANISM_PARTIAL**.
* **What GDINO does have, uniquely**, is the highest share of pool candidates that
  match no non-crowd COCO object (`R2` 0.718 vs 0.611 / 0.564) - boxes on nothing
  rather than boxes on a same-class competitor. Within GDINO that variable is not
  associated with harm either (+0.004 [-.024, .032]), so it is reported as an open
  description, not as a mechanism.

Consequences for how this report should be cited: keep "candidate-cardinality
degradation replicates and is amplified ~2.8x under a class-prompted detector, and it
cannot be a candidate-*count* effect because all three banks hold exactly 64 proposals
and are evaluated at the same presented K". Drop "the amplification is evidence for a
same-class semantic-competition mechanism": the direct test contradicts its premise.
The `composition, not count` phrase in section 5 should be read as *count is excluded;
the measured composition channel does not account for the between-family gap*.
