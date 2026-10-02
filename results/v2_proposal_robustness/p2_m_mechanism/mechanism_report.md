# V2-P2-M mechanism diagnostic - report

Protocol **V2-P2-M**, classification **DESCRIPTIVE_MECHANISM**. Config freeze
`results/v2_proposal_robustness/p2_m_mechanism_config_freeze.json` was committed as
`87009ac` **before any P2-M number existed**, including the four decision labels, the
0.50 shrinkage bar, the 2-of-3 families requirement and the three stop conditions.

Driver: `python -u scripts/p2_m_mechanism.py` (wall 330.28 s, CPU only).
New training parameters **0**, model forward passes **0**, new seeds **0**, new
features **0**, new data **0**. Bootstrap: frozen image-cluster design, 5,000
replicates, seed 0, 95 % CI. Cohorts (frozen common-K50, pooled testA+testB):
RPN 10,286 / DETR 9,665 / GDINO 10,402 expressions - all three clear the 8,000 stop
condition, and all three reproduce the V2-P2-C1 attrition table exactly.

## 1. Verdict

**MECHANISM_PARTIAL** - rule (a) holds, rule (b) fails.

| frozen rule | requirement | measured | outcome |
|---|---|---|---|
| (a) within-family dose response | rho(H1c, R1) > 0 with CI excluding 0 in >= 2 of 3 families | DETR +0.0894 [.0611, .1191], GDINO +0.0525 [.0242, .0808], RPN -0.0144 [-.0383, +.0095] | **PASS (2/3)** |
| (b) matched-strata shrinkage | >= 0.50 for GDINO-minus-RPN **and** GDINO-minus-DETR | -0.180 and -0.065 | **FAIL** |

So: within the two deep families, per-expression pool redundancy really does co-move
with per-expression pool-growth harm; but the *between-family* amplification measured
by V2-P2-C1 is **not** carried by the composition channel this protocol measured.
Matching expressions into R1 quintiles leaves the GDINO gap fully intact - it widens
it slightly.

## 2. The two orderings that rule (b) cannot reconcile

Family means on the frozen cohorts (from `mechanism_point.csv`; harm is the mean over
the three B3 seeds):

| family | R1 same-class distractors | R2 unmatched fraction | R4 pair IoU > .7 | R4 pair IoU > .9 | H1c net harm | rho(H1c, R1) |
|---|---|---|---|---|---|---|
| RPN | **11.51** | 0.564 | 0.0014 | 0.000034 | 0.3605 | -0.014 |
| DETR | 8.71 | 0.611 | 0.0297 | 0.0074 | 0.2957 | **+0.089** |
| GDINO | **6.23** (lowest) | **0.718** (highest) | 0.0094 | 0.0018 | **0.4016** (highest) | +0.053 |

GDINO suffers the **largest** pool-growth harm while carrying the **fewest** same-class
distractors and less geometric redundancy than DETR. Its distinctive measured property
is R2 - the share of pool members that match no non-crowd COCO object at IoU >= 0.5 -
i.e. proposals on *nothing* rather than proposals on a competitor of the same class.
And within GDINO, R2 does not behave like a harm channel either:
rho(H1c, R2) = +0.0036 [-0.024, +0.032], indistinguishable from zero.

## 3. Matched strata (S1), the whole table

R1 quintile edges computed once on the pooled three-family expression sample
(3 / 5 / 9 / 14); no bin fell below the frozen 200-row minimum, so the shrinkage
statistic is reported (`bins_dropped = 0`).

| bin | R1 range | n RPN | H1c RPN | n DETR | H1c DETR | n GDINO | H1c GDINO |
|---|---|---|---|---|---|---|---|
| 0 | [0, 3) | 673 | 0.3685 | 1,980 | 0.2406 | 2,925 | 0.3671 |
| 1 | [3, 5) | 1,014 | 0.3527 | 1,430 | 0.2821 | 2,181 | 0.4088 |
| 2 | [5, 9) | 2,667 | 0.3838 | 2,269 | 0.2781 | 2,415 | 0.3990 |
| 3 | [9, 14) | 2,400 | 0.3490 | 2,039 | 0.3082 | 1,759 | 0.4222 |
| 4 | [14, inf) | 3,532 | 0.3515 | 1,947 | 0.3691 | 1,122 | 0.4504 |

Reading: DETR and GDINO show the frozen within-family dose response (0.241 -> 0.369 and
0.367 -> 0.450 across the R1 quintiles); RPN is flat and slightly decreasing, consistent
with its negative point estimate. GDINO exceeds DETR in 5/5 bins and exceeds RPN in 4/5
bins (bin 0 is a tie: 0.3671 vs 0.3685). The gap is therefore *within* strata, not
*between* them - which is exactly what a shrinkage near zero says.

## 4. Full A1 grid (descriptive, `mechanism_association.csv`)

Spearman rho, mean over the three B3 seeds, image-cluster 95 % CI, n = the family
cohort in every cell (no row was lost: unknown-target share = 0.0000 in all three
families, so the R1 sub-cohort equals the full cohort). **Bold** = CI excludes 0.

| redundancy | RPN H1c | DETR H1c | GDINO H1c |
|---|---|---|---|
| R1 same-class distractors | -0.014 [-.038, +.010] | **+0.089** [+.061, +.119] | **+0.053** [+.024, +.081] |
| R2 unmatched fraction | **+0.039** [+.015, +.063] | **-0.043** [-.072, -.015] | +0.004 [-.024, +.032] |
| R4 pair IoU > .7 | **-0.024** [-.048, -.0001] | **+0.057** [+.029, +.085] | **+0.082** [+.054, +.109] |
| R4 pair IoU > .9 | **-0.027** [-.050, -.005] | **+0.055** [+.027, +.084] | **+0.063** [+.036, +.090] |
| R5 query_cos_spread (Class I) | +0.005 [-.016, +.027] | +0.009 [-.016, +.034] | +0.011 [-.013, +.033] |

Three honest observations. (i) Every magnitude is small: |rho| <= 0.09, i.e. the
composition variables explain a few percent of the rank variance of a single binary
indicator, so this is a weak association even where it is positive; the two negative
RPN cells that reach significance (-0.024, -0.027) are of the same order as the
positive DETR/GDINO cells and point the other way. (ii) The only Class-I CLIP-side
measure, `query_cos_spread`, is unassociated with harm in all three families. (iii)
The `H2a_delta_margin` rows (not part of any decision rule) are much larger - GDINO:
-0.251 with R4-mid, -0.155 with R1 - i.e. pool redundancy tracks the *collapse of the
confidence margin* far more sharply than it tracks the loss flip, and the sign is
family-dependent (DETR +0.175, GDINO -0.251 for the same cell), which is a reason to
report it and not to decide on it.

## 5. Structural identity found while checking the arithmetic

`H1b_gained_flip` is exactly 0 in every family and every seed. That is not an
empirical surprise but a property of the nested-K design: the K5 pool is a subset of
the K50 pool (`distractor_order[:4]` inside `distractor_order[:49]`) and the frozen
scorer is an `IndependentMLPScorer` - one logit per (query, candidate) pair, not a
list-level interaction - so growing the pool can only add competitors for candidate 0
and can never promote it into the top-1.
Consequently `H1c == H1a` identically, `H1c` equals the family accuracy drop
(GDINO 0.8463 - 0.4448 = 0.4016, RPN 0.7895 - 0.4290 = 0.3605, DETR
0.8806 - 0.5849 = 0.2957 - all match `c1_point.csv`), and the `H1a` and `H1c` rows of
the association table are bit-identical. The freeze already forbade using H1b to
cancel H1a; this records why there was nothing to cancel.

## 6. What this does and does not change

* **Unchanged**: every V2-P1 and V2-P2-C1 confirmatory number. This protocol is
  descriptive and, per its own classification, may not create, strengthen or weaken
  an existence claim. C1 replicates on three proposal families; that stands.
* **Withdrawn**: the mechanistic reading in `p2_c1_gdino/analysis_report.md` section 5
  ("the class-prompt bank fills its 64 slots with semantically redundant, same-class
  boxes ... strongest evidence so far for the semantic-competition mechanism"). The
  direct measurement contradicts its premise: GDINO's pool contains the *fewest*
  same-class distractors of the three banks, and stratifying on that variable does not
  shrink GDINO's excess harm at all. An addendum is appended there and the final table
  is corrected.
* **Still supported**: the arithmetic that excludes candidate *count* (all three banks
  hold exactly 64 proposals per image; all three are evaluated at the same presented K).
* **Open, not measured here**: GDINO's distinguishing measured property is the highest
  share of pool members matching no COCO object (R2 0.718) combined with near-zero
  geometric redundancy - a "many plausible boxes, few of them on a GT object" profile.
  This protocol cannot call that a mechanism: R2 is unassociated with harm within
  GDINO, and no frozen primitive available here measures semantic confusability of
  same-object duplicates directly. Any follow-up would need its own freeze.
* **Not claimed**: causality of any kind; that one detector is better than another;
  any CI on the shrinkage ratio (the freeze forbids promoting it to a test); any
  C4 / hard-regime statement about GDINO (`out_of_scope_confirmed` is copied from the
  freeze verbatim).

## 7. Reproduction

```
python -u scripts/p2_m_mechanism.py                 # 330 s, writes the four artifacts
python -u scripts/p2_m_mechanism.py --limit 400     # smoke: no CI, writes nothing
python -m pytest tests/test_p2_m_plumbing.py -q     # 14 tests: 4 protocol-identity, 4 estimator,
                                                   # 1 stratification, 5 result-layer
```

Declared deviations from the frozen helpers, both recorded in the freeze's own
favour: the replicate loop is local because `phase0a.paired_cluster_bootstrap` is
hard-wired to `metric(a) - metric(b)` over two prediction sets (the sampling design,
seed, replicate count and CI level are the frozen ones, and the replicate re-runs the
same mean-over-seeds estimator as the point value); R1 is estimated on its own row
mask, which coincides with the full cohort because the unknown-target share is 0.
Ranking uses `scipy.stats.rankdata` (average ties), verified against
`scipy.stats.spearmanr` to 1e-12 in `test_p2_m_plumbing.py`.
