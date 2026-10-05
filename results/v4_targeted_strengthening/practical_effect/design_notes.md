# V4 practical effect statistics: implementation contract

## Scope and provenance

This package is a deterministic metric layer over the already-saved V3 FineCops
predictions.  The input is the controlled common cohort of 7,004 expressions
from 2,888 images at K50, with B0 and B16 and three fixed seeds.  It does not
train, regenerate predictions, or decide how a test subgroup is formed.  The
high/low mask and correctness labels are supplied by the runner after the
protocol owner freezes their definitions.

FineCops has already been observed.  The V4 work is a results-driven,
precomputed targeted strengthening analysis, not an independent confirmation.
The utilities here cannot establish otherwise.

## Metric definitions

`practical_profile` reports correctness AUROC and the absolute fractional-risk
curve at coverage `[.2, .3, .4, .5, .6, .7, .8, .9]`.  A cutoff that splits a
score tie accepts the required fraction of that tie in expectation.  Coverage
zero has no defined risk; the profile records the first score group's error
rate as the explicitly tagged right limit.

`aurc_v4` analytically integrates that fractional risk over all coverage
`c in (0, 1]`.  For a score-tie group of size `g`, with `h` errors and
`p=h/g`, after `m` rows and `e` errors, its unnormalized contribution is

```text
p*g + (e - p*m) * log((m+g)/m)    when m > 0
p*g                               when m = 0
```

The sum is divided by the total row count.  This is named `AURC_V4`; it does
not overwrite the existing finite-grid/legacy trap E-AURC implementation.
The V4 oracle is `(1-a) + a*log(a)` for accuracy `a`, with endpoint values 1
at `a=0` and 0 at `a=1`.  `E-AURC_V4 = AURC_V4 - oracle`.  It is not described
as accuracy-independent.  When two score vectors use the same labels,
`Delta E-AURC_V4` equals `Delta AURC_V4`; the helper asserts this identity.

Weighted rows represent integer bootstrap multiplicities.  A shared-image
draw must assign the same multiplicity to all expressions from one image.
`image_draw_row_multiplicities` maps a sampled image-id list to aligned row
weights, after which profile, AURC, and AUROC helpers reproduce explicit
image-cluster resampling.

## Pair-credit reporting

For each seed and each Full versus S+Q comparison, `pairwise_credit_counts`
reports the true weighted counts of positive labels, negative labels, and
`P = n_positive * n_negative` positive-negative pairs, plus wins, ties, losses,
and credit (`wins + 0.5*ties`).  Single-class AUROC is undefined and is
represented as `None`; it is never replaced with zero.

`pairwise_transition_counts` computes the point-estimate 3-by-3 table from
SQ pair state (`loss/tie/win`) to Full pair state.  It traverses positive rows
in bounded blocks to keep memory fixed while preserving exact counts.  The
standard AUROC universe includes same-image and different-image expression
pairs.  It also reports regained pairs (`SQ loss -> Full win`), regressed pairs
(`SQ win -> Full loss`), and all four tie-changing transitions.  Net credit is
`Full credit - SQ credit`; the normalized value is `net_credit / P * 10,000`.

Formal bootstrap draws resample whole images, keeping each image's expressions
together.  They use the same draw for all endpoints and seeds, then compute
AUROC from weighted score groups in `O(n log n)`.  The all-pairs transition
table is a point-estimate audit and need not be rebuilt for every draw.  Each
seed's normalized effect is computed before averaging the three seeds; do not
pool pairs or reweight seeds by `P`.

## Intended runner contract after protocol freeze

For each B0/B16 configuration and fixed seed, use the saved labels and scores
on the same common K50 rows:

1. Call `practical_profile` for the absolute MSP and S (Stats) curves and the
   Full and S+Q absolute metrics.
2. Call `paired_aurc_effect(Full, S+Q, labels)` for primary A,
   `Delta AURC = Full - S+Q`.
3. Supply the frozen high-group mask to `high_low_auroc_effect` for primary
   A4, `Delta AUROC_high - Delta AUROC_low`.  Group construction remains a
   training-data-only protocol decision; this module does not infer
   `nn_cos_mean` tertiles or inspect test outcomes to make groups.
4. Call `pairwise_transition_counts` once for each seed/configuration as a
   descriptive pair-accounting audit.  Keep all raw counts and credits.
5. For each of 5,000 shared-image draws from 2,888 image clusters using seed
   0, convert the sampled image IDs to row multiplicities.  Compute the six
   prespecified V4 family effects per seed, then average the three seed
   effects within each draw.  Retain the exact same draws for the 95% interval
   and the familywise Bonferroni interval.

The six-member family is the two A endpoints (B0 and B16), the two A4
endpoints (B0 and B16), and the two pure-B primary AUROCs.  Its two-sided
Bonferroni interval is 99.1666667%, with percentile quantiles
`0.0041666667` and `0.9958333333`; the same draws also supply the 95% interval
at `0.025` and `0.975`.  The runner owns invalid-draw handling.  A draw with a
single correctness class has undefined AUROC and stays invalid; any primary
with invalid draws must be labeled conditional and cannot be called
SUPPORTED as a formal result.

No V4 endpoint, real sample heterogeneity, bootstrap draw, or performance
measurement was computed while this design remained unsealed.
