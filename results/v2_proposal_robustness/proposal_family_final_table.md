# Proposal-family final table (section 19), extended by V2-P2

P1 (RPN, DETR) columns are unchanged from commit `5bbbb73`-era text; the GDINO
column is added by **V2-P2-C1** and covers **C1 only** (C4 was never authorized
for GDINO - see `p2_c1_gdino_config_freeze.json`).

| Metric | RPN (frozen artifacts) | DETR (P1-F4/F5) | GDINO (P2-C1) |
|---|---:|---:|---:|
| target recall@0.5 (N=64) | 0.9859 | 0.9973 | 0.9880 [a] |
| target recall@0.7 (N=64) | 0.8998 | 0.9845 | not probed [a] |
| K50 availability (audit) | 0.9912 | 0.9224 | 0.9920 [a] |
| K50 availability (pooled-test rows) | 0.9867 | 0.9117 | 0.9863 |
| same-cat >=4 availability (audit) | 0.9003 | 0.7817 | **0.6892 (gate FAIL)** [a] |
| K5 grounding acc | 0.7907 | 0.8806 | 0.8463 [b] |
| K50 grounding acc | 0.4320 | 0.5849 | 0.4448 [b] |
| K5 MSP AUROC | 0.8426 | 0.7994 | 0.8308 [b] |
| K50 MSP AUROC | 0.7904 | 0.7142 | 0.5942 [b] |
| delta AUROC K5-K50 | 0.0522 | 0.0852 | **0.2366** [b] |
| E-AURC worsening (relative) | 2.2413 | 4.5547 | **8.6541** [b] |
| RER50 drop | 0.4560 | 0.2943 | **0.6678** [b] |
| delta Rand (E1b-R1 AUROC) | 0.0012 | 0.0066 | out of scope |
| delta Hard (E1b-R1 AUROC) | 0.0318 | 0.0213 | out of scope |
| Amplification (Hard-Rand) | 0.0306 | 0.0147 | out of scope |
| C1 verdict | YES | YES | **YES** |
| C4 verdict | YES | YES | not run (out of scope) |

[a] Probe-level (`p2_gdino_probe`, 100 images / 251 expressions, ref-target unit),
not a full audit; the same-category supply gate failed there (0.6892 vs the frozen
0.85 threshold), which is exactly why GDINO was admitted to C1 only.
[b] Common-K50 cohort, mean over the 3 frozen B3 seeds, from `p2_c1_gdino/c1_point.csv`
(cohort 10,402 rows vs RPN 10,286 / DETR 9,665). On that same cohort the recomputed
RPN delta AUROC is 0.0517 and DETR's is 0.0852, matching the frozen F4 rows to 1e-12;
the RPN cells above keep the original frozen-artifact values, so RPN's two
delta-AUROC cells differ only by cohort definition (0.0522 frozen vs 0.0517 recomputed).

C1 verdict (DETR): **YES**
C4 verdict (DETR): **YES**
C1 verdict (GDINO, V2-P2-C1): **YES - replicated and amplified (2.8x DETR, 4.6x RPN)**
Overall P1 verdict: **CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST** (YES)
P2 extension verdict: **C1_REPLICATED_ON_THIRD_PROPOSAL_FAMILY** (C1 only; C4 not tested)
P2 mechanism verdict (V2-P2-M): **MECHANISM_PARTIAL**

candidate-cardinality degradation and hard-semantic amplification both replicate
under DETR proposals. V2-P2 adds that candidate-cardinality degradation also
replicates under a class-prompted detector, where it is ~3x larger, and that the
amplification cannot be a candidate-*count* effect: all three banks hold exactly 64
proposals per image and all three are evaluated at the same presented K.

V2-P2-M then tested *why* it is larger, and the honest answer is "not for the reason
first guessed". Within family, pool composition is a real but weak harm channel
(rho(H1c, same-class distractors) = +0.089 DETR / +0.053 GDINO, both CIs excluding 0;
RPN shows none). Across families it does not carry the amplification: matching
expressions into same-class-redundancy quintiles leaves the GDINO gap completely
unshrunk (shrinkage -0.18 vs the frozen 0.50 bar), because GDINO has the *fewest*
same-class distractors of the three banks while suffering the *most* harm. See
`p2_m_mechanism/mechanism_report.md` and the addendum in
`p2_c1_gdino/analysis_report.md`.
