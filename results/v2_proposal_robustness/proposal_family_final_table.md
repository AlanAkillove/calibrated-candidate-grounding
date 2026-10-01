# P1 proposal-family final table (section 19)

| Metric | RPN (frozen artifacts) | DETR (P1-F4/F5) |
|---|---:|---:|
| target recall@0.5 (N=64) | 0.9859 | 0.9973 |
| target recall@0.7 (N=64) | 0.8998 | 0.9845 |
| K50 availability (audit) | 0.9912 | 0.9224 |
| K50 availability (pooled-test rows) | 0.9867 | 0.9117 |
| same-cat >=4 availability (audit) | 0.9003 | 0.7817 |
| K5 grounding acc | 0.7907 | 0.8806 |
| K50 grounding acc | 0.4320 | 0.5849 |
| K5 MSP AUROC | 0.8426 | 0.7994 |
| K50 MSP AUROC | 0.7904 | 0.7142 |
| delta AUROC K5-K50 | 0.0522 | 0.0852 |
| E-AURC worsening (relative) | 2.2413 | 4.5547 |
| RER50 drop | 0.4560 | 0.2943 |
| delta Rand (E1b-R1 AUROC) | 0.0012 | 0.0066 |
| delta Hard (E1b-R1 AUROC) | 0.0318 | 0.0213 |
| Amplification (Hard-Rand) | 0.0306 | 0.0147 |
| C1 verdict | YES | YES |
| C4 verdict | YES | YES |

C1 verdict (DETR): **YES**  
C4 verdict (DETR): **YES**  
Overall P1 verdict: **CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST** (YES)

candidate-cardinality degradation and hard-semantic amplification both replicate under DETR proposals.

