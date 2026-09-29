# Final result tables

_All values are derived from phase artifacts by `scripts/build_final_registry.py`; the consistency audit against `docs/experiment_log.md` passed._

## Table 1 — Candidate cardinality reliability (B3 Independent, random regime)

| K | Accuracy | AUROC_correct | E-AURC | RER@50 |
|---|---:|---:|---:|---:|
| 5 | 0.7907 | 0.8426 | 0.0378 | 0.8212 |
| 10 | 0.6763 | 0.8103 | 0.0705 | 0.6484 |
| 20 | 0.5643 | 0.7995 | 0.0961 | 0.5093 |
| 50 | 0.4320 | 0.7904 | 0.1226 | 0.3653 |

_Source: `phase0b_independent/aggregate.csv` (3-seed mean, corrected global-T, pooled cohort n=20799)._

> Note: under nested candidate sets the raw accuracy decline is partly structural; the paper emphasis is E-AURC / AUROC_correct / RER@coverage.


## Table 2 — Information sufficiency ladder (pooled test, 3-seed mean)

| Information | Model | Tier | K5 AUROC | K50 AUROC | K50 E-AURC | K50 RER50 |
|---|---|---|---:|---:|---:|---:|
| MSP | msp | score scalar | 0.8407 | 0.7890 | 0.1241 | 0.3588 |
| Score Stats | stats_logistic | score handcrafted stats | 0.8408 | 0.7915 | 0.1216 | 0.3611 |
| Score DeepSets | score_deepsets | full score set | 0.7380 | 0.5529 | 0.3047 | 0.1301 |
| Stats + Semantic | e1b_stats_semantic | score stats + semantic stats | 0.8423 | 0.8069 | 0.1140 | 0.3836 |
| Semantic E2 | e2_combined | learned winner-vs-competitor | 0.8267 | 0.7878 | 0.1240 | 0.3542 |
| Semantic E3 | e3_full | semantic full candidate set | 0.7979 | 0.7508 | 0.1505 | 0.3026 |

> Highlight: simple **semantic statistics (E1b)** beat the learned E2/E3 and the full score-set DeepSets; complexity does not help.


## Table 3 — Controlled hard competition (RefCOCO+, Phase 1F, b3_mean)

| Regime | Stats AUROC | E1b AUROC | dAUROC | E-AURC reduction | RER@50 gain |
|---|---:|---:|---:|---:|---:|
| Random K5 | 0.8370 | 0.8383 | +0.0012 | -0.22% | -1.21pp |
| SameCategory K5 | 0.8128 | 0.8446 | +0.0318 | +15.37% | +4.64pp |

**Dose-response (K=10, same8 cohort; m = number of same-category competitors):**

| m | dAUROC | E-AURC reduction | RER@50 gain |
|---:|---:|---:|---:|
| 0 | -0.0007 | -1.21% | -1.47pp |
| 2 | +0.0106 | +4.43% | +2.05pp |
| 4 | +0.0182 | +8.11% | +3.55pp |
| 8 | +0.0299 | +15.17% | +5.40pp |

diff-of-diffs dAUROC (hard−random) = +0.0306 [+0.0266, +0.0348]. A8.6 gate = CONFIRMED; manipulation_ok = True.


## Table 4 — External replication

| Dataset | Regime | Stats | Semantic | dAUROC | EAURC red. | RER50 gain | Manipulation valid? |
|---|---|---:|---:|---:|---:|---:|:---:|
| RefCOCO+ | random | 0.8370 | 0.8383 | +0.0012 | -0.22% | -1.21pp | YES |
| RefCOCO+ | same-category | 0.8128 | 0.8446 | +0.0318 | +15.37% | +4.64pp | YES |
| RefCOCOg strict | random | 0.8537 | 0.8542 | +0.0006 | -0.27% | -1.13pp | NO |
| RefCOCOg strict | same-category | 0.9176 | 0.9354 | +0.0178 | +20.42% | +1.02pp | NO |

> RefCOCO+ manipulation is valid (YES); the RefCOCOg same-category manipulation does NOT create additional semantic ambiguity (NO), so the RefCOCOg hard-minus-random gap must not be read as amplification replication. Q1 semantic transfer = YES; full verdict = Case D (EXTERNAL INCONCLUSIVE).
