| Dataset | Regime | Stats AUROC | E1b AUROC | dAUROC | EAURC reduction | RER50 gain |
|---|---|---:|---:|---:|---:|---:|
| RefCOCO+ | Random-K5 | 0.8370 | 0.8383 | +0.0012 | -0.2% | -1.21pp |
| RefCOCO+ | SameCat-K5 | 0.8128 | 0.8446 | +0.0318 | +15.4% | +4.64pp |
| RefCOCOg strict | Random-K5 | 0.8537 | 0.8542 | +0.0006 | -0.3% | -1.13pp |
| RefCOCOg strict | SameCat-K5 | 0.9176 | 0.9354 | +0.0178 | +20.4% | +1.02pp |

Values are 3-seed means of the per-seed image-cluster paired-bootstrap rows (5000 reps, 95% CI). RefCOCO+ rows are read from the frozen Phase-1F `bootstrap.csv`; RefCOCOg rows from `a11_results.json`.
