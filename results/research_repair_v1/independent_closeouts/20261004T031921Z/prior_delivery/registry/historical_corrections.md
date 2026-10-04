# Historical P2 report corrections

Source: `results/v2_proposal_robustness/p2_c1_gdino/c1_point.csv`, SHA256 `ef02bf2e6ee10fb513a1fc502cd56bb2b8f99321a3496de0e5c6d83b5ed9f70a`.

| Family | K5 accuracy | AUROC | E-AURC | RER@50 |
|---|---|---|---|---|
| RPN | 0.789520 | 0.840720 | 0.038092 | 0.824775 |
| DETR | 0.880600 | 0.799356 | 0.032105 | 0.680503 |
| GDINO | 0.846344 | 0.830757 | 0.031266 | 0.806466 |

Highest K5 RER@50: RPN.
GDINO minus DETR accuracy: -0.034256.
GDINO minus RPN accuracy: +0.056824.

E-AURC is selective utility, not a calibration metric; lowest E-AURC does not imply best calibration.
The original report remains immutable.

## Historical P1 C4 RER direction

The field `e1b_minus_r1_rer50_gain` contains R1−E1b. Actual E1b gain is E1b−R1, since RER is higher-is-better.

These are frozen historical point corrections; paired uncertainty is reported in the repaired statistical export.

| Family | Scorer | Regime | Historical field | Actual E1b RER50 gain |
|---|---|---|---|---|
| RPN | b3_seed1 | hard | -0.047995 | +0.047995 |
| RPN | b3_seed1 | random | +0.010485 | -0.010485 |
| RPN | b3_seed2 | hard | -0.054342 | +0.054342 |
| RPN | b3_seed2 | random | +0.009661 | -0.009661 |
| RPN | b3_seed3 | hard | -0.040389 | +0.040389 |
| RPN | b3_seed3 | random | +0.013711 | -0.013711 |
| DETR | b3_seed1 | hard | -0.034252 | +0.034252 |
| DETR | b3_seed1 | random | -0.019486 | +0.019486 |
| DETR | b3_seed2 | hard | -0.031408 | +0.031408 |
| DETR | b3_seed2 | random | -0.010536 | +0.010536 |
| DETR | b3_seed3 | hard | -0.030744 | +0.030744 |
| DETR | b3_seed3 | random | -0.008666 | +0.008666 |
