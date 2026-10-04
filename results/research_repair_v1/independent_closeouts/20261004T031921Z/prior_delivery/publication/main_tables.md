# Main evidence tables

## Table 1 — frozen B3 MSP under candidate expansion

Pooled testA+testB, global temperature; fixed-three-seed means and shared 95% image-cluster intervals.

| K | Accuracy | AUROC | Adaptive ECE | E-AURC | RER50 | Rows / images |
|---|---|---|---|---|---|---|
| 5 | 0.789520 [0.777603, 0.801466] | 0.840720 [0.831534, 0.850170] | 0.010401 [0.010831, 0.018070] | 0.038092 [0.034229, 0.042037] | 0.824775 [0.792797, 0.853906] | 10286 / 1490 |
| 10 | 0.675611 [0.661861, 0.689429] | 0.810275 [0.800499, 0.820406] | 0.019556 [0.017987, 0.031088] | 0.071018 [0.064948, 0.077167] | 0.651547 [0.618234, 0.682025] | 10286 / 1490 |
| 20 | 0.562707 [0.548416, 0.577488] | 0.799487 [0.790316, 0.808596] | 0.040287 [0.032949, 0.050506] | 0.096938 [0.090626, 0.103438] | 0.505755 [0.477721, 0.532521] | 10286 / 1490 |
| 50 | 0.428997 [0.415011, 0.443047] | 0.789022 [0.778694, 0.798714] | 0.069660 [0.060497, 0.080193] | 0.124054 [0.116575, 0.131801] | 0.358815 [0.334891, 0.381343] | 10286 / 1490 |

E-AURC and RER measure selective utility and depend on accuracy. Absolute intervals are accompanied by [paired effect intervals](formal_statistics.csv). The [full axis tables](statistics.md) retain each model, cohort, source and recovery status.

Adaptive ECE bins are recalculated within each draw. Its observed point can lie below the reported percentile interval; the estimates and intervals are retained as computed.

## Table 2 — propositions, evidence conditions and limits

| Proposition | Current evidence | Conditions and limits |
|---|---|---|
| Expansion changes reliability | Stored B3 endpoint and paired-effect intervals | GT-assisted target-present common cohort; independent candidate scoring establishes accuracy monotonicity, while AUROC direction is empirical. |
| Incremental candidate–candidate V | Incremental V signal is not uniformly resolved across reported cells. The central claim is limited to the measured cellwise effects; no general candidate-interaction conclusion follows. | Fixed S/S+Q/S+V/Full groups and fixed grounding; cellwise conditional effects with test-set reuse. |
| Score-model extrapolation | Formal small/large-K contrasts and DoD stored. | Equal train/tune row budgets; tested model failures do not prove score-information absence. |
| Same-category construction | 138 / 21026 category mismatches; sensitivity FORMAL_COMPLETE; independent raw numerical QA PASS. | Proposal multiplicity differs from distinct annotated objects; preserve source-specific availability intersections. |
| Four-corner decomposition | Observed label path means have opposite signs in RPN, DETR, GDINO. The signed Shapley mean can conceal this cancellation; the accounting does not identify a cause. | Frozen prediction accounting; H1c addresses accuracy harm and does not identify AUROC-gap causes. |

Source snapshots and formal method checks are recorded in [evidence.json](evidence.json).
