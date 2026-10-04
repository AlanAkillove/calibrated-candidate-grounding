# statistics — artifact-derived repair tables

These tables render stored repair outputs. File existence alone does not
establish formal completion; consult STATUS.json and the acceptance report.

## pilot/summary.json

Source: `results/research_repair_v1/statistics/pilot/summary.json`; SHA256 `98beeb0efd5f7ab1f5654ab30556312ccbe670bd6f4a964e0a9ca4606d09897b`.

| Field | Stored value |
|---|---|
| schema | ccg.research_repair_v1.statistics.summary.v1 |
| created_utc | 2026-10-03T02:22:36.360513+00:00 |
| classification | RESULT_DRIVEN_SUPPLEMENTARY_REPAIR |
| baseline_manifest | results/research_repair_v1/input_manifest.json |
| baseline_manifest_sha256 | 511499c47659bde0bdf207befac4e6efb41975a8b4f1623b10e469c0b89a334f |
| bootstrap.resample_unit | image_cluster |
| bootstrap.replicates | 5000 |
| bootstrap.seed | 0 |
| bootstrap.ci_level | 0.95 |
| bootstrap.method | percentile |
| inference_scope | conditional test-sample uncertainty for the three fixed trained seeds; seed SD is reported separately and does not estimate training randomness |

## summary.json

Source: `results/research_repair_v1/statistics/summary.json`; SHA256 `99985fb91509d25c837d1d6b0aef9a3a5d6fbaf153fac83ddcfeb3de75e46114`.

| Field | Stored value |
|---|---|
| schema | ccg.research_repair_v1.statistics.summary.v1 |
| created_utc | 2026-10-03T02:21:49.192053+00:00 |
| classification | RESULT_DRIVEN_SUPPLEMENTARY_REPAIR |
| baseline_manifest | results/research_repair_v1/input_manifest.json |
| baseline_manifest_sha256 | 511499c47659bde0bdf207befac4e6efb41975a8b4f1623b10e469c0b89a334f |
| bootstrap.resample_unit | image_cluster |
| bootstrap.replicates | 5000 |
| bootstrap.seed | 0 |
| bootstrap.ci_level | 0.95 |
| bootstrap.method | percentile |
| bootstrap.validity_sufficiency_rule.minimum_valid_fraction_for_exploratory_ci | 0.95 |
| bootstrap.validity_sufficiency_rule.minimum_valid_replicates | ceil(0.95 * planned_replicates) |
| bootstrap.validity_sufficiency_rule.gate_rule | all planned replicates must be valid for gate_eligible=true; any CI with invalid draws is conditional on finite draws and is not gate-eligible |
| bootstrap.validity_sufficiency_rule.rationale | invalid draws can remove probability mass from the 2.5%/97.5% tails; below 95% valid draws the interval is explicitly uncertainty-insufficient |
| inference_scope | conditional test-sample uncertainty for each job's fixed model/seed set; report n_seeds from each estimate and show seed SD separately, which does not estimate training randomness |

