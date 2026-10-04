# M3 tolerance provenance (v1)

Created 2026-10-04T03:33:58.949723+00:00. This is a read-only provenance check; it does not change the historical records or construct intervals.

## Finding

The original M3/M3.1 artifacts explicitly specify **1e-9 for the frozen Phase-A K5 anchor** and **1e-4 for the V2-G dose-response rescore anchor**. The original M2 runner separately defines a **1e-4 confidence-score STOP**. The M3 B0 runner’s **1e-9 `REPRO_TOL` is used for M1/M2 expert confidence reproduction**.

The preregistered M3 protocol and frozen M3.1 amendment do not state a tolerance for replaying the committed M3 point-metric table. The current recovery wrapper compares those rows at 1e-9; that is an attempt-specific strict replay criterion, not automatically an original M3 protocol threshold. The `original_tolerance` fields in the recovery failure records are preserved as written; this report clarifies how their values are supported by source provenance.

All three parent scopes and all nine seed records remain `UNVERIFIABLE` for exact committed-point replay. Their strict mismatch does not recompute or negate the original scientific M3 gates. No formal bootstrap ran and no CI is supplied.

## Thresholds and sources

| Check | Tolerance | What it checks | Provenance |
|---|---:|---|---|
| M3.1 frozen Phase-A K5 | 1e-9 | E_random K5 AUROC against frozen V2-G Phase-A store | `amendment_v2m31.json` `/anchors/e_random_vs_phase_a`; `run_v2m_m3_conf.py` `ANCHOR_STORE_TOL` |
| M3.1 dose-response rescore | 1e-4 | Re-scored per-level AUROC where V2-G per-row dose scores were not saved | `amendment_v2m31.json` `/anchors/e_random_vs_dose_response`; source explains ~2.8e-7 float32 drift |
| M2 validation STOP | 1e-4 | Maximum confidence-score rescore difference | `run_v2m_m2.py` `STOP_TOL`; separately recorded in M3.1 checks |
| B0 expert confidence | 1e-9 | Frozen M1/M2 confidence vectors | `run_v2m_m3_b0.py` `REPRO_TOL`; not a point-metric tolerance |
| Current committed M3 point replay | 1e-9 | Recomputed M3 point rows versus committed CSV | `repair_legacy_model_recovery.py`; repair-attempt criterion only; original M3 protocol does not define this replay tolerance |

## Existing failure evidence

| Scope | Parent failure JSON (SHA-256) | Seed first-mismatch absolute errors | No-checkpoint inventory |
|---|---|---|---|
| m3_b0_severity_macro | `results/research_repair_v1/statistics/recovery/runs/20261003T095000Z/failures/m3_b0_severity_macro.json` `cdc50357c3f531f36bddbaa842a648dfe4f4caf4f429c635224e7f69c2d2e5f9` | 1: 4.765e-07, 2: 9.49e-08, 3: 4.766e-07 | `results/research_repair_v1/statistics/recovery/runs/20261003T095000Z/inventory/m3_b0_checkpoint_inventory.json` `89ad522665d226a040336b169015d06704d53ae3f1a400366bd278bec9a5eeeb` |
| m3_b1_severity_macro | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/m3_b1_severity_macro.json` `35aabd9e94b4185ec0cf2c8455d004579c65c10e290181ecc69945769a971842` | 1: 9.254e-08, 2: 9.182e-08, 3: 2.753e-07 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/inventory/m3_b1_checkpoint_inventory.json` `83719446a7e4dbfa39f9a6c606f1fc904c7dc7526b596b1aaa2ae6f6999f5543` |
| m3_b2_severity_macro | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/m3_b2_severity_macro.json` `b70db4f738c19fb44edd4d53cce66fcdc5035c8772f325b15d3a0e3d1b575dec` | 1: 1.807e-07, 2: 8.855e-08, 3: 1.282e-09 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/inventory/m3_b2_checkpoint_inventory.json` `bc0b4b8592f9e1c9cdf77a725d1f76018ec4b02e4f8e2568b8723da0bd6a780c` |

Each seed failure JSON and its associated log was independently SHA-256 checked against its parent record. The nine JSON references and verified hashes follow; the existing failure JSON/log bytes were not edited.

### m3_b0_severity_macro

| Seed scope | Failure JSON | JSON SHA-256 | Observed first mismatch | Log SHA-256 |
|---|---|---|---|---|
| v2m_m3_b0_seed_1 | `results/research_repair_v1/statistics/recovery/runs/20261003T095000Z/failures/v2m_m3_b0_seed_1.json` | `42ddb1b9e2f5f712e3e2127c788c9dc129df5fcecb1f140d4efdeb82f3230f1e` | 4.765e-07: M3_B0_committed_point_metrics ('m0', 'AdaptiveMix', 1)/auroc_correct: observed=0.83556321140940337 expected=0.83556273495669664 abs_error=4.765e-07 > 1e-09 | `3d38031959e5b2f25a5158d197456b86193abc5c58611085ea44c76314324a8e` |
| v2m_m3_b0_seed_2 | `results/research_repair_v1/statistics/recovery/runs/20261003T095000Z/failures/v2m_m3_b0_seed_2.json` | `1302532a814a9b6340477e30e4f94f7105516117bd8130e4a18fc9bdb91c093a` | 9.49e-08: M3_B0_committed_point_metrics ('m0', 'StaticMix', 2)/auroc_correct: observed=0.83838825178801757 expected=0.83838834668962781 abs_error=9.490e-08 > 1e-09 | `f4fecb2cb291faaf64957af3ac8ddd89fab0c9a2da0d2e156232863c884d6dc5` |
| v2m_m3_b0_seed_3 | `results/research_repair_v1/statistics/recovery/runs/20261003T095000Z/failures/v2m_m3_b0_seed_3.json` | `0815279cc5e6a534a9a011b50f33259b1fdd0259cc109ddf9ac4579ff56e2e46` | 4.766e-07: M3_B0_committed_point_metrics ('m0', 'AdaptiveMix', 3)/auroc_correct: observed=0.83519531345323217 expected=0.83519483683716045 abs_error=4.766e-07 > 1e-09 | `9d10f916ae2a26b5c570c0c423fad9324e92dceb80c5c1ea1cca690061c4bb76` |

### m3_b1_severity_macro

| Seed scope | Failure JSON | JSON SHA-256 | Observed first mismatch | Log SHA-256 |
|---|---|---|---|---|
| v2m_m3_conf_b1_seed_1 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/v2m_m3_conf_b1_seed_1.json` | `6575cd0bfd5a34f5f9c1e336f209f703229878077a39c40a3b753ca0850e0f1e` | 9.254e-08: M3_1_committed_point_metrics ('m0', 'E_curriculum', 1)/auroc_correct: observed=0.82784455131171097 expected=0.82784445877347601 abs_error=9.254e-08 > 1e-09 | `dca50fef74c4b0b1152fb6e9d9b2f2c3873092163cfbafa9040c1c62de3c7a29` |
| v2m_m3_conf_b1_seed_2 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/v2m_m3_conf_b1_seed_2.json` | `e37f6af34c73c6d8deaae8c00c1bbed06a616a3324ffba471ff838f95b64a083` | 9.182e-08: M3_1_committed_point_metrics ('m0', 'E_curriculum', 2)/auroc_correct: observed=0.82284197408010107 expected=0.82284188226259269 abs_error=9.182e-08 > 1e-09 | `a43f90bec5967d354f5abdacb9919aaf42f151ba96a5dab48bb49db08cda8ba7` |
| v2m_m3_conf_b1_seed_3 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/v2m_m3_conf_b1_seed_3.json` | `589fc3bcd8992e1b0450856b544e4303b6c7da57c910baae4e7d1902b2f4242d` | 2.753e-07: M3_1_committed_point_metrics ('m0', 'E_curriculum', 3)/auroc_correct: observed=0.83023033518105915 expected=0.83023005989869059 abs_error=2.753e-07 > 1e-09 | `3317da5b6185c2b333f14628122542d3a443e655603804e6b483f039ca817187` |

### m3_b2_severity_macro

| Seed scope | Failure JSON | JSON SHA-256 | Observed first mismatch | Log SHA-256 |
|---|---|---|---|---|
| v2m_m3_conf_b2_seed_1 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/v2m_m3_conf_b2_seed_1.json` | `c423c18a7bf8d0b05291ee6891a462486daf8df841fb231b0966082c0d771dd0` | 1.807e-07: M3_1_committed_point_metrics ('m0', 'StaticMix', 1)/auroc_correct: observed=0.83527067830597723 expected=0.83527049758630811 abs_error=1.807e-07 > 1e-09 | `0928f1781ee9438863de20faddaa22725d255d46d66e2190601058d1fa79051e` |
| v2m_m3_conf_b2_seed_2 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/v2m_m3_conf_b2_seed_2.json` | `3698fcf3cd33f884604c9ef8aedd5c97021ae208a762e163917cfe7c16f07ae9` | 8.855e-08: M3_1_committed_point_metrics ('m0', 'E_curriculum', 2)/auroc_correct: observed=0.83442950212180977 expected=0.83442941357471268 abs_error=8.855e-08 > 1e-09 | `7763bb2b59880f7e2058433f2be87e6732e45c3925efa19699889b6b757d589f` |
| v2m_m3_conf_b2_seed_3 | `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/v2m_m3_conf_b2_seed_3.json` | `0bd47f7b547f3c80f34af17dd2cef81d2987588ed59cb347aa565b82348e1f6a` | 1.282e-09: M3_1_committed_point_metrics ('m0', 'E_curriculum', 3)/e_aurc: observed=0.056762275645773394 expected=0.056762276928089314 abs_error=1.282e-09 > 1e-09 | `bbc2f544a2335670f3df0e1f21fb4db7069c6da34255a19f24baf7327c620b41` |

## Hash-bound sources

The JSON sidecar lists SHA-256 and byte length for the original protocol, amendment, original runners, metadata, repair source, and review documents. The M3.1 amendment freeze hashes match the current protocol and original runner files; B0 and M3.1 metadata source hashes match the corresponding protocol/amendment files.

The original parent and seed failure files remain untouched. The sidecar itself records all three aggregate failure hashes, nine child JSON hashes, associated log hashes, inventory hashes, source hashes, and the 4.3 MB scope-ledger hash.
