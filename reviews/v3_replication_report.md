# V3 OpenCLIP B16 S/Q/V replication

## Status and interpretation

The B16 fit, source-score replay, cached-feature evaluation, point estimates, and parent-allocated formal bootstrap are complete. This is a developmental/configuration replication on the already-observed COCO test cohort, not independent confirmation and not FineCops confirmation C3. All reported intervals are marginal 95% descriptive intervals; the 98.75% familywise interval convention remains reserved for the four FineCops confirmation comparisons.

## Frozen setup

The scorer is the three frozen V2-G B16 checkpoints at `results/v2_backbone_generalization/b1_phase0b/seed_{1,2,3}/model.npz`, using corrected temperatures 1.1711449916929477, 1.0707236210403277, and 1.057321688843383. Cached OpenCLIP ViT-B-16 / laion2b_s34b_b88k features and semantic embeddings were reused. K5/K10 semantic features were recomputed in chunks from the existing embedding NPZs; K50 and alternate candidate cells were summarized from the existing B16 feature H5 files. No visual feature extraction or grounding training/refit was run.

The direct replication input manifest records 38 files and reuses 30 SHA/size entries from the previously verified manifest; the parent-run historical image audit independently passed all 22,094 discovered images with no decode failures (`results/v3_final_validation/exposure/historical_fingerprint_status.json`).

Feature columns and order come from frozen `ccg.repairs.information.FEATURE_GROUPS`: S17, S+Q25, S+V25, and Full33. Each scorer seed fit all four logistic reliability models on the original 523-image train / 224-image tune split, pooling K5 and K10. Each fit used training-only standardization, CPU BLAS/torch limits of two threads, and the frozen C grid [0.1, 1, 10] with original tie selection. All 12 models selected C=0.1. Each seed used 7,372 combined train rows.

| Seed | S tune AUROC | S+Q | S+V | Full |
|---|---:|---:|---:|---:|
| 1 | 0.819699 | 0.819776 | 0.823162 | 0.823264 |
| 2 | 0.821535 | 0.821040 | 0.825198 | 0.826580 |
| 3 | 0.825122 | 0.822939 | 0.828245 | 0.826855 |

Portable checkpoints for downstream C3 inference are at `results/v3_final_validation/replication/models/b3_seed{1,2,3}/{S,S+Q,S+V,Full}.json`. Their schema is `v3_b16_s_q_v_replication_logistic` version 1, with feature names, fitted normalization mean/std/keys/dimension, selected logistic coefficients/intercept/C, tune grid, source split, and frozen temperature metadata. `ccg.v3.replication.load_model_json(path)` returns the normalization, predictor, and payload; inference applies `feature_block(stats17, sem16, group)`, `normalize_apply`, then `predict_proba`.

## Source scorer anchors and identity checks

All six CPU replays against the frozen raw-score anchors passed `ccg.semantic.hard_scores.verify_against_raw_scores` at the original absolute tolerance 1e-4. Each comparison covers 20,799 rows:

| Seed | K5 max absolute score error | K10 max absolute score error |
|---|---:|---:|
| 1 | 1.3351440e-05 | 1.5258789e-05 |
| 2 | 1.1444092e-05 | 1.3351440e-05 |
| 3 | 1.1444092e-05 | 1.2397766e-05 |

The exact candidate manifests remained in their frozen order with target in local column 0. Random K5 versus hard K5 target scores differed by at most 1.1444092e-05; dose K10 m0 versus m8 target scores were identical. Every evaluated row matched frozen sentence, reference, image, and split identities. Phase 0A exposes 21,373 generic candidate-eligible rows; the fixed scorer cohort is a verified 20,799-row subset, so the 574 non-cohort rows were excluded before feature construction.

## COCO developmental point estimates

All numbers below are means of the three fixed-seed effects. AUROC improvements are positive; risk improvements are negative. Risk uses the shared `fractional_boundary_tie` convention. The authoritative per-seed and absolute metrics are in `metrics.csv` and paired effects in `point_effects.csv`.

| Cell | Rows / images | Full−S+Q AUROC | Full−S+Q Risk@50 | Full−S+Q Risk@80 |
|---|---:|---:|---:|---:|
| Random K5 | 10,286 / 1,490 | −0.000343 | +0.001231 | −0.001086 |
| Random K10 | 10,286 / 1,490 | +0.004189 | −0.002463 | −0.003127 |
| Random K50 | 10,286 / 1,490 | +0.014322 | −0.011537 | −0.005145 |
| Matched hard K5 | 9,487 / 1,424 | +0.015801 | −0.010400 | −0.009425 |
| Dose m0 K10 | 7,410 / 1,189 | +0.000612 | +0.001799 | −0.000337 |
| Dose m8 K10 | 7,410 / 1,189 | +0.017719 | −0.016284 | −0.007816 |

For the primary developmental K50 cell, absolute point estimates with 95% marginal image-bootstrap intervals were:

| Group | Correctness AUROC | Risk@50 | Risk@80 |
|---|---:|---:|---:|
| S | 0.787661 [0.777474, 0.797109] | 0.376499 [0.357108, 0.397518] | 0.513173 [0.497041, 0.529378] |
| S+Q | 0.784732 [0.774347, 0.794328] | 0.380193 [0.359099, 0.401152] | 0.515563 [0.499340, 0.531704] |
| S+V | 0.801975 [0.792310, 0.810887] | 0.365157 [0.344880, 0.386159] | 0.507065 [0.491055, 0.523857] |
| Full | 0.799054 [0.789083, 0.808218] | 0.368656 [0.349023, 0.388955] | 0.510419 [0.493866, 0.526595] |

The 5,000 shared image-cluster draws yielded the primary `Full−S+Q` estimates: AUROC +0.014322 [0.011671, 0.016967], Risk@50 −0.011537 [−0.014607, −0.007334], and Risk@80 −0.005145 [−0.007053, −0.003472]. The direct paired contrasts were:

The direct paired difference of Full−S+Q, using the same sampled images for both cells, was:

| Paired contrast | AUROC difference of differences | Risk@50 | Risk@80 |
|---|---:|---:|---:|
| Hard K5 − random K5 | +0.016336 [0.014405, 0.018254] | −0.010716 [−0.014310, −0.008056] | −0.008845 [−0.011654, −0.006793] |
| Dose m8 − m0 K10 | +0.017107 [0.014326, 0.020165] | −0.018084 [−0.021706, −0.013372] | −0.007479 [−0.010495, −0.004528] |

The group ordering limits the interpretation: at random K50, S+V has higher absolute AUROC (0.801975) and lower Risk@50/Risk@80 (0.365157/0.507065) than Full (0.799054; 0.368656/0.510419), while S+Q is below S on AUROC (0.784732 vs 0.787661). The positive Full−S+Q contrast supports a conditional incremental difference when V is added to the S+Q baseline on this cohort. It does not establish that Full is the best group, that Q always helps, or that an interaction is causal. No post hoc model or feature tuning was performed.

All 111 mean estimates and their per-seed estimates had 5,000 valid draws and zero invalid draws. The bootstrap JSON contains compact intervals, per-seed summaries, invalid counts, and references to raw draws; the NPZ contains all 444 raw mean/per-seed arrays, each of length 5,000. The NPZ SHA-256 is `9ff33ce1a0b7c98d8003ff9f69d9abee9890777fe7f259f351a2dd039133b97b`. The exporter cleanup removed duplicated per-seed arrays from JSON after the run and verified the NPZ arrays and mean-draw aggregation without rerunning any statistics. The primary V3 claim remains reserved for independent FineCops C3. These already-observed COCO intervals are descriptive configuration evidence only.

## Artifacts and validation

`results/v3_final_validation/replication/` contains the input SHA/size manifest, split reuse record, source anchor errors, exact candidate identity checks, portable models, per-sentence probabilities and identities, absolute metrics, paired point effects, bootstrap summary, raw draws and raw-draw SHA manifest, plus final acceptance evidence and run recovery history. The 84 test metric rows cover 3 seeds × 7 cells × 4 groups. The six synthetic replication tests pass, including group definitions, train-only standardization, frozen-anchor tolerance, candidate identity and target-column checks, fractional tied-boundary risk, and portable JSON inference round-trip.

The first two full-run attempts stopped on implementation assertions (Windows slash normalization in a manifest-key lookup, then an overly strict Phase 0A-versus-scorer row-set equality check). Both were corrected; the successful run required the exact canonical scorer IDs to be a subset of Phase 0A rows and additionally verified ref/image/split equality for every used row. `run_attempts.json` preserves recovery history; the successful point and bootstrap stages are logged in `replication.log`.
