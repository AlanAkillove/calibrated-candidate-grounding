# Required statistical scope coverage

This inventory includes recovery-pending scopes. A registered runner or a completed subset does not establish complete coverage.

| Scope | Status | Formal completed / required | Unverifiable recovery jobs | Recovery route | Anchor |
|---|---|---|---|---|---|
| phase0a | FORMAL_5000_COMPLETE | 6 / 6 | none | Frozen native-ID prediction NPZ; no recovery. | PASS_FOR_COMPLETED_JOBS |
| phase0b | FORMAL_5000_COMPLETE | 6 / 6 | none | Frozen native-ID three-seed predictions; no recovery. | PASS_FOR_COMPLETED_JOBS |
| phase05 | FORMAL_5000_COMPLETE | 4 / 4 | none | Frozen prediction CSVs read-only; preserve original model identity and old scaler leakage note. | PASS_FOR_COMPLETED_JOBS |
| phase1 | FORMAL_5000_COMPLETE | 8 / 8 | none | Fixed Stats/E1b sourced from frozen model bundle plus original feature rows; selected CSV retains per-seed selected model identity and is not substituted for fixed E1b. | PASS_FOR_COMPLETED_JOBS |
| phase1f | FORMAL_5000_COMPLETE | 3 / 3 | none | Frozen native-ID NPZs; no recovery. | PASS_FOR_COMPLETED_JOBS |
| refcocog | FORMAL_5000_COMPLETE | 1 / 1 | none | Read-only frozen bundle and external native-ID predictions; no refit when bundle load exists. | PASS_FOR_COMPLETED_JOBS |
| v2g_g3 | FORMAL_5000_COMPLETE | 12 / 12 | none | READ_ONLY_PREDICTION_ARTIFACT | PASS_FOR_COMPLETED_JOBS |
| v2g_g4 | FORMAL_5000_COMPLETE | 2 / 2 | none | Frozen B3 forward under original PhaseA/PhaseB candidates; no training; forward pending. | PASS_FOR_COMPLETED_JOBS |
| d1 | FORMAL_5000_COMPLETE | 3 / 3 | none | READ_ONLY_NATIVE_ID_PREDICTIONS | PASS_FOR_COMPLETED_JOBS |
| d2_c1 | FORMAL_5000_COMPLETE | 1 / 1 | none | READ_ONLY_PREDICTION_ARTIFACT | PASS_FOR_COMPLETED_JOBS |
| d2_c4 | FORMAL_5000_COMPLETE | 1 / 1 | none | READ_ONLY_NATIVE_ID_PREDICTIONS | PASS_FOR_COMPLETED_JOBS |
| p | FORMAL_5000_COMPLETE | 5 / 5 | none | READ_ONLY_NATIVE_ID_PREDICTIONS | PASS_FOR_COMPLETED_JOBS |
| m1 | FORMAL_5000_COMPLETE | 5 / 5 | none | READ_ONLY_CONFIDENCE_ARCHIVE_WITH_VERIFIED_ID_BRIDGE | PASS_FOR_COMPLETED_JOBS |
| m2 | FORMAL_5000_COMPLETE | 1 / 1 | none | READ_ONLY_CONFIDENCE_ARCHIVE_WITH_VERIFIED_NATIVE_ID_BRIDGE | PASS_FOR_COMPLETED_JOBS |
| m25 | FORMAL_5000_COMPLETE | 1 / 1 | none | Frozen M1/M2 model forward; recovery delegated to B, no fitting. | PASS_FOR_COMPLETED_JOBS |
| m3 | UNVERIFIABLE | 0 / 0 | m3_b0_severity_macro, m3_b1_severity_macro, m3_b2_severity_macro | Original deterministic checkpoint-load/inference/refit routes attempted; original 1e-9 point-anchor tolerance retained. No historical prediction archive was substituted. | STRICT_FAILURE_EVIDENCE_FOR_ALL_REQUIRED_JOBS |

- m3/m3_b0_severity_macro: UNVERIFIABLE. Evidence: `results/research_repair_v1/statistics/recovery/runs/20261003T095000Z/failures/m3_b0_severity_macro.json`. This recovery limitation does not supply an effect estimate or a gate pass.
  Original tolerances: `{"M1_M2_confidence": 1e-09, "M2_point_metrics": 1e-09, "M3_B0_committed_point_metrics": 1e-09}`. Observed error: `4.766e-07`. Failure reason: The original deterministic M3-B0 route completed for all 3 seeds, but the committed point-metric table was not reproduced within the unchanged 1e-9 tolerance for 3/3 seeds; largest observed absolute difference is 4.766e-07. M1/M2 confidence anchors passed; no saved per-seed mixer checkpoint exists for exact load-and-replay.

- m3/m3_b1_severity_macro: UNVERIFIABLE. Evidence: `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/m3_b1_severity_macro.json`. This recovery limitation does not supply an effect estimate or a gate pass.
  Original tolerances: `{"committed_point_metrics": 1e-09, "dose_rescore": 0.0001, "m2_validation_stop": "original M2 STOP_TOL", "phase_a_k5_store": 1e-09}`. Observed error: `2.753e-07`. Failure reason: B1 original M3.1 deterministic fit/inference completed for all 3 seeds and passed the frozen Phase-A/dose/STOP checks, but the committed point-metric table was not reproduced within 1e-9 for any seed; maximum first-mismatch absolute error=2.753e-07. The historical curriculum/mixer/per-sample outputs are not loadable from a saved checkpoint.

- m3/m3_b2_severity_macro: UNVERIFIABLE. Evidence: `results/research_repair_v1/statistics/recovery/runs/20261003T130700Z/failures/m3_b2_severity_macro.json`. This recovery limitation does not supply an effect estimate or a gate pass.
  Original tolerances: `{"committed_point_metrics": 1e-09, "dose_rescore": 0.0001, "m2_validation_stop": "original M2 STOP_TOL", "phase_a_k5_store": 1e-09}`. Observed error: `1.807e-07`. Failure reason: B2 original M3.1 deterministic fit/inference completed for all 3 seeds and passed the frozen Phase-A/dose/STOP checks, but the committed point-metric table was not reproduced within 1e-9 for any seed; maximum first-mismatch absolute error=1.807e-07. The historical curriculum/mixer/per-sample outputs are not loadable from a saved checkpoint.
