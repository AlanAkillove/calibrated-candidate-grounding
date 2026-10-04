# Candidate category and object audit

Verified run: `20261003_Cslot_audit4`; status **AUDIT_COMPLETE_MISMATCH_FOUND**.

True annotation category differs from the target proposal's assigned category in 138 of 21026 audited expressions (rate 0.006563).

| Family | Expressions | Mismatches | Rate | Unknown proposal category |
|---|---|---|---|---|
| RPN | 10425 | 86 | 0.008249 | 0 |
| DETR | 10601 | 52 | 0.004905 | 0 |

Historical hard/dose source cohorts remain separate. Proposal counts are distinct from GT object counts.

| Family | Cell | Candidate source | Source expressions | Category audit coverage | Mismatches |
|---|---|---|---|---|---|
| RPN | hard5 | phase1f_frozen_candidate_archive | 9487 | 9487 | 54 |
| RPN | hard10 | phase1f_frozen_candidate_archive | 6765 | 6765 | 24 |
| RPN | expb_m0 | phase1f_frozen_candidate_archive | 7410 | 7410 | 27 |
| RPN | expb_m2 | phase1f_frozen_candidate_archive | 7410 | 7410 | 27 |
| RPN | expb_m4 | phase1f_frozen_candidate_archive | 7410 | 7410 | 27 |
| RPN | expb_m8 | phase1f_frozen_candidate_archive | 7410 | 7410 | 27 |
| RPN | hard5 | v2_p1_frozen_hard_k5_predictions | 9623 | 9623 | 54 |
| DETR | hard5 | v2_p1_frozen_hard_k5_predictions | 8376 | 8376 | 11 |

Geometry, objectness, IoU, duplicates and distinct-object summaries:

- [candidate geometry summary](../candidates/audit_run_20261003_Cslot_audit4/candidate_geometry_historical_source_summary.csv)
- [candidate geometry supply intersection summary](../candidates/audit_run_20261003_Cslot_audit4/candidate_geometry_supply_intersection_summary.csv)
- [candidate geometry sensitivity summary](../candidates/audit_run_20261003_Cslot_audit4/candidate_geometry_sensitivity_summary.csv)
- [availability intersection](../candidates/audit_run_20261003_Cslot_audit4/availability_intersection.csv)

The mismatch requires a frozen old/new sensitivity analysis on paired availability intersections. Formal paired sensitivity intervals are stored; their independent raw numerical QA is recorded separately.

Sensitivity stage: **FORMAL_COMPLETE**.

Independent raw numerical QA: **PASS**.

[Paired sensitivity estimates](candidate_sensitivity.md).
