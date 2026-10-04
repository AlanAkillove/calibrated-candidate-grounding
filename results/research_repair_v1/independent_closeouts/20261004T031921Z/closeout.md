# Independent-review correction closeout

Completed 2026-10-04 (Asia/Shanghai) by the three specified **gpt-6-luna / max** agents. All three required corrections are implemented and regression-verified. Versioned execution acceptance is **PASS**. The original independent **CONDITIONAL_ACCEPTANCE** verdict remains unmodified; this is an execution closeout and does not claim a new independent scientific verdict.

| Required correction | Verified behavior |
|---|---|
| Finite-sample E-AURC | Preserves trapezoidal AURC minus the continuous oracle reference. [0.8,0.3] / [1,0] gives -0.028426409720027357; negative values are not clipped. |
| Risk-only metric request | Risk50/80 requests return defined risks; risk+RER and all-correct RER policies pass. All 6,732 public metric values across 396 actual B3/M2 settings match the previous implementation exactly. |
| Candidate-audit counting units | 21,026 family-expression records (RPN10,425, DETR10,601), 10,607 distinct sentence IDs, 138 mismatch records (RPN86, DETR52), and 47 mismatched family-target-object pairs. Rows and bootstrap draws are preserved. |

Full suite: **1,238 passed**, **two original opt-in skips**, zero failures/errors, **263 unchanged source/config files**. Five manifests cover **834 unchanged inputs**. All **9,955 stored estimates** pass arithmetic verification and **77 raw archives** retain their prior path/size/SHA identities. All 53 preservation copies and original review/test/acceptance records pass SHA checks. The scientific registry retains 304 SHA-bound scientific artifact identities; size-only registry entries are explicitly not treated as SHA proofs. No formal models were retrained and no new bootstrap draws were generated.

M3 remains **UNVERIFIABLE**. Original M3.1 frozen-store 1e-9 and dose-rescore 1e-4 checks are separate from the repair wrapper's committed-point 1e-9 replay criterion. The original protocol did not prescribe that point cutoff. All three parent/nine seed failures, logs and inventories remain intact; no approximate CI or relaxed tolerance replaces exact replay.

The early A launch fingerprint and complete earlier pre-derive raw hashes remain unavailable. This closeout SHA comparison does not fill those historical gaps. Adaptive ECE percentile intervals have unestablished nominal coverage near zero. V gains depend on the measured cell/model/cohort, with unresolved random K5 and dose m0; positive high-K/hard/dose contrasts do not establish universal gain or causal interaction. Test-set reuse, fixed trained seeds, GT-assisted target-present cohorts and the shared COCO visual domain remain limitations.

- [Complete closeout and SHA-bound evidence](closeout.json)
- [Versioned execution acceptance](logs/acceptance_check.json)
- [Full-suite verification](logs/pytest_full_final_verification.json)
- [Final source, preservation and scientific identity checks](logs/final_closeout_identity_verification.json)
- [Before/after metric endpoints](statistics/metric_endpoint_parity.json) and [final docstring source binding](statistics/metric_endpoint_parity_docstring_addendum.json)
- [Derived candidate counts](candidates/candidate_audit_count_evidence.json)
- [M3 tolerance provenance](recovery/m3_tolerance_provenance_v2.md)
- [Preserved prior delivery](prior_delivery/) and [preservation manifest](preservation_manifest.json)

The original `results/research_repair_v1/logs/` evidence describes the earlier delivery. The current execution checker is invoked with `--evidence-dir results/research_repair_v1/independent_closeouts/20261004T031921Z/logs`. A fresh full-suite rerun requires new evidence and basetemp paths; the runner refuses to overwrite existing closeout XML/snapshot or an existing basetemp.
