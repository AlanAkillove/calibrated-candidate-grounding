# Test correction rationale

`test_rq4_m1_material_bank.py` and the material-bank reads in
`test_v2_governance_closeout.py` now check the preserved historical appendix.
Their original value, identity, boundary, and frozen-registry invariants remain
unchanged. Requiring every historical label, duplicate table, and old frozen-status
sentence in the current scientific summary would contradict the authorized repair
and promote obsolete interpretation. Current evidence regeneration and scientific
boundaries receive separate integration tests. This change does not loosen any
numerical tolerance, remove tests, or modify the baseline.

The same rationale applies to `test_v2_p_program_summary.py`'s material-bank identity test: its historical bilingual labels, artifact values, axis boundaries and withdrawal assertions now read the preserved pre-repair appendix. The current summary deliberately uses corrected intervals and conditional claims rather than reproducing every historical gate label. No numeric or boundary assertion was removed.

The first full suite ran all 1175 collected tests with local weights/offline settings: 1165 passed, 3 failed, 5 setup errors, 2 skipped (224.970 s). The new sensitivity fixture had two failures (floating-point identity comparison and absent fixture manifest); C owns their repair. Five hard-competition setup errors reproduce the already-recorded constrained-thread refit failure, not a change to the frozen artifact. B will retain strict original tolerances and all model/split/normalization invariants while using the original hash-verified frozen load path. Final acceptance still requires the completed full suite.

`test_hard_competition.py` now tests the original frozen model through its load-and-verify interface at the original 1e-9 tolerance. It retains original coefficient, confidence, normalization, train-row mask, candidate-order and input-immutability checks, and adds explicit selected-C consistency. The refit function remains unchanged and its constrained-thread failure is recorded separately; frozen-model identity does not depend on a successful new optimizer run. Targeted five affected tests and the full twelve-test file passed.
