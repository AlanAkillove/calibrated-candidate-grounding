# A5.4 post-hoc gate source review

Read-only review; no bootstrap or source edits.

Amendment A5.4 is a real post-hoc Reliability GO/replication criterion, distinct from Q1/Q2/Q3 and from a nonexistent `B0_temperature_gate`. Its historical scope is Phase0A B1 cosine and Phase0B B3 Independent, both on the 20,799-row / 2,981-image `__pooled__` all-common cohort (validation plus test; no train rows), using corrected global-T confidence. Phase0A uses one fixed cosine scorer; Phase0B uses three fixed model seeds. The repair equivalents are `phase0a___all_common__` and `phase0b___all_common__`; do not substitute `__pooled_test__`.

Route A is mechanically defined: K5→K20 or K50, E-AURC relative worsening ≥20% with image-bootstrap 95% CI excluding zero, and either RER@50 or RER@80 drop ≥10 percentage points with CI excluding zero. Positive repair signs are `E_AURC(Kx)-E_AURC(K5)` as a relative ratio and `RER(K5)-RER(Kx)`; RER output in summary is a fraction and is multiplied by 100 for pp. Route B requires K5−K20/K50 AUROC_correct drop ≥0.03 with CI excluding zero plus a stable corrected-global-T reliability-map shift. The map clause is qualitative: it has no numerical cutoff. Keep its historical manual adjudication linked to the source maps, or record a fresh manual review with its rationale; do not invent a threshold or CI.

The original logs recorded Phase0A Route A pass / Route B fail, Phase0B Route A pass / Route B pass, and an overall A5.4 GO / Case A. The composite claim is post-hoc and is not Gate Q1/Q2/Q3. The current repaired K50 numeric estimates still pass Route A for both scorers: see JSON for exact points, CIs, seedwise values, and source hashes. Route A suffices for the overall GO, so the qualitative map condition need not be converted into a new numeric rule.

Exact existing K50 references:

- Both jobs: `crossK_relative_eaurc_worsening__global_T`, `crossK_K5_minus_K50__global_T::rer_at_50`, and `crossK_K5_minus_K50__global_T::auroc_correct`.
- Phase0A job: `phase0a___all_common__`; Phase0B job: `phase0b___all_common__`.

K20 / RER80 are conditions in both all-common jobs, but the current named contrast estimates are K5→K50 only. Their same-call NPZ archives retain K5/K20/K50 condition replicates from the shared image draws, so the missing K20 contrasts can be derived without resampling if desired. Since the current K50 Route A witnesses pass, this omission does not change the recorded A5.4 composite decision.

Historical source and repair artifact SHA256 values, route definitions, estimate metadata, and manual-map limitations are in `a54_gate_review.json`.
