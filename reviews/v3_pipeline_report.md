# V3 FineCops natural and controlled pipeline

## Scope and safeguards

The V3 runner prepares proposal and CLIP feature caches separately from scorer inference. Natural candidates are the stable objectness-ranked crop-valid proposal prefix for each requested K; they do not force a target, remove another valid target, or deduplicate proposals. Controlled candidates use the frozen nested random construction with the highest-IoU target and all other IoU-at-least-0.5 proposals excluded from the distractor pool. Both modes reuse the same candidate identities across B0, B16, and scorer seeds.

Rows with zero candidates remain explicit failed-grounding rows with null correctness and confidence. Rows with one to four candidates retain grounding results and have no reliability confidence. Natural evaluation keeps the complete cohort; only reliability metrics require effective K of at least five. Bank-wide and presented target counts and coverage are both retained. Raw candidate IDs and variable-width scorer logits are written losslessly and linked row by row.

Confirmation preparation requires the parent-issued content-bound prepare authorization. Confirmation scoring requires a verified `v3-freeze-v1` and claims the write-once run ledger before loading or forwarding any scorer. No confirmation prediction, reliability performance, or bootstrap was run in this development pass.

## Development verification

The audited development cohort contains 522 official positive train expressions across 200 images. The actual image fingerprints and exposure records passed CPU preflight. All actual JPEG dimensions matched the source dimensions. Eight selected expressions have a one-pixel right or bottom box overhang; their released COCO boxes were preserved without clipping or rescaling.

Development preparation completed with the frozen COCO_V1/N64 RPN, and new B0/B16 feature caches under `cache/v3/dev/`. The resulting bank has no zero-proposal images. The preparation manifest records no scorer forward, reliability forward, or performance aggregation.

The source replay checks passed for all three B0 seeds. The scorer replay checked 72 rows with maximum absolute logit difference `7.6294e-06` against the `1e-4` source tolerance. Repaired logistic S/Full probabilities matched 1,536 source rows within `3.33e-16`; ScoreDeepSets small/large probabilities matched 1,536 rows within `1.12e-16`, against the `1e-9` replay criterion. The report binds each source model checkpoint path and SHA256. B16 scorer and portable reliability artifacts loaded for all three seeds with the frozen source temperatures.

Dev natural and controlled inference completed across 4,176 expression-by-mode-by-K cells. It wrote 25,056 prediction rows for both backbones and all three seeds, plus 503,244 raw scorer logits with matching candidate-width offsets. The output includes 1,368 controlled empty-candidate rows; all are retained as failed-grounding flow, and no rows are padded. Controlled K50 common-cohort point estimates are exploratory development results only; the formal 5,000-draw bootstrap remains unrun.

The actual content-bound dev acceptance report passed all row-grid, candidate nesting, shared-identity, source-anchor, flow-preservation, and lossless-logit alignment checks:

- Acceptance: `results/v3_final_validation/pipeline/dev_acceptance.json` (SHA256 `d7280342ce1b82c75c78e92e66ba64893ee51ff812c4ed160b539a2fcfb331f`)
- Predictions: `results/v3_final_validation/pipeline/dev/inference/predictions.jsonl`
- Raw scorer outputs: `results/v3_final_validation/pipeline/dev/inference/scorer_logits.npz`
- Natural and controlled summaries: `results/v3_final_validation/pipeline/dev/inference/summary.json`
- Source replay: `results/v3_final_validation/pipeline/dev/source_model_anchors.json`
- Proposal and feature preparation inventory: `results/v3_final_validation/pipeline/dev/preparation_manifest.json`

## Tests

The V3 test modules passed: 23 tests. The full repository test suite also completed with exit code 0. Pytest reported existing unregistered `slow` and `gpu` marker warnings. Checks used `E:\conda\envs\deepminer\python.exe`; Python compilation passed for the V3 pipeline modules, runner, and tests.

## Confirmation status

The confirmation runner remains gated by the parent freeze. The image audit found one official positive val expression whose released box has zero intersection with its image despite matching the source object box. The parent directed that structurally unusable image to be excluded before confirmation while its audit record remains preserved. The replacement confirmation cohort and final stage acceptances must pass before a freeze can be sealed; this report does not claim confirmation results.


## Parent closeout — 2026-10-05 local time

The preceding report describes the original development pass. The metadata-only dev revalidation, final exposure acceptance and confirmation preparation acceptance subsequently passed. The parent sealed the input freeze before the first scorer forward. Confirmation inference completed once, with an empty corrections list, and the formal controlled/natural 5000-draw analyses completed from saved predictions. Final acceptance independently verifies all candidate identities, stable score winners, GT IoU labels, grid alignment, uncertainty arrays and natural signed ranking accounting. See results/v3_final_validation/final_acceptance.json and reviews/v3_completion.md for actual completion and narrowed conclusions. The original dev acceptance and source-anchor reports remain preserved.
