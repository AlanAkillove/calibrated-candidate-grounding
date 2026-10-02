# V2-P2-C1 commit B - GDINO Proposal-C bank extraction + plumbing

Protocol: `results/v2_proposal_robustness/p2_c1_gdino_config_freeze.json`
(commit A, `91f2758`). Scope: **C1 only**; GDINO same_category / hard_k5 / C4
remain forbidden by the freeze and are not produced anywhere here.

## 1. Extractor

`scripts/p2_extract_gdino_proposals.py` - thin clone of the frozen
`scripts/p1_extract_detr_proposals.py`: identical bank-v1 writer, atomic
resume, corruption check / `--repair`, progress flush, stats artefact. Only
the generator module is swapped to `ccg.data.gdino` (class-prompt,
query-independent; score = top-N by max class-token sigmoid prob, `[CLS]`
global column excluded, no threshold, never padded). Sanitisation and top-N
selection are the shared `ccg.data.detr` code path with the per-image
row-exact cross-check (`assert_sanitisation_matches`) firing inside every
`extract_gdino_proposals` call - a drift between the two families aborts the
image, and zero images aborted.

Checkpoint identity is read from the pinned
`p2_gdino_probe/checkpoint_identity.json` (revision `12bdfa3…`,
model.safetensors sha256 `5548f844c928c4b6…`) and stamped into the bank attrs.

## 2. Full-bank outcome (19,992 images)

| item | value |
|---|---|
| images banked | **19,992 / 19,992** (full RefCOCO+ universe) |
| failures | **0** |
| per-image K | min = max = mean = **64** (histogram {64: 19992}) |
| total crops | 1,279,488 |
| throughput | 1.31 img/s fp32 (probe 0.55 s/img -> prefetch hides ~2x) |
| wall clock | 15,254 s (~4.2 h) |
| peak VRAM | **2,521 MB** (gate <= 6 GB, same margin as the probe) |
| device | RTX 4060 Laptop GPU, torch 2.5.1+cu121, transformers 4.57.6 |

Full stats artefact: `bank_extraction_stats.json` (copy of
`cache/proposals_gdino_stats.json`).

Mixing guard verified live: pointing `--out` at the frozen RPN bank raises
`RuntimeError: ... is not a GDINO bank (model_name=fasterrcnn_resnet50_fpn);
refusing to mix` before any write.

## 3. Plumbing (no new numbers)

- `scripts/p1_replication_core.py`: `FAMILIES["GDINO"]` =
  features `cache/features_gdino`, manifests `cache/gdino_stage/manifests`,
  bank `cache/proposals_gdino.h5` (paths byte-identical to the freeze; RPN /
  DETR entries untouched).
- `scripts/p1_frozen_inference.py`: `--family` accepts `GDINO`; `"both"`
  semantics remain exactly `["RPN", "DETR"]` (GDINO cannot sneak into a P1
  re-run). The driver skips the same_category/hard job for GDINO
  (C1-only freeze), recorded in its report as `out_of_scope`.
- `scripts/p2_c1_cardinality.py` (lands with commit D): imports
  `run_c1_for / common_k50_ids / secondary_raw_auroc / _effect_summary /
  _intersect_effects` **verbatim** from frozen F4; RPN/DETR point rows are
  re-asserted against `p1_f4_c1/c1_point.csv` to 1e-12 before anything is
  written; cohort < 8000 triggers the frozen STOP. Zero new numeric code.

## 4. Tests

`tests/test_p2_c1_plumbing.py` (8 tests): freeze-file scope, FAMILIES entry ==
freeze paths, CLI choice GDINO + both-stays-pair, hard-job skip guard,
extractor defaults == freeze (out / identity / N / manifest), identity-json
parsers, frozen score-definition string, static zero-fit scan of the
extractor. Full suite: **999 passed, 2 skipped** (was 991 + 2).

## 5. Next (commit C, per freeze git_plan)

1. `scripts/build_candidate_sets.py --bank cache/proposals_gdino.h5 --out
   cache/gdino_stage --regime random --seed 20260927`
2. `scripts/extract_features.py --bank cache/proposals_gdino.h5 --out
   cache/features_gdino` (OpenCLIP identity check enforced)
3. `scripts/p1_frozen_inference.py --family GDINO` (4 random jobs x 3 frozen
   B3 seeds; RPN predictions reused byte-for-byte)
