# V2-P2-C1 commit C - GDINO manifests + feature caches + frozen inference

Protocol: `../p2_c1_gdino_config_freeze.json` (commit A, `91f2758`).
Everything here is plumbing over frozen code: `scripts/build_candidate_sets.py`,
`scripts/extract_features.py`, `scripts/p1_frozen_inference.py` were NOT modified
for this stage (the only P2 edits to the driver landed in commit B: a `--family`
choice plus the GDINO hard-job skip guard).

## 1. Candidate manifests (random regime only)

Frozen command: `--bank cache/proposals_gdino.h5 --out cache/gdino_stage
--regime random --seed 20260927`.

* 49,856 refs over 19,992 images, `bank_fingerprint=2c643cf195321ac1…`,
  `top_n=64`, schema `manifests-v1`.
* Per split (refs / target-present / common-K50): train 42278 / 42082 / 41624,
  val 3805 / 3783 / 3754 (val_select 1917 / 1887, val_calib 1888 / 1867),
  testA 1975 / 1965 / 1957, testB 1798 / 1780 / 1738.
* `same_category` deliberately **not built** - C4 is out of scope by the freeze,
  so no GDINO same-category artifact exists anywhere on disk.
* Provenance archived: `manifest_meta/random_*.meta.json`.

## 2. OpenCLIP feature caches

Frozen command: `--bank cache/proposals_gdino.h5 --out cache/features_gdino
--batch-size 128 --precision fp16 --skip-global --resume`.

| item | GDINO | frozen DETR run |
|---|---|---|
| region crops encoded | 1,279,488 (19,992 imgs) | 1,279,488 (19,992) |
| crops/s | 209.7 | 173.0 |
| region seconds | 5,182 | 7,398 |
| invalid crops | **0** (rate 0.0) | 0 |
| text cache | 141,564 sentences, 80.5 s | identical corpus |
| peak VRAM | 833 MB | - |
| checkpoint sha256 | `1bd3c7172de5b207…` | **same** (ViT-B/32 laion2b_s34b_b79k) |
| `assert_openclip_identity` | **PASS** | PASS |

`region_features.h5` is 1,317,691,272 bytes - byte-size identical to the DETR
region cache (same N, same images, same fp16/512-dim layout).

Note on `--skip-global`: the freeze demands parameters "identical to the frozen
DETR feature run", and `cache/features_detr_r50` carries **no** `global_features.h5`
(the RPN-era cache does). A code survey of the whole C1 path
(`ccg.semantic.*`, `ccg.reliability.*`, `p1_replication_core`, `p1_f3`) returns
zero readers of global features, so the global stage was an unused 43-minute
detour; the run was relaunched with `--skip-global --resume` (streaming resume
picked up at `cached=3008`, at most 64 unflushed images redone). The first,
aborted attempt also produced a useful operational finding: without `python -u`
a redirected driver log stays empty, and an open HDF5 `.tmp` is exclusively
locked (its directory-entry size/mtime lag), so progress must be read from
unbuffered log lines plus final file sizes.

## 3. Frozen inference (GDINO, C1 only)

`scripts/p1_frozen_inference.py --family GDINO` - 4 random jobs x 3 frozen B3
seeds = 12 prediction files (`p1__GDINO__random_k{5,10,20,50}__b3_seed{1,2,3}.npz`,
gitignored like the P1 ones). Static zero-fit scan and the `fit_is_forbidden`
tripwire ran inside the driver; `new_training_parameters = 0`.
RPN was **not** re-run: its frozen predictions are reused as stored.

Attrition, the three families side by side (expression-level pooled test):

| family | target-present rows | eligible K5/K10/K20/K50 | common-K50 | K50 availability | images | refs |
|---|---|---|---|---|---|---|
| RPN | 10,425 | 10425 / 10425 / 10425 / 10,286 | 10,286 | 0.9867 | 1,490 | 3,647 |
| DETR | 10,601 | 10601 / 10601 / 10570 / 9,665 | 9,665 | 0.9117 | 1,421 | 3,440 |
| **GDINO** | **10,547** | 10547 / 10547 / 10547 / **10,402** | **10,402** | **0.9863** | 1,489 | 3,695 |

* Frozen stop condition "common-K50 cohort < 8000 -> STOP" is satisfied with
  margin: **10,402 >= 8,000**.
* GDINO shows **no attrition before K50** (like RPN, unlike DETR), because every
  bank image carries exactly 64 valid proposals - the cardinality-side strength
  the probe already measured, now confirmed at manifest/cohort level.
* The driver records `same_category: "out_of_scope (V2-P2-C1 freeze: C1 only)"`
  for GDINO, i.e. the C4 prohibition is enforced in code, not just in prose.

## 4. Frozen-artifact protection

The driver writes its report to `results/v2_proposal_robustness/inference_report.json`,
which is P1's committed artifact. The GDINO report was moved to
`p2_gdino_stage/inference_report.json` and P1's file restored from git, verified
byte-identical: sha256 `2a013ef6cb6703560b8f715aed096ab4c7364e8659f3b28b3fb6932d97842e2f`
(before and after).

## 5. Next (commit D)

`scripts/p2_c1_cardinality.py` (already committed in B, not yet run) produces
`C1_GDINO` by importing the frozen F4 numeric stack verbatim, re-asserting the
RPN/DETR point rows against `p1_f4_c1/c1_point.csv` to 1e-12, rendering the
three-family figures and the descriptive intersections.
