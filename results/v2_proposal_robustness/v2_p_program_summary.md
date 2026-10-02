# V2-P program summary - the proposal-family robustness axis

**Status: axis closed.** This file is an *index*, not a result. It adds no measurement,
no re-analysis, no new metric, no new label and no new number: every row either quotes
a frozen artifact or points at one. Its purpose is to state, in one place, what the
V2-P axis is entitled to claim after V2-P1, V2-P2-C1 and V2-P2-M, what it has withdrawn,
and what it leaves open.

* **Axis scope.** Does the frozen reliability finding survive when the *proposal family*
  changes? RPN (class-agnostic, frozen reference) -> DETR-R50 (set-based, one-stage) ->
  Grounding DINO base (class-prompt, open-vocabulary).
* **Not to be confused with `V2-M`.** "V2-P2-M" is the mechanism diagnostic *inside the
  V2-P axis*. The `V2-M` protocol in `docs/experiment_log.md` and
  `results/v2_local_competition/` is the separate Local Competition Reliability Module.
* **Machine-checked.** `tests/test_v2_p_program_summary.py` verifies that every verdict
  label and every headline number quoted below equals the artifact it cites, that every
  path mentioned here exists, and that the forbidden wordings of section 5 do not appear
  in this file's claim statements.

---

## 1. What was held fixed across the whole axis

Nothing below was re-tuned between families; the P2 freeze records each item as
`immutable_reuse_from_v2_p1` and the RPN/DETR reuse is verified numerically
(`p2_c1_verdict.json: rpn_detr_reuse_verbatim_check = "PASSED (all frozen F4 point rows
matched to 1e-12)"`).

| invariant | value |
|---|---|
| backbone | single frozen OpenCLIP ViT-B/32 |
| scorer | frozen B3 `IndependentMLPScorer`, seeds {1, 2, 3}, evaluated per seed then mean +/- std |
| calibration | per-seed global temperature, `global_T_corrected` variant; never refit per family |
| cohort | common-K50 (target present AND >= 49 valid distractors), nested K in {5, 10, 20, 50}, baseline K5, primary pair (5, 50) |
| candidate construction | exact V1 seeded-random regime (`--regime random --seed 20260927`, target slot 0); detector confidence ordering is bank provenance only (amendment **P1-A0**) |
| proposal supply | N = 64 proposals per image in **all three** banks |
| bootstrap | image-cluster paired, 5 000 replicates, seed 0, 95 % percentile CI |
| metrics | Accuracy / AUROC / E-AURC / RER@50 / RER@80 via `ccg.reliability.evaluate` |
| new training | 0 parameters, 0 new checkpoints; no NMS, no truncation change, no threshold change |

Cohort sizes (pooled testA + testB, common-K50): RPN 10 286 / DETR 9 665 / GDINO 10 402
expressions. DETR attrition 10 601 -> {10 601, 10 601, 10 570, 9 665}; RPN 10 425 -> 10 286.

## 2. Claim ledger

| ID | claim, stated exactly as supported | verdict | artifact | what it does **not** say |
|---|---|---|---|---|
| P-01 | Candidate-cardinality degradation replicates under DETR-R50 proposals | **YES** | `p1_f4_c1/f4_verdict.json` | not that DETR is worse; both families degrade |
| P-02 | Hard-semantic amplification replicates under DETR-R50 proposals (manipulation valid 3/3 seeds) | **YES** | `p1_f5_c4/f5_verdict.json` | no causal claim; same-category is a GT-assisted diagnostic regime |
| P-03 | The core findings survive the proposal-family change | **CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST** (YES) | `p1_f5_c4/f5_verdict.json`, `p1_final_summary.json` | "survive" = replicate, not "explained" |
| P-04 | Higher absolute grounding accuracy does not remove cardinality degradation (DETR K5 0.8806 > RPN 0.7895, yet C1 replicates) | interpretation case, `c1_replicated = true` | `p1_final_summary.json: interpretation_case` | explicitly "(no causal claim)" |
| P-05 | C1 also replicates on a class-prompted open-vocabulary family, and there it is several times larger | **YES** - `gdino_C1_PROPOSAL_FAMILY_REPLICATED` | `p2_c1_gdino/p2_c1_verdict.json` | C1 only; no GDINO hard-regime statement exists anywhere |
| P-06 | The GDINO amplification cannot be a candidate-**count** effect | arithmetic, not a test | three banks hold `top_n = 64`; identical presented K | says nothing about *which* composition channel does carry it |
| P-07 | The candidate-**composition** channel measured here does not carry the between-family amplification | **MECHANISM_PARTIAL** (rule a PASS 2/3, rule b FAIL) | `p2_m_mechanism/p2_m_verdict.json` | not a refutation of composition in general; see section 6 |
| P-08 | Within family, per-expression same-class distractor count co-moves weakly with per-expression pool-growth harm in the two deep families | rho(H1c, R1) DETR **+0.0894** [+0.0611, +0.1191], GDINO **+0.0525** [+0.0242, +0.0808], RPN -0.0144 [-0.0383, +0.0095] | `p2_m_mechanism/p2_m_verdict.json`, `mechanism_association.csv` | descriptive; every cell of the H1c grid has abs(rho) <= 0.09, no functional form claimed (the H2a rows, outside every decision rule, are larger - see section 6.3) |
| P-09 | `H1b_gained_flip` is exactly 0 in every family and seed, so `H1c == H1a` == the family accuracy drop | structural identity | `mechanism_point.csv` (H1b = 0.0), `hard_scores.py` | not an empirical finding; consequence of nested pools + per-pair scoring |

## 3. Headline numbers, one row per family

K5 -> K50, mean over the three frozen seeds, each family on its own common-K50 cohort
(`c1_point.csv` / `f4_verdict.json`; RPN's frozen-artifact value in brackets).

| family | K5 acc | K50 acc | dAUROC | E-AURC worsening (relative) | RER@50 drop | C1 |
|---|---|---|---|---|---|---|
| RPN | 0.7895 | 0.4290 | +0.0517 [0.0522] | 2.257 | 0.4660 | YES |
| DETR | 0.8806 | 0.5849 | +0.0852 | 4.555 | 0.2943 | YES |
| GDINO | 0.8463 | 0.4448 | **+0.2366** | **8.654** | **0.6678** | **YES** |

"E-AURC worsening (relative)" is the fractional increase of E-AURC from K5 to K50
(+225.7 %, +455.5 %, +865.4 %), not a ratio of raw values; it is the frozen Route-B
quantity, reproduced from `c1_point.csv` to 1e-3.

Derived ratios of the published means: GDINO/DETR = 2.78, GDINO/RPN = 4.58 (the
"~2.8x / ~4.6x" of `proposal_family_final_table.md`). On matched-expression
intersections the ordering survives cohort differences (descriptive, no CI): GDINO u RPN
n = 10 125 -> 0.2312 vs 0.0534 (gap +0.1778); GDINO u DETR n = 9 589 -> 0.2212 vs 0.0863
(gap +0.1350).

Mechanism-side family means on the same cohorts (`p2_m_mechanism/mechanism_point.csv`):

| family | R1 same-class distractors | R2 unmatched fraction | H1c net harm | rho(H1c, R1) |
|---|---|---|---|---|
| RPN | 11.51 | 0.564 | 0.3605 | -0.014 |
| DETR | 8.71 | 0.611 | 0.2957 | +0.089 |
| GDINO | **6.23** (lowest) | **0.718** (highest) | **0.4016** (highest) | +0.053 |

## 4. Withdrawn and corrected (dated ledger)

| item | date | what was said | what is now said | where |
|---|---|---|---|---|
| **P2-A0** | 2026-10-02 (commit `0c60104`) | GDINO's 64 slots are filled with *same-class* redundant boxes and that is what amplifies C1 - "the strongest evidence in the program so far for the semantic-competition mechanism" | **withdrawn**. GDINO carries the fewest same-class distractors of the three banks; R1-quintile matching leaves the gap unshrunk (shrinkage -0.180 vs RPN, -0.065 vs DETR, against the frozen 0.50 bar) | `p2_amendment_history.csv`; marked pointer + section 9 in `p2_c1_gdino/analysis_report.md`; rewritten closing paragraph in `proposal_family_final_table.md` |
| P2-A0, surviving part | same | the candidate-*count* exclusion | **kept** - it is arithmetic (64 proposals per image in all three banks, same presented K) | as above |
| P2-A0, untouched part | same | every confirmatory C1/C4 number of sections 1-5 of the P2-C1 report | **unchanged**; P2-M is `DESCRIPTIVE_MECHANISM` and by its own classification may not create, strengthen or weaken an existence claim | `p2_m_mechanism_config_freeze.json: classification_meaning` |

No gate, threshold, bin edge, seed, cohort rule or verdict label was edited after its own
results became visible. P2-A0 is a *prose* correction of an interpretation, executed after
the correcting measurement was pre-registered (`87009ac`) and run; the classification of
each measurement is unchanged.

## 5. Boundary register - wording this axis never licenses

* No detector ranking. Absolute accuracy differs; C1 is a statement about the *shape in K*.
* No causal language. P2-M is associational within family and stratified across families;
  `honesty_clause` in the P2-M freeze applies verbatim.
* No GDINO hard-regime / C4 statement of any kind. The E1 probe failed the frozen
  same-category supply gate (0.6892 < 0.85), so a GDINO C4 would have been structurally
  uninterpretable; `out_of_scope_confirmed` in `p2_c1_verdict.json` records the prohibition,
  and no `same_category` manifest, `hard_k5` job or C4 verdict for GDINO exists on disk.
* No CI on the shrinkage ratio (the freeze forbids promoting it to a test); no CI on the
  matched-intersection gaps (defined as secondary and descriptive).
* "Not established" is not "explained by something else". P2-M leaves the between-family
  gap unaccounted for by the channels it measured; that is not evidence for another channel.
* No cross-visual-domain claim. Every V2-P family shares the COCO image universe; the
  external-validity language of V2-D1 / V2-D2 is not this axis's to use.
* No new scorer, feature function, seed, dataset, bank, threshold or NMS appears anywhere
  in this axis; the P2-M smoke path (`--limit`) writes no artifacts.

## 6. Open and explicitly unclaimed (each needs its own pre-result freeze)

1. **The `nothing-in-COCO` profile.** GDINO's distinctive measured property is the highest
   share of pool members matching no non-crowd COCO object (R2 0.718) together with
   near-zero geometric redundancy. P2-M cannot call this a mechanism: within GDINO
   rho(H1c, R2) = +0.0036 [-0.024, +0.032], and no frozen primitive here measures semantic
   confusability of same-object duplicates directly.
2. **The confidence-signal collapse.** GDINO's raw top1-top2 margin predictor falls to
   0.587 at K50 (near chance) while its raw top-1 ranking degrades least among the deep
   families. Reported in `c1_secondary_raw_auroc.csv`, never gate-bearing, never explained.
3. **`H2a_delta_margin` associations are an order of magnitude larger than the H1c ones and
   flip sign across families** (GDINO -0.251 / DETR +0.175 against R4-mid). Reported, not
   decided on; a family-dependent sign is a reason to freeze a new protocol rather than to
   pick a direction post hoc.
4. Dose-response across families, target-absence regimes, E2 query-conditioned GDINO
   top-K, and further proposal families (DINO-DETR / DETR-R101) remain out of scope by the
   P2-C1 freeze.

## 7. Lineage

| stage | protocol | classification | config freeze (authored before its own results) | verdict artifact | prose report |
|---|---|---|---|---|---|
| bank + scorer compatibility | V2-P1 F0-F3 | engineering | - | `p1_detr_r50/proposal_summary.json`, `phase_b_feasibility_audit.json` | `p1_detr_r50/protocol.json` |
| C1 replication | V2-P1 P1-F4 | CONFIRMATORY | `p1_f4_f5_config_freeze.json` (`1285e95`) | `p1_f4_c1/f4_verdict.json` | `proposal_family_final_table.md` |
| C4 replication | V2-P1 P1-F5 | CONFIRMATORY | same freeze | `p1_f5_c4/f5_verdict.json` | same |
| axis summary | V2-P1 | CONFIRMATORY | - | `p1_final_summary.json` (`7441ab1`) | `proposal_family_final_table.md` |
| third-family probe | V2-P2 B0/B1 | feasibility | audit + probe gate set | `p2_gdino_probe/probe_report.json` | `phase_b_feasibility_audit.md` |
| C1 on GDINO | V2-P2-C1 | CONFIRMATORY | `p2_c1_gdino_config_freeze.json` (`91f2758`) | `p2_c1_gdino/p2_c1_verdict.json` | `p2_c1_gdino/analysis_report.md` (+ section 9 addendum) |
| mechanism | V2-P2-M | DESCRIPTIVE_MECHANISM | `p2_m_mechanism_config_freeze.json` (`87009ac`) | `p2_m_mechanism/p2_m_verdict.json` | `p2_m_mechanism/mechanism_report.md` |
| axis close-out | this file | index | - | `p2_amendment_history.csv` | `v2_p_program_summary.md` |

Commit chain: `2249206` -> `bcdc2cd` -> `8a831f1` (P1-A0) -> `1285e95` -> `feb40d8` ->
`6496119` -> `7441ab1` -> `9dd14e0` -> `6cfc437` -> `75ce48d` -> `91f2758` (P2-C1 freeze)
-> `6c1850c` -> `e9cd139` -> `7905aad` -> `87009ac` (P2-M freeze) -> `0c60104` -> `9b010e6`.

## 8. Paper-facing sentences that are safe to copy

1. "Under a frozen scorer stack and identical candidate-construction regime, candidate
   cardinality degrades selective reliability for three different proposal families -
   class-agnostic RPN, set-based DETR-R50, and a class-prompted open-vocabulary detector -
   with delta AUROC (K5-K50) of +0.052, +0.085 and +0.237 respectively."
2. "All three proposal banks supply exactly 64 candidates per image and all three families
   are evaluated at the same presented K, so the amplification is not a candidate-count
   effect."
3. "A pre-registered descriptive diagnostic establishes neither more nor less than this:
   within the two deep families, same-class pool redundancy co-moves weakly with
   per-expression harm (Spearman rho +0.089 and +0.053, CIs excluding 0; RPN none), but
   matching expressions into redundancy quintiles leaves the between-family gap unshrunk
   (shrinkage -0.18 against a 0.50 bar). The composition account measured here does not
   explain the amplification."
4. "Hard-semantic amplification replicates under DETR proposals (amplification +0.0147,
   CI low +0.0108, manipulation valid in 3/3 seeds), so the two core effects of the mainline
   are not artifacts of one proposal family."

Downstream text that this axis made stale, and has now corrected: `docs/final_result_summary.md`
used to list "single proposal family" / "单一 proposal 家族" among its limitations. That sentence
predated V2-P. It was replaced when this axis closed: the material bank now carries a V2 robustness
section (V2-G, V2-D1, V2-D2, V2-M/V2-M3, V2-P) in both its Chinese and its English summary, its
limitations state this axis's actual boundary (section 5 above - three families sharing one image
domain and one frozen scorer stack, GDINO in C1 only at probe 0.6892 < 0.85, the amplification
mechanism left open), and it records the P2-A0 withdrawal. The correction was made in that document
itself; this index still does not duplicate its numbers, and the paper-facing text must keep citing
the per-axis artifacts under `results/` rather than this file, because only the artifacts carry the
frozen protocol each value was produced under.

## 9. Integrity record and reproduction

* Tests: 1 034 passed / 2 skipped with this close-out (1 019 / 2 at `0c60104`, plus the 15
  traceability tests in `tests/test_v2_p_program_summary.py`). The V2-P axis is covered by
  `tests/test_p1_detr_proposals.py`, `tests/test_p1_f3_route_f.py`,
  `tests/test_p1_replication.py`, `tests/test_p2_gdino.py`,
  `tests/test_p2_c1_plumbing.py`, `tests/test_p2_c1_verdict.py`,
  `tests/test_p2_m_plumbing.py` (14 tests) and this file's traceability tests.
* Frozen artifacts untouched by V2-P2: `inference_report.json` sha256
  `2a013ef6cb6703560b8f715aed096ab4c7364e8659f3b28b3fb6932d97842e2f` verified unchanged
  after the P2 runs; RPN/DETR point rows reproduced to 1e-12.
* Zero-cost close-out: this axis consumed no additional model training. P2-M ran in
  330.28 s on CPU with 0 model forward passes.
* Downstream correction: `docs/final_result_summary.md` (paper material bank) had its
  "single proposal family" limitation replaced by this axis's real boundary and gained a V2
  robustness section, together with the P2-A0 withdrawal. No number in it was touched, and
  `results/final_registry/` was not regenerated - the bank now states that its V2 values come
  from the per-axis artifacts. This file remains an index and does not restate the bank.

```
E:\conda\envs\deepminer\python.exe    scripts/p1_frozen_inference.py --family DETR|GDINO
E:\conda\envs\deepminer\python.exe -u scripts/p2_c1_cardinality.py          # C1 gate, 3 families
E:\conda\envs\deepminer\python.exe -u scripts/p2_m_mechanism.py             # mechanism, ~5.5 min
E:\conda\envs\deepminer\python.exe -m pytest tests/test_v2_p_program_summary.py -q
```

Full per-stage commands: `p2_c1_gdino/analysis_report.md` section 8,
`p2_m_mechanism/mechanism_report.md` section 7.
