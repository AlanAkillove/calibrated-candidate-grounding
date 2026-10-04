# V2-M legacy gate and provenance review

Audit time: `2026-10-03T13:48:31.699771+00:00`. Read-only review: no legacy runner, repair runner, fit, or bootstrap was changed/run.

## M1

Predicate in `scripts/run_v2m_m1.py::_build_gate` (source SHA in the ledger): SameCategory-K5 LCR−E1b AUROC delta ≥ 0.010 with CI low > 0, plus E-AURC reduction ≥ 5% or RER@50 gain ≥ 3 pp, and Random-K5 guard delta ≥ −0.005. It is a conjunction.
Historical decision is **NO-GO**: AUROC delta -0.029442, CI low -0.032442; E-AURC reduction -0.168746; RER50 gain -4.435 pp. Random guard passes at -0.001267. The audit reports passed=True, exact point/bootstrap rebuild (both max delta 0).

## M2 m=8 structure and selective branches

The m=8 primary rule is conjunctive: LCR−E1b AUROC ≥ 0.010 and CI low > 0; LCR−Aggregate-MLP AUROC > 0 and CI low > 0. Historical M2 is **NO-GO** because the first contrast is −0.001858 (CI [−0.005304, 0.001643]); structural contrast is +0.035743 (CI [0.030282, 0.041361]) and passes. This isolates the negative result to the proposed LCR advantage over curriculum E1b despite its positive contrast against the matched Aggregate-MLP capacity control.
Selective metrics are a separate diagnostic, not an M2_GO condition: E-AURC reduction -0.033957, RER50 gain -0.551 pp, RER80 gain 0.739 pp. The actual `method_signal_mixed` predicate also requires the LCR−E1b AUROC condition and all three selective measures < 0, so it is false. No separate numeric selective threshold is listed in the M2 preregistration. Gate contribution LCR−noGate is -0.002241, CI [-0.004286, -0.000181], significantly favoring noGate. M2 audit passed with exact point/bootstrap rebuild.

## M2.5 diagnostic pattern

`_pattern` is not a success gate. It defines low-severity random specialization as any of m0/m2 CI_high < 0 and high-severity curriculum specialization as m8 CI_low > 0. Both hold: m0 delta(Curriculum−Random) = −0.008902, CI [−0.012809, −0.005096]; m2 CI overlaps zero; m8 = +0.025830, CI [+0.022435, +0.029320]. Result: **SPECIALIST TRADEOFF PRESENT**, a diagnostic that permits consideration of a new M3 amendment. M2.5 audit passed.

## M3 case definition

B0 is labeled POST-HOC DEVELOPMENTAL and `no_gate=true`; its Adaptive−Static macro result is +0.0000893, CI [+0.0000151, +0.0001647], but it is not confirmatory. Frozen V2-M3.1 evaluates B1 and B2 separately: primary macro AUROC delta ≥ 0.003 and CI low > 0; both extreme-safety deltas ≥ −0.003; pass is primary AND safety. Selective support (E-AURC relative improvement ≥ 3% OR RER50 gain ≥ 1 pp) labels a passing result only and does not enter pass.
B1 fails primary (delta 0.000148, CI low -0.000056) and m0 safety; B2 fails primary (delta 0.000430, CI low -0.000388) and both safety checks. Neither passes. Cross-backbone result is **ADAPTIVE MIXTURE NOT SUPPORTED** (0/2 pass, stop=True).

## Refresh runner audit

`scripts/refresh_repair_gate_decisions.py` does **not** recalculate M1/M2/M2.5/M3 decisions. It reads only the repair summary/old gate-decision file plus Phase05, Phase1, Phase1F, and RefCOCOg source artifacts. For M1/M2/M2.5/M3 it supplies hardcoded provenance strings, iterates existing repair rows, copies them, then marks the new decision pending. It does not read the referenced V2-M JSON, call `_build_gate`/`_pattern`/`_backbone_gate`, or populate missing old fields. The current repair decision rows have null `old_decision`/`old_thresholds`, so refresh will preserve nulls.

Two M2 provenance pointers are invalid: `gates/M2_SELECTIVE_SIGNAL` should be `gates/selective_metrics`; `gates/M2_GATE_CONTRIBUTION` should be `gates/gate_contribution`. M3 metadata points at `::gate`, but the actual predicates are `_backbone_gate` and `_cross_backbone_verdict`. The script is an annotation/pending scaffold for these scopes, not an evaluator.

## Source and artifact SHA-256 ledger

| Path | SHA-256 |
|---|---|
| `results/v2_local_competition/gate.json` | `d1138bc1b801f0a0b1d52c2c6c5d43166029e3bf99d73a6ac44d8ddbf4337091` |
| `results/v2_local_competition/m1_audit.json` | `d4fa5480aab387108b8a66c8d9e7f5cfb282ca1dfb5110a5fdb0b1d44707812e` |
| `results/v2_local_competition/protocol.json` | `90085940f75ed8cbf6ac3c2f72219b9ff34bde80c97aad89362580bf5fedd88b` |
| `scripts/run_v2m_m1.py` | `438928d78ebb63c06874efcd0a1dedf04d410f79ff13458de20ad1a06f992a0f` |
| `scripts/audit_v2m_m1.py` | `73bb4bcd69970e7322dd947eddc73007c9f71a8d879ec176a7ca43d326c72d44` |
| `results/v2_local_competition/m2_curriculum/gate.json` | `183b3314f5638a394c66eda4eea9b5af70a501372a3364a08284ce11c80a1944` |
| `results/v2_local_competition/m2_curriculum/m2_audit.json` | `d37357cdae3b0f586037efbb1de31105d37ec1c89f17183ca52a626066656e3d` |
| `results/v2_local_competition/m2_curriculum/metadata.json` | `761cb59404435387ad3362fadbd317d6118abdb58adac4e9befb7881ad4a8599` |
| `scripts/run_v2m_m2.py` | `f1b01a9f073710e14900fb825e88b4ffacba1ad822030515542720984495336a` |
| `scripts/audit_v2m_m2.py` | `5df6ba607e54edf19d608bba3e78a296808579972bee7958c8e043d08a6f1078` |
| `results/v2_local_competition/m25_specialist_audit/verdict.json` | `0ae386ce479f67deeea628243a96335122574eafc16c30974614668841d0e766` |
| `results/v2_local_competition/m25_specialist_audit/m25_audit.json` | `0d32e6e8325982836a28fda09777a918e9e7efc20c68d42d72dc40ca7f0da9fb` |
| `results/v2_local_competition/m25_specialist_audit/metadata.json` | `c46d682ec5637d537719e396b0a0c13cd5913b91d46beb5d7cd174053826ad66` |
| `scripts/run_v2m_m25.py` | `d54528ba808b0e5beae57704409f51233ca9c2383210aff5d20f0ad8d8d3899a` |
| `scripts/audit_v2m_m25.py` | `379c15e08d1b742e62971192fb9845350b51eca737f4cebf18ca3144c8816103` |
| `results/v2_local_competition/m3_mixture/b0/summary.json` | `1286a4448cc2313a0c79eb272b76a90a100a7ab6e367b0c5bf7f0181e14e727a` |
| `results/v2_local_competition/m3_mixture/b0/m3_b0_audit.json` | `ce09fc2519ec245f066462993ed0f224d5b86f0b3f9ca97e5226ba9bc9ffcc7a` |
| `results/v2_local_competition/m3_mixture/amendment_v2m31.json` | `d4e75eebbb9db3cd4e104d182d386533f0df62e0e6d5ec0933270744d81b7c85` |
| `results/v2_local_competition/m3_mixture/protocol_m3.json` | `d5e17d6bba56471bf9382c2c3ebf534a98550ed9dbd5fe1f826f2831e9a371da` |
| `results/v2_local_competition/m3_mixture/conf/b1/gate.json` | `f826a089f9e37bb85957fbae8b704e39a0f2cca4f11f450b845e5877efb0a2b9` |
| `results/v2_local_competition/m3_mixture/conf/b2/gate.json` | `c953c9d9dca7af90bbb4a217195e35a3f2ef4c9b0eed4257bf5cba963b77352f` |
| `results/v2_local_competition/m3_mixture/conf/cross_backbone_verdict.json` | `a6954a7f7ff9bef6afb0f1a67f9f3391169f697d6d3ca1025b69cb9f3f81384a` |
| `scripts/run_v2m_m3_b0.py` | `580c19e53b119a106c44fbd4b9806223a13fefa2548902269de3ad968a11348e` |
| `scripts/run_v2m_m3_conf.py` | `a2aae27c54ebb72d9df7d73816e5a3fa85f4cf3f730bcb8bba5f8315a138eee0` |
| `scripts/refresh_repair_gate_decisions.py` | `2bb4783c9fc6d9638ddae48dfe29c1579a34c10d2f4d91690a73a078f75d1d61` |
| `results/research_repair_v1/statistics/gate_decisions.json` | `1f0037c0193b788a5a2527987641331b790ae61c488039a312db1a7279ab0a09` |

## Exact JSON pointers

### m1
- `results/v2_local_competition/gate.json#/gates/M1_GO`
- `results/v2_local_competition/protocol.json#/M1_gates/GO_gate`
- `results/v2_local_competition/protocol.json#/M1_gates/primary_cell`

### m2
- `results/v2_local_competition/m2_curriculum/gate.json#/gates/M2_GO`
- `results/v2_local_competition/m2_curriculum/gate.json#/gates/selective_metrics`
- `results/v2_local_competition/m2_curriculum/gate.json#/gates/gate_contribution`
- `results/v2_local_competition/m2_curriculum/gate.json#/gates/capacity_case`
- `results/v2_local_competition/protocol.json#/M2_preregistration/gate`

### m25
- `results/v2_local_competition/m25_specialist_audit/verdict.json#/delta_definition`
- `results/v2_local_competition/m25_specialist_audit/verdict.json#/severity/m0/auroc_correct`
- `results/v2_local_competition/m25_specialist_audit/verdict.json#/severity/m2/auroc_correct`
- `results/v2_local_competition/m25_specialist_audit/verdict.json#/severity/m8/auroc_correct`
- `results/v2_local_competition/m25_specialist_audit/verdict.json#/pattern`
- `results/v2_local_competition/m25_specialist_audit/verdict.json#/conditions`
- `results/v2_local_competition/m25_specialist_audit/verdict.json#/interpretation`

### m3
- `results/v2_local_competition/m3_mixture/b0/summary.json#/no_gate`
- `results/v2_local_competition/m3_mixture/b0/summary.json#/label`
- `results/v2_local_competition/m3_mixture/b0/summary.json#/b0_questions/adaptive_vs_static`
- `results/v2_local_competition/m3_mixture/amendment_v2m31.json#/frozen_gates`
- `results/v2_local_competition/m3_mixture/amendment_v2m31.json#/frozen_verdict_rule`
- `results/v2_local_competition/m3_mixture/conf/b1/gate.json#/primary_gate`
- `results/v2_local_competition/m3_mixture/conf/b1/gate.json#/extreme_safety`
- `results/v2_local_competition/m3_mixture/conf/b1/gate.json#/selective_support`
- `results/v2_local_competition/m3_mixture/conf/b2/gate.json#/primary_gate`
- `results/v2_local_competition/m3_mixture/conf/b2/gate.json#/extreme_safety`
- `results/v2_local_competition/m3_mixture/conf/b2/gate.json#/selective_support`
- `results/v2_local_competition/m3_mixture/conf/cross_backbone_verdict.json#/verdict`

