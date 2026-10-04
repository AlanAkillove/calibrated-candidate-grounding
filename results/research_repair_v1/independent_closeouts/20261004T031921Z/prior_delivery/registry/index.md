# Research Repair v1 evidence index

Current machine status: **COMPLETE**; completed=True.

This is a result-driven supplementary repair. Original inputs remain frozen.
Full configuration freezes, input identity freezes, post-result corrections,
and target redefinitions are distinct provenance categories.

| Scientific question / axis | Historical evidence | Repair tables | Stage |
|---|---|---|---|
| RQ1 main uncertainty and temperature | Phase05/1/1F, RefCOCOg, V2-G/D/P, LCR/M25/M3 | [statistics](statistics_tables.md) | COMPLETE_59_JOBS_6646_ESTIMATES_23_TERMINAL_GATES_M3_UNVERIFIABLE |
| RQ2 feature sources and ID/OOD | Phase05 and Phase1/1F | [information](information_tables.md) | COMPLETE_9_CALLS_1860_ESTIMATES_RAW_QA_PASS |
| Candidate category and proposal/object audit | frozen manifests and proposal banks | [candidates](candidates_tables.md) | COMPLETE_15_GROUPS_24_MAPPINGS_1404_ESTIMATES_METADATA_RAW_QA_PASS |
| RQ3 four corners, paths and interaction | V2-P2-M and RQ4-M1 | [mechanism](mechanism_tables.md) | COMPLETE_45_ESTIMATES_RAW_QA_PASS |

The historical [V1 registry](../../final_registry/) was not regenerated.
V2 backbone/data/proposal/model evidence and the historical mechanism record
retain their per-axis paths in the indexed repair tables. Gates remain
operational history; current conclusions require corrected effects and limits.

- [Artifact identities](artifact_index.json)
- [Frozen baseline manifest](../input_manifest.json)
- [Machine state](../STATUS.json)
- [Logs and acceptance evidence](../logs/)
- [Test correction rationale](test_corrections.md)

- [Historical report corrections](historical_corrections.md)

- [Artifact-derived statistical export and figure](../publication/statistics.md)
- [Publication source identity](../publication/evidence.json)

Regenerate this index without training or modifying the baseline:

```powershell
& 'E:/conda/envs/deepminer/python.exe' -B scripts/build_repair_registry.py
```
