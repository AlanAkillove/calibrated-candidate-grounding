# Current result summary — Research Repair v1

Research Repair v1 is complete under the frozen execution protocol. Formal artifacts,
full tests, stored-array arithmetic and immutable inputs passed acceptance.
M3's B0/B1/B2 historical results could not be recovered within the unchanged
1e-9 point tolerance; their independent failure evidence is retained, with no new
effect CI. See `results/research_repair_v1/STATUS.json` and the evidence index.

The pre-repair material bank is preserved in
`docs/appendix/final_result_summary_pre_repair.md`. Its historical gate labels and
intervals are records of the prior analysis, not current scientific conclusions.
The historical `results/final_registry/` has not been regenerated for V2 or repair;
original mechanism sources remain in `results/v2_rq4_mechanism/m1_transition_confidence/`.

## Scientific scope and questions

GT-assisted target-present candidate grounding within the shared COCO visual domain.
RQ1: expansion effects on accuracy, discrimination, calibration and selective utility.
RQ2: score extrapolation limits and query–crop versus candidate–candidate information.
RQ3: four-corner accounting, paths, interaction and explanation limits.

E-AURC and RER remain accuracy-dependent selective utility measures. Temperature
preserves within-expression argmax but can change cross-expression MSP ranking.
Failure of tested score models does not establish insufficient score information.
Small symmetric Shapley label components may conceal opposing path effects.
Exact statistical accounting decomposition, not a causal identification of confidence
ranking changes, is the mechanism interpretation boundary. H1c measures accuracy
harm; it cannot stand in for AUROC-gap explanation. Threshold-unmatched proposals
are unmatched to the annotations used, not necessarily empty background.

## Training and provenance

V2-G freezes encoders but trains corresponding scorer/reliability heads. V2-P and
historical mechanism accounting reuse a frozen scorer. V2-M/M3 train or fit models.
RefCOCO language transfer shares images; strict RefCOCOg is image-disjoint within
COCO and its invalid hard manipulation cannot confirm amplification.
Full pre-result configuration freezes, input identity freezes (D1/D2), retrospective
corrections and D2 target redefinition have different evidential status. A staged
protocol does not erase adaptivity after observing prior test results. Repair analyses
are result-driven supplements, not independently preregistered confirmation.

<!-- artifact-derived-repair-evidence -->

## Artifact-derived evidence

Machine status: **COMPLETE**; completed=true.
The current formal export contains 6646 stored estimates from 59 jobs.
Only stored 5000-draw, seed-0, 95% image-cluster jobs enter that export.

- [Formal statistical tables](../results/research_repair_v1/publication/statistics.md)
- [Main metrics and proposition tables](../results/research_repair_v1/publication/main_tables.md)
- [Complete estimate export](../results/research_repair_v1/publication/formal_statistics.csv)
- [Required scope coverage](../results/research_repair_v1/publication/coverage.md)
- [Historical and repaired operational gates](../results/research_repair_v1/publication/gates.md)
- [Full minus S+Q paired evidence](../results/research_repair_v1/publication/information.md)
- [Candidate category, object and availability audit](../results/research_repair_v1/publication/candidates.md)
- [Four corners, paths and interaction](../results/research_repair_v1/publication/mechanism.md)
- [Source identity and rendering record](../results/research_repair_v1/publication/evidence.json)
- [All experimental artifacts and acceptance state](../results/research_repair_v1/registry/index.md)

Incremental V signal is not uniformly resolved across reported cells. The central claim is limited to the measured cellwise effects; no general candidate-interaction conclusion follows.

Verified candidate audit: 138 category mismatches among 21026 expressions. Sensitivity stage: **FORMAL_COMPLETE**; independent raw numerical QA: **PASS**.

Observed label path means have opposite signs in RPN, DETR, GDINO. The signed Shapley mean can conceal this cancellation; the accounting does not identify a cause.
