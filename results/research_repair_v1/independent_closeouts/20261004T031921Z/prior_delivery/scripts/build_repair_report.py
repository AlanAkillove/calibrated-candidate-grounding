"""Render publication evidence from stored repair outputs without doing analysis.

This entry point never trains, resamples, alters inputs, or declares completion.
It includes only stored 5000-draw jobs in the statistical evidence export.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_repair_v1"
MARKER = "<!-- artifact-derived-repair-evidence -->"
METRICS = ("accuracy", "auroc_correct", "e_aurc", "rer_at_50")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_statistics(summary: dict) -> list[dict]:
    """No pilot estimates or CI endpoint averaging enter the rendered evidence."""
    rows = []
    for job in summary.get("jobs", []):
        method = job.get("result", {})
        if (job.get("status") != "RECOMPUTED_5000"
                or method.get("n_replicates") != 5000
                or method.get("seed") != 0
                or method.get("ci_level") != 0.95
                or method.get("resample_unit") != "image_cluster"):
            continue
        for estimate in job.get("estimates", []):
            rows.append({
                "job_id": job["job_id"], "scope": estimate.get("scope", job.get("scope")),
                "cohort": estimate.get("cohort", job.get("cohort")),
                "estimate": estimate["name"], "metric": estimate.get("metric"),
                "point": estimate.get("point"), "ci_low": estimate.get("ci_low"),
                "ci_high": estimate.get("ci_high"),
                "n_seeds": len(estimate.get("seed_estimates", {})),
                "seed_standard_deviation": estimate.get("seed_standard_deviation"),
                "n_rows": estimate.get("n_rows"), "n_images": estimate.get("n_images"),
                "n_replicates": estimate.get("n_replicates"),
                "valid_replicates": estimate.get("valid_replicates"),
                "invalid_replicates": estimate.get("invalid_replicates"),
                "anchor_status": estimate.get("anchor_status"),
                "recovery_status": estimate.get("recovery_status"),
                "raw_replicates": estimate.get("raw_replicates"),
                "operation": estimate.get("operation"), "scale": estimate.get("scale"),
                "formula": estimate.get("formula"), "unit": estimate.get("unit"),
                "heldout_test_only": job.get("cohort_metadata", {}).get("heldout_test_only"),
                "cohort_split_counts": json.dumps(job.get("cohort_metadata", {}).get("split_counts", {}), sort_keys=True),
            })
    return rows


def numeric(value: object) -> str:
    if value is None:
        return "unavailable"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return str(value)
    return f"{number:.6f}" if math.isfinite(number) else "unavailable"


def scope_and_gate_evidence(destination: Path, formal_job_ids: set[str] | None = None) -> dict:
    """Expose scope gaps and historical operational decisions in the appendix."""
    evidence = {}
    coverage_path = OUT / "statistics/required_scope_coverage.json"
    lines = ["# Required statistical scope coverage", "",
             "This inventory includes recovery-pending scopes. A registered runner or a completed subset does not establish complete coverage.", "",
             "| Scope | Status | Formal completed / required | Unverifiable recovery jobs | Recovery route | Anchor |",
             "|---|---|---|---|---|---|"]
    if coverage_path.exists():
        coverage, source_hash = json_snapshot(coverage_path)
        evidence.update(coverage_source=coverage_path.relative_to(ROOT).as_posix(), coverage_sha256=source_hash)
        for row in coverage.get("scopes", []):
            complete = (len(set(row.get("job_ids", [])) & formal_job_ids) if formal_job_ids is not None
                        else row.get("formal_jobs_complete", 0))
            cells = (row["scope_id"], row.get("status", "PENDING"),
                     f"{complete} / {row.get('formal_jobs_required', 'unknown')}",
                     ", ".join(item.get("job_id", "unknown") for item in row.get("unverifiable_jobs", [])) or "none",
                     row.get("recovery_method", ""), row.get("anchor_status", ""))
            lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
        for row in coverage.get("scopes", []):
            for item in row.get("unverifiable_jobs", []):
                lines += ["", f"- {row['scope_id']}/{item.get('job_id')}: {item.get('reason_code', 'UNVERIFIABLE')}. "
                          f"Evidence: `{item.get('evidence_path', 'pending')}`. "
                          "This recovery limitation does not supply an effect estimate or a gate pass."]
                relative = item.get("evidence_path")
                failure_path = (ROOT / relative).resolve() if relative else None
                if failure_path and failure_path.is_relative_to(OUT.resolve()) and failure_path.is_file():
                    failure, failure_hash = json_snapshot(failure_path)
                    if failure.get("schema") == "research-repair-v1-recovery-failure-v1":
                        evidence.setdefault("recovery_failures", []).append({
                            "job_id": item.get("job_id"), "path": relative, "sha256": failure_hash})
                        lines += [f"  Original tolerances: `{json.dumps(failure.get('original_tolerance'), sort_keys=True)}`. "
                                  f"Observed error: `{failure.get('observed_error', 'see seed records')}`. "
                                  f"Failure reason: {failure.get('failure_reason', 'see source record')}"]
    else:
        lines += ["", "No required-scope inventory is available."]
    (destination / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decisions_path = OUT / "statistics/gate_decisions.json"
    lines = ["# Historical operational gates and repaired decisions", "",
             "Original thresholds remain fixed. These are operational predicates, interpreted alongside the scientific scope and metric limitations.", "",
             "A GO through an E-AURC/RER branch, including A5.4 Route A, does not establish an accuracy-independent "
             "discrimination decline or a score-information limit.", "",
             "| Scope | Gate | Original decision | Repaired decision | Status | Original thresholds | Predicate source |",
             "|---|---|---|---|---|---|---|"]
    if decisions_path.exists():
        decisions, source_hash = json_snapshot(decisions_path)
        evidence.update(gates_source=decisions_path.relative_to(ROOT).as_posix(), gates_sha256=source_hash)
        for row in decisions.get("decisions", []):
            cells = (row.get("scope_id", ""), row.get("gate_id", ""), row.get("old_decision"),
                     row.get("new_decision", "PENDING"), row.get("status", "PENDING"),
                     json.dumps(row.get("old_thresholds"), ensure_ascii=False, sort_keys=True),
                     row.get("source", row.get("predicate_source", "")))
            lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
        lines += ["", "## Predicate execution and interpretation notes", ""]
        for row in decisions.get("decisions", []):
            notes = [value for key, value in row.items() if isinstance(value, str)
                     and (key in ("predicate_note", "decision_basis", "map_clause_handling", "scientific_interpretation")
                          or "discrepancy" in key or "interpretation" in key)]
            if notes:
                lines += [f"**{row.get('scope_id')}/{row.get('gate_id')}**: " + " ".join(notes), ""]
    else:
        lines += ["", "No repaired gate decision artifact is available."]
    (destination / "gates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return evidence


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_snapshot(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    return rows, hashlib.sha256(raw).hexdigest()


def json_snapshot(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def controlled_interface_figure(destination: Path) -> str:
    """Design schematic; experimental estimates are rendered in separate figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")
    labels = (
        (0.2, 2.9, "Image + expression\nShared COCO domain"),
        (3.2, 2.9, "GT assisted candidates\nTarget present\nMaximum-K feasibility"),
        (6.2, 2.9, "Cached encoder features\nNo new extraction\nFrozen within comparison"),
        (9.2, 2.9, "Fixed grounding scorer\nLogits + selected proposal\nCorrectness per cell/seed"),
        (0.2, 0.9, "Reliability feature blocks\nS: score statistics\nQ: query–crop; V: crop–crop"),
        (3.2, 0.9, "Train / tune selection\nTrain-only normalization\nFixed grids and row budgets"),
        (6.2, 0.9, "Final held-out evaluation\nRandom K; matched hard\nFixed-K proposal dose"),
        (9.2, 0.9, "Image-cluster uncertainty\nSame draw across fixed seeds\nEffect per seed, then mean"),
    )
    for x, y, label in labels:
        axis.add_patch(FancyBboxPatch((x, y), 2.6, 1.3, boxstyle="round,pad=0.08",
                                     facecolor="#e8f1f5", edgecolor="#27647b", linewidth=1.2))
        axis.text(x + 1.3, y + 0.65, label, ha="center", va="center", fontsize=9.4)
    for y in (3.55, 1.55):
        for x in (2.8, 5.8, 8.8):
            axis.annotate("", xy=(x + 0.3, y), xytext=(x, y),
                          arrowprops={"arrowstyle": "->", "color": "#27647b", "lw": 1.3})
    axis.text(6, 4.6, "Controlled target-present evaluation interface", ha="center", fontsize=15)
    axis.plot([10.5, 10.5, 1.5], [2.81, 2.5, 2.5], color="#27647b", linewidth=1.2)
    axis.annotate("", xy=(1.5, 2.28), xytext=(1.5, 2.5),
                  arrowprops={"arrowstyle": "->", "color": "#27647b", "lw": 1.3})
    axis.text(6, 4.34, "Candidate category, geometry and distinct-object audits accompany composition claims.",
              ha="center", fontsize=9)
    axis.text(6, 0.35,
              "Scorers/heads can be trained for a specified axis before being frozen for its comparison; axes do not form a factorial study.",
              ha="center", fontsize=8.9)
    fig.tight_layout()
    filename = "figure_controlled_interface.png"
    fig.savefig(destination / filename, dpi=220, bbox_inches="tight")
    fig.savefig(destination / filename.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    return filename


def effect_page(path: Path, rows: list[dict], *, title: str, columns: list[str], context: str | None = None) -> None:
    lines = [f"# {title}", "",
             "Stored shared image-cluster intervals, 5000 draws, seed 0, 95% percentile.", ""]
    if context:
        lines += [context, ""]
    lines += ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")).replace("|", "\\|") for k in columns) + " |")
    if not rows:
        lines += ["", "No completed formal source is available yet."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidate_evidence(destination: Path) -> dict:
    """An audit is displayed only through its verified published run pointer."""
    pointer_path = OUT / "candidates/current_audit.json"
    evidence = {"source": None, "audit_status": "NO_VERIFIED_AUDIT", "sensitivity_status": "PENDING"}
    lines = ["# Candidate category and object audit", ""]
    if pointer_path.exists():
        pointer, pointer_hash = json_snapshot(pointer_path)
        source = (ROOT / pointer["summary_path"]).resolve()
        candidate_root = (OUT / "candidates").resolve()
        if not source.is_relative_to(candidate_root):
            raise ValueError("candidate audit pointer escapes the repair candidate directory")
        summary, source_hash = json_snapshot(source)
        if (source_hash != pointer["summary_sha256"]
                or summary.get("audit_run_id") != pointer.get("audit_run_id")
                or summary.get("status") not in ("AUDIT_COMPLETE_MISMATCH_FOUND", "AUDIT_COMPLETE_NO_MISMATCH")):
            raise ValueError("candidate audit pointer does not identify a completed matching snapshot")
        evidence.update(source=source.relative_to(ROOT).as_posix(), source_sha256=source_hash,
                        pointer_source=pointer_path.relative_to(ROOT).as_posix(), pointer_sha256=pointer_hash,
                        audit_status=summary["status"], audit_run_id=summary["audit_run_id"])
        mismatch = summary["mismatch"]
        evidence["mismatch"] = mismatch
        lines += [f"Verified run: `{summary['audit_run_id']}`; status **{summary['status']}**.", "",
                  f"True annotation category differs from the target proposal's assigned category in "
                  f"{mismatch['total_mismatches']} of {mismatch['total_expressions']} audited expressions "
                  f"(rate {numeric(mismatch['overall_rate'])}).", "",
                  "| Family | Expressions | Mismatches | Rate | Unknown proposal category |",
                  "|---|---|---|---|---|"]
        for family, row in mismatch["by_family"].items():
            lines.append(f"| {family} | {row['n_expressions']} | {row['n_mismatches']} | "
                         f"{numeric(row['mismatch_rate'])} | {row.get('n_unknown_target_proposal_category', '')} |")
        lines += ["", "Historical hard/dose source cohorts remain separate. Proposal counts are distinct from GT object counts.", "",
                  "| Family | Cell | Candidate source | Source expressions | Category audit coverage | Mismatches |",
                  "|---|---|---|---|---|---|"]
        for row in summary["historical_candidate_source_scope"]["sources"]:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in (
                "family", "cell", "candidate_source", "source_rows",
                "source_rows_covered_by_category_audit", "target_proposal_mismatches")) + " |")
        lines += ["", "Geometry, objectness, IoU, duplicates and distinct-object summaries:", ""]
        for label in ("candidate_geometry_summary", "candidate_geometry_supply_intersection_summary",
                      "candidate_geometry_sensitivity_summary", "availability_intersection"):
            filename = summary["output_files"].get(label)
            if filename:
                artifact = source.parent / filename
                if not artifact.is_file():
                    raise ValueError(f"completed candidate audit lacks {label}")
                relative = Path("..") / artifact.relative_to(OUT)
                lines.append(f"- [{label.replace('_', ' ')}]({relative.as_posix()})")
        if mismatch["total_mismatches"]:
            sensitivity = candidate_sensitivity_evidence(source, source_hash, destination)
            evidence.update(sensitivity)
            lines += ["", "The mismatch requires a frozen old/new sensitivity analysis on paired availability intersections. "
                      + ("Formal paired sensitivity intervals are stored; their independent raw numerical QA is recorded separately."
                         if evidence["sensitivity_status"] == "FORMAL_COMPLETE" else
                         "Completed candidate construction and geometry do not establish an effect on predictions; sensitivity intervals remain pending.")]
            lines += ["", f"Sensitivity stage: **{evidence['sensitivity_status']}**.", "",
                      f"Independent raw numerical QA: **{evidence.get('raw_numerical_qa_status', 'PENDING')}**.", "",
                      "[Paired sensitivity estimates](candidate_sensitivity.md)."]
        else:
            evidence["sensitivity_status"] = "NOT_REQUIRED_ZERO_MISMATCH"
            lines += ["", "Zero measured mismatch: the protocol does not require new sensitivity candidates."]
    else:
        lines += ["No completed audit has been published through the verified run pointer.",
                  "Partial files and failed attempts do not establish audit completion."]
    (destination / "candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return evidence


def candidate_sensitivity_evidence(audit_source: Path, audit_hash: str, destination: Path) -> dict:
    forward_source = audit_source.parent / "forward/summary.json"
    source = audit_source.parent / "forward/sensitivity_bootstrap/summary.json"
    evidence = {"sensitivity_status": "PENDING", "sensitivity_rows": 0, "raw_numerical_qa_status": "PENDING"}
    rows = []
    if forward_source.exists():
        forward, forward_hash = json_snapshot(forward_source)
        if forward.get("status") == "COMPLETE":
            evidence["sensitivity_status"] = "FORWARD_COMPLETE_INTERVALS_PENDING"
            if source.exists():
                summary, source_hash = json_snapshot(source)
                method = summary.get("bootstrap", summary.get("bootstrap_contract", {}))
                if (summary.get("status") == "COMPLETE" and method.get("n_replicates") == 5000
                        and method.get("seed") == 0 and method.get("ci_level") == 0.95
                        and method.get("resample_unit") == "image_cluster"):
                    if (summary.get("audit_run_id") != audit_source.parent.name.removeprefix("audit_run_")
                            or summary.get("candidate_audit_summary_sha256") != audit_hash
                            or summary.get("candidate_forward_summary_sha256") != forward_hash):
                        raise ValueError("formal candidate sensitivity is not bound to this audit and frozen forward")
                    csv_source = source.parent / summary["estimates_csv"]
                    csv_rows, csv_hash = csv_snapshot(csv_source)
                    rows = [row for row in csv_rows if int(row["n_replicates"]) == 5000]
                    evidence.update(sensitivity_status="FORMAL_COMPLETE", sensitivity_rows=len(rows),
                                    sensitivity_source=source.relative_to(ROOT).as_posix(),
                                    sensitivity_source_sha256=source_hash,
                                    sensitivity_csv=csv_source.relative_to(ROOT).as_posix(),
                                    sensitivity_csv_sha256=csv_hash,
                                    forward_source_sha256=forward_hash)
                    qa_source = OUT / "logs/stored_estimate_verification.json"
                    if qa_source.exists():
                        qa, qa_hash = json_snapshot(qa_source)
                        if (qa.get("status") == "PASS" and not qa.get("failures")
                                and "candidates" in qa.get("axes", [])
                                and qa.get("estimate_counts", {}).get("candidates") == len(rows)
                                and qa.get("estimate_source_sha256", {}).get(csv_source.relative_to(OUT).as_posix()) == csv_hash):
                            evidence.update(raw_numerical_qa_status="PASS",
                                            raw_numerical_qa_source=qa_source.relative_to(ROOT).as_posix(),
                                            raw_numerical_qa_sha256=qa_hash)
    headline = [row for row in rows if "old_minus_new" in row.get("estimate", "")
                and row.get("metric") in ("accuracy", "auroc_correct", "e_aurc", "rer_at_50")]
    effect_page(destination / "candidate_sensitivity.md", headline, title="Frozen paired candidate sensitivity",
                columns=["family", "candidate_source", "cell", "eval_split", "confidence_head", "metric", "contrast", "units",
                         "point", "ci_low", "ci_high", "seed_standard_deviation", "n_rows", "n_images",
                         "valid_replicates", "invalid_replicates"])
    if rows:
        with (destination / "candidate_sensitivity.md").open("a", encoding="utf-8") as handle:
            handle.write("\nOld minus corrected candidate effects are conditional on the source-specific paired availability intersection. "
                         "The full source CSV retains matched-random interactions and joint dose macros formed within each shared draw. "
                         "Intervals containing zero do not establish equivalence.\n")
    return evidence


def mechanism_evidence(destination: Path) -> dict:
    source = OUT / "mechanism/summary.json"
    ci_source = OUT / "mechanism/bootstrap_ci.csv"
    evidence = {"source": None, "rows": 0, "figures": []}
    rows = []
    if source.exists() and ci_source.exists():
        summary, source_hash = json_snapshot(source)
        bootstrap = summary.get("bootstrap", {})
        if (summary.get("status") == "COMPLETE" and bootstrap.get("n_replicates") == 5000
                and bootstrap.get("seed") == 0 and bootstrap.get("ci_level") == 0.95
                and bootstrap.get("resample_unit") == "image_cluster"):
            csv_rows, ci_hash = csv_snapshot(ci_source)
            rows = [r for r in csv_rows if int(r["n_replicates"]) == 5000]
            evidence.update(source=source.relative_to(ROOT).as_posix(), source_sha256=source_hash,
                            ci_source=ci_source.relative_to(ROOT).as_posix(), ci_source_sha256=ci_hash,
                            rows=len(rows))
    effect_page(destination / "mechanism.md", rows, title="Four corners, signed paths and interaction",
                columns=["family", "estimate", "formula", "point_mean3", "ci_low_mean3", "ci_high_mean3",
                         "seed_standard_deviation", "valid_replicates", "invalid_replicates"])
    if not rows:
        return evidence
    family_rows = {family: {row["estimate"]: row for row in rows if row["family"] == family}
                   for family in ("RPN", "DETR", "GDINO")}
    opposite = [family for family, current in family_rows.items()
                if float(current["L_at_p5"]["point_mean3"]) < 0
                and float(current["L_at_p50"]["point_mean3"]) > 0]
    if opposite:
        evidence["path_interpretation"] = (
            "Observed label path means have opposite signs in " + ", ".join(opposite)
            + ". The signed Shapley mean can conceal this cancellation; the accounting does not identify a cause.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ("L_at_p5", "L_at_p50", "C_at_r5", "C_at_r50", "shapley_label_delta",
             "shapley_confidence_delta", "interaction_I")
    labels = ("Label at p5", "Label at p50", "Confidence at r5", "Confidence at r50",
              "Shapley label", "Shapley confidence", "Interaction I")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis, family in zip(axes, ("RPN", "DETR", "GDINO")):
        lookup = {r["estimate"]: r for r in rows if r["family"] == family}
        for index, name in enumerate(names):
            row = lookup.get(name)
            if row is None or any(row[k] in ("", "None", "null", "nan")
                                  for k in ("point_mean3", "ci_low_mean3", "ci_high_mean3")):
                continue
            point, low, high = (float(row[k]) for k in ("point_mean3", "ci_low_mean3", "ci_high_mean3"))
            axis.plot([low, high], [index, index], color="#27647b", linewidth=2)
            axis.plot(point, index, "o", color="#27647b")
        axis.set_title(family)
        axis.set_yticks(range(len(names)), labels)
        axis.set_xlabel("Signed AUROC change")
        axis.axvline(0, color="#888888", linewidth=0.8)
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen K5→K50 accounting: fixed-seed mean and shared 95% CI")
    fig.tight_layout()
    filename = "figure_mechanism_paths.png"
    fig.savefig(destination / filename, dpi=220, bbox_inches="tight")
    fig.savefig(destination / filename.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    evidence["figures"].append(filename)
    with (destination / "mechanism.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n![Signed paths and interaction]({filename})\n\nExact accounting; no causal identification.\n")
        if evidence.get("path_interpretation"):
            handle.write("\n" + evidence["path_interpretation"] + "\n")
    return evidence


def information_evidence(destination: Path) -> dict:
    source = OUT / "information/summary.json"
    ci_source = OUT / "information/information_bootstrap.csv"
    method_source = OUT / "information/information_bootstrap.json"
    evidence = {"source": None, "rows": 0, "figures": []}
    rows = []
    if source.exists() and ci_source.exists() and method_source.exists():
        summary, source_hash = json_snapshot(source)
        method, method_hash = json_snapshot(method_source)
        if (summary.get("formal_bootstrap_complete") is True and summary.get("formal_replicates") == 5000
                and method.get("status") == "COMPLETE" and method.get("n_replicates") == 5000
                and method.get("bootstrap_seed") == 0 and method.get("ci_level") == 0.95
                and method.get("resample_unit") == "image_cluster"):
            csv_rows, ci_hash = csv_snapshot(ci_source)
            rows = [r for r in csv_rows if int(r["n_replicates"]) == 5000]
            evidence.update(source=source.relative_to(ROOT).as_posix(), source_sha256=source_hash,
                            ci_source=ci_source.relative_to(ROOT).as_posix(), ci_source_sha256=ci_hash,
                            method_source=method_source.relative_to(ROOT).as_posix(), method_source_sha256=method_hash,
                            rows=len(rows))
    primary = [r for r in rows if r.get("contrast") == "Full_minus_S_plus_Q"
               and r.get("metric") == "auroc_correct"]
    effect_page(destination / "information.md", primary, title="Incremental candidate–candidate information",
                columns=["scope", "cell", "K", "eval_split", "contrast", "metric", "point", "ci_low",
                         "ci_high", "valid_replicates", "invalid_replicates"],
                context="Effects are computed within each of three fixed B3 model seeds and then averaged. "
                        "Grounding is fixed within each feature comparison on the GT-assisted target-present common cohort. "
                        "These are conditional cellwise estimates on previously observed test distributions.")
    if primary:
        # This descriptive classification does not pool cells, count significance
        # votes, or claim a multiplicity-adjusted global hypothesis test.
        if all(int(r.get("invalid_replicates", 0)) == 0 for r in primary) \
                and all(float(r["ci_low"]) > 0 for r in primary if r.get("ci_low") not in ("", "None")) \
                and all(r.get("ci_low") not in ("", "None") for r in primary):
            conclusion = "All reported primary cell intervals are above zero; cellwise conditional incremental evidence is retained. No multiplicity-adjusted global claim is made."
        else:
            conclusion = "Incremental V signal is not uniformly resolved across reported cells. The central claim is limited to the measured cellwise effects; no general candidate-interaction conclusion follows."
        with (destination / "information.md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + conclusion + "\n")
        evidence["cellwise_interpretation"] = conclusion
        evidence["figures"] = information_figures(rows, destination)
    return evidence


def information_figures(rows: list[dict], destination: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    primary = [r for r in rows if r.get("eval_split") == "__pooled_test__"
               and r.get("contrast") == "Full_minus_S_plus_Q" and r.get("metric") == "auroc_correct"]
    if primary:
        fig, axis = plt.subplots(figsize=(8, max(3.7, len(primary) * 0.3 + 1.3)))
        for index, row in enumerate(primary):
            try:
                point, low, high = (float(row[k]) for k in ("point", "ci_low", "ci_high"))
            except (ValueError, TypeError):
                continue
            axis.plot([low, high], [index, index], color="#27647b", linewidth=2)
            axis.plot(point, index, "o", color="#27647b")
        axis.set_yticks(range(len(primary)), [f"{r['scope']} / {r['cell']}" for r in primary])
        axis.invert_yaxis()
        axis.axvline(0, color="#888888", linewidth=0.8)
        axis.set_xlabel("AUROC(Full) − AUROC(S+Q)")
        axis.set_title("Incremental V: pooled test, fixed-seed mean and shared 95% CI")
        axis.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        filename = "figure_information_sources.png"
        fig.savefig(destination / filename, dpi=220, bbox_inches="tight")
        fig.savefig(destination / filename.replace(".png", ".svg"), bbox_inches="tight")
        plt.close(fig)
        figures.append(filename)
    direct = [r for r in rows if r.get("scope") == "random" and r.get("eval_split") == "__pooled_test__"
              and not r.get("contrast") and r.get("metric") == "auroc_correct"]
    if direct:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for axis, model in zip(axes, ("StatsLogistic", "ScoreDeepSets")):
            for regime, color in (("smallK5_10", "#27647b"), ("largeK20_50", "#d28034")):
                selected = sorted([r for r in direct if r.get("model") == f"{model}_{regime}"
                                   and r.get("training_regime") == regime], key=lambda r: int(r["K"]))
                if not selected:
                    continue
                ks = [int(r["K"]) for r in selected]
                axis.plot(ks, [float(r["point"]) for r in selected], "o-", label=regime, color=color)
                for k, row in zip(ks, selected):
                    try:
                        low, high = (float(row[field]) for field in ("ci_low", "ci_high"))
                    except (ValueError, TypeError):
                        continue
                    axis.plot([k, k], [low, high], color=color, linewidth=1.2)
            axis.set_title(model)
            axis.set_xticks((5, 10, 20, 50))
            axis.set_xlabel("Evaluation K")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        axes[0].set_ylabel("AUROC_correct")
        fig.suptitle("Small/large-K training: equal row budgets, pooled test, 95% CI")
        fig.tight_layout()
        filename = "figure_cross_k_training.png"
        fig.savefig(destination / filename, dpi=220, bbox_inches="tight")
        fig.savefig(destination / filename.replace(".png", ".svg"), bbox_inches="tight")
        plt.close(fig)
        figures.append(filename)
    with (destination / "information.md").open("a", encoding="utf-8") as handle:
        for filename in figures:
            handle.write(f"\n![Information experiment]({filename})\n")
        handle.write("\nAll test data are evaluated after tune selection. Plotted absolute intervals do not replace paired regime/DoD intervals.\n")
    return figures


def temperature_figure(rows: list[dict], destination: Path) -> str | None:
    selected = [r for r in rows if r["scope"] == "Phase0A_B0_temperature"
                and r["cohort"] in ("testA", "testB")
                and r["estimate"].startswith("crossK_K5_minus_K50__")
                and r["metric"] == "auroc_correct"]
    if not selected:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True, sharey=True)
    conditions = ("native", "global_T", "validation_perK_diagnostic")
    labels = ("Native", "Frozen global T", "Validation per-K T")
    for axis, cohort in zip(axes, ("testA", "testB")):
        lookup = {r["estimate"].split("__", 1)[1].split("::", 1)[0]: r
                  for r in selected if r["cohort"] == cohort}
        for index, condition in enumerate(conditions):
            row = lookup.get(condition)
            if row is None or any(row[k] is None for k in ("point", "ci_low", "ci_high")):
                continue
            point, low, high = (float(row[k]) for k in ("point", "ci_low", "ci_high"))
            axis.plot([low, high], [index, index], color="#27647b", linewidth=2)
            axis.plot(point, index, "o", color="#27647b")
        axis.axvline(0, color="#888888", linewidth=0.8)
        axis.set_title(cohort)
        axis.set_yticks(range(3), labels)
        axis.set_xlabel("AUROC(K5) − AUROC(K50)")
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle("Cosine scorer temperature diagnostic: shared image bootstrap, 95% CI")
    fig.tight_layout()
    filename = "figure_temperature_gap.png"
    fig.savefig(destination / filename, dpi=220, bbox_inches="tight")
    fig.savefig(destination / filename.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    return filename


def expansion_figure(rows: list[dict], destination: Path) -> str | None:
    selected = [row for row in rows if row["scope"] == "Phase0B_temperature"
                and row["cohort"] == "__pooled_test__"
                and "__global_T::" in row["estimate"]
                and row["estimate"].startswith("K")]
    if not selected:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = (("accuracy", "Grounding accuracy"), ("auroc_correct", "Correctness AUROC"),
              ("ece_adaptive", "Adaptive ECE"), ("e_aurc", "E-AURC"),
              ("rer_at_50", "RER at 50% coverage"), ("rer_at_80", "RER at 80% coverage"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for axis, (metric, title) in zip(axes.flat, fields):
        current = sorted([row for row in selected if row["metric"] == metric],
                         key=lambda row: int(row["estimate"].split("__", 1)[0][1:]))
        ks = [int(row["estimate"].split("__", 1)[0][1:]) for row in current]
        axis.plot(ks, [float(row["point"]) for row in current], "o-", color="#27647b")
        for k, row in zip(ks, current):
            if row["ci_low"] is not None and row["ci_high"] is not None:
                axis.plot([k, k], [row["ci_low"], row["ci_high"]], color="#27647b", linewidth=1.4)
        axis.set_title(title)
        axis.set_xticks((5, 10, 20, 50))
        axis.set_xlabel("Candidate K")
        axis.grid(alpha=0.2)
    fig.suptitle("Frozen B3 MSP, global temperature: pooled test, fixed-seed mean and 95% CI")
    fig.text(0.5, 0.015, "GT-assisted target-present common cohort; E-AURC/RER depend on accuracy. "
             "Paired effects are in the full estimate export.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    filename = "figure_expansion_endpoints.png"
    fig.savefig(destination / filename, dpi=220, bbox_inches="tight")
    fig.savefig(destination / filename.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    return filename


def main_tables(rows: list[dict], information: dict, candidates: dict, mechanism: dict,
                destination: Path) -> None:
    endpoints = [row for row in rows if row["scope"] == "Phase0B_temperature"
                 and row["cohort"] == "__pooled_test__" and "__global_T::" in row["estimate"]
                 and row["estimate"].startswith("K")]
    lines = ["# Main evidence tables", "", "## Table 1 — frozen B3 MSP under candidate expansion", "",
             "Pooled testA+testB, global temperature; fixed-three-seed means and shared 95% image-cluster intervals.", "",
             "| K | Accuracy | AUROC | Adaptive ECE | E-AURC | RER50 | Rows / images |",
             "|---|---|---|---|---|---|---|"]
    for k in (5, 10, 20, 50):
        lookup = {row["metric"]: row for row in endpoints if row["estimate"].startswith(f"K{k}__")}
        if not lookup:
            continue
        values = []
        for metric in ("accuracy", "auroc_correct", "ece_adaptive", "e_aurc", "rer_at_50"):
            row = lookup[metric]
            values.append(f"{numeric(row['point'])} [{numeric(row['ci_low'])}, {numeric(row['ci_high'])}]")
        first = next(iter(lookup.values()))
        lines.append(f"| {k} | " + " | ".join(values)
                     + f" | {first['n_rows']} / {first['n_images']} |")
    lines += ["", "E-AURC and RER measure selective utility and depend on accuracy. "
              "Absolute intervals are accompanied by [paired effect intervals](formal_statistics.csv). "
              "The [full axis tables](statistics.md) retain each model, cohort, source and recovery status.", "",
              "Adaptive ECE bins are recalculated within each draw. Its observed point can lie below the reported "
              "percentile interval; the estimates and intervals are retained as computed.", "",
              "## Table 2 — propositions, evidence conditions and limits", "",
              "| Proposition | Current evidence | Conditions and limits |", "|---|---|---|"]
    lines.append("| Expansion changes reliability | Stored B3 endpoint and paired-effect intervals | "
                 "GT-assisted target-present common cohort; independent candidate scoring establishes accuracy monotonicity, "
                 "while AUROC direction is empirical. |")
    lines.append("| Incremental candidate–candidate V | "
                 + information.get("cellwise_interpretation", "Formal paired intervals pending.")
                 + " | Fixed S/S+Q/S+V/Full groups and fixed grounding; cellwise conditional effects with test-set reuse. |")
    lines.append("| Score-model extrapolation | "
                 + ("Formal small/large-K contrasts and DoD stored." if information.get("rows")
                    else "Training completed; formal contrast intervals pending.")
                 + " | Equal train/tune row budgets; tested model failures do not prove score-information absence. |")
    mismatch = candidates.get("mismatch")
    candidate_description = (f"{mismatch['total_mismatches']} / {mismatch['total_expressions']} category mismatches; "
                             f"sensitivity {candidates['sensitivity_status']}; "
                             f"independent raw numerical QA {candidates.get('raw_numerical_qa_status', 'PENDING')}."
                             if mismatch else "Verified audit pending.")
    lines.append("| Same-category construction | " + candidate_description
                 + " | Proposal multiplicity differs from distinct annotated objects; preserve source-specific availability intersections. |")
    lines.append("| Four-corner decomposition | "
                 + mechanism.get("path_interpretation", "Formal four-corner/path intervals pending.")
                 + " | Frozen prediction accounting; H1c addresses accuracy harm and does not identify AUROC-gap causes. |")
    lines += ["", "Source snapshots and formal method checks are recorded in [evidence.json](evidence.json).", ""]
    (destination / "main_tables.md").write_text("\n".join(lines), encoding="utf-8")


def build() -> dict:
    destination = OUT / "publication"
    destination.mkdir(exist_ok=True)
    source = OUT / "statistics/summary.json"
    summary, source_hash = json_snapshot(source) if source.exists() else ({}, None)
    rows = formal_statistics(summary)
    fields = ["job_id", "scope", "cohort", "estimate", "metric", "point", "ci_low", "ci_high",
              "n_seeds", "seed_standard_deviation", "n_rows", "n_images", "n_replicates",
              "valid_replicates", "invalid_replicates", "anchor_status", "recovery_status", "raw_replicates",
              "operation", "scale", "formula", "unit", "heldout_test_only", "cohort_split_counts"]
    csv_write(destination / "formal_statistics.csv", rows, fields)
    # Direct endpoints form a descriptive table. Derived contrasts remain in
    # the full export, so this table cannot replace the actual paired effect CIs.
    direct = [r for r in rows if (r["heldout_test_only"] is True
                                 or r["cohort"] in ("testA", "testB", "__pooled_test__"))
              and r["metric"] in (*METRICS, "ece_adaptive")
              and (r["operation"] == "condition" or (r["operation"] is None and "::" in r["estimate"]
                   and not any(k in r["estimate"].lower() for k in
                               ("minus", "crossk", "relative", "drop", "reduction", "worsening", "gain", "dod", "macro", "contrast", "amplification"))))]
    lines = ["# Artifact-derived formal statistical evidence", "",
             "Absolute endpoints are descriptive; paired contrasts and their intervals",
             "are preserved in [the complete export](formal_statistics.csv).", "",
             "| Scope | Cohort | Endpoint | Unit | Point | 95% CI | Seeds | Valid / invalid draws | Anchor |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in direct:
        lines.append(f"| {row['scope']} | {row['cohort']} | {row['estimate']} | {row['unit'] or 'metadata pending'} | {numeric(row['point'])} | "
                     f"[{numeric(row['ci_low'])}, {numeric(row['ci_high'])}] | {row['n_seeds']} | "
                     f"{row['valid_replicates']} / {row['invalid_replicates']} | {row['anchor_status']} |")
    figure = temperature_figure(rows, destination)
    expansion = expansion_figure(rows, destination)
    if figure:
        lines += ["", f"![Cosine scorer temperature diagnostic]({figure})", "",
                  "The historical Phase0A_B0_temperature scope and B0 seed key refer to the frozen cosine scorer "
                  "in this source, identified as B1 in A5.4. Source identities take precedence over legacy aliases. "
                  "Per-K temperature is a validation-selected diagnostic.",
                  "These are conditional test-sample intervals, not independent confirmation."]
        contrasts = [r for r in rows if r["scope"] == "Phase0A_B0_temperature"
                     and r["cohort"] in ("testA", "testB")
                     and r["estimate"].startswith("crossK_K5_minus_K50__")
                     and r["metric"] == "auroc_correct"]
        lines += ["", "| Cohort | Temperature condition | K5−K50 AUROC effect | 95% paired CI |",
                  "|---|---|---|---|"]
        for row in contrasts:
            condition = row["estimate"].split("__", 1)[1].split("::", 1)[0]
            lines.append(f"| {row['cohort']} | {condition} | {numeric(row['point'])} | "
                         f"[{numeric(row['ci_low'])}, {numeric(row['ci_high'])}] |")
    lines += ["", "No rendering operation marks the repair complete.", ""]
    if expansion:
        lines += [f"![Expansion endpoints]({expansion})", ""]
    (destination / "statistics.md").write_text("\n".join(lines), encoding="utf-8")
    state = json.loads((OUT / "STATUS.json").read_text(encoding="utf-8"))
    interface = controlled_interface_figure(destination)
    information = information_evidence(destination)
    candidates = candidate_evidence(destination)
    mechanism = mechanism_evidence(destination)
    scope_evidence = scope_and_gate_evidence(destination, {row["job_id"] for row in rows})
    main_tables(rows, information, candidates, mechanism, destination)
    record = {"schema": "research-repair-publication-v1", "status_snapshot": state["status"],
              "completion_snapshot": state["completed"], "formal_estimates": len(rows),
              "formal_jobs": sorted({r["job_id"] for r in rows}),
              "source": source.relative_to(ROOT).as_posix() if source.exists() else None,
              "source_sha256": source_hash,
              "figures": [interface] + ([figure] if figure else []) + ([expansion] if expansion else [])
                         + information["figures"] + mechanism["figures"],
              "interface_protocol_source": "reviews/repair_protocol.md",
              "interface_protocol_sha256": digest(ROOT / "reviews/repair_protocol.md"),
              "information": information, "candidates": candidates, "mechanism": mechanism,
              "scope_and_gate_evidence": scope_evidence}
    (destination / "evidence.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    document = ROOT / "docs/final_result_summary.md"
    if document.exists():
        prefix = document.read_text(encoding="utf-8").split(MARKER, 1)[0].rstrip()
        generated = [MARKER, "", "## Artifact-derived evidence", "",
                     f"Machine status: **{state['status']}**; completed={str(state['completed']).lower()}.",
                     f"The current formal export contains {len(rows)} stored estimates from {len(record['formal_jobs'])} jobs.",
                     "Only stored 5000-draw, seed-0, 95% image-cluster jobs enter that export.", "",
                     "- [Formal statistical tables](../results/research_repair_v1/publication/statistics.md)",
                     "- [Main metrics and proposition tables](../results/research_repair_v1/publication/main_tables.md)",
                     "- [Complete estimate export](../results/research_repair_v1/publication/formal_statistics.csv)",
                     "- [Required scope coverage](../results/research_repair_v1/publication/coverage.md)",
                     "- [Historical and repaired operational gates](../results/research_repair_v1/publication/gates.md)",
                     "- [Full minus S+Q paired evidence](../results/research_repair_v1/publication/information.md)",
                     "- [Candidate category, object and availability audit](../results/research_repair_v1/publication/candidates.md)",
                     "- [Four corners, paths and interaction](../results/research_repair_v1/publication/mechanism.md)",
                     "- [Source identity and rendering record](../results/research_repair_v1/publication/evidence.json)",
                     "- [All experimental artifacts and acceptance state](../results/research_repair_v1/registry/index.md)", ""]
        if information.get("cellwise_interpretation"):
            generated += [information["cellwise_interpretation"], ""]
        if candidates.get("mismatch"):
            mismatch = candidates["mismatch"]
            generated += [f"Verified candidate audit: {mismatch['total_mismatches']} category mismatches among "
                          f"{mismatch['total_expressions']} expressions. Sensitivity stage: "
                          f"**{candidates['sensitivity_status']}**; independent raw numerical QA: "
                          f"**{candidates.get('raw_numerical_qa_status', 'PENDING')}**.", ""]
        if mechanism.get("path_interpretation"):
            generated += [mechanism["path_interpretation"], ""]
        document.write_text(prefix + "\n\n" + "\n".join(generated), encoding="utf-8")
    return record


if __name__ == "__main__":
    report = build()
    print(f"Rendered {report['formal_estimates']} stored formal estimates; completion unchanged")
