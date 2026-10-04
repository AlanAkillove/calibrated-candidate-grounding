"""Read-only artifact indexing and table rendering for Research Repair v1.

Never changes historical results, runs inference, resamples, or sets completion.
Current tables are rendered from formal repair CSV/JSON files with source identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_repair_v1"
AXES = ("statistics", "information", "candidates", "mechanism")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def table_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def scalar_rows(value: object, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        rows = []
        for key, child in value.items():
            rows.extend(scalar_rows(child, f"{prefix}.{key}" if prefix else str(key)))
        return rows
    if isinstance(value, list):
        if len(value) <= 20 and all(not isinstance(x, (dict, list)) for x in value):
            return [(prefix, json.dumps(value, ensure_ascii=False))]
        return []
    return [(prefix, value)]


def historical_corrections(registry: Path) -> None:
    """Correct reported P2 comparisons directly from frozen point rows."""
    path = ROOT / "results/v2_proposal_robustness/p2_c1_gdino/c1_point.csv"
    if not path.exists():
        return
    _, rows = table_rows(path)
    means = {}
    for family in ("RPN", "DETR", "GDINO"):
        selected = [r for r in rows if r["family"] == family and r["K"] == "5"
                    and r["seed"] in ("seed1", "seed2", "seed3")]
        if len(selected) != 3:
            raise ValueError(f"Expected three fixed seeds for {family} K5, got {len(selected)}")
        means[family] = {key: sum(float(r[key]) for r in selected) / 3
                         for key in ("accuracy", "auroc_correct", "e_aurc", "rer_at_50")}
    correction = {
        "source": path.relative_to(ROOT).as_posix(), "source_sha256": digest(path),
        "historical_report": "results/v2_proposal_robustness/p2_c1_gdino/analysis_report.md",
        "k5_seed_mean": means,
        "highest_k5_rer50": max(means, key=lambda k: means[k]["rer_at_50"]),
        "gdino_minus_detr_accuracy": means["GDINO"]["accuracy"] - means["DETR"]["accuracy"],
        "gdino_minus_rpn_accuracy": means["GDINO"]["accuracy"] - means["RPN"]["accuracy"],
        "interpretation": "E-AURC is selective utility, not a calibration metric; lowest E-AURC does not imply best calibration.",
        "original_report_unchanged": True,
    }
    lines = ["# Historical P2 report corrections", "",
             f"Source: `{correction['source']}`, SHA256 `{correction['source_sha256']}`.", "",
             "| Family | K5 accuracy | AUROC | E-AURC | RER@50 |", "|---|---|---|---|---|"]
    lines += [f"| {f} | " + " | ".join(f"{v:.6f}" for v in values.values()) + " |"
              for f, values in means.items()]
    lines += ["", f"Highest K5 RER@50: {correction['highest_k5_rer50']}.",
              f"GDINO minus DETR accuracy: {correction['gdino_minus_detr_accuracy']:+.6f}.",
              f"GDINO minus RPN accuracy: {correction['gdino_minus_rpn_accuracy']:+.6f}.", "",
              correction["interpretation"], "The original report remains immutable.", ""]
    selective_path = ROOT / "results/v2_proposal_robustness/p1_f5_c4/c4_selective.csv"
    if selective_path.exists():
        raw = selective_path.read_bytes()
        selective_rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
        corrected = []
        for row in selective_rows:
            actual_gain = float(row["rer50_e1b"]) - float(row["rer50_r1"])
            legacy_value = float(row["e1b_minus_r1_rer50_gain"])
            if abs(actual_gain + legacy_value) > 1e-12:
                raise ValueError("Historical C4 RER field does not match its original R1-minus-E1b implementation")
            corrected.append({"family": row["family"], "scorer": row["scorer"], "regime": row["regime"],
                              "legacy_field_value": legacy_value, "true_e1b_minus_r1_rer50_gain": actual_gain})
        correction["p1_c4_rer_sign"] = {
            "source": selective_path.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "historical_field": "e1b_minus_r1_rer50_gain",
            "actual_historical_formula": "RER50(R1) - RER50(E1b)",
            "correct_gain_formula": "RER50(E1b) - RER50(R1)", "rows": corrected,
            "interpretation": "RER is higher-is-better. This corrects a historical point-column label; formal intervals come from repaired statistics."}
        lines += ["## Historical P1 C4 RER direction", "",
                  "The field `e1b_minus_r1_rer50_gain` contains R1−E1b. Actual E1b gain is E1b−R1, since RER is higher-is-better.", "",
                  "These are frozen historical point corrections; paired uncertainty is reported in the repaired statistical export.", "",
                  "| Family | Scorer | Regime | Historical field | Actual E1b RER50 gain |",
                  "|---|---|---|---|---|"]
        lines += [f"| {row['family']} | {row['scorer']} | {row['regime']} | {row['legacy_field_value']:+.6f} | "
                  f"{row['true_e1b_minus_r1_rer50_gain']:+.6f} |" for row in corrected]
        lines += [""]
    (registry / "historical_corrections.json").write_text(
        json.dumps(correction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (registry / "historical_corrections.md").write_text("\n".join(lines), encoding="utf-8")


def build() -> dict:
    manifest = json.loads((OUT / "input_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((OUT / "STATUS.json").read_text(encoding="utf-8"))
    registry = OUT / "registry"
    registry.mkdir(exist_ok=True)
    historical_corrections(registry)
    records = []
    pages = []
    for axis in AXES:
        files = sorted(p for p in (OUT / axis).rglob("*") if p.is_file())
        lines = [f"# {axis} — artifact-derived repair tables", "",
                 "These tables render stored repair outputs. File existence alone does not",
                 "establish formal completion; consult STATUS.json and the acceptance report.", ""]
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            record = {"axis": axis, "path": rel, "size_bytes": path.stat().st_size}
            snapshot = None
            # Large arrays and checkpoints are indexed without unnecessary re-reading.
            if path.suffix in (".json", ".csv", ".md", ".txt", ".log"):
                snapshot = path.read_bytes()
                record["size_bytes"] = len(snapshot)
                record["sha256"] = hashlib.sha256(snapshot).hexdigest()
            records.append(record)
            if path.suffix == ".csv" and not any(
                word in path.stem.lower() for word in
                ("prediction", "by_expression", "per_expression", "by_image", "by_object", "epoch", "curve")
            ):
                reader = csv.DictReader(io.StringIO(snapshot.decode("utf-8-sig"), newline=""))
                columns, rows = list(reader.fieldnames or []), list(reader)
                record["n_rows"] = len(rows)
                if len(rows) <= 1000 and len(columns) <= 40:
                    lines += [f"## {path.relative_to(OUT / axis).as_posix()}", "",
                              f"Source: `{rel}`; SHA256 `{record['sha256']}`.", "",
                              "| " + " | ".join(map(cell, columns)) + " |",
                              "| " + " | ".join("---" for _ in columns) + " |"]
                    lines += ["| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |"
                              for row in rows]
                    lines.append("")
            elif path.name == "summary.json":
                value = json.loads(snapshot.decode("utf-8-sig"))
                lines += [f"## {path.relative_to(OUT / axis).as_posix()}", "",
                          f"Source: `{rel}`; SHA256 `{record['sha256']}`.", "",
                          "| Field | Stored value |", "|---|---|"]
                lines += [f"| {cell(key)} | {cell(val)} |" for key, val in scalar_rows(value)]
                lines.append("")
        target = registry / f"{axis}_tables.md"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pages.append(target.name)
    for path in sorted(p for p in (OUT / "publication").rglob("*") if p.is_file()):
        records.append({"axis": "publication", "path": path.relative_to(ROOT).as_posix(),
                        "size_bytes": path.stat().st_size, "sha256": digest(path)})
    inventory = {
        "schema": "research-repair-registry-v1",
        "baseline_commit": manifest["base_commit"],
        "baseline_manifest_sha256": digest(OUT / "input_manifest.json"),
        "baseline_n_files": manifest["n_files"],
        "status_snapshot": status,
        "classification": "RESULT_DRIVEN_SUPPLEMENTARY_REPAIR",
        "artifacts": records,
        "table_pages": pages,
    }
    (registry / "artifact_index.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Research Repair v1 evidence index", "",
             f"Current machine status: **{status['status']}**; completed={status['completed']}.", "",
             "This is a result-driven supplementary repair. Original inputs remain frozen.",
             "Full configuration freezes, input identity freezes, post-result corrections,",
             "and target redefinitions are distinct provenance categories.", "",
             "| Scientific question / axis | Historical evidence | Repair tables | Stage |",
             "|---|---|---|---|"]
    mapping = [
        ("RQ1 main uncertainty and temperature", "Phase05/1/1F, RefCOCOg, V2-G/D/P, LCR/M25/M3", "statistics"),
        ("RQ2 feature sources and ID/OOD", "Phase05 and Phase1/1F", "information"),
        ("Candidate category and proposal/object audit", "frozen manifests and proposal banks", "candidates"),
        ("RQ3 four corners, paths and interaction", "V2-P2-M and RQ4-M1", "mechanism"),
    ]
    for question, legacy, axis in mapping:
        lines.append(f"| {question} | {legacy} | [{axis}]({axis}_tables.md) | {status['stages'][axis]} |")
    lines += ["", "The historical [V1 registry](../../final_registry/) was not regenerated.",
              "V2 backbone/data/proposal/model evidence and the historical mechanism record",
              "retain their per-axis paths in the indexed repair tables. Gates remain",
              "operational history; current conclusions require corrected effects and limits.", "",
              "- [Artifact identities](artifact_index.json)",
              "- [Frozen baseline manifest](../input_manifest.json)",
              "- [Machine state](../STATUS.json)",
              "- [Logs and acceptance evidence](../logs/)",
              "- [Test correction rationale](test_corrections.md)", "",
              "- [Historical report corrections](historical_corrections.md)", "",
              "- [Artifact-derived statistical export and figure](../publication/statistics.md)",
              "- [Publication source identity](../publication/evidence.json)", "",
              "Regenerate this index without training or modifying the baseline:", "",
              "```powershell", "& 'E:/conda/envs/deepminer/python.exe' -B scripts/build_repair_registry.py", "```", ""]
    (registry / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return inventory


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = build()
    print(f"Indexed {len(result['artifacts'])} repair artifacts; completion status unchanged")
