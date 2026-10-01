"""P1 final proposal-family summary (protocol sections 19 / 20 / 22).

Assembles the frozen-evidence table AFTER F4 + F5 have produced their verdict
artifacts.  The RPN column is loaded exclusively from frozen artifacts -
``results/phase0b_independent`` (point / bootstrap CSVs),
``results/phase1f_hard_semantic/gate.json``,
``results/v2_backbone_generalization/g3_cardinality_gate.json`` and the V1
proposal audit under ``results/proposal_audit`` - never hardcoded.  The DETR
column comes from ``p1_f4_c1/f4_verdict.json`` and ``p1_f5_c4/f5_verdict.json``.
Everything here is read-only analysis: **0 new parameters, no fitting**.

Run::

    conda activate deepminer
    python scripts/p1_summary.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from scripts import p1_replication_core as core  # noqa: E402

P1_DIR = ROOT / "results" / "v2_proposal_robustness"
F4_VERDICT = P1_DIR / "p1_f4_c1" / "f4_verdict.json"
F5_VERDICT = P1_DIR / "p1_f5_c4" / "f5_verdict.json"
INFERENCE_REPORT = P1_DIR / "inference_report.json"
CONFIG_FREEZE = P1_DIR / "p1_f4_f5_config_freeze.json"
RPN_AUDIT = ROOT / "results" / "proposal_audit"
DETR_AUDIT = P1_DIR / "p1_detr_r50"
F3_DIR = ROOT / "results" / "v2_p1_f3_route_f"
V2G_G3_GATE = ROOT / "results" / "v2_backbone_generalization" / "g3_cardinality_gate.json"
OUT_JSON = P1_DIR / "p1_final_summary.json"
OUT_MD = P1_DIR / "proposal_family_final_table.md"
OUT_DIST = P1_DIR / "p1_distribution_diagnostic.csv"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p1-summary] {msg}", flush=True)


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _audit_row(path: Path, **want: Any) -> Dict[str, str]:
    """First row of an audit CSV whose columns match ``want`` (e.g. N=64)."""
    for row in _csv_rows(path):
        if all(row.get(str(k)) == str(v) for k, v in want.items()):
            return row
    return {}


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _mean_scorer(rows: List[Dict[str, str]], family: str, metric: str) -> Optional[float]:
    vals = [float(r["mean"]) for r in rows
            if r.get("family") == family and r.get("metric") == metric]
    return sum(vals) / len(vals) if vals else None


def _msp_worsening_from_ratio(ratio_diff: Optional[float]) -> Optional[float]:
    """phase0b ratio r=(K5-K50)/K50 -> K5-baselined worsening w=(K50-K5)/K5."""
    if ratio_diff is None or abs(1.0 + ratio_diff) < 1e-12:
        return None
    return -ratio_diff / (1.0 + ratio_diff)


def final_table_rows(f4: Dict[str, Any], f5: Dict[str, Any]) -> List[List[Any]]:
    rpn_c1 = core.load_rpn_c1_reference()
    rpn_c4 = core.load_rpn_c4_reference()
    point = rpn_c1.get("point_by_k", {})
    p5, p50 = point.get("5", {}), point.get("50", {})

    rpn_recall = _audit_row(RPN_AUDIT / "recall_by_N.csv", N=64)
    detr_recall = _audit_row(DETR_AUDIT / "recall_by_N.csv", N=64)
    rpn_avail = _audit_row(RPN_AUDIT / "candidate_availability_by_N.csv", N=64, K=50)
    detr_avail = _audit_row(DETR_AUDIT / "candidate_availability_by_N.csv", N=64, K=50)
    rpn_same = _audit_row(RPN_AUDIT / "same_category_availability.csv", N=64, threshold=4)
    detr_same = _audit_row(DETR_AUDIT / "same_category_availability.csv", N=64, threshold=4)

    d_c1 = f4["c1_by_family"]["DETR"]
    d_gate, d_eff = d_c1["gate"], d_c1["effects"]["mean_std"]
    d_pt = {(int(r["K"]), r["seed"]): r for r in _f4_point_rows()
            if r["family"] == "DETR"}
    d_k5 = d_pt.get((5, "mean"), {})
    d_k50 = d_pt.get((50, "mean"), {})
    d_c4 = f5["c4_by_family"]["DETR"]["gate"]

    g3 = json.loads(V2G_G3_GATE.read_text(encoding="utf-8"))["backbones"]
    rpn_c1_verdict = "YES" if any(
        bb.get("route_a_passed") or bb.get("route_b_passed") for bb in g3.values()) else "NO"
    rpn_c4_verdict = {"CONFIRMED": "YES"}.get(str(rpn_c4.get("verdict")), str(rpn_c4.get("verdict")))

    return [
        ["target recall@0.5 (N=64)", _f(rpn_recall, "ref_target_recall@0.5"),
         _f(detr_recall, "ref_target_recall@0.5")],
        ["target recall@0.7 (N=64)", _f(rpn_recall, "ref_target_recall@0.7"),
         _f(detr_recall, "ref_target_recall@0.7")],
        ["K50 availability (audit)", _f(rpn_avail, "frac_available"), _f(detr_avail, "frac_available")],
        ["K50 availability (pooled-test rows)",
         _attrition_availability(f4, "RPN"), _attrition_availability(f4, "DETR")],
        ["same-cat >=4 availability (audit)", _f(rpn_same, "frac"), _f(detr_same, "frac")],
        ["K5 grounding acc", p5.get("accuracy"), _f(d_k5, "accuracy")],
        ["K50 grounding acc", p50.get("accuracy"), _f(d_k50, "accuracy")],
        ["K5 MSP AUROC", p5.get("auroc_correct"), _f(d_k5, "auroc_correct")],
        ["K50 MSP AUROC", p50.get("auroc_correct"), _f(d_k50, "auroc_correct")],
        ["delta AUROC K5-K50",
         rpn_c1.get("auroc_correct|absolute", {}).get("diff_mean"), d_eff["delta_auroc"]["mean"]],
        ["E-AURC worsening (relative)",
         _msp_worsening_from_ratio(rpn_c1.get("e_aurc|relative", {}).get("diff_mean")),
         d_gate["eaurc_worsening_mean"]],
        ["RER50 drop", rpn_c1.get("rer_at_50|absolute", {}).get("diff_mean"),
         d_gate["rer50_drop_mean"]],
        ["delta Rand (E1b-R1 AUROC)", rpn_c4.get("delta_rand"), d_c4["delta_rand_mean"]],
        ["delta Hard (E1b-R1 AUROC)", rpn_c4.get("delta_hard"), d_c4["delta_hard_mean"]],
        ["Amplification (Hard-Rand)", rpn_c4.get("amplification", {}).get("diff"),
         d_c4["amplification_mean"]],
        ["C1 verdict", rpn_c1_verdict, f4["detr_C1_PROPOSAL_FAMILY_REPLICATED"]],
        ["C4 verdict", rpn_c4_verdict, f5["detr_C4_PROPOSAL_FAMILY_REPLICATED"]],
    ]


def _f4_point_rows() -> List[Dict[str, str]]:
    path = P1_DIR / "p1_f4_c1" / "c1_point.csv"
    return _csv_rows(path) if path.exists() else []


def _attrition_availability(f4: Dict[str, Any], family: str) -> Optional[float]:
    rep = f4.get("c1_by_family", {}).get(family, {}).get("attrition", {}).get("random", {})
    return rep.get("k50_availability")


def distribution_diagnostic(f4: Dict[str, Any], f5: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Section 20 (descriptive only): which structural quantities survive the shift."""
    rows: List[Dict[str, Any]] = []

    def add(metric: str, rpn: Any, detr: Any, stable_note: str = "") -> None:
        rows.append({"metric": metric, "RPN": rpn, "DETR": detr, "note": stable_note})

    rpn_iou = _audit_row(RPN_AUDIT / "proposal_iou_statistics.csv", N=64)
    detr_iou = _audit_row(DETR_AUDIT / "proposal_iou_statistics.csv", N=64)
    add("target max IoU (median)", _f(rpn_iou, "target_max_iou_median"),
        _f(detr_iou, "target_max_iou_median"))
    add("GT max IoU (median)", _f(rpn_iou, "gt_max_iou_median"), _f(detr_iou, "gt_max_iou_median"))
    add("pairwise redundancy frac>0.7", _f(rpn_iou, "redundancy_frac_gt_0.7"),
        _f(detr_iou, "redundancy_frac_gt_0.7"))
    add("pairwise redundancy frac>0.9", _f(rpn_iou, "redundancy_frac_gt_0.9"),
        _f(detr_iou, "redundancy_frac_gt_0.9"))

    cd = {r["metric"]: r for r in _csv_rows(DETR_AUDIT / "candidate_distribution.csv")}
    for key, label in (("mean_area_px2", "box area mean (px^2)"),
                       ("median_area_px2", "box area median (px^2)"),
                       ("mean_aspect_ratio", "aspect ratio mean")):
        row = cd.get(key, {})
        add(label, _f(row, "rpn"), _f(row, "detr_r50"))

    rpn_same = _audit_row(RPN_AUDIT / "same_category_availability.csv", N=64, threshold=4)
    detr_same = _audit_row(DETR_AUDIT / "same_category_availability.csv", N=64, threshold=4)
    add("same-category supply (frac >= 4)", _f(rpn_same, "frac"), _f(detr_same, "frac"))

    score_rows = _csv_rows(F3_DIR / "score_distribution.csv") if (F3_DIR / "score_distribution.csv").exists() else []
    add("raw margin12 mean (K5)", _mean_scorer(score_rows, "RPN", "margin12_raw"),
        _mean_scorer(score_rows, "DETR", "margin12_raw"))
    add("MSP mean (K5)", _mean_scorer(score_rows, "RPN", "msp"),
        _mean_scorer(score_rows, "DETR", "msp"))

    cmp_rows = {r["Metric"]: r for r in _csv_rows(F3_DIR / "comparison_table.csv")} \
        if (F3_DIR / "comparison_table.csv").exists() else {}
    add("grounding accuracy (natural K5)", _f(cmp_rows.get("Accuracy", {}), "RPN_K5"),
        _f(cmp_rows.get("Accuracy", {}), "DETR_K5"))

    d_eff = f4["c1_by_family"]["DETR"]["effects"]["mean_std"]
    rpn_c1 = core.load_rpn_c1_reference()
    add("C1 effect size: delta AUROC K5-K50",
        rpn_c1.get("auroc_correct|absolute", {}).get("diff_mean"), d_eff["delta_auroc"]["mean"],
        "structural: reliability degrades in both families")
    rpn_c4 = core.load_rpn_c4_reference()
    add("C4 amplification (Hard-Rand)", rpn_c4.get("amplification", {}).get("diff"),
        f5["c4_by_family"]["DETR"]["gate"]["amplification_mean"],
        "structural: semantic gain keeps its hard-competition boost")
    inter = f4.get("matched_proposal_family_intersection_secondary", {})
    if inter:
        add("C1 delta AUROC on matched intersection",
            inter.get("RPN", {}).get("delta_auroc_K5_K50_mean"),
            inter.get("DETR", {}).get("delta_auroc_K5_K50_mean"),
            "same expressions, both families")
    return rows


def run() -> Dict[str, Any]:
    f4 = json.loads(F4_VERDICT.read_text(encoding="utf-8"))
    f5 = json.loads(F5_VERDICT.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_FREEZE.read_text(encoding="utf-8"))

    rows = final_table_rows(f4, f5)
    dist = distribution_diagnostic(f4, f5)

    c1_yes = f4["detr_C1_PROPOSAL_FAMILY_REPLICATED"] == "YES"
    c4_label = f5["detr_C4_PROPOSAL_FAMILY_REPLICATED"]
    overall = core.overall_p1_verdict(c1_replicated=c1_yes, c4_verdict=c4_label)

    lines = ["# P1 proposal-family final table (section 19)", "",
             "| Metric | RPN (frozen artifacts) | DETR (P1-F4/F5) |", "|---|---:|---:|"]
    for name, rpn, detr in rows:
        fmt = lambda v: "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(f"| {name} | {fmt(rpn)} | {fmt(detr)} |")
    lines += ["", f"C1 verdict (DETR): **{f4['detr_C1_PROPOSAL_FAMILY_REPLICATED']}**  ",
              f"C4 verdict (DETR): **{c4_label}**  ",
              f"Overall P1 verdict: **{overall['label']}** ({overall['value']})", "",
              overall["detail"], ""]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with OUT_DIST.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "RPN", "DETR", "note"])
        w.writeheader()
        for row in dist:
            w.writerow(row)

    payload = {
        "artifact": "v2_p1_final_summary",
        "protocol": "V2-P1 sections 19 / 20 / 22",
        "classification": "CONFIRMATORY",
        "rpn_column_source": "frozen artifacts only (phase0b CSVs, phase1f gate.json, "
                             "v2_backbone_generalization g3 gate, proposal_audit)",
        "final_table": [{"metric": m, "RPN": r, "DETR": d} for m, r, d in rows],
        "distribution_diagnostic_descriptive_only": dist,
        "verdicts": {"detr_C1": f4["detr_C1_PROPOSAL_FAMILY_REPLICATED"],
                     "detr_C4": c4_label,
                     "rpn_recomputed_crosscheck": {
                         "C1": f4["c1_by_family"]["RPN"]["verdict"],
                         "C4": f5["c4_by_family"]["RPN"]["c4_verdict"]}},
        "overall_p1_verdict": overall,
        "interpretation_case": f4.get("interpretation_case", {}),
        "five_questions": {
            "q1_cardinality_replication": f4["detr_C1_PROPOSAL_FAMILY_REPLICATED"],
            "q2_hard_amplification_replication": c4_label,
            "q3_matched_effect_sizes": f4.get("matched_proposal_family_intersection_secondary", {}),
            "q4_core_findings_survive": overall["label"],
            "q5_accuracy_vs_reliability": f4.get("interpretation_case", {}),
        },
        "new_training_parameters": 0,
        "preconditions_from_frozen_config": config.get("preconditions", {}),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    _log(f"overall verdict: {overall['label']} ({overall['value']})")
    _log(f"wrote {OUT_MD.name}, {OUT_JSON.name}, {OUT_DIST.name}")
    return payload


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
