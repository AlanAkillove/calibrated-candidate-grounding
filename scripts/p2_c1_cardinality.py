"""V2-P2-C1: C1 cardinality replication under the Grounding DINO family (third family).

Question (frozen in results/v2_proposal_robustness/p2_c1_gdino_config_freeze.json)::

    Does candidate-cardinality reliability degradation replicate under the
    query-independent Grounding DINO class-prompt proposals (C1 ONLY)?

Zero new numeric code: every computation is IMPORTED VERBATIM from the frozen
P1-F4 stack - ``run_c1_for`` / ``common_k50_ids`` / ``secondary_raw_auroc`` /
``_effect_summary`` / ``_intersect_effects`` and ``core.c1_gate`` - and simply
invoked with ``family="GDINO"`` alongside byte-identical re-reads of the frozen
RPN / DETR numbers (asserted equal to ``p1_f4_c1/c1_point.csv``, never recomputed
differently).  RPN / DETR predictions are reused from
``results/v2_proposal_robustness/predictions`` exactly as stored; GDINO adds one
family string, nothing else.

Hard regime / C4 is OUT OF SCOPE for GDINO by the freeze; this script never loads
a same_category manifest.

Run::

    conda activate deepminer
    python scripts/p2_c1_cardinality.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from scripts import p1_replication_core as core  # noqa: E402
from scripts import p1_f4_cardinality as f4  # frozen numeric stack, verbatim reuse  # noqa: E402

OUT_DIR = ROOT / "results" / "v2_proposal_robustness" / "p2_c1_gdino"
PRED_DIR = ROOT / "results" / "v2_proposal_robustness" / "predictions"
F4_DIR = ROOT / "results" / "v2_proposal_robustness" / "p1_f4_c1"
P1_INFERENCE_REPORT = ROOT / "results" / "v2_proposal_robustness" / "inference_report.json"
P2_INFERENCE_REPORT = (
    ROOT / "results" / "v2_proposal_robustness" / "p2_gdino_stage" / "inference_report.json"
)
FREEZE_PATH = ROOT / "results" / "v2_proposal_robustness" / "p2_c1_gdino_config_freeze.json"

MIN_COHORT_ROWS = 8000  # freeze stop condition: cohort < 8000 -> STOP, report, no rescue
GDINO_COLOR = "#4c72b0"  # distinct third colour (RPN grey, DETR red unchanged)

FAMILIES3 = ("RPN", "DETR", "GDINO")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p2-c1] {msg}", flush=True)


def _assert_reuse_verbatim(point_rows: List[Dict[str, Any]], family: str) -> None:
    """RPN/DETR rows recomputed through the frozen function must equal the frozen CSV."""
    import csv

    frozen: Dict[str, Dict[str, float]] = {}
    with (F4_DIR / "c1_point.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["family"] == family:
                frozen[f"{r['seed']}|{r['K']}"] = {
                    m: float(r[m]) for m in
                    ("accuracy", "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
                }
    checked = 0
    for row in point_rows:
        if row["family"] != family:
            continue
        key = f"{row['seed']}|{row['K']}"
        ref = frozen.get(key)
        if ref is None:
            raise AssertionError(f"{family}: unexpected frozen row {key}")
        for metric, value in ref.items():
            if not np.isclose(float(row[metric]), value, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"{family} {key} {metric}: reuse drifted ({row[metric]} vs frozen {value})"
                )
        checked += 1
    if checked != len(frozen):
        raise AssertionError(f"{family}: checked {checked} frozen rows, expected {len(frozen)}")


def make_three_family_figures(
    fam_curves: Dict[str, Dict[int, Dict[str, float]]], fig_dir: Path
) -> List[str]:
    """Same axes / metrics / dpi as f4.make_figures, with the GDINO curve added."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    ks = list(core.KS)
    colors = {"RPN": "#8f9fa8", "DETR": "#b4635f", "GDINO": GDINO_COLOR}

    for metric, fname, title, ylabel in (
        ("auroc_correct", "fig1_k_vs_auroc_three_families.png",
         "P2-Fig1 - AUROC_correct vs K (RPN vs DETR vs GDINO)", "AUROC(correct)"),
        ("e_aurc", "fig2_k_vs_eaurc_three_families.png",
         "P2-Fig2 - E-AURC vs K (RPN vs DETR vs GDINO)", "E-AURC (lower is better)"),
    ):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for fam in FAMILIES3:
            curve = fam_curves[fam]
            ax.plot(ks, [curve[k][metric] for k in ks], marker="o",
                    color=colors[fam], label=f"{fam} (common-K50)")
        ax.set_xticks(ks)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        p = fig_dir / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p.relative_to(ROOT)))
    return written


def _intersection(pair: List[str], ids: Set[int],
                  curves: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"families": list(pair), "n_expressions": int(len(ids))}
    for family in pair:
        _, _, stage = f4.run_c1_for(PRED_DIR, family, ids)
        out[family] = f4._intersect_effects(stage)
    a, b = pair
    out[f"delta_auroc_K5_K50_gap_{b}_minus_{a}"] = (
        out[b]["delta_auroc_K5_K50_mean"] - out[a]["delta_auroc_K5_K50_mean"])
    out[f"rer50_drop_gap_{b}_minus_{a}"] = (
        out[b]["rer50_drop_mean"] - out[a]["rer50_drop_mean"])
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.out_dir)
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["protocol"] == "V2-P2-C1"

    attrition: Dict[str, Any] = {}
    for report_path, families in (
        (P1_INFERENCE_REPORT, ("RPN", "DETR")),
        (P2_INFERENCE_REPORT, ("GDINO",)),
    ):
        rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
        for f in rep.get("families", []):
            if f["family"] in families:
                attrition[f["family"]] = f.get("attrition", {})

    results: Dict[str, Any] = {}
    fam_curves: Dict[str, Dict[int, Dict[str, float]]] = {}
    point_all: List[Dict[str, Any]] = []
    boot_all: List[Dict[str, Any]] = []
    secondary_all: List[Dict[str, Any]] = []
    common_ids: Dict[str, Set[int]] = {}

    for family in FAMILIES3:
        ids = f4.common_k50_ids(family)
        if family == "GDINO" and len(ids) < MIN_COHORT_ROWS:
            raise RuntimeError(
                f"freeze stop condition: GDINO common-K50 cohort {len(ids)} < {MIN_COHORT_ROWS}; "
                "STOP and report, do not rescue by changing parameters"
            )
        common_ids[family] = ids
        _log(f"{family}: common-K50 rows={len(ids)}")
        point_rows, boot_rows, stage = f4.run_c1_for(PRED_DIR, family, ids)
        gate = core.c1_gate(stage)
        if family in ("RPN", "DETR"):
            _assert_reuse_verbatim(point_rows, family)
        point_all += point_rows
        boot_all += boot_rows
        secondary_all += f4.secondary_raw_auroc(PRED_DIR, family, ids)
        fam_curves[family] = {
            int(r["K"]): {"auroc_correct": r["auroc_correct"], "e_aurc": r["e_aurc"],
                          "accuracy": r["accuracy"]}
            for r in point_rows if r["seed"] == "mean"
        }
        results[family] = {
            "cohort_rows": int(len(ids)),
            "attrition": attrition.get(family, {}),
            "gate": gate,
            "effects": f4._effect_summary(stage),
            "verdict": "YES" if gate["replicated"] else "NO",
        }
        _log(f"{family} C1 verdict={results[family]['verdict']} "
             f"dAUROC(K5-K50)={results[family]['effects']['mean_std']['delta_auroc']['mean']:+.4f}")

    # descriptive intersections (freeze: GDINO-union-RPN and GDINO-union-DETR; the
    # RPN-DETR pair from the frozen F4 report is included for the three-family table)
    intersections = {
        "GDINO_union_RPN": _intersection(["RPN", "GDINO"],
                                         core.matched_expression_ids(common_ids["RPN"], common_ids["GDINO"]),
                                         fam_curves),
        "GDINO_union_DETR": _intersection(["DETR", "GDINO"],
                                          core.matched_expression_ids(common_ids["DETR"], common_ids["GDINO"]),
                                          fam_curves),
    }

    figures = make_three_family_figures(fam_curves, out / "figures")

    f4._write_csv(out / "c1_point.csv", point_all, f4.C1_POINT_COLS)
    f4._write_csv(out / "c1_bootstrap.csv", boot_all, f4.C1_BOOT_COLS)
    f4._write_csv(out / "c1_secondary_raw_auroc.csv", secondary_all, f4.SECONDARY_COLS)

    payload = {
        "artifact": "v2_p2_c1_gdino_cardinality",
        "protocol": "V2-P2-C1 (C1 replication on the GDINO class-prompt family; C4 out of scope)",
        "classification": "CONFIRMATORY",
        "config_freeze": str(FREEZE_PATH.relative_to(ROOT)),
        "regime": "random (nested seeded-random, P1-A0 applies unchanged)",
        "primary_cohort": "common-K50 (identical rows across K)",
        "nested_ks": list(core.KS),
        "bootstrap": {"replicates": core.BOOTSTRAP_REPLICATES, "seed": core.BOOTSTRAP_SEED,
                      "ci": core.BOOTSTRAP_CI, "resample_unit": "image", "paired": True},
        "new_training_parameters": 0,
        "c1_by_family": results,
        "c1_verdicts": {f: results[f]["verdict"] for f in FAMILIES3},
        "gdino_C1_PROPOSAL_FAMILY_REPLICATED": results["GDINO"]["verdict"],
        "rpn_detr_reuse_verbatim_check": "PASSED (all frozen F4 point rows matched to 1e-12)",
        "intersections_secondary": intersections,
        "figures": figures,
        "out_of_scope_confirmed": [
            "no GDINO same_category manifest, hard_k5 job, or C4 verdict exists or is reported",
        ],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    f4._write_json(out / "p2_c1_verdict.json", payload)
    _log(f"C1_GDINO={results['GDINO']['verdict']}  (RPN={results['RPN']['verdict']}, "
         f"DETR={results['DETR']['verdict']})")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V2-P2-C1 GDINO cardinality replication (C1 only)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
