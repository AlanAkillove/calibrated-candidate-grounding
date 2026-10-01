"""P1-F4 - C1 proposal-family cardinality replication (CONFIRMATORY).

Question (protocol sections 2-9)::

    Does candidate-cardinality reliability degradation replicate under DETR proposals?

The frozen V2-G G3 gate is re-applied **verbatim** to the DETR proposal family on the
*nested* ``K = 5, 10, 20, 50`` seeded-random cohorts (P1-A0: the primary random regime
is the exact V1 seeded-random construction, never DETR confidence top-K), evaluated on
the **common-K50** cohort so the same rows carry every K (``C5`` ⊂ ``C10`` ⊂ ``C20`` ⊂
``C50``).  The only thing that changes vs the frozen RPN analysis is the proposal bank;
the base reliability (``conf_msp`` = temperature-scaled B3), the frozen R1 / E1b heads,
the per-seed temperature, the image-cluster paired bootstrap (5000, seed 0, CI 0.95) and
the gate thresholds are all untouched.  **0 new training parameters.**

Secondary (result-frozen, never in the gate): a calibration-independent raw top-1 /
raw margin AUROC profile separates cardinality degradation from frozen-temperature
calibration shift, and an RPN-DETR common-expression intersection isolates effect-size
differences from cohort-composition differences.

Run::

    conda activate deepminer
    python scripts/p1_f4_cardinality.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from ccg.experiment.phase0a import SampleStats  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402

from scripts import p1_replication_core as core  # noqa: E402

OUT_DIR = ROOT / "results" / "v2_proposal_robustness" / "p1_f4_c1"
PRED_DIR = ROOT / "results" / "v2_proposal_robustness" / "predictions"
INFERENCE_REPORT = ROOT / "results" / "v2_proposal_robustness" / "inference_report.json"

C1_POINT_COLS = ["family", "model", "seed", "K", "n", "n_images",
                 "accuracy", "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"]
C1_BOOT_COLS = ["family", "seed", "K_a", "K_b", "metric", "diff_kind",
                "n", "n_clusters", "mean_a", "mean_b", "diff", "ci_low", "ci_high",
                "ci_level", "n_replicates", "std_diff"]
SECONDARY_COLS = ["family", "seed", "K", "raw_top1_auroc", "raw_margin12_auroc"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p1-f4] {msg}", flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=float) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def common_k50_ids(family: str) -> Set[int]:
    """Family's own common-K50 cohort (target present AND >= 49 valid distractors)."""
    c = core.build_corpus(family, "random")
    try:
        return core.nested_cohort_ids(c.eval_records(core.POOLED_TEST, ks=core.KS), core.PRIMARY_KB)
    finally:
        c.close()


def run_c1_for(pred_dir: Path, family: str, keep_ids: Set[int]
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Any, Any]]:
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    stage: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}

    for seed in core.B3_SEEDS:
        sk = f"seed{seed}"
        scorer = f"b3_seed{seed}"
        per_k: Dict[int, Dict[str, np.ndarray]] = {}
        for k in core.KS:
            d = core.load_pred(pred_dir, family, f"random_k{k}", scorer, keep_ids)
            if not len(d["sentence_id"]):
                raise RuntimeError(f"{family} seed{seed} K{k}: no rows in cohort")
            per_k[k] = d
            row = core.c1_point_row(d["conf_msp"], d["correct"], d["image_id"], k, sk)
            row["family"] = family
            point_rows.append(row)
        base = per_k[core.K_BASELINE]
        clusters = base["image_id"].astype(np.int64)
        for k_b in (10, 20, core.PRIMARY_KB):
            other = per_k[k_b]
            if not np.array_equal(base["sentence_id"], other["sentence_id"]):
                raise AssertionError(f"{family} {sk}: K5/K{k_b} rows not matched on the common cohort")
            sa = SampleStats.from_conf_correct(base["conf_msp"], base["correct"])
            sb = SampleStats.from_conf_correct(other["conf_msp"], other["correct"])
            for metric, kind in core.C1_METRIC_SPECS:
                fn = reval._metric_fn(metric)
                res = core._c1_bootstrap(fn, sa, sb, clusters, kind, metric)
                boot_rows.append({
                    "family": family, "seed": sk, "K_a": core.K_BASELINE, "K_b": int(k_b),
                    "metric": metric, "diff_kind": kind,
                    "n": int(res["n"]), "n_clusters": int(res["n_clusters"]),
                    "mean_a": float(res["mean_a"]), "mean_b": float(res["mean_b"]),
                    "diff": float(res["diff"]), "ci_low": float(res["ci_low"]),
                    "ci_high": float(res["ci_high"]), "ci_level": float(res["ci_level"]),
                    "n_replicates": int(res["n_replicates"]),
                    "std_diff": float(res.get("std_diff", float("nan"))),
                })
                stage.setdefault((f"{metric}|{kind}", k_b), {})[sk] = {
                    "diff": float(res["diff"]), "ci_low": float(res["ci_low"]),
                    "ci_high": float(res["ci_high"]),
                    "mean_a": float(res["mean_a"]), "mean_b": float(res["mean_b"]),
                }

    # 3-seed mean point rows (per K) for reporting / figures.
    for k in core.KS:
        per_seed = [r for r in point_rows if r["K"] == k]
        point_rows.append({
            "family": family, "model": "global_T_corrected", "seed": "mean", "K": int(k),
            "n": int(round(np.mean([r["n"] for r in per_seed]))),
            "n_images": int(round(np.mean([r["n_images"] for r in per_seed]))),
            "accuracy": core._mean([r["accuracy"] for r in per_seed]),
            "auroc_correct": core._mean([r["auroc_correct"] for r in per_seed]),
            "e_aurc": core._mean([r["e_aurc"] for r in per_seed]),
            "rer_at_50": core._mean([r["rer_at_50"] for r in per_seed]),
            "rer_at_80": core._mean([r["rer_at_80"] for r in per_seed]),
        })
    return point_rows, boot_rows, stage


def secondary_raw_auroc(pred_dir: Path, family: str, keep_ids: Set[int]) -> List[Dict[str, Any]]:
    """Calibration-independent raw top-1 / raw margin AUROC by K (section 7, secondary)."""
    rows: List[Dict[str, Any]] = []
    auc_fn = reval._metric_fn("auroc_correct")
    for seed in core.B3_SEEDS:
        sk = f"seed{seed}"
        scorer = f"b3_seed{seed}"
        for k in core.KS:
            d = core.load_pred(pred_dir, family, f"random_k{k}", scorer, keep_ids)
            rows.append({
                "family": family, "seed": sk, "K": int(k),
                "raw_top1_auroc": float(auc_fn(SampleStats.from_conf_correct(d["raw_top1"], d["correct"]))),
                "raw_margin12_auroc": float(auc_fn(
                    SampleStats.from_conf_correct(d["raw_margin12"], d["correct"]))),
            })
    return rows


def _effect_summary(stage: Dict[Any, Any]) -> Dict[str, Any]:
    """Per-seed + mean K5->K50 effects used by the report (section 5)."""
    auc_key = ("auroc_correct|absolute", core.PRIMARY_KB)
    rer50_key = ("rer_at_50|absolute", core.PRIMARY_KB)
    rer80_key = ("rer_at_80|absolute", core.PRIMARY_KB)
    eaurc_key = ("e_aurc|relative", core.PRIMARY_KB)
    per_seed = {
        f"seed{s}": {
            "delta_auroc": stage[auc_key][f"seed{s}"]["diff"],
            "rer50_drop": stage[rer50_key][f"seed{s}"]["diff"],
            "rer80_drop": stage[rer80_key][f"seed{s}"]["diff"],
            "eaurc_worsening": core._worsening_from_rel(
                stage[eaurc_key][f"seed{s}"]["mean_a"], stage[eaurc_key][f"seed{s}"]["mean_b"],
                stage[eaurc_key][f"seed{s}"]["ci_low"], stage[eaurc_key][f"seed{s}"]["ci_high"])[0],
        }
        for s in core.B3_SEEDS
    }
    def _ms(key: str) -> Dict[str, float]:
        vals = [v[key] for v in per_seed.values()]
        return {"mean": core._mean(vals), "std": core._std(vals)}
    return {"per_seed": per_seed,
            "mean_std": {k: _ms(k) for k in ("delta_auroc", "rer50_drop", "rer80_drop", "eaurc_worsening")}}


def _intersect_effects(stage: Dict[Any, Any]) -> Dict[str, Any]:
    return {
        "delta_auroc_K5_K50_mean": core._mean([stage[("auroc_correct|absolute", core.PRIMARY_KB)][f"seed{s}"]["diff"] for s in core.B3_SEEDS]),
        "rer50_drop_mean": core._mean([stage[("rer_at_50|absolute", core.PRIMARY_KB)][f"seed{s}"]["diff"] for s in core.B3_SEEDS]),
    }


def make_figures(fam_curves: Dict[str, Dict[int, Dict[str, float]]], fig_dir: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    ks = list(core.KS)
    colors = {"RPN": "#8f9fa8", "DETR": "#b4635f"}

    for metric, fname, title, ylabel in (
        ("auroc_correct", "fig1_k_vs_auroc.png", "P1-Fig1 - AUROC_correct vs K (RPN vs DETR)", "AUROC(correct)"),
        ("e_aurc", "fig2_k_vs_eaurc.png", "P1-Fig2 - E-AURC vs K (RPN vs DETR)", "E-AURC (lower is better)"),
    ):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for fam in ("RPN", "DETR"):
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.out_dir)
    pred = Path(args.pred_dir)
    inference = json.loads(INFERENCE_REPORT.read_text(encoding="utf-8")) if INFERENCE_REPORT.exists() else {}
    attrition_by_family = {
        rep["family"]: rep.get("attrition", {}) for rep in inference.get("families", [])
    }

    results: Dict[str, Any] = {}
    fam_curves: Dict[str, Dict[int, Dict[str, float]]] = {}
    boot_all: List[Dict[str, Any]] = []
    point_all: List[Dict[str, Any]] = []
    secondary_all: List[Dict[str, Any]] = []

    common_ids_by_family: Dict[str, Set[int]] = {}
    for family in ("RPN", "DETR"):
        ids = common_k50_ids(family)
        common_ids_by_family[family] = ids
        _log(f"{family}: common-K50 rows={len(ids)}")
        point_rows, boot_rows, stage = run_c1_for(pred, family, ids)
        gate = core.c1_gate(stage)
        point_all += point_rows
        boot_all += boot_rows
        secondary_all += secondary_raw_auroc(pred, family, ids)
        fam_curves[family] = {
            int(r["K"]): {"auroc_correct": r["auroc_correct"], "e_aurc": r["e_aurc"],
                          "accuracy": r["accuracy"]}
            for r in point_rows if r["seed"] == "mean"
        }
        results[family] = {
            "cohort_rows": int(len(ids)),
            "attrition": attrition_by_family.get(family, {}),
            "gate": gate,
            "effects": _effect_summary(stage),
            "verdict": "YES" if gate["replicated"] else "NO",
        }

    # matched RPN-DETR common-expression intersection (section 8, secondary)
    inter_ids = core.matched_expression_ids(common_ids_by_family["RPN"], common_ids_by_family["DETR"])
    inter: Dict[str, Any] = {"n_expressions": int(len(inter_ids))}
    for family in ("RPN", "DETR"):
        _, _, stage = run_c1_for(pred, family, inter_ids)
        inter[family] = _intersect_effects(stage)
    inter["delta_auroc_gap_DETR_minus_RPN"] = (
        inter["DETR"]["delta_auroc_K5_K50_mean"] - inter["RPN"]["delta_auroc_K5_K50_mean"])
    inter["rer50_drop_gap_DETR_minus_RPN"] = (
        inter["DETR"]["rer50_drop_mean"] - inter["RPN"]["rer50_drop_mean"])

    rpn_ref = core.load_rpn_c1_reference()

    figures = make_figures(fam_curves, out / "figures")

    _write_csv(out / "c1_point.csv", point_all, C1_POINT_COLS)
    _write_csv(out / "c1_bootstrap.csv", boot_all, C1_BOOT_COLS)
    _write_csv(out / "c1_secondary_raw_auroc.csv", secondary_all, SECONDARY_COLS)

    # interpretation case (section 9 / 21): DETR K5 accuracy vs RPN, still replicated?
    detr_k5_acc = fam_curves["DETR"][5]["accuracy"]
    rpn_k5_acc = fam_curves["RPN"][5]["accuracy"]
    interp = {
        "detr_k5_accuracy": detr_k5_acc,
        "rpn_k5_accuracy": rpn_k5_acc,
        "detr_k5_higher": bool(detr_k5_acc > rpn_k5_acc),
        "c1_replicated": results["DETR"]["gate"]["replicated"],
    }
    if interp["detr_k5_higher"] and interp["c1_replicated"]:
        interp["note"] = ("higher absolute grounding accuracy does not remove "
                          "candidate-cardinality reliability degradation (no causal claim)")
    else:
        interp["note"] = "see gate + attrition; no causal interpretation asserted"

    payload = {
        "artifact": "v2_p1_f4_c1_proposal_family_cardinality",
        "protocol": "V2-P1 P1-F4 C1 replication (V2-G G3 applied verbatim on DETR proposals)",
        "classification": "CONFIRMATORY",
        "regime": "random (nested seeded-random, P1-A0)",
        "primary_cohort": "common-K50 (identical rows across K)",
        "nested_ks": list(core.KS),
        "bootstrap": {"replicates": core.BOOTSTRAP_REPLICATES, "seed": core.BOOTSTRAP_SEED,
                      "ci": core.BOOTSTRAP_CI, "resample_unit": "image", "paired": True},
        "new_training_parameters": 0,
        "c1_by_family": results,
        "c1_verdicts": {f: results[f]["verdict"] for f in ("RPN", "DETR")},
        "detr_C1_PROPOSAL_FAMILY_REPLICATED": results["DETR"]["verdict"],
        "matched_proposal_family_intersection_secondary": inter,
        "rpn_frozen_reference_from_artifacts": rpn_ref,
        "interpretation_case": interp,
        "figures": figures,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out / "f4_verdict.json", payload)
    _log(f"RPN C1 verdict={results['RPN']['verdict']}  DETR C1 verdict={results['DETR']['verdict']}")
    _log(f"DETR dAUROC(K5-K50)={results['DETR']['effects']['mean_std']['delta_auroc']['mean']:+.4f} "
         f"routeA={results['DETR']['gate']['route_a_passed']} routeB={results['DETR']['gate']['route_b_passed']}")
    _log(f"intersection gap dAUROC DETR-RPN={inter['delta_auroc_gap_DETR_minus_RPN']:+.4f}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P1-F4 C1 proposal-family cardinality replication")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
