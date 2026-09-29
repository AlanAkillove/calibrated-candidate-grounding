"""A11 final report artefacts: main replication table + two figures (protocol 37-39).

Reads the two frozen result sources and emits the deliverables the amendment asked
for, with **zero re-fitting** and **no hard-coded RefCOCO+ numbers**:

* ``results/phase1e_refcocog_external/a11_results.json`` - the RefCOCOg strict
  external analysis written by :mod:`scripts.a11_external_analysis`.
* ``results/phase1f_hard_semantic/bootstrap.csv``       - the frozen RefCOCO+
  Phase-1F K=5 paired-bootstrap rows (Random-K5 / SameCategory-K5). The RefCOCO+
  cells of the main table are *read* from here (protocol section 37).

Aggregation convention is identical to Phase 1F's ``gate.json``: each cell is the
3-seed mean of the per-seed paired-bootstrap point and its CI endpoints, so the
RefCOCO+ and RefCOCOg bars are directly comparable.

Outputs (all under ``results/phase1e_refcocog_external/``):
  main_table.md                        the protocol section 37 table
  figures/a11_fig1_delta_auroc.png     the core replication figure (section 38)
  figures/a11_fig2_eaurc_rer50.png     E-AURC reduction / RER@50 gain (section 39)
  a11_report.json                      consolidated machine-readable summary

    conda activate deepminer
    python scripts/a11_external_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
# Phase 1F bootstrap.csv column indices (stable, from the file header).
COL = {
    "K": 1, "metric": 4, "diff": 5, "ci_low": 6, "ci_high": 7,
    "mean_a": 8, "mean_b": 9, "cell": 16, "scorer": 15, "row_kind": 22, "reduction": 25,
}


def _f(row: List[str], key: str) -> float:
    return float(row[COL[key]])


def _phase1f_cells(bootstrap_csv: Path) -> Dict[str, Dict[str, float]]:
    """RefCOCO+ K=5 {rand5, hard5} cells: 3-seed mean of per-seed bootstrap rows."""
    per: Dict[str, Dict[str, Dict[str, float]]] = {
        "rand5": {"s1": {}, "s2": {}, "s3": {}},
        "hard5": {"s1": {}, "s2": {}, "s3": {}},
    }
    with bootstrap_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if len(row) <= COL["reduction"] or row[COL["K"]] != "5":
                continue
            cell = row[COL["cell"]]
            if cell not in ("rand5", "hard5"):
                continue
            pos = SEEDS.index(row[COL["scorer"]])
            seed_key = f"s{pos + 1}"
            kind = row[COL["row_kind"]]
            metric = row[COL["metric"]]
            if kind == "pairwise" and metric == "auroc_correct":
                per[cell][seed_key]["e1b_auroc"] = _f(row, "mean_a")
                per[cell][seed_key]["stats_auroc"] = _f(row, "mean_b")
                per[cell][seed_key]["delta_auroc"] = _f(row, "diff")
                per[cell][seed_key]["delta_auroc_ci_low"] = _f(row, "ci_low")
                per[cell][seed_key]["delta_auroc_ci_high"] = _f(row, "ci_high")
            elif kind == "pairwise" and metric == "rer_at_50":
                per[cell][seed_key]["rer50_gain_pp"] = _f(row, "diff") * 100.0
            elif kind == "pairwise_ratio" and metric == "e_aurc":
                per[cell][seed_key]["e_aurc_reduction"] = _f(row, "reduction")

    out: Dict[str, Dict[str, float]] = {}
    fields = ("e1b_auroc", "stats_auroc", "delta_auroc", "delta_auroc_ci_low",
              "delta_auroc_ci_high", "rer50_gain_pp", "e_aurc_reduction")
    for cell in ("rand5", "hard5"):
        out[cell] = {
            fld: float(np.mean([per[cell][sk][fld] for sk in ("s1", "s2", "s3")]))
            for fld in fields
        }
    return out


def _external_cells(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """RefCOCOg strict cells: main_table means + 3-seed mean of per-seed Delta CI."""
    table = results["main_table"]
    per_seed = results["per_seed"]
    out: Dict[str, Dict[str, float]] = {}
    # regime name in main_table/per_seed -> cell key used across both datasets.
    for regime, cell_key in (("random", "rand5"), ("same_category", "hard5")):
        row = dict(table[regime])
        # CI endpoints come from the per-seed metric blocks (mean of 3 seeds),
        # mirroring the Phase 1F gate.json aggregation convention.
        row["delta_auroc_ci_low"] = float(np.mean(
            [per_seed[s][regime]["delta_auroc_ci_low"] for s in SEEDS]))
        row["delta_auroc_ci_high"] = float(np.mean(
            [per_seed[s][regime]["delta_auroc_ci_high"] for s in SEEDS]))
        out[cell_key] = row
    return out


def _render_table(ph1f: Dict[str, Dict[str, float]], ext: Dict[str, Dict[str, float]]) -> str:
    def fmt(cell: Dict[str, float]) -> str:
        return (f"{cell['stats_auroc']:.4f} | {cell['e1b_auroc']:.4f} | "
                f"{cell['delta_auroc']:+.4f} | {cell['e_aurc_reduction'] * 100:+.1f}% | "
                f"{cell['rer50_gain_pp']:+.2f}pp")

    lines = [
        "| Dataset | Regime | Stats AUROC | E1b AUROC | dAUROC | EAURC reduction | RER50 gain |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| RefCOCO+ | Random-K5 | {fmt(ph1f['rand5'])} |",
        f"| RefCOCO+ | SameCat-K5 | {fmt(ph1f['hard5'])} |",
        f"| RefCOCOg strict | Random-K5 | {fmt(ext['rand5'])} |",
        f"| RefCOCOg strict | SameCat-K5 | {fmt(ext['hard5'])} |",
        "",
        "Values are 3-seed means of the per-seed image-cluster paired-bootstrap rows "
        "(5000 reps, 95% CI). RefCOCO+ rows are read from the frozen Phase-1F "
        "`bootstrap.csv`; RefCOCOg rows from `a11_results.json`.",
    ]
    return "\n".join(lines) + "\n"


def _bar_group(ax, cells_by_source, field, *, scale=1.0, decimals=3, ylabel):
    """Plot 4 bars: (RefCOCO+, RefCOCOg) x (Random, Same-category) with 95% CI.

    ``scale`` converts the stored field to the plotted unit: e_aurc_reduction is a
    fraction (scale=100 -> %), rer50_gain_pp is already pp (scale=1), delta_auroc is
    an AUROC difference (scale=1).
    """
    labels = ["Random\nK5", "Same-category\nK5"]
    colors = ["#7f9fc4", "#b4635f"]
    xbase = np.array([0.0, 1.0])
    width = 0.36
    fmt = f"{{:+.{decimals}f}}"
    for i, (src, cells) in enumerate(cells_by_source):
        point = np.array([cells["rand5"][field], cells["hard5"][field]]) * scale
        if field == "delta_auroc":
            lo = point - np.array([cells["rand5"]["delta_auroc_ci_low"],
                                   cells["hard5"]["delta_auroc_ci_low"]]) * scale
            hi = np.array([cells["rand5"]["delta_auroc_ci_high"],
                           cells["hard5"]["delta_auroc_ci_high"]]) * scale - point
            err = np.vstack([lo, hi])
        else:
            err = None
        pos = xbase + (i - 0.5) * width
        ax.bar(pos, point, width=width, color=colors[i], label=src,
               yerr=err, capsize=4, error_kw={"ecolor": "0.35", "lw": 1.2})
        for p, v in zip(pos, point):
            ax.text(p, v, fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(xbase)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)


def _figure1(out_dir: Path, ph1f, ext, q1_label, q2_label) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    _bar_group(ax, [("RefCOCO+", ph1f), ("RefCOCOg strict", ext)], "delta_auroc",
               scale=1.0, decimals=3, ylabel="dAUROC  (E1b - Stats)")
    ax.set_title("Fig 1 - semantic reliability increment by regime\n"
                 f"{q1_label}  |  {q2_label}", fontsize=9)
    fig.tight_layout()
    path = out_dir / "figures" / "a11_fig1_delta_auroc.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _figure2(out_dir: Path, ph1f, ext) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    _bar_group(axes[0], [("RefCOCO+", ph1f), ("RefCOCOg strict", ext)],
               "e_aurc_reduction", scale=100.0, decimals=1, ylabel="E-AURC reduction (%)")
    axes[0].set_title("E-AURC reduction (Stats -> E1b)")
    _bar_group(axes[1], [("RefCOCO+", ph1f), ("RefCOCOg strict", ext)],
               "rer50_gain_pp", scale=1.0, decimals=1, ylabel="RER@50 gain (pp)")
    axes[1].set_title("RER@50 gain (Stats -> E1b)")
    for ax in axes:
        ax.set_xticklabels(["Random\nK5", "Same-category\nK5"])
    fig.suptitle("Fig 2 - selective metrics by regime: RefCOCO+ (internal) vs RefCOCOg strict (external)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "figures" / "a11_fig2_eaurc_rer50.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(args: argparse.Namespace, log) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    ph1f = _phase1f_cells(Path(args.phase1f_bootstrap))
    ext = _external_cells(results)

    table_md = _render_table(ph1f, ext)
    (out_dir / "main_table.md").write_text(table_md, encoding="utf-8")
    log("[a11-report] wrote main_table.md")

    q1 = results["q1_semantic_transfer"]["decision"]
    q2 = results["q2_hard_amplification"]["decision"]
    q2_short = {"YES": "YES", "NO": "NO",
                "INVALID_MANIPULATION": "INVALID (manipulation)"}.get(q2, q2)
    q1_label = f"RefCOCOg Q1 semantic transfer = {q1}"
    q2_label = f"Q2 hard amplification = {q2_short}"
    fig1 = _figure1(out_dir, ph1f, ext, q1_label, q2_label)
    fig2 = _figure2(out_dir, ph1f, ext)
    log(f"[a11-report] wrote {fig1.name}, {fig2.name}")

    report = {
        "protocol": "A11",
        "schema": "a11-report-v1",
        "aggregation": "3-seed mean of per-seed paired-bootstrap rows and CI endpoints",
        "q1_semantic_transfer": results["q1_semantic_transfer"],
        "q2_hard_amplification": results["q2_hard_amplification"],
        "full_verdict": results["full_verdict"],
        "manipulation": {
            "valid": results["manipulation"]["valid"],
            "per_seed_significant": {
                s: results["manipulation"]["per_seed"][s]["n_criterion_significant"]
                for s in SEEDS},
        },
        "cells": {"refcococo_plus": ph1f, "refcocog_strict": ext},
        "diagnostics": results.get("diagnostics", {}),
        "main_table_markdown": table_md,
    }
    (out_dir / "a11_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log("[a11-report] wrote a11_report.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path,
                        default=ROOT / "results/phase1e_refcocog_external/a11_results.json")
    parser.add_argument("--phase1f-bootstrap", type=Path,
                        default=ROOT / "results/phase1f_hard_semantic/bootstrap.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results/phase1e_refcocog_external")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args, print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
