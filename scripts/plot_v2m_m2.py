#!/usr/bin/env python
"""V2-M M2 core figure (round-request section 12): the competition severity curve.

Reads ``m2_curriculum/severity_curve.csv`` + ``bootstrap_pairs.csv`` and draws

* left  panel: absolute AUROC of every model vs the competition severity
  ``m in {0, 2, 4, 8}`` (seed mean +/- seed std);
* right panel: ``AUROC(model) - AUROC(E1b)`` vs ``m`` -- LCR carries the paired
  image-cluster bootstrap CI (seed-mean bounds, house convention), while
  LCR-noGate / Aggregate-MLP / R1 are drawn as point-estimate controls; the
  frozen M2 GO threshold (+0.010) and the zero line are drawn for reference.

The figure is *descriptive*: monotonicity is not required, the question is
whether the structured model accumulates an advantage as severity grows.

Usage
-----
    python scripts/plot_v2m_m2.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
M2_DIR = _REPO / "results/v2_local_competition/m2_curriculum"
CURVE_PATH = M2_DIR / "severity_curve.csv"
BOOT_PATH = M2_DIR / "bootstrap_pairs.csv"
OUT_PATH = M2_DIR / "figures" / "m2_severity_curve.png"

MODELS = ("R1", "E1b", "Aggregate-MLP", "LCR-noGate", "LCR")
LEVELS = ("m0", "m2", "m4", "m8")
X = np.arange(len(LEVELS), dtype=float)
GO_THRESHOLD = 0.010
COLOURS = {
    "R1": "#7f7f7f",
    "E1b": "#1f77b4",
    "Aggregate-MLP": "#ff7f0e",
    "LCR-noGate": "#2ca02c",
    "LCR": "#d62728",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lcr_ci(boot_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Seed-mean paired CI bounds of LCR-E1b (auroc) per level."""
    ci: Dict[str, Any] = {}
    for level in LEVELS:
        sel = [
            r for r in boot_rows
            if r["cohort"] == level and r["model_a"] == "LCR" and r["model_b"] == "E1b"
            and r["metric"] == "auroc_correct"
        ]
        if not sel:
            continue
        sel.sort(key=lambda r: int(r["seed"]))
        ci[level] = (
            float(np.mean([float(r["ci_low"]) for r in sel])),
            float(np.mean([float(r["ci_high"]) for r in sel])),
        )
    return ci


def main() -> int:
    curve = _read_csv(CURVE_PATH)
    boots = _read_csv(BOOT_PATH)
    auroc = {(r["model"], r["level"]): float(r["auroc_mean"]) for r in curve}
    auroc_std = {(r["model"], r["level"]): float(r["auroc_std"]) for r in curve}
    ci = _lcr_ci(boots)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    ax_a, ax_b = axes

    # ---- left: absolute AUROC -------------------------------------------------
    for model in MODELS:
        y = np.array([auroc[(model, level)] for level in LEVELS])
        err = np.array([auroc_std[(model, level)] for level in LEVELS])
        ax_a.errorbar(
            X, y, yerr=err, marker="o", capsize=3.0, linewidth=1.7,
            color=COLOURS[model], label=model,
        )
    ax_a.set_xticks(X)
    ax_a.set_xticklabels([level.replace("m", "m=") for level in LEVELS])
    ax_a.set_xlabel("competition severity m (same-category distractors), K=10")
    ax_a.set_ylabel("AUROC(correct), frozen test split")
    ax_a.set_title("Absolute reliability AUROC vs severity (3-seed mean +/- std)")
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=8, loc="lower right")

    # ---- right: delta vs E1b --------------------------------------------------
    for model in ("R1", "Aggregate-MLP", "LCR-noGate", "LCR"):
        delta = np.array([auroc[(model, level)] - auroc[("E1b", level)] for level in LEVELS])
        if model == "LCR" and ci:
            lows = np.array([ci[level][0] for level in LEVELS])
            highs = np.array([ci[level][1] for level in LEVELS])
            yerr = np.vstack([delta - lows, highs - delta])
            ax_b.errorbar(
                X, delta, yerr=yerr, marker="o", capsize=3.5, linewidth=2.0,
                color=COLOURS[model], label="LCR - E1b (95% paired CI)",
            )
        else:
            ax_b.plot(
                X, delta, marker="o", linewidth=1.4, color=COLOURS[model],
                label=f"{model} - E1b",
            )
    ax_b.axhline(0.0, color="black", linewidth=0.9)
    ax_b.axhline(GO_THRESHOLD, color="crimson", linewidth=1.1, linestyle="--")
    ax_b.text(
        X[0] - 0.30, GO_THRESHOLD + 0.0015, f"M2 GO threshold = +{GO_THRESHOLD:.3f}",
        color="crimson", fontsize=8.5, ha="left", va="bottom",
    )
    ax_b.set_xticks(X)
    ax_b.set_xticklabels([level.replace("m", "m=") for level in LEVELS])
    ax_b.set_xlabel("competition severity m (same-category distractors), K=10")
    ax_b.set_ylabel("AUROC(model) - AUROC(E1b)")
    ax_b.set_title("Delta vs semantic baseline (m=8 is completely unseen)")
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=8, loc="best")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(f"V2M_M2_FIGURE_OK {OUT_PATH}")
    for model in MODELS:
        values = ", ".join(
            f"{level}:{auroc[(model, level)]:.4f}" for level in LEVELS
        )
        print(f"  {model:14s} AUROC {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
