#!/usr/bin/env python
"""V2-M M1 figure: per-cohort reliability AUROC and paired deltas (section 33 analogue).

Reads only frozen M1 artifacts (never re-computes anything):

* ``results/v2_local_competition/m1_random_only/point_metrics.csv`` -- 3-seed
  point metrics of every model and cohort;
* ``results/v2_local_competition/bootstrap/pairs.csv`` -- paired image-cluster
  bootstrap rows (seed-mean delta and seed-mean CI bounds, house convention).

Panel A: AUROC_correct per cohort (3-seed mean) for R1 / E1b / Aggregate-MLP /
LCR-noGate / LCR.  Panel B: paired AUROC deltas vs E1b -- LCR and
LCR-noGate with bootstrap CI whiskers, Aggregate-MLP as the capacity control
(its pairwise bootstrap vs E1b is not part of the frozen M1 plan, so it is drawn
as a point estimate only).

Writes ``results/v2_local_competition/figures/m1_cohort_delta_auroc.png``.

Usage
-----
    python scripts/plot_v2m_m1.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("results/v2_local_competition")
POINT_CSV = ROOT / "m1_random_only" / "point_metrics.csv"
PAIRS_CSV = ROOT / "bootstrap" / "pairs.csv"
OUT_PATH = ROOT / "figures" / "m1_cohort_delta_auroc.png"

COHORTS: Tuple[str, ...] = (
    "Random-K5",
    "Random-K5-matched",
    "SameCategory-K5",
    "K20-random",
    "K50-random",
)
LABELS = {
    "Random-K5": "Random-K5\nm=0 analogue",
    "Random-K5-matched": "Random-K5-matched\n(matched rows)",
    "SameCategory-K5": "SameCategory-K5\nunseen hard",
    "K20-random": "K20-random\n(never trained)",
    "K50-random": "K50-random\n(never trained)",
}
MODEL_STYLE = {
    "R1": dict(color="#8c8c8c", marker="o", ls="--", label="R1 stats logistic"),
    "E1b": dict(color="#1f4e79", marker="s", ls="-", label="E1b stats+semantic (31d)"),
    "Aggregate-MLP": dict(color="#c00000", marker="^", ls="-.", label="Aggregate-MLP",
                          mfc="white"),
    "LCR-noGate": dict(color="#7f7f7f", marker="v", ls=":", label="LCR-noGate (ablation)"),
    "LCR": dict(color="#2e7d32", marker="D", ls="-", label="LCR"),
}
MODELS = ("R1", "E1b", "Aggregate-MLP", "LCR-noGate", "LCR")
GO_THRESHOLD = 0.010
GUARD_THRESHOLD = -0.005


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def point_auroc() -> Dict[Tuple[str, str], float]:
    acc: Dict[Tuple[str, str], List[float]] = {}
    for row in _read(POINT_CSV):
        acc.setdefault((row["cohort"], row["model"]), []).append(float(row["auroc_correct"]))
    return {key: float(np.mean(values)) for key, values in acc.items()}


def pair_summary(pair: Mapping[str, str], cohort: str) -> Dict[str, float] | None:
    rows = [
        r for r in _read(PAIRS_CSV)
        if r["cohort"] == cohort and r["metric"] == "auroc_correct"
        and r["model_a"] == pair[0] and r["model_b"] == pair[1]
    ]
    if not rows:
        return None
    return {
        "delta": float(np.mean([float(r["diff"]) for r in rows])),
        "ci_low": float(np.mean([float(r["ci_low"]) for r in rows])),
        "ci_high": float(np.mean([float(r["ci_high"]) for r in rows])),
    }


def main() -> int:
    point = point_auroc()
    x = np.arange(len(COHORTS))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=160)

    # -- panel A: AUROC levels -------------------------------------------------
    for model in MODELS:
        style = dict(MODEL_STYLE[model])
        label = style.pop("label")
        ax_a.plot(x, [point[(c, model)] for c in COHORTS], label=label, **style)
    ax_a.set_xticks(x, [LABELS[c] for c in COHORTS], fontsize=8)
    ax_a.set_ylabel("AUROC(correct) — 3-seed mean")
    ax_a.set_title("Reliability AUROC by frozen test cohort (B0, random-only training)", fontsize=10)
    ax_a.grid(alpha=0.25, axis="y")
    ax_a.legend(fontsize=8, frameon=False)

    # -- panel B: paired deltas vs E1b -----------------------------------------
    width = 0.27
    lcr = [pair_summary(("LCR", "E1b"), c) for c in COHORTS]
    nogate = [pair_summary(("LCR-noGate", "E1b"), c) for c in COHORTS]
    agg = [point[(c, "Aggregate-MLP")] - point[(c, "E1b")] for c in COHORTS]

    def _bars(offset: float, colour: str, values, label: str, hatch: str | None = None):
        # ``values`` entries: dict (paired CI), float (point estimate) or None
        # (comparison absent from the frozen plan, e.g. LCR-noGate outside K5).
        xs = x + offset
        entries = [v if isinstance(v, dict) else (None if v is None else {"delta": float(v)}) for v in values]
        vals = np.array([np.nan if e is None else e["delta"] for e in entries], dtype=np.float64)
        shown = np.isfinite(vals)
        ax_b.bar(
            xs[shown], vals[shown], width=width, color=colour, edgecolor="black",
            linewidth=0.6, hatch=hatch, label=label,
        )
        ci = np.array([shown[i] and "ci_low" in e for i, e in enumerate(entries)], dtype=bool)
        if ci.any():
            lows = np.array([entries[i]["ci_low"] for i in np.flatnonzero(ci)])
            highs = np.array([entries[i]["ci_high"] for i in np.flatnonzero(ci)])
            mids = vals[ci]
            ax_b.errorbar(xs[ci], mids, yerr=np.vstack([mids - lows, highs - mids]),
                          fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5)

    _bars(-width, "#2e7d32", lcr, "LCR − E1b (paired CI)")
    _bars(0.0, "#c00000", agg, "Aggregate-MLP − E1b (point only)", hatch="//")
    _bars(+width, "#7f7f7f", nogate, "LCR-noGate − E1b (paired CI)")

    ax_b.axhline(0.0, color="black", linewidth=0.9)
    ax_b.axhline(GO_THRESHOLD, color="#1f4e79", linewidth=0.8, linestyle="--")
    ax_b.axhline(GUARD_THRESHOLD, color="#b26a00", linewidth=0.8, linestyle=":")
    ax_b.set_ylim(-0.038, 0.015)
    ax_b.text(x[0] - 0.42, GO_THRESHOLD + 0.0012, "M1 GO threshold +0.010",
              color="#1f4e79", fontsize=7, ha="left")
    ax_b.text(x[0] - 0.42, GUARD_THRESHOLD - 0.0042, "Random-K5 guard −0.005",
              color="#b26a00", fontsize=7, ha="left")
    ax_b.set_xticks(x, [LABELS[c] for c in COHORTS], fontsize=8)
    ax_b.set_ylabel("Δ AUROC vs E1b (seed-mean, paired bootstrap CI)")
    ax_b.set_title("Primary comparison: LCR − E1b (capacity control: Aggregate-MLP)", fontsize=10)
    ax_b.grid(alpha=0.25, axis="y")
    ax_b.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"V2M_M1_FIGURE_WRITTEN {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
