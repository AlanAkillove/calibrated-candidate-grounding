"""Static publication figures from stored formal V3 estimates only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_final_validation"
PUB = OUT / "publication"


def save(fig, name):
    PUB.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(PUB / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def forest(ax, labels, items, *, title, level):
    for i, item in enumerate(items):
        ci = item.get("bonferroni_9875") if level == .9875 else item
        point = item.get("point")
        if point is None or ci.get("ci_low") is None:
            ax.text(0, i, "undefined", va="center")
            continue
        ax.plot([ci["ci_low"], ci["ci_high"]], [i, i], color="#245e86", lw=2)
        ax.scatter([point], [i], color="#245e86", s=35, zorder=3)
    ax.axvline(0, color=".55", lw=1, ls="--")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", alpha=.2)


def replication():
    path = OUT / "replication/bootstrap_summary.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "COMPLETE" or data.get("replicates") != 5000:
        return
    groups = ("S", "S+Q", "S+V", "Full")
    colors = ("#777777", "#d7a127", "#388a76", "#88489b")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4), gridspec_kw={"width_ratios": [1, 1.25]})
    estimates = data["cohorts"]["random_all"]["estimates"]
    for group, color in zip(groups, colors):
        items = [estimates[f"absolute__randomK{k}__{group}__auroc_correct"] for k in (5, 10, 50)]
        axes[0].plot([5, 10, 50], [item["point"] for item in items], "o-", label=group, color=color)
        axes[0].fill_between([5, 10, 50], [item["ci_low"] for item in items], [item["ci_high"] for item in items], color=color, alpha=.09)
    axes[0].set(xlabel="Requested candidate count K", ylabel="Correctness AUROC", title="Four absolute feature-group performances", xticks=[5, 10, 50])
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=.2)
    specs = (("random_all", "Full_minus_SQ__randomK5__auroc_correct", "Random K5: Full - SQ"),
             ("random_all", "Full_minus_SQ__randomK50__auroc_correct", "Random K50: Full - SQ"),
             ("same4", "Hard_minus_random__Full_minus_SQ__auroc_correct", "Hard - random: gain difference"),
             ("same8", "m8_minus_m0__Full_minus_SQ__auroc_correct", "m8 - m0: gain difference"))
    forest(axes[1], [item[2] for item in specs], [data["cohorts"][cohort]["estimates"][name] for cohort, name, label in specs],
           title="Conditional V-block increments (95% CI)", level=.95)
    axes[1].set_xlabel("AUROC difference; positive favors additional V")
    fig.suptitle("B/16 COCO replication: developmental evidence, not independent confirmation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .95])
    save(fig, "figure_b16_source_replication")


def confirmation():
    base = OUT / "pipeline/confirmation/bootstrap"
    path = base / "controlled/summary.json"
    if not path.exists():
        return
    controlled = json.loads(path.read_text(encoding="utf-8"))
    if controlled.get("status") != "COMPLETE" or controlled.get("n_replicates") != 5000:
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    labels = ["C1: B0 MSP K5 - K50", "C2: B0 Full - SQ at K50", "C3: B/16 Full - SQ at K50", "C4: B0 SDS large - small at K50"]
    forest(ax, labels, [controlled["estimates"][f"C{i}/auroc_correct"] for i in range(1, 5)],
           title="FineCops independent confirmation (98.75% percentile CI)", level=.9875)
    ax.set_xlabel("AUROC difference (three fixed-seed effects averaged)")
    fig.tight_layout()
    save(fig, "figure_independent_confirmation")
    path = base / "natural/summary.json"
    if not path.exists():
        return
    natural = json.loads(path.read_text(encoding="utf-8"))
    if natural.get("status") != "COMPLETE" or natural.get("n_replicates") != 5000:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4), sharex=True)
    for row, backbone in enumerate(("b0", "b16")):
        for column, names in enumerate((("flow/target_coverage", "flow/accuracy_all", "flow/accuracy_covered"),
                                         ("paired_overall/Full_minus_SQ/auroc_correct", "paired_covered/Full_minus_SQ/auroc_correct"),
                                         ("paired_overall/Full_minus_SQ/risk50", "paired_covered/Full_minus_SQ/risk50"))):
            for name in names:
                items = [natural["estimates"][f"{backbone}/K{k}/{name}"] for k in (5, 10, 20, 50)]
                label = name.replace("flow/", "").replace("paired_", "").replace("/Full_minus_SQ/", ": ")
                axes[row, column].plot([5, 10, 20, 50], [item["point"] for item in items], "o-", label=label)
                low = [np.nan if item["ci_low"] is None else item["ci_low"] for item in items]
                high = [np.nan if item["ci_high"] is None else item["ci_high"] for item in items]
                axes[row, column].fill_between([5, 10, 20, 50], low, high, alpha=.1)
            axes[row, column].set_title(f"{backbone.upper()}: " + ("coverage and accuracy" if column == 0 else "Full - SQ AUROC" if column == 1 else "Full - SQ Risk@50"))
            axes[row, column].grid(alpha=.2)
            axes[row, column].legend(frameon=False, fontsize=7)
            if column:
                axes[row, column].axhline(0, color=".5", lw=1, ls="--")
            if row == 1:
                axes[row, column].set_xlabel("Requested K (actual candidate counts retained)")
    fig.suptitle("FineCops natural top-K: whole-flow and covered-cohort views (95% CI)")
    fig.tight_layout(rect=[0, 0, 1, .95])
    save(fig, "figure_natural_interface")


if __name__ == "__main__":
    replication()
    confirmation()
