#!/usr/bin/env python
"""V2-G final consistency audit + main-table generator (item 31 / sections 3-4).

Reads ONLY the raw G3 / G4-PhaseA / G4-PhaseB / G5 / V1-frozen artifacts and:

  1. Recomputes every headline value from its own per-seed / per-cell raw rows
     (never from the chat report text) and asserts internal consistency.  Any
     mismatch beyond tolerance raises STOP and exits non-zero.
  2. Emits results/v2_backbone_generalization/final_summary.csv.  The B0 (V1
     OpenCLIP B/32) row is loaded from the V1 frozen artifacts, never hard-coded.

Unit note: g3 stores rer50_drop as an ABSOLUTE probability drop (fraction);
phase05 stores it in percentage points.  The table column RER50_drop_pp is
percentage points for every backbone (g3 value x 100).  E-AURC worsening is a
dimensionless relative-worsening ratio for every backbone.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "results" / "v2_backbone_generalization"
_V1_PHASE0B = _REPO / "results" / "phase0b_independent"
_V1_PHASE05 = _REPO / "results" / "phase05_score_sufficiency"
_V1_PHASE1F = _REPO / "results" / "phase1f_hard_semantic"
_OUT_CSV = _ROOT / "final_summary.csv"

_TOL = 1e-9  # internal recompute tolerance (values are literal means of stored rows)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _agg_metrics(path: Path) -> Dict[tuple, float]:
    """aggregate.csv -> {(eval_split, K, variant, metric): mean}."""
    out: Dict[tuple, float] = {}
    for row in _load_csv(path):
        key = (row["eval_split"], int(float(row["K"])), row["variant"], row["metric"])
        out[key] = float(row["mean"])
    return out


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(list(xs)))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho via mean-rank Pearson (matches run_v2g_hard._spearman)."""
    def ranks(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size, dtype=np.float64)
        i = 0
        while i < a.size:
            j = i
            while j + 1 < a.size and a[order[j + 1]] == a[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


class Auditor:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.checks: int = 0

    def eq(self, label: str, got: float, want: float, tol: float = _TOL) -> None:
        self.checks += 1
        if not (math.isfinite(got) and math.isfinite(want)) or abs(got - want) > tol:
            self.failures.append(f"{label}: got={got!r} vs want={want!r} (tol={tol})")

    def true(self, label: str, cond: bool) -> None:
        self.checks += 1
        if not cond:
            self.failures.append(f"{label}: expected True, got False")

    def close(self, label: str, got: float, want: float, tol: float) -> None:
        self.eq(label, got, want, tol=tol)


# ---------------------------------------------------------------------------
# Cardinality (G3 for B1/B2; phase05 for B0)
# ---------------------------------------------------------------------------
def audit_g3(aud: Auditor, g3: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for bb, blk in g3["backbones"].items():
        for metric in ("auc_drop_K5_K50", "eaurc_worsening_K5_K50", "rer50_drop_K5_K50"):
            d = blk[metric]
            recomputed = _mean(d["per_seed_effects"])
            aud.eq(f"g3/{bb}/{metric} effect_mean", recomputed, d["effect_mean"])
        # route pass requires all seeds CI exclude 0 on the metrics that gate each route
        aud.true(f"g3/{bb} auc all_seeds_ci_exclude_0", blk["auc_drop_K5_K50"]["all_seeds_ci_exclude_0"])
        out[bb] = {
            "auc_drop": blk["auc_drop_K5_K50"]["effect_mean"],
            "eaurc_worsening": blk["eaurc_worsening_K5_K50"]["effect_mean"],
            "rer50_drop_pp": blk["rer50_drop_K5_K50"]["effect_mean"] * 100.0,
            "cardinality": bool(blk["CARDINALITY_REPLICATED"]),
        }
    return out


# ---------------------------------------------------------------------------
# Hard semantic (G4 Phase B) for one backbone
# ---------------------------------------------------------------------------
def audit_phaseB(aud: Auditor, tag: str, gate: Mapping[str, Any],
                 amp_rows: List[Dict[str, str]], manip_rows: List[Dict[str, str]],
                 dose_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    # 1) matched-cell row identity: every K5 cell shares one n, every K10 another
    n_k5 = {int(r["n"]) for r in amp_rows if int(r["K"]) == 5}
    n_k10 = {int(r["n"]) for r in amp_rows if int(r["K"]) == 10}
    aud.true(f"phaseB/{tag} K5 matched n consistent", len(n_k5) == 1)
    aud.true(f"phaseB/{tag} K10 matched n consistent", len(n_k10) == 1)
    aud.true(f"phaseB/{tag} K5!=K10 n", n_k5 != n_k10)
    # 2) amplification = delta_hard - delta_rand (rowwise)
    for r in amp_rows:
        aud.eq(f"phaseB/{tag} amp=Δhard-Δrand {r['scorer']} K{r['K']}",
               float(r["amplification"]),
               float(r["delta_hard"]) - float(r["delta_rand"]), tol=1e-12)
    # 3) K5 3-seed means == gate headline
    k5 = [r for r in amp_rows if int(r["K"]) == 5]
    aud.eq(f"phaseB/{tag} delta_hard_mean", _mean([float(r["delta_hard"]) for r in k5]),
           gate["delta_hard_mean"])
    aud.eq(f"phaseB/{tag} amplification_mean", _mean([float(r["amplification"]) for r in k5]),
           gate["amplification_mean"])
    aud.eq(f"phaseB/{tag} delta_hard_ci_low_mean", _mean([float(r["delta_hard_ci_low"]) for r in k5]),
           gate["delta_hard_ci_low_mean"])
    aud.eq(f"phaseB/{tag} amplification_ci_low_mean", _mean([float(r["amplification_ci_low"]) for r in k5]),
           gate["amplification_ci_low_mean"])
    # 4) gate thresholds match the frozen contract verdict logic
    pass_dh = gate["delta_hard_mean"] >= 0.015 and gate["delta_hard_ci_low_mean"] > 0.0
    pass_amp = gate["amplification_mean"] >= 0.01 and gate["amplification_ci_low_mean"] > 0.0
    aud.true(f"phaseB/{tag} pass_delta_hard flag", pass_dh == gate["pass_delta_hard"])
    aud.true(f"phaseB/{tag} pass_amplification flag", pass_amp == gate["pass_amplification"])
    expected_verdict = ("NOT_ASSESSABLE" if not gate["manipulation_ok"]
                        else "HARD_SEMANTIC_REPLICATED" if (pass_dh and pass_amp)
                        else "NOT_REPLICATED")
    aud.true(f"phaseB/{tag} verdict", expected_verdict == gate["verdict"])
    # 5) 3 seeds present, distinct, reported per-seed (never pooled-only)
    aud.true(f"phaseB/{tag} 3 seeds in gate", len(gate["per_seed_delta_hard"]) == 3)
    aud.true(f"phaseB/{tag} 3 seeds in amp csv", len({r["scorer"] for r in amp_rows}) == 3)
    # 6) manipulation recompute (freeze item 7): per-seed >=2/3 dir AND >=1 CI-excl
    seed_metric: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in manip_rows:
        s = seed_metric.setdefault(r["scorer"], {})
        s[r["feature"]] = {"dir": int(r["correct_direction"] in ("True", "true", "1")),
                           "ci": int(r["ci_excludes_0"] in ("True", "true", "1"))}
    n_valid = 0
    for scorer, mm in seed_metric.items():
        dir_ok = sum(v["dir"] for v in mm.values())
        ci_ok = sum(v["ci"] for v in mm.values())
        seed_valid = dir_ok >= 2 and ci_ok >= 1
        n_valid += int(seed_valid)
    recomputed_manip_ok = n_valid >= 2  # >=2/3 seeds valid
    aud.true(f"phaseB/{tag} manipulation_ok recompute", recomputed_manip_ok == gate["manipulation_ok"])
    # shared image-cluster draws: one cluster unit (image_id) across all paired shifts
    aud.true(f"phaseB/{tag} shared image clusters (single n_clusters)",
             len({int(r["n_clusters"]) for r in manip_rows}) == 1)
    # manipulation features must be backbone-neutral primary features only
    from ccg.v2.semantic_features import MANIPULATION_METRICS  # noqa: E402
    allowed = {name for name, _ in MANIPULATION_METRICS}
    aud.true(f"phaseB/{tag} manipulation features subset of primary",
             all(r["feature"] in allowed for r in manip_rows))
    # 7) dose response recompute (3-seed mean delta per level, spearman on levels)
    by_level: Dict[int, List[float]] = {}
    for r in dose_rows:
        by_level.setdefault(int(r["level"]), []).append(float(r["delta_auroc"]))
    levels = sorted(by_level)
    delta_by_level = [_mean(by_level[m]) for m in levels]
    recomputed_rho = _spearman(levels, delta_by_level)
    aud.eq(f"phaseB/{tag} dose spearman_rho", recomputed_rho, gate["dose"]["spearman_rho"])
    for m in levels:
        aud.eq(f"phaseB/{tag} dose delta at m={m}", delta_by_level[levels.index(m)],
               gate["dose"]["delta_mean_by_level"][str(m)])
    # 8) STOP reproduction fidelity (A8.4): every max|delta| <= 1e-4
    for key, val in gate["stop_checks_max_abs_delta"].items():
        aud.true(f"phaseB/{tag} STOP {key} <=1e-4", float(val) <= 1e-4)
    # 9) dose cohort immutable: identical n across all seeds/levels (same manifest)
    nset = {int(r["n"]) for r in dose_rows}
    aud.true(f"phaseB/{tag} dose cohort immutable (single n)", len(nset) == 1)
    return {
        "delta_rand": _mean([float(r["delta_rand"]) for r in k5]),
        "delta_hard": gate["delta_hard_mean"],
        "amplification": gate["amplification_mean"],
        "manipulation": bool(gate["manipulation_ok"]),
        "dose_rho": gate["dose"]["spearman_rho"],
        "hard": gate["verdict"] == "HARD_SEMANTIC_REPLICATED",
        "verdict": gate["verdict"],
    }


# ---------------------------------------------------------------------------
# B0 row from V1 frozen artifacts
# ---------------------------------------------------------------------------
def load_b0(aud: Auditor) -> Dict[str, Any]:
    p0b = _agg_metrics(_V1_PHASE0B / "aggregate.csv")
    p05 = _load_json(_V1_PHASE05 / "sufficiency_gate.json")
    p1f = _load_json(_V1_PHASE1F / "gate.json")
    k5a = p0b[("__pooled__", 5, "global_T_corrected", "auroc_correct")]
    k50a = p0b[("__pooled__", 50, "global_T_corrected", "auroc_correct")]
    k5acc = p0b[("__pooled__", 5, "global_T_corrected", "accuracy")]
    k50acc = p0b[("__pooled__", 50, "global_T_corrected", "accuracy")]
    # B0 cardinality from the frozen V1 score-sufficiency gate (pooled-test K=50)
    cell = next(c for c in p05["cells_seed_mean"]
                if c["split"] == "__pooled_test__" and int(c["K"]) == 50)
    # B0 hard from phase1f (V1 A8 comparison; different definitions - see doc)
    dh = p1f["cell_samecat_k5"]["delta_auroc"]
    dod = p1f["diff_of_diffs_auroc"]["diff"]
    drand = dh - dod
    aud.true("B0 cardinality GO", p05["verdict_seed_mean"]["verdict"] == "GO_candidate_embeddings")
    aud.true("B0 hard manipulation_ok", p1f["manipulation"]["manipulation_ok"])
    return {
        "k5_acc": k5acc, "k50_acc": k50acc, "k5_auroc": k5a, "k50_auroc": k50a,
        "eaurc_worsening": cell["e_aurc_worsening"], "rer50_drop_pp": cell["rer50_drop"],
        "cardinality": True, "delta_rand": drand, "delta_hard": dh,
        "amplification": dod, "manipulation": True, "dose_rho": None,
        "hard": p1f["verdict"]["confirmed"], "verdict": "CONFIRMED (V1 A8)",
    }


def main() -> int:
    sys.path.insert(0, str(_REPO / "src"))
    aud = Auditor()

    g3 = _load_json(_ROOT / "g3_cardinality_gate.json")
    card = audit_g3(aud, g3)

    phaseB: Dict[str, Dict[str, Any]] = {}
    for tag in ("b1", "b2"):
        d = _ROOT / "g4_phaseB" / tag
        phaseB[tag] = audit_phaseB(
            aud, tag, _load_json(d / "gate.json"),
            _load_csv(d / "amplification.csv"),
            _load_csv(d / "manipulation_check.csv"),
            _load_csv(d / "dose_response.csv"),
        )

    b0 = load_b0(aud)

    # G5 overall recompute from the per-backbone booleans (deterministic rule)
    g5 = _load_json(_ROOT / "g4_phaseB" / "g5_overall_gate.json")
    rows_by_bb = {r["backbone"]: r for r in g5["cross_backbone"]}
    n_both = sum(int(rows_by_bb[b]["cardinality_replicated"] and rows_by_bb[b]["hard_semantic_replicated"])
                 for b in rows_by_bb)
    authorized = n_both >= math.ceil(2 / 3 * len(rows_by_bb))
    aud.eq("g5 n_both_replicated", n_both, g5["n_both_replicated"])
    aud.true("g5 authorized recompute", authorized == g5["V2_METHOD_DEVELOPMENT_AUTHORIZED"])
    # B1/B2 entries must match gate + g3 (loaded, not assumed)
    for tag, key in (("b1", "B1_openclip_b16"), ("b2", "B2_siglip_b16")):
        aud.true(f"g5 {key} cardinality", rows_by_bb[key]["cardinality_replicated"] == card["B1_openclip_b16" if tag == "b1" else "B2_siglip_b16"]["cardinality"])
        aud.true(f"g5 {key} hard", rows_by_bb[key]["hard_semantic_replicated"] == phaseB[tag]["hard"])

    # ---- final_summary.csv ----
    def phase0b_row(tag: str) -> Dict[str, float]:
        m = _agg_metrics(_ROOT / f"{tag}_phase0b" / "aggregate.csv")
        return {
            "k5_acc": m[("__pooled__", 5, "global_T_corrected", "accuracy")],
            "k50_acc": m[("__pooled__", 50, "global_T_corrected", "accuracy")],
            "k5_auroc": m[("__pooled__", 5, "global_T_corrected", "auroc_correct")],
            "k50_auroc": m[("__pooled__", 50, "global_T_corrected", "auroc_correct")],
        }

    table = [
        {"Backbone": "B0_openclip_b32", **b0},
        {"Backbone": "B1_openclip_b16", **card["B1_openclip_b16"], **phaseB["b1"], **phase0b_row("b1")},
        {"Backbone": "B2_siglip_b16", **card["B2_siglip_b16"], **phaseB["b2"], **phase0b_row("b2")},
    ]
    cols = ["Backbone", "k5_acc", "k50_acc", "k5_auroc", "k50_auroc",
            "eaurc_worsening", "rer50_drop_pp", "cardinality",
            "delta_rand", "delta_hard", "amplification", "manipulation",
            "dose_rho", "hard"]
    header = ["Backbone", "K5_Acc", "K50_Acc", "K5_AUROC", "K50_AUROC",
              "E_AURC_worsening", "RER50_drop_pp", "Cardinality",
              "DeltaRand", "DeltaHard", "Amplification", "Manipulation",
              "Dose_rho", "HardSemantic"]
    with open(_OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in table:
            w.writerow([(f"{row[c]:.6f}" if isinstance(row[c], float) else
                         ("YES" if row[c] is True else "NO" if row[c] is False else
                          ("NA_V1" if row[c] is None else row[c]))) for c in cols])

    # ---- report ----
    print("V2-G FINAL SUMMARY (recomputed from raw artifacts)")
    print(_OUT_CSV.relative_to(_REPO).as_posix())
    for row in table:
        rho = "NA_V1" if row["dose_rho"] is None else f"{row['dose_rho']:.3f}"
        print(f"  {row['Backbone']:<16} K5_AUROC={row['k5_auroc']:.4f} "
              f"K50_AUROC={row['k50_auroc']:.4f} dRand={row['delta_rand']:+.5f} "
              f"dHard={row['delta_hard']:+.5f} A={row['amplification']:+.5f} "
              f"card={row['cardinality']} manip={row['manipulation']} rho={rho} hard={row['hard']}")
    print(f"\nchecks_run={aud.checks} failures={len(aud.failures)}")
    if aud.failures:
        print("\n*** AUDIT STOP *** inconsistent headline values:")
        for f in aud.failures:
            print("  - " + f)
        return 3
    print("AUDIT_OK: every headline value recomputed from raw artifacts is consistent.")
    print(f"authorized={authorized}  n_both_replicated={n_both}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
