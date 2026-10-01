"""P1-F5 - C4 proposal-family hard-semantic replication (CONFIRMATORY).

Question (protocol sections 10-18)::

    Does hard-competition semantic amplification replicate under DETR proposals?

On the matched Random-K5 / SameCategory-K5 cohort (same expression / image / target
proposal / scorer seed / K - only distractor composition differs) the frozen R1 -> E1b
semantic gain is measured and amplified by hard competition:

    d_rand  = AUROC(E1b) - AUROC(R1)   on the matched random control
    d_hard  = AUROC(E1b) - AUROC(R1)   on the same-category cohort
    A       = d_hard - d_rand          (diff-of-diffs, shared image-cluster draws)

A frozen backbone-neutral manipulation check (``winner_competitor_max_cos`` up,
``winner_top2_cos`` up, ``q_margin12`` down; V2 14-d semantic space) runs first; an
invalid manipulation yields ``C4 = NOT_ASSESSABLE`` (never FAIL).  When valid, the V2-G
G4 gate (d_hard >= 0.015 and amplification >= 0.01, CI lower bounds > 0) decides
replication.  R1 / E1b are the frozen heads - nothing is retrained. **0 new parameters.**

Run::

    conda activate deepminer
    python scripts/p1_f5_hard_semantic.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.semantic import hard_eval as heval  # noqa: E402

from scripts import p1_replication_core as core  # noqa: E402

OUT_DIR = ROOT / "results" / "v2_proposal_robustness" / "p1_f5_c4"
PRED_DIR = ROOT / "results" / "v2_proposal_robustness" / "predictions"
INFERENCE_REPORT = ROOT / "results" / "v2_proposal_robustness" / "inference_report.json"

MANIP_COLS = ["family", "scorer", "feature", "direction", "mean_hard", "mean_rand",
              "diff", "ci_low", "ci_high", "correct_direction", "ci_excludes_0",
              "n", "n_clusters"]
AMP_COLS = ["family", "scorer", "n", "delta_hard", "delta_hard_ci_low", "delta_hard_ci_high",
            "delta_rand", "delta_rand_ci_low", "delta_rand_ci_high",
            "amplification", "amplification_ci_low", "amplification_ci_high",
            "auroc_r1_hard", "auroc_e1b_hard", "auroc_r1_rand", "auroc_e1b_rand"]
SELECTIVE_COLS = ["family", "scorer", "regime", "e_aurc_r1", "e_aurc_e1b",
                  "rer50_r1", "rer50_e1b", "rer80_r1", "rer80_e1b",
                  "e1b_minus_r1_e_aurc_reduction", "e1b_minus_r1_rer50_gain"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p1-f5] {msg}", flush=True)


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


def matched_hard_ids(family: str) -> Set[int]:
    return core.build_hard_samples(family)["matched_ids"]


def _auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    from ccg.experiment.phase0a import SampleStats
    return reval._metric_fn("auroc_correct")(SampleStats.from_conf_correct(conf, correct))


def _selective_row(family: str, scorer: str, regime: str, hard: Dict[str, np.ndarray],
                   rand: Dict[str, np.ndarray]) -> Dict[str, Any]:
    def _point(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
        r1 = reval.point_metric_row(d["conf_stats"], d["correct"], probability=d["conf_stats"])
        e1b = reval.point_metric_row(d["conf_e1b"], d["correct"], probability=d["conf_e1b"])
        return {
            "e_aurc_r1": float(r1["e_aurc"]), "e_aurc_e1b": float(e1b["e_aurc"]),
            "rer50_r1": float(r1["rer_at_50"]), "rer50_e1b": float(e1b["rer_at_50"]),
            "rer80_r1": float(r1["rer_at_80"]), "rer80_e1b": float(e1b["rer_at_80"]),
            "e1b_minus_r1_e_aurc_reduction": float(r1["e_aurc"] - e1b["e_aurc"]),
            "e1b_minus_r1_rer50_gain": float(r1["rer_at_50"] - e1b["rer_at_50"]),
        }
    d = hard if regime == "hard" else rand
    return {"family": family, "scorer": scorer, "regime": regime, **_point(d)}


def run_c4_for(pred_dir: Path, family: str, matched_ids: Set[int]
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any],
                          Dict[str, Any], List[Dict[str, Any]]]:
    auroc_fn = reval._metric_fn("auroc_correct")
    manip_rows: List[Dict[str, Any]] = []
    amp_rows: List[Dict[str, Any]] = []
    selective_rows: List[Dict[str, Any]] = []
    per_seed: Dict[str, Dict[str, float]] = {}
    seed_valid_map: Dict[str, bool] = {}

    for scorer in core.SCORERS:
        hard = core.load_pred(pred_dir, family, "hard_k5", scorer, matched_ids)
        rand = core.load_pred(pred_dir, family, "random_k5", scorer, matched_ids)
        if not np.array_equal(hard["sentence_id"], rand["sentence_id"]):
            raise AssertionError(f"{family} {scorer}: hard/random rows not matched on the cohort")
        clusters = hard["image_id"].astype(np.int64)

        dir_ok = ci_excl = 0
        for feature, direction in core.MANIPULATION:
            col = core.V2_SEM_INDEX[feature]
            res = heval.paired_shift_bootstrap(
                hard["sem14"][:, col], rand["sem14"][:, col], clusters,
                n_replicates=core.BOOTSTRAP_REPLICATES, seed=core.BOOTSTRAP_SEED,
                ci=core.BOOTSTRAP_CI, name=feature)
            diff, lo, hi = float(res["diff"]), float(res["ci_low"]), float(res["ci_high"])
            correct_dir = diff > 0.0 if direction == "up" else diff < 0.0
            excludes = lo > 0.0 if direction == "up" else hi < 0.0
            dir_ok += int(correct_dir)
            ci_excl += int(excludes)
            manip_rows.append({
                "family": family, "scorer": scorer, "feature": feature, "direction": direction,
                "mean_hard": float(res["mean_hard"]), "mean_rand": float(res["mean_rand"]),
                "diff": diff, "ci_low": lo, "ci_high": hi,
                "correct_direction": bool(correct_dir), "ci_excludes_0": bool(excludes),
                "n": int(res["n"]), "n_clusters": int(res["n_clusters"]),
            })
        seed_valid_map[scorer] = core.manipulation_valid_seed(dir_ok, ci_excl)

        dh = reval.model_vs_model_bootstrap_row(
            hard["conf_e1b"], hard["correct"], hard["conf_stats"], hard["correct"], clusters,
            eval_split=core.POOLED, K=5, model_a=core.E1B_NAME, model_b=core.STATS_NAME,
            metrics=("auroc_correct",), replicates=core.BOOTSTRAP_REPLICATES,
            seed=core.BOOTSTRAP_SEED, ci=core.BOOTSTRAP_CI)[0]
        dr = reval.model_vs_model_bootstrap_row(
            rand["conf_e1b"], rand["correct"], rand["conf_stats"], rand["correct"], clusters,
            eval_split=core.POOLED, K=5, model_a=core.E1B_NAME, model_b=core.STATS_NAME,
            metrics=("auroc_correct",), replicates=core.BOOTSTRAP_REPLICATES,
            seed=core.BOOTSTRAP_SEED, ci=core.BOOTSTRAP_CI)[0]
        dod = heval.paired_diff_of_diffs_bootstrap(
            auroc_fn, (hard["conf_e1b"], hard["correct"]), (hard["conf_stats"], hard["correct"]),
            (rand["conf_e1b"], rand["correct"]), (rand["conf_stats"], rand["correct"]), clusters,
            n_replicates=core.BOOTSTRAP_REPLICATES, seed=core.BOOTSTRAP_SEED, ci=core.BOOTSTRAP_CI,
            metric_name="auroc_correct")
        amp_rows.append({
            "family": family, "scorer": scorer, "n": int(dh["n"]),
            "delta_hard": float(dh["diff"]), "delta_hard_ci_low": float(dh["ci_low"]),
            "delta_hard_ci_high": float(dh["ci_high"]),
            "delta_rand": float(dr["diff"]), "delta_rand_ci_low": float(dr["ci_low"]),
            "delta_rand_ci_high": float(dr["ci_high"]),
            "amplification": float(dod["diff"]), "amplification_ci_low": float(dod["ci_low"]),
            "amplification_ci_high": float(dod["ci_high"]),
            "auroc_r1_hard": _auroc(hard["conf_stats"], hard["correct"]),
            "auroc_e1b_hard": _auroc(hard["conf_e1b"], hard["correct"]),
            "auroc_r1_rand": _auroc(rand["conf_stats"], rand["correct"]),
            "auroc_e1b_rand": _auroc(rand["conf_e1b"], rand["correct"]),
        })
        per_seed[scorer] = {
            "delta_hard": float(dh["diff"]), "delta_hard_ci_low": float(dh["ci_low"]),
            "delta_rand": float(dr["diff"]),
            "amplification": float(dod["diff"]), "amplification_ci_low": float(dod["ci_low"]),
        }
        selective_rows.append(_selective_row(family, scorer, "hard", hard, rand))
        selective_rows.append(_selective_row(family, scorer, "random", hard, rand))

    n_valid = int(sum(seed_valid_map.values()))
    manipulation_ok = core.manipulation_aggregate_ok(n_valid, len(core.SCORERS))
    gate = core.c4_hard_gate(per_seed, manipulation_ok)
    manip_summary = {
        "metrics": [[f, d] for f, d in core.MANIPULATION],
        "per_seed_valid": seed_valid_map,
        "n_seed_valid": n_valid, "n_seeds": len(core.SCORERS),
        "rule": "per-seed: >=2/3 metrics correct dir AND >=1 CI excludes 0; ok = >=2/3 seeds",
        "manipulation_valid": bool(manipulation_ok),
    }
    return manip_rows, amp_rows, gate, manip_summary, selective_rows


def make_fig3(families: Dict[str, Dict[str, float]], fig_dir: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = ["DeltaRand", "DeltaHard"]
    rpn = [families["RPN"]["delta_rand_mean"], families["RPN"]["delta_hard_mean"]]
    detr = [families["DETR"]["delta_rand_mean"], families["DETR"]["delta_hard_mean"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - 0.18, rpn, width=0.34, color="#8f9fa8", label="RPN (frozen)")
    ax.bar(x + 0.18, detr, width=0.34, color="#b4635f", label="DETR (proposal-B)")
    ax.set_xticks(x, labels)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel("AUROC(E1b) - AUROC(R1)")
    ax.set_title("P1-Fig3 - Random vs Hard semantic incremental AUROC")
    ax.legend()
    p = fig_dir / "fig3_random_vs_hard_increment.png"
    fig.tight_layout()
    p_png = p
    fig.savefig(p_png, dpi=150)
    plt.close(fig)
    return [str(p_png.relative_to(ROOT))]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.out_dir)
    pred = Path(args.pred_dir)
    inference = json.loads(INFERENCE_REPORT.read_text(encoding="utf-8")) if INFERENCE_REPORT.exists() else {}
    hard_attr = {rep["family"]: rep.get("attrition", {}).get("same_category", {})
                 for rep in inference.get("families", [])}

    results: Dict[str, Any] = {}
    all_manip: List[Dict[str, Any]] = []
    all_amp: List[Dict[str, Any]] = []
    all_sel: List[Dict[str, Any]] = []
    matched_by_family: Dict[str, Set[int]] = {}

    for family in ("RPN", "DETR"):
        ids = matched_hard_ids(family)
        matched_by_family[family] = ids
        _log(f"{family}: matched hard/random K5 rows={len(ids)}")
        manip, amp, gate, msum, sel = run_c4_for(pred, family, ids)
        all_manip += manip
        all_amp += amp
        all_sel += sel
        results[family] = {
            "matched_rows": int(len(ids)),
            "attrition": hard_attr.get(family, {}),
            "manipulation": msum,
            "gate": gate,
            "c4_verdict": core.c4_verdict_label(gate),
        }

    # matched proposal-family intersection (section 18, secondary)
    inter_ids = core.matched_expression_ids(matched_by_family["RPN"], matched_by_family["DETR"])
    inter: Dict[str, Any] = {"n_expressions": int(len(inter_ids))}
    for family in ("RPN", "DETR"):
        _, _, gate, _, _ = run_c4_for(pred, family, inter_ids)
        inter[family] = {
            "delta_rand_mean": gate["delta_rand_mean"],
            "delta_hard_mean": gate["delta_hard_mean"],
            "amplification_mean": gate["amplification_mean"],
            "manipulation_ok": gate["manipulation_ok"],
        }

    rpn_c4_ref = core.load_rpn_c4_reference()

    fig_fam = {
        f: {"delta_rand_mean": results[f]["gate"]["delta_rand_mean"],
            "delta_hard_mean": results[f]["gate"]["delta_hard_mean"]}
        for f in ("RPN", "DETR")
    }
    figures = make_fig3(fig_fam, out / "figures")

    _write_csv(out / "c4_manipulation.csv", all_manip, MANIP_COLS)
    _write_csv(out / "c4_amplification.csv", all_amp, AMP_COLS)
    _write_csv(out / "c4_selective.csv", all_sel, SELECTIVE_COLS)

    overall = core.overall_p1_verdict(
        c1_replicated=True,  # placeholder replaced below from f4 verdict file if present
        c4_verdict=results["DETR"]["c4_verdict"],
    )
    f4_file = ROOT / "results" / "v2_proposal_robustness" / "p1_f4_c1" / "f4_verdict.json"
    if f4_file.exists():
        c1_rep = bool(json.loads(f4_file.read_text(encoding="utf-8"))
                      ["c1_by_family"]["DETR"]["gate"]["replicated"])
    else:
        c1_rep = False
    overall = core.overall_p1_verdict(c1_replicated=c1_rep, c4_verdict=results["DETR"]["c4_verdict"])

    payload = {
        "artifact": "v2_p1_f5_c4_proposal_family_hard_semantic",
        "protocol": "V2-P1 P1-F5 C4 replication (V2-G G4 applied on DETR proposals)",
        "classification": "CONFIRMATORY",
        "cohort": "matched Random-K5 / SameCategory-K5 (distractor composition only)",
        "bootstrap": {"replicates": core.BOOTSTRAP_REPLICATES, "seed": core.BOOTSTRAP_SEED,
                      "ci": core.BOOTSTRAP_CI, "resample_unit": "image",
                      "shared_cluster_draws_for_hard_vs_random": True},
        "new_training_parameters": 0,
        "c4_by_family": {f: {k: results[f][k] for k in
                             ("matched_rows", "attrition", "manipulation", "gate", "c4_verdict")}
                         for f in ("RPN", "DETR")},
        "detr_C4_PROPOSAL_FAMILY_REPLICATED": results["DETR"]["c4_verdict"],
        "rpn_C4_replication_reference": results["RPN"]["c4_verdict"],
        "matched_proposal_family_intersection_secondary": inter,
        "rpn_frozen_reference_from_artifacts": rpn_c4_ref,
        "overall_p1_verdict": overall,
        "figures": figures,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out / "f5_verdict.json", payload)
    _log(f"RPN C4 verdict={results['RPN']['c4_verdict']}  "
         f"DETR C4 verdict={results['DETR']['c4_verdict']}  "
         f"manip_valid(DETR)={results['DETR']['manipulation']['manipulation_valid']}")
    _log(f"DETR d_hard={results['DETR']['gate']['delta_hard_mean']:+.4f} "
         f"amplification={results['DETR']['gate']['amplification_mean']:+.4f}")
    print(json.dumps(overall, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P1-F5 C4 proposal-family hard-semantic replication")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
