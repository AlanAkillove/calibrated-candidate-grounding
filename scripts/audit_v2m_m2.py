#!/usr/bin/env python
"""V2-M M2 consistency audit: recompute the frozen M2 record from raw confidences.

Recompute chain (no model is trained, no score is regenerated):

    raw per-seed ``confidences.npz`` -> correctness / image clusters
    -> point metrics (5 models x 4 severity levels) -> paired image-cluster
    bootstrap (5000 reps, seed 0, shared draws) -> severity tables
    -> M2 gate verdicts

and compare every recomputed value against the committed M2 artifacts
(``m2_curriculum/point_metrics.csv`` / ``bootstrap_pairs.csv`` /
``severity_curve.csv`` / ``gate_diagnostics.csv`` / ``gate.json``).

The explicitly checked m=8 comparisons (Phase 0.2 audit list of the M2.5
round): LCR-E1b, LCR-Aggregate-MLP, LCR-LCR-noGate, LCR-R1 -- plus every
other severity level as a completeness guard, the full gate tree, and the
re-computable g/delta diagnostics.

Writes ``results/v2_local_competition/m2_curriculum/m2_audit.json``;
exit code 0 = PASS.

Usage
-----
    python scripts/audit_v2m_m2.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.reliability import evaluate as reval  # noqa: E402

__all__ = ["main"]

TOLERANCE = 1e-10
PRIMARY = "m8"
M8_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("LCR", "E1b"),
    ("LCR", "Aggregate-MLP"),
    ("LCR", "LCR-noGate"),
    ("LCR", "R1"),
)

POINT_FIELDS = (
    "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80",
    "aurc", "aurc_oracle", "ece_adaptive", "brier_binary", "nll_binary",
)
BOOT_FIELDS = ("diff", "ci_low", "ci_high", "mean_a", "mean_b")
BOOT_INT_FIELDS = ("n", "n_clusters", "n_replicates")
DIAG_FIELDS = ("mean_abs_gd", "mean_g", "mean_abs_d")


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_v2m_m2", _REPO / "scripts" / "run_v2m_m2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _abs(path: Any) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO / p


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Audit:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {name}" + (f" ({detail})" if detail else ""), flush=True)

    def close(self, name: str, recomputed: float, stored: float) -> float:
        delta = abs(float(recomputed) - float(stored))
        self.check(name, delta <= TOLERANCE, f"max|delta|={delta:.3e}")
        return delta


def _leaf_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
        b, (int, float, np.integer, np.floating)
    ):
        return abs(float(a) - float(b)) <= TOLERANCE
    return a == b


def _tree_diffs(
    prefix: str, recomputed: Any, stored: Any, out: List[str], limit: int = 40
) -> None:
    """Collect mismatching leaves between two nested JSON-like structures."""
    if len(out) >= limit:
        return
    if isinstance(recomputed, Mapping):
        if not isinstance(stored, Mapping):
            out.append(f"{prefix}: type mismatch ({type(recomputed)} vs {type(stored)})")
            return
        rec_map = {str(key): value for key, value in recomputed.items()}
        sto_map = {str(key): value for key, value in stored.items()}
        keys = sorted(set(rec_map) | set(sto_map))
        for key in keys:
            step = f"{prefix}.{key}" if prefix else key
            if key not in rec_map:
                out.append(f"{step}: missing in recomputed")
            elif key not in sto_map:
                out.append(f"{step}: missing in stored")
            else:
                _tree_diffs(step, rec_map[key], sto_map[key], out, limit)
        return
    if isinstance(recomputed, (list, tuple)):
        if not isinstance(stored, (list, tuple)):
            out.append(f"{prefix}: type mismatch ({type(recomputed)} vs {type(stored)})")
            return
        if len(recomputed) != len(stored):
            out.append(f"{prefix}: length {len(recomputed)} vs {len(stored)}")
            return
        for index, (item_r, item_s) in enumerate(zip(recomputed, stored)):
            _tree_diffs(f"{prefix}[{index}]", item_r, item_s, out, limit)
        return
    if not _leaf_equal(recomputed, stored):
        out.append(f"{prefix}: {recomputed!r} != {stored!r}")


class FaultyCounter:
    def __init__(self) -> None:
        self.count = 0

    def bump(self, steps: int = 1) -> None:
        self.count += steps


def main() -> int:
    started = time.perf_counter()
    runner = _load_runner()
    root = _abs(runner.OUT_ROOT)
    m2_dir = root / "m2_curriculum"
    point_csv = m2_dir / "point_metrics.csv"
    boot_csv = m2_dir / "bootstrap_pairs.csv"
    sev_csv = m2_dir / "severity_curve.csv"
    diag_csv = m2_dir / "gate_diagnostics.csv"
    gate_path = m2_dir / "gate.json"

    print("V2-M M2 consistency audit")
    print(f"  protocol sha256: {runner._sha256_file(_abs(runner.PROTOCOL_PATH))}")
    print(f"  manifest freeze sha256: {runner._sha256_file(m2_dir / 'manifests' / 'manifest_freeze.json')}")

    stored_point = {(r["cohort"], int(r["seed"]), r["model"]): r for r in _read_csv(point_csv)}
    stored_boot = {
        (r["cohort"], int(r["seed"]), f"{r['model_a']}-{r['model_b']}", r["metric"]): r
        for r in _read_csv(boot_csv)
    }
    stored_sev = {(r["level"], r["model"]): r for r in _read_csv(sev_csv)}
    diag_rows = [
        {
            "level": r["level"], "seed": int(r["seed"]), "n": int(r["n"]),
            **{key: float(r[key]) for key in (
                "mean_abs_gd", "mean_g", "mean_abs_d",
                "corr_winner_competitor_max_cos", "corr_winner_top2_cos", "corr_q_margin12",
            )},
        }
        for r in _read_csv(diag_csv)
    ]
    stored_diag = {(r["level"], int(r["seed"])): r for r in _read_csv(diag_csv)}
    stored_gate = json.loads(gate_path.read_text(encoding="utf-8"))

    audit = Audit()
    point_faults = FaultyCounter()
    boot_faults = FaultyCounter()
    max_point_delta = 0.0
    max_boot_delta = 0.0
    max_diag_delta = 0.0
    m8_records: Dict[str, Any] = {}

    print("[derivation]")
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    for seed in runner.SEEDS:
        z_path = m2_dir / f"seed_{int(seed)}" / "confidences.npz"
        with np.load(z_path) as z:
            present = set(z.files)
            expected = set()
            for level in runner.LEVELS_TEST:
                cohort = f"m{int(level)}"
                expected |= {f"{cohort}__{model}" for model in runner.MODELS}
                expected |= {f"{cohort}__correct", f"{cohort}__image_id",
                             f"{cohort}__gate", f"{cohort}__delta"}
            audit.check(
                f"seed{seed} confidences.npz key universe complete",
                present == expected,
                f"missing={sorted(expected - present)[:3]} extra={sorted(present - expected)[:3]}",
            )
            for level in runner.LEVELS_TEST:
                cohort = f"m{int(level)}"
                correct = np.asarray(z[f"{cohort}__correct"], dtype=np.uint8).astype(bool)
                clusters = np.asarray(z[f"{cohort}__image_id"], dtype=np.int64)
                gate_arr = np.asarray(z[f"{cohort}__gate"], dtype=np.float64)
                delta_arr = np.asarray(z[f"{cohort}__delta"], dtype=np.float64)
                rec_diag = {
                    "mean_abs_gd": float(np.mean(np.abs(gate_arr * delta_arr))),
                    "mean_g": float(np.mean(gate_arr)),
                    "mean_abs_d": float(np.mean(np.abs(delta_arr))),
                }
                stored = stored_diag[(cohort, int(seed))]
                for key, value in rec_diag.items():
                    max_diag_delta = max(
                        max_diag_delta, abs(value - float(stored[key]))
                    )
                for model in runner.MODELS:
                    conf = np.asarray(z[f"{cohort}__{model}"], dtype=np.float64)
                    row = reval.point_metric_row(conf, correct, probability=conf)
                    row.update({
                        "cohort": cohort, "K": int(runner.K_LEVEL), "seed": int(seed),
                        "model": model, "n": int(correct.size),
                        "n_images": int(np.unique(clusters).size),
                        "b3_accuracy": float(correct.mean()),
                    })
                    point_rows.append(row)
                    stored_row = stored_point[(cohort, int(seed), model)]
                    for key in ("n", "n_images") + POINT_FIELDS:
                        delta = abs(float(row[key]) - float(stored_row[key]))
                        if delta > TOLERANCE:
                            point_faults.bump()
                        max_point_delta = max(max_point_delta, delta)
                    const_delta = abs(float(row["b3_accuracy"]) - float(stored_row["b3_accuracy"]))
                    if const_delta > TOLERANCE:
                        point_faults.bump()
                    max_point_delta = max(max_point_delta, const_delta)
                for pair in runner.PAIR_PLAN[int(level)]:
                    conf_a = np.asarray(z[f"{cohort}__{pair[0]}"], dtype=np.float64)
                    conf_b = np.asarray(z[f"{cohort}__{pair[1]}"], dtype=np.float64)
                    rows = reval.model_vs_model_bootstrap_row(
                        conf_a, correct, conf_b, correct, clusters,
                        eval_split=runner.POOLED, K=int(runner.K_LEVEL),
                        model_a=pair[0], model_b=pair[1], metrics=runner.BOOT_METRICS,
                        replicates=int(runner.BOOT_REPLICATES), seed=int(runner.BOOT_SEED),
                        ci=float(runner.BOOT_CI),
                    )
                    for row in rows:
                        pair_label = f"{pair[0]}-{pair[1]}"
                        stored_row = stored_boot[(cohort, int(seed), pair_label, row["metric"])]
                        for key in BOOT_FIELDS + ("ci_level",):
                            delta = abs(float(row[key]) - float(stored_row[key]))
                            if delta > TOLERANCE:
                                boot_faults.bump()
                            max_boot_delta = max(max_boot_delta, delta)
                        for key in BOOT_INT_FIELDS:
                            if int(row[key]) != int(stored_row[key]):
                                boot_faults.bump()
                        row.update({"cohort": cohort, "seed": int(seed)})
                        boot_rows.append(row)
                        if cohort == PRIMARY and pair in M8_PAIRS:
                            m8_records.setdefault(pair_label, {}).setdefault(
                                f"seed{int(seed)}", {}
                            )[row["metric"]] = {
                                "recomputed_diff": float(row["diff"]),
                                "stored_diff": float(stored_row["diff"]),
                                "recomputed_ci": [float(row["ci_low"]), float(row["ci_high"])],
                                "stored_ci": [float(stored_row["ci_low"]), float(stored_row["ci_high"])],
                            }

    print("[recomputation vs stored artifacts]")
    audit.check(
        "point metrics reproduce (4 levels x 3 seeds x 5 models)",
        point_faults.count == 0,
        f"faulty values={point_faults.count} max|delta|={max_point_delta:.3e}",
    )
    audit.check(
        "paired bootstrap reproduces (10 pair-levels x 3 seeds x 4 metrics)",
        boot_faults.count == 0,
        f"faulty values={boot_faults.count} max|delta|={max_boot_delta:.3e}",
    )
    audit.check(
        "gate diagnostics (g/delta) reproduce from raw confidences",
        max_diag_delta <= TOLERANCE,
        f"max|delta|={max_diag_delta:.3e}",
    )

    # ---- m8 headline comparisons (explicitly listed) ------------------------
    print("[m=8 headline comparisons]")
    for pair in M8_PAIRS:
        label = f"{pair[0]}-{pair[1]}"
        worst = 0.0
        for metrics in m8_records[label].values():
            entry = metrics["auroc_correct"]
            worst = max(
                worst,
                abs(entry["recomputed_diff"] - entry["stored_diff"]),
                abs(entry["recomputed_ci"][0] - entry["stored_ci"][0]),
                abs(entry["recomputed_ci"][1] - entry["stored_ci"][1]),
            )
        audit.check(
            f"m8 {label} (auroc diff + CI reproduce)",
            worst <= TOLERANCE,
            f"max|delta|={worst:.3e}",
        )

    # ---- severity curve -----------------------------------------------------
    print("[severity curve]")
    rec_sev = {f"{r['level']}|{r['model']}": r for r in runner._severity_rows(point_rows, list(runner.SEEDS))}
    sev_faults = 0
    max_sev_delta = 0.0
    for key, rec_row in rec_sev.items():
        stored_row = stored_sev[tuple(key.split("|"))]
        for field in ("auroc_mean", "auroc_std", "e_aurc_mean", "e_aurc_std",
                      "rer50_mean", "rer50_std", "rer80_mean", "rer80_std"):
            delta = abs(float(rec_row[field]) - float(stored_row[field]))
            if delta > TOLERANCE:
                sev_faults += 1
            max_sev_delta = max(max_sev_delta, delta)
        for field in ("n_rows", "n_seeds"):
            if int(rec_row[field]) != int(stored_row[field]):
                sev_faults += 1
    audit.check(
        "severity_curve.csv reproduces (4 levels x 5 models)",
        sev_faults == 0,
        f"faulty values={sev_faults} max|delta|={max_sev_delta:.3e}",
    )

    # ---- rebuilt gate -------------------------------------------------------
    print("[gate rebuild]")
    rec_gate = runner._build_gate(point_rows, boot_rows, diag_rows, list(runner.SEEDS))
    # JSON round-trip normalises int dict keys (per_seed) exactly like the stored file.
    rec_gate = json.loads(json.dumps(rec_gate, default=runner._json_default))
    tree_out: List[str] = []
    _tree_diffs("", rec_gate, stored_gate, tree_out, limit=40)
    audit.check(
        "M2 gate tree reproduces (verdicts + comparisons + diagnostics)",
        not tree_out,
        f"mismatches={len(tree_out)}" + (f" first={tree_out[0]}" if tree_out else ""),
    )
    audit.close(
        "M2_GO.delta1_LCR_vs_E1b.delta",
        rec_gate["gates"]["M2_GO"]["delta1_LCR_vs_E1b"]["delta"],
        stored_gate["gates"]["M2_GO"]["delta1_LCR_vs_E1b"]["delta"],
    )
    audit.close(
        "M2_GO.delta2_LCR_vs_AggregateMLP.delta",
        rec_gate["gates"]["M2_GO"]["delta2_LCR_vs_AggregateMLP"]["delta"],
        stored_gate["gates"]["M2_GO"]["delta2_LCR_vs_AggregateMLP"]["delta"],
    )
    for name, stored, recomputed in (
        ("M2_GO.verdict", stored_gate["gates"]["M2_GO"]["verdict"], rec_gate["gates"]["M2_GO"]["verdict"]),
        ("M2_GO.condition_delta1", stored_gate["gates"]["M2_GO"]["condition_delta1_ge_0.010_and_ci_low_gt_0"], rec_gate["gates"]["M2_GO"]["condition_delta1_ge_0.010_and_ci_low_gt_0"]),
        ("M2_GO.condition_delta2", stored_gate["gates"]["M2_GO"]["condition_delta2_gt_0_and_ci_low_gt_0"], rec_gate["gates"]["M2_GO"]["condition_delta2_gt_0_and_ci_low_gt_0"]),
        ("capacity_case.case", stored_gate["gates"]["capacity_case"]["case"], rec_gate["gates"]["capacity_case"]["case"]),
        ("gate_contribution.unsupported", stored_gate["gates"]["gate_contribution"]["unsupported"], rec_gate["gates"]["gate_contribution"]["unsupported"]),
        ("selective_metrics.method_signal_mixed", stored_gate["gates"]["selective_metrics"]["method_signal_mixed"], rec_gate["gates"]["selective_metrics"]["method_signal_mixed"]),
        ("authorization.V2-MG", stored_gate["authorization"]["V2-MG"], rec_gate["authorization"]["V2-MG"]),
        ("authorization.B1/B2", stored_gate["authorization"]["B1/B2"], rec_gate["authorization"]["B1/B2"]),
        ("authorization.v2mg_not_authorized", stored_gate["authorization"]["v2mg_not_authorized"], rec_gate["authorization"]["v2mg_not_authorized"]),
    ):
        audit.check(f"{name}: stored={stored} recomputed={recomputed}", stored == recomputed)

    passed = all(entry["ok"] for entry in audit.checks)
    payload = {
        "artifact": "v2m_m2_audit",
        "passed": bool(passed),
        "protocol_sha256": runner._sha256_file(_abs(runner.PROTOCOL_PATH)),
        "manifest_freeze_sha256": runner._sha256_file(m2_dir / "manifests" / "manifest_freeze.json"),
        "bootstrap": {
            "replicates": int(runner.BOOT_REPLICATES), "seed": int(runner.BOOT_SEED),
            "ci": float(runner.BOOT_CI), "metrics": list(runner.BOOT_METRICS),
        },
        "m8_headline": m8_records,
        "max_point_delta": max_point_delta,
        "max_bootstrap_delta": max_boot_delta,
        "max_diag_delta": max_diag_delta,
        "max_severity_delta": max_sev_delta,
        "gate_tree_mismatches": tree_out,
        "checks": audit.checks,
        "runtime_sec": round(time.perf_counter() - started, 1),
    }
    out_path = m2_dir / "m2_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"V2M_M2_AUDIT_{'PASS' if passed else 'FAIL'} "
        f"checks={len(audit.checks)} failed={sum(1 for c in audit.checks if not c['ok'])} "
        f"max_point_delta={max_point_delta:.3e} max_boot_delta={max_boot_delta:.3e} "
        f"max_diag_delta={max_diag_delta:.3e} runtime={time.perf_counter() - started:.1f}s"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
