#!/usr/bin/env python
"""V2-M3 B0 consistency audit: recompute the developmental audit and compare.

Recompute chain (read-only; experts frozen; no expert training, no tuning):

    frozen M1/M2 artifacts -> p_R / p_C on the frozen K=10 cells
    -> per-backbone CompetitionIndex (train-only CDF)
    -> StaticMix / AdaptiveMix fitted on reliability_tune m={0,2,4}
    -> frozen test cells m in {0,2,4,8}: 5-model inference + metrics
    -> paired image-cluster bootstrap (5000 reps, seed 0)
    -> macro bootstrap (shared draws across severity)

The whole B0 pipeline is re-run into a scratch directory and every value is
compared against the committed B0 artifacts
(``m3_mixture/b0/point_metrics.csv`` / ``macro_metrics.csv`` /
``bootstrap_pairs.csv`` / ``alpha_diagnostics.csv`` / ``regret.csv`` /
``calibration.csv`` / ``summary.json``).

Writes ``results/v2_local_competition/m3_mixture/b0/m3_b0_audit.json``;
exit code 0 = PASS.

Usage
-----
    python scripts/audit_v2m_m3_b0.py
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

__all__ = ["main"]

TOLERANCE = 1e-10
M3_DIR = Path("results/v2_local_competition/m3_mixture")
B0_DIR = M3_DIR / "b0"
RECOMPUTE_DIR = B0_DIR / "_audit_recompute"
RECOMPUTE_LOG = Path("logs_v2m_m3_b0_audit_recompute.txt")

POINT_FIELDS = (
    "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80",
    "ece_adaptive", "brier_binary", "nll_binary",
)
MACRO_FIELDS = ("macro_auroc", "macro_e_aurc", "macro_rer50", "macro_rer80")
BOOT_FLOAT_FIELDS = ("diff", "ci_low", "ci_high", "mean_a", "mean_b", "ci_level")
BOOT_INT_FIELDS = ("K", "n", "n_clusters", "n_replicates")
ALPHA_FIELDS = ("a_mean", "a_median", "a_p10", "a_p90", "alpha_mean", "alpha_median")
REGRET_FIELDS = ("oracle_auroc", "regret")
CALIB_FIELDS = ("mean_pred", "accuracy", "ece_adaptive", "brier_binary", "nll_binary")


def _load_module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        for key in sorted(set(rec_map) | set(sto_map)):
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


def _compare_rows(
    audit: Audit,
    name: str,
    rec_rows: Sequence[Mapping[str, Any]],
    sto_rows: Sequence[Mapping[str, Any]],
    *,
    key_fn: Any,
    float_fields: Sequence[str],
    int_fields: Sequence[str] = (),
) -> float:
    rec_map = {key_fn(r): r for r in rec_rows}
    sto_map = {key_fn(r): r for r in sto_rows}
    if set(rec_map) != set(sto_map):
        audit.check(
            name, False,
            f"row universe differs: recomputed={len(rec_map)} stored={len(sto_map)}",
        )
        return float("inf")
    faults = 0
    max_delta = 0.0
    for key, rec in rec_map.items():
        sto = sto_map[key]
        for field in float_fields:
            delta = abs(float(rec[field]) - float(sto[field]))
            if delta > TOLERANCE:
                faults += 1
            max_delta = max(max_delta, delta)
        for field in int_fields:
            if int(rec[field]) != int(sto[field]):
                faults += 1
    audit.check(name, faults == 0, f"rows={len(rec_map)} faults={faults} max|delta|={max_delta:.3e}")
    return max_delta


def main() -> int:
    started = time.perf_counter()
    print("V2-M3 B0 consistency audit")
    m3 = _load_module("run_v2m_m3_b0", "scripts/run_v2m_m3_b0.py")

    # ---- recompute (read-only chain; identical frozen inputs) -----------------
    print("[recompute]")
    print(f"  scratch dir: {RECOMPUTE_DIR}")
    t0 = time.perf_counter()
    with open(_REPO / RECOMPUTE_LOG, "w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
        rc = m3.main(
            [
                "--out-dir", str(RECOMPUTE_DIR),
                "--bootstrap-replicates", "5000",
                "--bootstrap-seed", "0",
                "--ci", "0.95",
            ]
        )
    if rc != 0:
        print(f"  recompute failed with rc={rc}; see {RECOMPUTE_LOG}")
        return 1
    print(f"  recompute done ({time.perf_counter() - t0:.1f}s; log {RECOMPUTE_LOG})")

    audit = Audit()
    print("[recomputation vs committed artifacts]")

    max_point = _compare_rows(
        audit,
        "point_metrics.csv reproduces (4 levels x 3 seeds x 5 models)",
        _read_csv(RECOMPUTE_DIR / "point_metrics.csv"),
        _read_csv(B0_DIR / "point_metrics.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"]), r["model"]),
        float_fields=POINT_FIELDS,
        int_fields=("K", "n", "n_images"),
    )
    max_macro = _compare_rows(
        audit,
        "macro_metrics.csv reproduces (3 seeds x 5 models)",
        _read_csv(RECOMPUTE_DIR / "macro_metrics.csv"),
        _read_csv(B0_DIR / "macro_metrics.csv"),
        key_fn=lambda r: (int(r["seed"]), r["model"]),
        float_fields=MACRO_FIELDS,
    )
    max_boot = _compare_rows(
        audit,
        "bootstrap_pairs.csv reproduces (per-level pairs + macro)",
        _read_csv(RECOMPUTE_DIR / "bootstrap_pairs.csv"),
        _read_csv(B0_DIR / "bootstrap_pairs.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"]), r["model_a"], r["model_b"], r["metric"]),
        float_fields=BOOT_FLOAT_FIELDS,
        int_fields=BOOT_INT_FIELDS,
    )
    max_alpha = _compare_rows(
        audit,
        "alpha_diagnostics.csv reproduces (4 levels x 3 seeds)",
        _read_csv(RECOMPUTE_DIR / "alpha_diagnostics.csv"),
        _read_csv(B0_DIR / "alpha_diagnostics.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"])),
        float_fields=ALPHA_FIELDS,
        int_fields=("n",),
    )
    max_regret = _compare_rows(
        audit,
        "regret.csv reproduces (4 levels x 3 seeds x 5 models)",
        _read_csv(RECOMPUTE_DIR / "regret.csv"),
        _read_csv(B0_DIR / "regret.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"]), r["model"]),
        float_fields=REGRET_FIELDS,
    )
    max_calib = _compare_rows(
        audit,
        "calibration.csv reproduces (4 levels x 3 seeds x 5 models)",
        _read_csv(RECOMPUTE_DIR / "calibration.csv"),
        _read_csv(B0_DIR / "calibration.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"]), r["model"]),
        float_fields=CALIB_FIELDS,
    )

    # ---- summary tree (fits / delta_macro / safety / alpha / regret / gates) --
    rec_summary = json.loads((RECOMPUTE_DIR / "summary.json").read_text(encoding="utf-8"))
    sto_summary = json.loads((B0_DIR / "summary.json").read_text(encoding="utf-8"))
    tree_out: List[str] = []
    _tree_diffs("", rec_summary, sto_summary, tree_out, limit=40)
    audit.check(
        "summary.json tree reproduces (fits + delta_macro + safety + alpha + regret)",
        not tree_out,
        f"mismatches={len(tree_out)}" + (f" first={tree_out[0]}" if tree_out else ""),
    )

    print("[B0 developmental headline]")
    delta = rec_summary["delta_macro"]["vs_static"]
    safety0 = rec_summary["extreme_regime_safety"]["m0_adaptive_minus_random"]
    safety8 = rec_summary["extreme_regime_safety"]["m8_adaptive_minus_curriculum"]
    print(
        f"  delta_macro(A-S)={float(delta['delta']):+.4f} "
        f"CI[{float(delta['ci_low']):+.4f}, {float(delta['ci_high']):+.4f}] "
        f"pos/neg={delta['n_positive']}/{delta['n_negative']}"
    )
    print(
        f"  extreme safety: m0(A-R)={float(safety0['delta']):+.4f} "
        f"m8(A-C)={float(safety8['delta']):+.4f}"
    )
    print(
        "  alpha mean by m: "
        + " ".join(
            f"{m}:{rec_summary['alpha_trend']['alpha_mean_by_m'][m]:.4f}"
            for m in ("m0", "m2", "m4", "m8")
        )
    )
    audit.check(
        "B0 label: POST-HOC DEVELOPMENTAL, no_gate=True",
        rec_summary["label"] == "POST-HOC DEVELOPMENTAL" and bool(rec_summary["no_gate"]),
    )
    audit.check(
        "repro checks: E_random vs M1 <= 1e-9 and E_curriculum vs M2 <= 1e-9",
        float(rec_summary["repro_checks"]["random_vs_m1_confidences_max"]) <= 1e-9
        and float(rec_summary["repro_checks"]["curriculum_vs_m2_confidences_max"]) <= 1e-9,
    )

    passed = all(entry["ok"] for entry in audit.checks)
    payload = {
        "artifact": "v2m_m3_b0_audit",
        "passed": bool(passed),
        "label": "B0 DEVELOPMENTAL RESULT — AdaptiveMix ~ StaticMix (no gate)",
        "recompute_dir": str(RECOMPUTE_DIR),
        "bootstrap": {
            "replicates": 5000, "seed": 0, "ci": 0.95,
            "metrics": ["auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"],
            "shared_draws_across_severity": True,
        },
        "headline": {
            "delta_macro_adaptive_minus_static": {
                "delta": float(delta["delta"]),
                "ci_low": float(delta["ci_low"]),
                "ci_high": float(delta["ci_high"]),
                "per_seed": delta["per_seed"],
            },
            "extreme_safety": {
                "m0_adaptive_minus_random": float(safety0["delta"]),
                "m8_adaptive_minus_curriculum": float(safety8["delta"]),
                "threshold": -0.003,
            },
            "alpha_mean_by_m": rec_summary["alpha_trend"]["alpha_mean_by_m"],
            "fits": {
                key: rec_summary["fits"][key]["mean"]
                for key in (
                    "static_c", "adaptive_theta", "adaptive_tau", "adaptive_beta",
                )
            },
        },
        "max_deltas": {
            "point_metrics": max_point,
            "macro_metrics": max_macro,
            "bootstrap_pairs": max_boot,
            "alpha_diagnostics": max_alpha,
            "regret": max_regret,
            "calibration": max_calib,
        },
        "summary_tree_mismatches": tree_out,
        "checks": audit.checks,
        "runtime_sec": round(time.perf_counter() - started, 1),
    }
    out_path = B0_DIR / "m3_b0_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"V2M_M3_B0_AUDIT_{'PASS' if passed else 'FAIL'} "
        f"checks={len(audit.checks)} failed={sum(1 for c in audit.checks if not c['ok'])} "
        f"max_point={max_point:.3e} max_boot={max_boot:.3e} "
        f"runtime={time.perf_counter() - started:.1f}s"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
