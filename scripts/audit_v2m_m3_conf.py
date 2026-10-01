#!/usr/bin/env python
"""V2-M3.1 confirmatory consistency audit: recompute B1/B2 and compare.

Recompute chain (read-only; frozen experts; the only training is the frozen
curriculum protocol; no B0-driven change):

    frozen V2-G Phase A models + per-backbone stores -> E_random anchors
    frozen M2 val cells -> E_curriculum (C grid 0.1/1/10, tune-only selection)
    frozen K=10 test cells m in {0,2,4,8} (cross-checked against the frozen B0
    expb arrays) -> per-backbone train-only CDF -> StaticMix / AdaptiveMix on
    reliability_tune m={0,2,4} -> five-model inference -> metrics -> paired
    image-cluster bootstrap (5000 reps, seed 0) -> macro bootstrap with shared
    draws -> frozen gates -> cross-backbone verdict.

The whole confirmatory pipeline is re-run into a scratch directory and every
committed artifact under ``m3_mixture/conf/`` is compared value-by-value:

    conf/{b1,b2}/{point_metrics,macro_metrics,bootstrap_pairs,
    alpha_diagnostics,regret,calibration,dose_anchor}.csv
    conf/{b1,b2}/gate.json          (volatile path/runtime keys stripped)
    conf/cross_backbone_verdict.json
    conf/metadata.json              (started_utc / runtime_sec stripped)

plus the frozen-hash check (mixer / bootstrap / protocol / conf script / V2-G
freeze) so the recompute provably used the frozen code.

Writes ``results/v2_local_competition/m3_mixture/conf/m3_conf_audit.json``;
exit code 0 = PASS.

Usage
-----
    python -u scripts/audit_v2m_m3_conf.py
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

__all__ = ["main"]

TOLERANCE = 1e-10
M3_DIR = Path("results/v2_local_competition/m3_mixture")
CONF_DIR = M3_DIR / "conf"
RECOMPUTE_DIR = CONF_DIR / "_audit_recompute"
RECOMPUTE_LOG = Path("logs_v2m_m3_conf_audit_recompute.txt")
BACKBONES: Tuple[str, ...] = ("b1", "b2")

CSV_NAMES: Tuple[str, ...] = (
    "point_metrics.csv", "macro_metrics.csv", "bootstrap_pairs.csv",
    "alpha_diagnostics.csv", "regret.csv", "calibration.csv", "dose_anchor.csv",
)
CSV_KEYS: Dict[str, Tuple[str, ...]] = {
    "point_metrics.csv": ("level", "seed", "model"),
    "macro_metrics.csv": ("seed", "model"),
    "bootstrap_pairs.csv": ("level", "seed", "model_a", "model_b", "metric"),
    "alpha_diagnostics.csv": ("level", "seed"),
    "regret.csv": ("level", "seed", "model"),
    "calibration.csv": ("level", "seed", "model"),
    "dose_anchor.csv": ("backbone", "seed", "level"),
}
#: Keys whose values legitimately differ between run and recompute (paths,
#: wall-clock fields) -- everything else must reproduce value-by-value.
SKIP_TREE_KEYS = frozenset({"figures", "backbone_dir", "runtime_sec", "started_utc"})
FROZEN_HASH_SOURCES: Dict[str, Path] = {
    "src/ccg/mixture/mixer.py": Path("src/ccg/mixture/mixer.py"),
    "src/ccg/mixture/bootstrap.py": Path("src/ccg/mixture/bootstrap.py"),
    "results/v2_local_competition/m3_mixture/protocol_m3.json": M3_DIR / "protocol_m3.json",
    "results/v2_backbone_generalization/g4_protocol_freeze.json":
        Path("results/v2_backbone_generalization/g4_protocol_freeze.json"),
    "scripts/run_v2m_m3_conf.py": Path("scripts/run_v2m_m3_conf.py"),
}


def _load_module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _strip(obj: Any, skip: frozenset) -> Any:
    if isinstance(obj, Mapping):
        return {key: _strip(value, skip) for key, value in obj.items() if str(key) not in skip}
    if isinstance(obj, list):
        return [_strip(value, skip) for value in obj]
    return obj


def _compare_csv(
    audit: Audit, name: str, rec_path: Path, sto_path: Path, key_cols: Sequence[str]
) -> float:
    """Compare every column of one CSV value-by-value (numeric vs string aware)."""
    rec_rows = _read_csv(rec_path)
    sto_rows = _read_csv(sto_path)
    if not rec_rows or not sto_rows:
        audit.check(name, False, "empty csv")
        return float("inf")
    if list(rec_rows[0].keys()) != list(sto_rows[0].keys()):
        audit.check(name, False, f"header differs: {list(rec_rows[0])} vs {list(sto_rows[0])}")
        return float("inf")
    key_fn: Callable[[Mapping[str, str]], Tuple[str, ...]] = lambda r: tuple(  # noqa: E731
        r[col] for col in key_cols
    )
    rec_map = {key_fn(r): r for r in rec_rows}
    sto_map = {key_fn(r): r for r in sto_rows}
    if set(rec_map) != set(sto_map):
        only_rec = sorted(set(rec_map) - set(sto_map))[:4]
        only_sto = sorted(set(sto_map) - set(rec_map))[:4]
        audit.check(
            name, False,
            f"row universe differs: rec={len(rec_map)} sto={len(sto_map)} "
            f"only_rec={only_rec} only_sto={only_sto}",
        )
        return float("inf")
    fields = [col for col in sto_rows[0].keys() if col not in key_cols]
    faults = 0
    max_delta = 0.0
    examples: List[str] = []
    for key, rec in rec_map.items():
        sto = sto_map[key]
        for field in fields:
            raw_a, raw_b = rec[field], sto[field]
            try:
                fa, fb = float(raw_a), float(raw_b)
            except (TypeError, ValueError):
                if raw_a != raw_b:
                    faults += 1
                    if len(examples) < 3:
                        examples.append(f"{key}:{field} {raw_a!r} != {raw_b!r}")
                continue
            delta = abs(fa - fb)
            max_delta = max(max_delta, delta)
            if delta > TOLERANCE:
                faults += 1
                if len(examples) < 3:
                    examples.append(f"{key}:{field} {fa:.6e} vs {fb:.6e}")
    detail = f"rows={len(rec_map)} faults={faults} max|delta|={max_delta:.3e}"
    if examples:
        detail += " first=" + "; ".join(examples)
    audit.check(name, faults == 0, detail)
    return max_delta


def main() -> int:
    started = time.perf_counter()
    print("V2-M3.1 confirmatory consistency audit")
    conf = _load_module("run_v2m_m3_conf", "scripts/run_v2m_m3_conf.py")

    # ---- frozen-hash check: the recompute must run the frozen code ------------
    print("[frozen hashes]")
    amendment = json.loads((M3_DIR / "amendment_v2m31.json").read_text(encoding="utf-8"))
    audit = Audit()
    for key, rel in FROZEN_HASH_SOURCES.items():
        expected = str(amendment["freeze_hashes"][key])
        actual = _sha256_file(_REPO / rel)
        audit.check(
            f"freeze hash {key}", actual == expected,
            f"{actual[:16]}... == {expected[:16]}...",
        )

    # ---- recompute (identical frozen inputs; scratch output dir) --------------
    print("[recompute]")
    print(f"  scratch dir: {RECOMPUTE_DIR}")
    t0 = time.perf_counter()
    RECOMPUTE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_REPO / RECOMPUTE_LOG, "w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
        rc = conf.main(
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

    print("[recomputation vs committed artifacts]")
    max_deltas: Dict[str, float] = {}
    for tag in BACKBONES:
        for csv_name in CSV_NAMES:
            max_deltas[f"{tag}/{csv_name}"] = _compare_csv(
                audit,
                f"{tag}/{csv_name} reproduces",
                RECOMPUTE_DIR / tag / csv_name,
                CONF_DIR / tag / csv_name,
                key_cols=CSV_KEYS[csv_name],
            )
        rec_gate = _strip(
            json.loads((RECOMPUTE_DIR / tag / "gate.json").read_text(encoding="utf-8")),
            SKIP_TREE_KEYS,
        )
        sto_gate = _strip(
            json.loads((CONF_DIR / tag / "gate.json").read_text(encoding="utf-8")),
            SKIP_TREE_KEYS,
        )
        diffs: List[str] = []
        _tree_diffs("", rec_gate, sto_gate, diffs, limit=40)
        audit.check(
            f"{tag}/gate.json tree reproduces (primary + safety + selective + alpha + fits)",
            not diffs,
            f"mismatches={len(diffs)}" + (f" first={diffs[0]}" if diffs else ""),
        )

    rec_verdict = json.loads((RECOMPUTE_DIR / "cross_backbone_verdict.json").read_text(encoding="utf-8"))
    sto_verdict = json.loads((CONF_DIR / "cross_backbone_verdict.json").read_text(encoding="utf-8"))
    diffs = []
    _tree_diffs("", rec_verdict, sto_verdict, diffs, limit=40)
    audit.check(
        "cross_backbone_verdict.json reproduces (incl. stop rule)",
        not diffs,
        f"mismatches={len(diffs)}" + (f" first={diffs[0]}" if diffs else ""),
    )

    rec_meta = _strip(
        json.loads((RECOMPUTE_DIR / "metadata.json").read_text(encoding="utf-8")),
        frozenset({"started_utc", "runtime_sec"}),
    )
    sto_meta = _strip(
        json.loads((CONF_DIR / "metadata.json").read_text(encoding="utf-8")),
        frozenset({"started_utc", "runtime_sec"}),
    )
    diffs = []
    _tree_diffs("", rec_meta, sto_meta, diffs, limit=40)
    audit.check(
        "metadata.json core reproduces (frozen config + source hashes)",
        not diffs,
        f"mismatches={len(diffs)}" + (f" first={diffs[0]}" if diffs else ""),
    )

    # ---- headline + label + anchors ------------------------------------------
    print("[confirmatory headline]")
    headline: Dict[str, Any] = {}
    for tag in BACKBONES:
        gate = json.loads((RECOMPUTE_DIR / tag / "gate.json").read_text(encoding="utf-8"))
        delta = gate["delta_macro"]["vs_static"]
        safety = gate["extreme_safety"]
        selective = gate["selective_support"]
        headline[tag] = {
            "pass": bool(gate["pass"]),
            "delta_macro_adaptive_minus_static": {
                "delta": float(delta["delta"]),
                "ci_low": float(delta["ci_low"]),
                "ci_high": float(delta["ci_high"]),
                "per_seed": {str(k): float(v) for k, v in delta["per_seed"].items()},
            },
            "extreme_safety": {
                "m0_adaptive_minus_random": float(safety["m0_adaptive_minus_random"]["delta"]),
                "m0_pass": bool(safety["m0_pass"]),
                "m8_adaptive_minus_curriculum": float(safety["m8_adaptive_minus_curriculum"]["delta"]),
                "m8_pass": bool(safety["m8_pass"]),
                "threshold": float(safety["threshold"]),
            },
            "selective_support": {
                "e_aurc_relative_improvement": selective["macro_e_aurc"]["relative_improvement"],
                "rer50_gain_pp": float(selective["macro_rer50"]["gain_pp"]),
                "satisfied": bool(selective["draft_rule_satisfied"]),
            },
            "alpha_m8_minus_m0": float(gate["alpha_trend"]["alpha_m8_minus_m0"]),
            "static_c_mean": float(gate["fits"]["static_c"]["mean"]),
            "adaptive_beta_mean": float(gate["fits"]["adaptive_beta"]["mean"]),
            "adaptive_tau_mean": float(gate["fits"]["adaptive_tau"]["mean"]),
        }
        print(
            f"  {tag}: dmacro={headline[tag]['delta_macro_adaptive_minus_static']['delta']:+.6f} "
            f"CI[{delta['ci_low']:+.6f},{delta['ci_high']:+.6f}] "
            f"m0={headline[tag]['extreme_safety']['m0_adaptive_minus_random']:+.6f} "
            f"m8={headline[tag]['extreme_safety']['m8_adaptive_minus_curriculum']:+.6f} "
            f"pass={headline[tag]['pass']}"
        )
    print(f"  verdict: {rec_verdict['verdict']} ({rec_verdict['n_pass']}/{rec_verdict['n_backbones']})")
    audit.check(
        "verdict: ADAPTIVE MIXTURE NOT SUPPORTED (0/2), stop=True",
        rec_verdict["verdict"] == "ADAPTIVE MIXTURE NOT SUPPORTED"
        and int(rec_verdict["n_pass"]) == 0
        and bool(rec_verdict["stop"]),
    )
    audit.check(
        "label: CONFIRMATORY / amendment V2-M3.1",
        rec_meta["label"] == "CONFIRMATORY" and rec_meta["amendment"] == "V2-M3.1",
    )
    for tag in BACKBONES:
        gate = json.loads((RECOMPUTE_DIR / tag / "gate.json").read_text(encoding="utf-8"))
        anchors = gate["anchors"]
        tolerances = anchors["tolerances"]
        k5 = float(anchors["phase_a_k5_max_delta"])
        dose = float(anchors["dose_r2_max_delta"])
        vstop = float(anchors["val_stop_max_delta"])
        ok = (
            k5 <= float(tolerances["phase_a_k5_store"])
            and dose <= float(tolerances["dose_rescore"])
            and vstop <= 1e-4  # frozen score-level STOP tolerance class
        )
        audit.check(
            f"{tag}: repro anchors within frozen tolerances", ok,
            f"k5={k5:.2e} dose={dose:.2e} val_stop={vstop:.2e}",
        )

    passed = all(entry["ok"] for entry in audit.checks)
    payload = {
        "artifact": "v2m_m3_conf_audit",
        "passed": bool(passed),
        "label": "CONFIRMATORY RESULT — adaptive mixture not supported",
        "amendment": "V2-M3.1",
        "recompute_dir": str(RECOMPUTE_DIR),
        "bootstrap": {
            "replicates": 5000, "seed": 0, "ci": 0.95,
            "metrics": ["auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"],
            "shared_draws_across_severity": True,
        },
        "headline": headline,
        "cross_backbone_verdict": rec_verdict["verdict"],
        "n_pass": int(rec_verdict["n_pass"]),
        "stop": bool(rec_verdict["stop"]),
        "max_deltas": max_deltas,
        "frozen_hashes_verified": {
            key: str(amendment["freeze_hashes"][key]) for key in FROZEN_HASH_SOURCES
        },
        "checks": audit.checks,
        "runtime_sec": round(time.perf_counter() - started, 1),
    }
    out_path = CONF_DIR / "m3_conf_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    worst = max([value for value in max_deltas.values() if np.isfinite(value)] or [0.0])
    print(
        f"V2M_M3_CONF_AUDIT_{'PASS' if passed else 'FAIL'} "
        f"checks={len(audit.checks)} failed={sum(1 for c in audit.checks if not c['ok'])} "
        f"worst_csv_delta={worst:.3e} runtime={time.perf_counter() - started:.1f}s"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
