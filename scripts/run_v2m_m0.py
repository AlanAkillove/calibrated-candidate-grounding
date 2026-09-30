#!/usr/bin/env python
"""V2-M M0 runner: implementation gate before any M1 training.

Runs the twelve-check unit suite (``tests/test_lcr_m0.py``) and then a set of
*real-artifact* integration audits, writing everything into
``results/v2_local_competition/m0_architecture.json``:

A. architecture freeze -- parameter counts (LCR 1333 / LCR-noGate 1329 /
   Aggregate-MLP 1569), budgets, frozen constants, protocol sha256;
B. embedding stores K5/K10/K20/K50 -- alignment with the frozen B0 scorer
   canonical rows (subset + ``target_local == 0``), L2-normalisation sanity,
   chunked frozen relation-feature extraction over *all* rows, bit-identical
   recomputation of the first chunk, ``winner == argmax``, rank column, and
   sha256 fingerprints of the score matrix before/after every stage
   (grounding invariance: LCR only changes the reliability score);
C. zero-hard-exposure -- frozen A6.2 train/tune masks on ``val_calib`` Random
   K5/K10 only, ``m1_training_guard`` over the combined masks, and the hard5
   sentence universe (SameCategory) verified disjoint from the training pool;
D. frozen SameCategory-K5 cell -- the Phase 1F ``hard5`` manifest reproduced
   row-for-row from ``ccg.semantic.hard.load_hard_cohort`` (candidate order and
   target indices), and its frozen per-seed predictions internally consistent
   (``correct == argmax == 0``).

Nothing here trains: M1 must only start when ``all_checks_passed`` is true.

Usage
-----
    python -u scripts/run_v2m_m0.py --log-file logs_v2m_m0.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.lcr import (  # noqa: E402
    M_COMPETITORS,
    AggregateMLP,
    LCR,
    LCRNoGate,
    relation_features,
)
from ccg.lcr.audit import (  # noqa: E402
    assert_grounding_unchanged,
    grounding_fingerprint,
    m1_training_guard,
)
from ccg.reliability import data as rdata  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402

__all__ = ["build_parser", "main"]

OUT_ROOT = Path("results/v2_local_competition")
PROTOCOL_PATH = OUT_ROOT / "protocol.json"
M0_PATH = OUT_ROOT / "m0_architecture.json"
B3_ROOT = rdata.B3_ROOT
EMB_ROOT = sdata.DEFAULT_OUT_ROOT
PHASE05_DIR = sdata.PHASE05_FEATURES_DIR
MANIFESTS = Path("cache/manifests")
HARD5_MANIFEST = Path("results/phase1f_hard_semantic/manifests/hard5_candidates.npz")
HARD5_PREDICTIONS = Path("results/phase1f_hard_semantic/predictions")

KS: Tuple[int, ...] = (5, 10, 20, 50)
M1_TRAIN_KS: Tuple[int, ...] = (5, 10)
SEEDS: Tuple[int, ...] = (1, 2, 3)
CHUNK: int = 2048
EXPECTED_PARAMS = {"LCR": 1333, "LCR-noGate": 1329, "Aggregate-MLP": 1569}


# ---------------------------------------------------------------------------
# small helpers (house style)
# ---------------------------------------------------------------------------
def _make_logger(log_file: Optional[Path]) -> Callable[[str], None]:
    stream = open(log_file, "a", encoding="utf-8") if log_file else None

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    return log


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO), capture_output=True, text=True, timeout=30
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _corrected_T(b3_root: Path, seed: int) -> float:
    """Mirror of the frozen per-seed corrected-T reader (V1 B0)."""
    em = json.loads(
        (Path(b3_root) / f"seed_{seed}" / "eval_metadata.json").read_text(encoding="utf-8")
    )
    t = float(em["temperature_corrected"])
    if not (np.isfinite(t) and t > 0):
        raise ValueError(f"{b3_root} seed_{seed}: invalid corrected T {t}")
    if not bool(em.get("corrected_fit", {}).get("interior", em.get("corrected_interior", True))):
        raise ValueError(f"{b3_root} seed_{seed}: corrected T fit is not interior")
    return t


# ---------------------------------------------------------------------------
# A. architecture freeze
# ---------------------------------------------------------------------------
def check_architecture(log: Callable[[str], None]) -> Dict[str, Any]:
    counts = {
        "LCR": LCR(seed=0).n_parameters(),
        "LCR-noGate": LCRNoGate(seed=0).n_parameters(),
        "Aggregate-MLP": AggregateMLP(seed=0).n_parameters(),
    }
    for name, expected in EXPECTED_PARAMS.items():
        if counts[name] != expected:
            raise AssertionError(
                f"{name}: parameter count {counts[name]} != frozen {expected}"
            )
    if not counts["LCR"] < 10_000:
        raise AssertionError("LCR exceeds the 10k parameter target")
    if not (abs(counts["Aggregate-MLP"] - counts["LCR"]) / counts["LCR"] <= 0.25):
        raise AssertionError("Aggregate-MLP is not within 25% of the LCR capacity")
    keys = {
        name: sorted(cls(seed=0).state_arrays().keys())
        for name, cls in (("LCR", LCR), ("LCR-noGate", LCRNoGate), ("Aggregate-MLP", AggregateMLP))
    }
    report = {
        "parameter_counts": counts,
        "expected_counts": dict(EXPECTED_PARAMS),
        "budget": {"hard_max": 50000, "target_10000_ok_LCR": bool(counts["LCR"] < 10_000)},
        "capacity_matching_ratio": float(abs(counts["Aggregate-MLP"] - counts["LCR"]) / counts["LCR"]),
        "frozen_constants": {
            "M": M_COMPETITORS,
            "relation_dim": 7,
            "gate_input_dim": 3,
            "aggregate_input_dim": 31,
        },
        "state_keys": keys,
    }
    log(f"[M0-A] params {counts} ok (LCR < 10k, AggMLP within 25%)")
    return report


# ---------------------------------------------------------------------------
# B. real stores: alignment, features, determinism, grounding invariance
# ---------------------------------------------------------------------------
def _extract_chunk(
    scores: np.ndarray, z_q: np.ndarray, z_i: np.ndarray, temperature: float
) -> Any:
    return relation_features(scores, z_q, z_i, temperature)


def check_store(k: int, T: float, log: Callable[[str], None]) -> Dict[str, Any]:
    started = time.perf_counter()
    store = sdata.load_embedding_store(k, out_root=EMB_ROOT)
    ref = sdata.load_scorer_canonical("b3_seed1", k, b3_root=B3_ROOT)
    pos = np.searchsorted(ref.sentence_id, store.sentence_id)
    if pos.size != len(store) or not np.array_equal(ref.sentence_id[pos], store.sentence_id):
        raise AssertionError(f"K={k}: store sentence ids are not aligned with b3_seed1 rows")
    target_local = np.asarray(ref.target_local[pos], dtype=np.int32)
    if not np.all(target_local == 0):
        raise AssertionError(f"K={k}: frozen target_local is not column 0 -- layout mismatch")
    scores = np.asarray(ref.scores[pos], dtype=np.float64)
    if store.z_i.shape != (len(store), k, return_dim(store)):
        raise AssertionError(f"K={k}: z_i shape {store.z_i.shape} unexpected")
    if not np.all(store.target_local == 0):
        raise AssertionError(f"K={k}: store target_local is not column 0")

    fingerprint_before = grounding_fingerprint(scores)
    n = int(scores.shape[0])
    ranks_expected = np.arange(2, M_COMPETITORS + 2, dtype=np.float64) / float(k)
    report: Dict[str, Any] = {
        "k": k,
        "n": n,
        "feature_dim": int(store.feature_dim),
        "store_file": str(sdata.embedding_file(k, out_root=EMB_ROOT)),
        "alignment": {"subset_of_b3_seed1": True, "target_local_zero": True},
        "chunk_size": CHUNK,
        "n_chunks": int((n + CHUNK - 1) // CHUNK),
        "provenance_sha256": {
            key: sdata.embedding_provenance(k, out_root=EMB_ROOT).get(key)
            for key in ("z_q_sha256", "z_i_sha256", "sentence_id_sha256")
        },
    }

    first: Optional[Any] = None
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        zq = np.asarray(store.z_q[start:stop], dtype=np.float64)
        zi = np.asarray(store.z_i[start:stop], dtype=np.float64)
        if start == 0:
            for name, block in (("z_q", zq), ("z_i", zi)):
                worst = float(np.abs(np.linalg.norm(block, axis=-1) - 1.0).max())
                if worst > 2e-3:
                    raise AssertionError(f"K={k}: {name} not L2-normalised (max dev {worst})")
            report["chunk0_l2_max_deviation"] = {
                "z_q": float(np.abs(np.linalg.norm(zq, axis=-1) - 1.0).max()),
                "z_i": float(np.abs(np.linalg.norm(zi, axis=-1) - 1.0).max()),
            }
        out = _extract_chunk(scores[start:stop], zq, zi, T)
        if out.r.shape != (stop - start, M_COMPETITORS, 7):
            raise AssertionError(f"K={k}: r shape {out.r.shape} unexpected")
        if not (np.all(np.isfinite(out.r)) and np.all(np.isfinite(out.gate))):
            raise AssertionError(f"K={k}: non-finite relation features")
        winners = np.argmax(scores[start:stop], axis=1)
        if not np.array_equal(out.winner_idx, winners):
            raise AssertionError(f"K={k}: winner_idx differs from np.argmax")
        expected_ranks = np.broadcast_to(ranks_expected, out.r[:, :, 6].shape)
        if not np.array_equal(out.r[:, :, 6], expected_ranks):
            raise AssertionError(f"K={k}: rank_j_over_k column wrong")
        for row in range(out.competitor_idx.shape[0]):
            if int(out.winner_idx[row]) in set(out.competitor_idx[row].tolist()):
                raise AssertionError(f"K={k}: winner leaked into the competitor set")
        if start == 0:
            first = out
        else:
            del out
        del zq, zi

    # bit-identical recomputation of the first chunk
    zq0 = np.asarray(store.z_q[: min(CHUNK, n)], dtype=np.float64)
    zi0 = np.asarray(store.z_i[: min(CHUNK, n)], dtype=np.float64)
    again = _extract_chunk(scores[: min(CHUNK, n)], zq0, zi0, T)
    for attr in ("order", "winner_idx", "competitor_idx", "r", "gate"):
        if not np.array_equal(getattr(first, attr), getattr(again, attr)):
            raise AssertionError(f"K={k}: feature extraction is not deterministic ({attr})")
    report["determinism_chunk0_bit_identical"] = True

    # the full LCR input path on real rows must leave the scores fingerprint intact
    n_first = int(first.winner_idx.shape[0])
    clf = LCR(seed=1)
    probs = clf.predict_proba({"b": np.zeros(n_first), "r": first.r, "gate": first.gate})
    if not (probs.shape == (n_first,) and np.all(np.isfinite(probs))):
        raise AssertionError(f"K={k}: LCR predict_proba on real rows failed")
    gate_values = clf.gate_values({"b": np.zeros(n_first), "r": first.r, "gate": first.gate})
    if not np.all((gate_values >= 0.0) & (gate_values <= 1.0)):
        raise AssertionError(f"K={k}: gate outside [0, 1] on real rows")
    report["lcr_predict_path_ok"] = True
    report["gate_range_real_rows"] = [float(gate_values.min()), float(gate_values.max())]

    fingerprint_after = grounding_fingerprint(scores)
    assert_grounding_unchanged(fingerprint_before, fingerprint_after, context=f"K={k}")
    report["scores_fingerprint_before"] = fingerprint_before
    report["scores_fingerprint_after"] = fingerprint_after
    report["scores_unchanged"] = True
    report["seconds"] = time.perf_counter() - started
    log(
        f"[M0-B] K={k}: {n} rows / {report['n_chunks']} chunks ok; "
        f"det+invariance ok ({report['seconds']:.1f}s)"
    )
    return report


def return_dim(store: Any) -> int:
    return int(store.z_i.shape[-1])


# ---------------------------------------------------------------------------
# C. zero-hard-exposure over the real frozen split
# ---------------------------------------------------------------------------
def check_zero_hard_exposure(
    hard_ids: np.ndarray, log: Callable[[str], None]
) -> Dict[str, Any]:
    stores = {k: sdata.load_embedding_store(k, out_root=EMB_ROOT) for k in M1_TRAIN_KS}
    base = stores[M1_TRAIN_KS[0]]
    split = rdata.build_reliability_split(
        base.eval_split,
        base.image_id,
        seed=rdata.DEFAULT_SPLIT_SEED,
        train_frac=rdata.DEFAULT_TRAIN_FRAC,
    )
    report: Dict[str, Any] = {
        "split_seed": int(split.seed),
        "train_frac": float(split.train_frac),
        "n_train_images": int(split.train_images.size),
        "n_tune_images": int(split.tune_images.size),
        "train_ks": list(M1_TRAIN_KS),
        "hard_universe_size": int(hard_ids.size),
        "per_k_rows": {},
        "guards": {},
    }
    for kind in ("reliability_train", "reliability_tune"):
        evals: List[np.ndarray] = []
        sids: List[np.ndarray] = []
        for k in M1_TRAIN_KS:
            store = stores[k]
            mask = split.row_mask(store.eval_split, store.image_id, kind=kind)
            images = np.unique(store.image_id[mask])
            if images.size != (
                split.train_images.size if kind == "reliability_train" else split.tune_images.size
            ):
                raise AssertionError(f"{kind}/K{k}: masked image count mismatch vs the split")
            rows = int(mask.sum())
            report["per_k_rows"].setdefault(k, {})[kind] = rows
            evals.append(np.asarray(store.eval_split[mask]))
            sids.append(np.asarray(store.sentence_id[mask]))
        guard = m1_training_guard(
            np.concatenate(evals),
            np.concatenate(sids),
            hard_ids,
            train_ks=M1_TRAIN_KS,
            seen_ks=M1_TRAIN_KS,
            context=f"M0/{kind}",
        )
        report["guards"][kind] = dict(guard)
        log(
            f"[M0-C] {kind}: rows K5={report['per_k_rows'][5][kind]} "
            f"K10={report['per_k_rows'][10][kind]} -> guard pass"
        )
    report["n_train_rows_total"] = int(
        sum(report["per_k_rows"][k]["reliability_train"] for k in M1_TRAIN_KS)
    )
    report["n_tune_rows_total"] = int(
        sum(report["per_k_rows"][k]["reliability_tune"] for k in M1_TRAIN_KS)
    )
    return report


# ---------------------------------------------------------------------------
# D. frozen SameCategory-K5 (hard5) cell
# ---------------------------------------------------------------------------
def check_hard5(log: Callable[[str], None]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "manifest": str(HARD5_MANIFEST),
        "manifest_sha256": _sha256_file(HARD5_MANIFEST),
    }
    cohort = shard.load_hard_cohort(features_dir=PHASE05_DIR, manifests_root=MANIFESTS, log=log)
    rows = np.flatnonzero(cohort.masks["same4"])
    hard_ids = np.asarray(cohort.sentence_id[rows], dtype=np.int64)
    with np.load(HARD5_MANIFEST) as manifest:
        manifest_ids = np.asarray(manifest["sentence_id"], dtype=np.int64)
        manifest_ci = np.asarray(manifest["candidate_indices"], dtype=np.int64)
        manifest_ti = np.asarray(manifest["target_index"], dtype=np.int64)

    if set(manifest_ids.tolist()) != set(hard_ids.tolist()):
        raise AssertionError("hard5 manifest sentence ids differ from the frozen same4 cohort")
    if manifest_ids.size != hard_ids.size:
        raise AssertionError("hard5 manifest row count differs from the same4 cohort")
    if not np.array_equal(manifest_ci[:, 0], manifest_ti):
        raise AssertionError("hard5: candidate_indices column 0 must be the target index")

    position = {int(s): int(i) for i, s in enumerate(manifest_ids.tolist())}
    rebuilt = np.empty_like(manifest_ci)
    for i, sid in enumerate(hard_ids.tolist()):
        rebuilt[position[sid]] = cohort.candidate_indices(int(rows[i]), "same_category", 5)
    if not np.array_equal(rebuilt, manifest_ci):
        raise AssertionError("hard5: frozen candidate_indices do not match the manifest rebuild")
    report["candidate_order_reproduced"] = True

    store5 = sdata.load_embedding_store(5, out_root=EMB_ROOT)
    overlaps = int(np.isin(hard_ids, store5.sentence_id).sum())
    if overlaps != hard_ids.size:
        raise AssertionError("hard5 rows are not all present in the K5 canonical store")
    pos = np.searchsorted(np.sort(store5.sentence_id), hard_ids)
    hard_splits = store5.eval_split[np.argsort(store5.sentence_id)[pos]]
    if not set(hard_splits.tolist()) <= {"testA", "testB"}:
        raise AssertionError("hard5 universe contains rows outside testA/testB")
    val_calib = store5.sentence_id[store5.eval_split == "val_calib"]
    if int(np.isin(hard_ids, val_calib).sum()) != 0:
        raise AssertionError("hard5 universe overlaps the val_calib training pool")
    report.update(
        {
            "n_rows": int(hard_ids.size),
            "n_images": int(np.unique(cohort.image_id[rows]).size),
            "eval_split_counts": {
                name: int((hard_splits == name).sum()) for name in ("testA", "testB")
            },
            "val_calib_overlap": 0,
        }
    )

    predictions: Dict[str, Any] = {}
    for seed in SEEDS:
        path = HARD5_PREDICTIONS / f"hard5__b3_seed{seed}.npz"
        with np.load(path) as pred:
            ids = np.asarray(pred["sentence_id"], dtype=np.int64)
            scores = np.asarray(pred["scores"], dtype=np.float32)
            correct = np.asarray(pred["correct"], dtype=np.int8)
        if not np.array_equal(ids, manifest_ids):
            raise AssertionError(f"hard5 seed{seed}: predictions are not row-aligned with the manifest")
        if scores.shape != (hard_ids.size, 5):
            raise AssertionError(f"hard5 seed{seed}: scores shape {scores.shape} unexpected")
        derived = (np.argmax(scores, axis=1) == 0).astype(np.int8)
        if not np.array_equal(derived, correct):
            raise AssertionError(f"hard5 seed{seed}: correct != (argmax == 0)")
        predictions[f"b3_seed{seed}"] = {
            "file": str(path),
            "sha256": _sha256_file(path),
            "b3_accuracy": float(np.mean(correct)),
        }
    report["predictions"] = predictions
    log(
        f"[M0-D] hard5: {hard_ids.size} rows / {report['n_images']} images; candidate order "
        f"reproduced; prediction accuracies "
        f"{[round(v['b3_accuracy'], 4) for v in predictions.values()]}"
    )
    return report, hard_ids


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log-file", default=None)
    p.add_argument("--skip-pytest", action="store_true", help="data checks only (not for M1 entry)")
    return p


def _run_pytest(log: Callable[[str], None]) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "tests/test_lcr_m0.py", "-q"]
    proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, timeout=3600)
    tail = [line for line in proc.stdout.strip().splitlines() if line.strip()][-3:]
    record = {
        "command": "python -m pytest tests/test_lcr_m0.py -q",
        "returncode": int(proc.returncode),
        "passed": bool(proc.returncode == 0),
        "stdout_tail": tail,
    }
    log(f"[M0] pytest: rc={proc.returncode} :: {tail[-1] if tail else ''}")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = _make_logger(Path(args.log_file) if args.log_file else None)
    started = time.perf_counter()
    log(f"[M0] start (branch={_git(['rev-parse', '--abbrev-ref', 'HEAD'])})")

    pytest_record = (
        {"skipped": True}
        if args.skip_pytest
        else _run_pytest(log)
    )
    if not args.skip_pytest and not pytest_record["passed"]:
        raise SystemExit("M0 unit tests failed -- fix tests/test_lcr_m0.py before data checks")

    T = _corrected_T(B3_ROOT, 1)
    log(f"[M0] corrected T (b3_seed1) = {T:.6f}")

    architecture = check_architecture(log)
    stores = {f"K{k}": check_store(k, T, log) for k in KS}

    hard5, hard_ids = check_hard5(log)
    guard = check_zero_hard_exposure(hard_ids, log)

    status = _git(["status", "--porcelain"])
    payload = {
        "artifact": "v2m_m0_architecture",
        "stage": "V2-M M0 (implementation gate before M1)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.perf_counter() - started,
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": _git(["rev-parse", "HEAD"]),
            "dirty": bool(status),
            "status_lines": len(status.splitlines()) if status else 0,
        },
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": _sha256_file(PROTOCOL_PATH),
            "base_commit": "0aaa29d",
        },
        "unit_tests": {**pytest_record, "file": "tests/test_lcr_m0.py"},
        "architecture": architecture,
        "stores": stores,
        "zero_hard_exposure": guard,
        "hard5_cell": hard5,
        "all_checks_passed": True,
    }
    _write_json(M0_PATH, payload)
    log(f"[M0] wrote {M0_PATH} (runtime {payload['runtime_seconds']:.1f}s)")
    print(
        "V2M_M0_COMPLETE "
        f"lcr_params={architecture['parameter_counts']['LCR']} "
        f"hard5_rows={hard5['n_rows']} all_checks_passed=True"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
