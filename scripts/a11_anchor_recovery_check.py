"""A11 frozen-anchor recovery test (protocol A11.10) - RefCOCO+ only.

Amendment A11 forbids any fitting on the RefCOCOg path: the external runner may
only ``load saved coefficients -> predict``.  Before a single RefCOCOg prediction
is produced this script proves that the *loaded* bundle reproduces the RefCOCO+
predictions that were frozen by Phase 1F (whose production path refit + sklearn
``predict_proba``).  Two independent questions, both fatal on failure:

1. **B3 ranking / score recovery** - the frozen B3 checkpoints, re-run on the
   RefCOCO+ anchor cohort, reproduce the stored raw logits: identical candidate
   ranking (argmax ordering) and every score within the A8.4 ``--stop-atol``.
   This reuses Phase 1F's own STOP check so nothing about the scorer is trusted.
2. **No-refit reliability recovery** - the closed-form
   :meth:`ccg.semantic.frozen_load.FrozenExternalModels.predict` (load the saved
   Stats-17 / E1b-33 coefficients + train-only normalisations, never refit)
   reproduces Phase 1F's stored ``conf_stats`` / ``conf_e1b`` on the same rows to
   within A8.4's ``_TOL``.  This is the exact forward path the RefCOCOg runner
   will use, validated against RefCOCO+ ground truth.

The cell rows, samples and batches are built by importing ``run_phase1f.py`` and
calling its own loaders, so the anchor exercises the *identical* construction the
frozen numbers came from - not a reimplementation that could drift.

    conda activate deepminer
    python scripts/a11_anchor_recovery_check.py            # rand5 + hard5, 3 seeds
    python scripts/a11_anchor_recovery_check.py --cells rand5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.semantic import frozen as sfrozen  # noqa: E402
from ccg.semantic import frozen_load as fl  # noqa: E402

PHASE1F_PATH = ROOT / "scripts" / "run_phase1f.py"
DEFAULT_OUT = Path("results/phase1e_refcocog_external")
DEFAULT_FROZEN = DEFAULT_OUT / "frozen_models"
ANCHOR_CELLS: Sequence[str] = ("rand5", "hard5")
B3_SEEDS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")


def _load_phase1f() -> Any:
    """Import ``run_phase1f.py`` as a module (it is guarded, so import is safe)."""
    spec = importlib.util.spec_from_file_location("a11_phase1f", PHASE1F_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {PHASE1F_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_logger(path: Path | None) -> Callable[[str], None]:
    handle = path.open("a", encoding="utf-8") if path is not None else None

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return log


def _ranking(scores: np.ndarray) -> np.ndarray:
    """Descending candidate order per row (ties broken by index), for identity."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), axis=1, kind="stable")


def _check_cell(
    cell: str,
    data: Any,  # phase1f CellData
    scorer: str,
    bundle: fl.FrozenExternalModels,
    reference: Dict[str, np.ndarray],
    *,
    stop_atol: float,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Compare one rebuilt (cell, seed) against the frozen Phase 1F npz arrays."""
    stored_scores = np.asarray(reference["scores"], dtype=np.float32)
    our_scores = np.asarray(data.scores, dtype=np.float32)

    if our_scores.shape != stored_scores.shape:
        raise AssertionError(
            f"{cell}/{scorer}: rebuilt {our_scores.shape} != stored {stored_scores.shape}"
        )
    for field in ("sentence_id", "ref_id", "image_id"):
        ours = np.asarray(getattr(data, field), dtype=np.int64)
        theirs = np.asarray(reference[field], dtype=np.int64)
        if not np.array_equal(ours, theirs):
            raise AssertionError(f"{cell}/{scorer}: {field} row alignment drifted from Phase 1F")

    # (1) B3 candidate ranking identical + score within stop_atol.
    if not np.array_equal(_ranking(our_scores), _ranking(stored_scores)):
        raise AssertionError(
            f"{cell}/{scorer}: candidate ranking is NOT identical to the frozen raw logits"
        )
    score_delta = float(np.abs(our_scores.astype(np.float64) - stored_scores.astype(np.float64)).max())
    if not score_delta <= stop_atol:
        raise AssertionError(
            f"{cell}/{scorer}: raw-score max|delta|={score_delta:.3e} exceeds stop-atol {stop_atol:.1e}"
        )

    # (2) No-refit closed-form bundle reproduces the stored conf_stats / conf_e1b.
    stats_conf, e1b_conf = bundle.predict(scorer, data.stats17, data.sem16)
    stats_delta = float(np.abs(stats_conf - np.asarray(reference["conf_stats"], np.float64)).max())
    e1b_delta = float(np.abs(e1b_conf - np.asarray(reference["conf_e1b"], np.float64)).max())
    tol = float(sfrozen._TOL)
    for name, value in (("conf_stats", stats_delta), ("conf_e1b", e1b_delta)):
        if not value <= tol:
            raise AssertionError(
                f"{cell}/{scorer}: loaded-bundle {name} max|delta|={value:.3e} exceeds {tol:.0e} "
                "- the no-refit path does not reproduce Phase 1F"
            )
    correct_match = bool(np.array_equal(np.argmax(our_scores, axis=1) == 0,
                                        np.asarray(reference["correct"], dtype=bool)))

    out = {
        "cell": cell,
        "scorer": scorer,
        "n": int(our_scores.shape[0]),
        "ranking_identical": True,
        "correct_identical": correct_match,
        "raw_score_max_abs": score_delta,
        "conf_stats_max_abs": stats_delta,
        "conf_e1b_max_abs": e1b_delta,
        "stop_atol": float(stop_atol),
        "tolerance": tol,
    }
    log(
        f"[a11-anchor] {cell}/{scorer}: n={out['n']} ranking=identical "
        f"raw={score_delta:.2e} conf_stats={stats_delta:.2e} conf_e1b={e1b_delta:.2e}"
    )
    return out


def run_anchor(
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Rebuild the anchor cells and check them against the frozen Phase 1F files."""
    p1f = _load_phase1f()

    # The bundle is the only reliability object this path may touch; loading it
    # (and never recover_frozen_models) is what A11.6/A11.7 are about.
    bundle = fl.load_models(args.frozen_models)
    missing = [s for s in args.seeds if str(s) not in bundle.seeds]
    if missing:
        raise AssertionError(f"bundle is missing requested seeds: {missing}")
    log(f"[a11-anchor] loaded frozen bundle {args.frozen_models}: seeds {bundle.scorers}")

    # Phase 1F's own loader builds the identical cohort / corpus / scorers and
    # runs its A8.4 recovery (RefCOCO+ side; this never touches RefCOCOg).  We
    # reuse only its rows/samples/batches, then predict through the loaded bundle.
    loaded = p1f._load_stage(args, log)
    cohort = loaded["cohort"]
    corpus = loaded["corpus"]
    models = loaded["models"]
    frozen_rel = loaded["frozen"]
    text_cache: Dict[int, np.ndarray] = {}

    cells = list(args.cells)
    unknown = [c for c in cells if c not in p1f.CELL_BY_NAME]
    if unknown:
        raise ValueError(f"unknown anchor cells: {unknown}")
    variant_by_name = {variant.variant: variant for variant in p1f.VARIANTS}

    results: List[Dict[str, Any]] = []
    for cell_name in cells:
        cell_spec = p1f.CELL_BY_NAME[cell_name]
        variant = variant_by_name[cell_spec.source]
        rows = p1f._variant_rows(cohort, variant)
        samples = p1f._variant_samples(cohort, variant, rows)
        batch = p1f.hscores.materialise_examples(corpus, samples, cell_spec.k, text_cache=text_cache)
        local = p1f._cell_local(cohort, rows, cell_spec)
        for scorer in args.seeds:
            scores_full = p1f.hscores.score_examples(models[scorer], batch, batch_size=args.batch_size)
            data = p1f._build_cell(
                cell_spec, batch, scores_full, local, cohort, rows, frozen_rel.seeds[scorer]
            )
            npz = Path(args.phase1f_root) / "predictions" / f"{cell_name}__{scorer}.npz"
            if not npz.exists():
                raise FileNotFoundError(f"{npz} missing - Phase 1F must run before the anchor check")
            with np.load(npz, allow_pickle=False) as store:
                reference = {key: store[key] for key in store.files}
            results.append(
                _check_cell(cell_name, data, scorer, bundle, reference,
                            stop_atol=float(args.stop_atol), log=log)
            )
            del data, scores_full
        del batch

    aggregate = {
        "raw_score_max_abs": float(max(r["raw_score_max_abs"] for r in results)),
        "conf_stats_max_abs": float(max(r["conf_stats_max_abs"] for r in results)),
        "conf_e1b_max_abs": float(max(r["conf_e1b_max_abs"] for r in results)),
    }
    ok = all(r["ranking_identical"] and r["correct_identical"] for r in results)
    report = {
        "schema_version": fl.ARTIFACT_SCHEMA,
        "protocol": "A11.10",
        "passed": bool(ok),
        "cells": cells,
        "seeds": [str(s) for s in args.seeds],
        "stop_atol": float(args.stop_atol),
        "tolerance": float(sfrozen._TOL),
        "aggregate_max_abs": aggregate,
        "checks": results,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", nargs="+", default=list(ANCHOR_CELLS), choices=list(ANCHOR_CELLS))
    parser.add_argument("--seeds", nargs="+", default=list(B3_SEEDS))
    parser.add_argument("--frozen-models", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--phase1f-root", type=Path, default=Path("results/phase1f_hard_semantic"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stop-atol", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    # RefCOCO+ artifact roots (Phase 1F defaults) - the anchor never reads RefCOCOg.
    parser.add_argument("--features", type=Path, default=Path("cache/features"))
    parser.add_argument("--manifests", type=Path, default=Path("cache/manifests"))
    parser.add_argument("--bank", type=Path, default=Path("cache/proposals.h5"))
    parser.add_argument("--refs", type=Path, default=Path("data/raw/refcoco+/refcoco+/refs(unc).p"))
    parser.add_argument("--image-sizes", type=Path, default=Path("cache/image_sizes.npz"))
    parser.add_argument("--b3-root", type=Path, default=Path("results/phase0b_independent"))
    parser.add_argument("--phase05", type=Path, default=Path("results/phase05_score_sufficiency"))
    parser.add_argument("--phase1", type=Path, default=Path("results/phase1_semantic_sufficiency"))
    parser.add_argument("--embeddings", type=Path, default=Path("cache/semantic_phase1"))
    parser.add_argument("--log-file", type=Path, default=None)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.log_file = args.log_file if args.log_file is not None else args.out / "anchor_recovery.json.log"
    log = _make_logger(Path(args.log_file))
    started = time.perf_counter()
    log("[a11-anchor] starting RefCOCO+ frozen-anchor recovery (no RefCOCOg is read)")
    report = run_anchor(args, log)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report["seconds"] = round(time.perf_counter() - started, 2)
    target = out / "anchor_recovery.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(
        f"[a11-anchor] {'PASS' if report['passed'] else 'FAIL'} "
        f"raw={report['aggregate_max_abs']['raw_score_max_abs']:.2e} "
        f"conf_stats={report['aggregate_max_abs']['conf_stats_max_abs']:.2e} "
        f"conf_e1b={report['aggregate_max_abs']['conf_e1b_max_abs']:.2e} -> {target}"
    )
    if not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
