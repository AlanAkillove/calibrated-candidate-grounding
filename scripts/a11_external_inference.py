"""A11 frozen inference over the RefCOCOg external cohort (protocol A11.6/A11.7/A11.36).

This writes the *per-row raw predictions* Amendment A11.36 requires for every
scorer seed and both matched regimes: the frozen Phase-0B B3 checkpoints re-run
on the extracted RefCOCOg features, together with the Stats / E1b reliabilities
produced by the no-refit closed-form bundle (``FrozenExternalModels.predict`` --
load saved coefficients + train-only normalisations + the per-seed temperature,
never refit).  Every aggregate statistic (manipulation / difficulty checks, the
diff-of-diffs, the bootstraps) is computed downstream from these files, so
nothing later can silently re-touch a grounding or reliability model.

No parameter is fitted here: the scorer set is exactly the three Phase-0B seeds
loaded read-only, the candidate sets are the A10 frozen manifests materialised
through :class:`RefCOCOGFeatureCorpus`, and the reliability forward is the
load-and-predict twin validated on the RefCOCO+ anchor (A11.10).  Runtime and
peak VRAM are recorded (A11.42).

    conda activate deepminer
    # full external inference (needs the full cache/refcocog_external extraction):
    python scripts/a11_external_inference.py
    # smoke against the tiny aligned extraction to validate the path only:
    python scripts/a11_external_inference.py --cache-root cache/_smoke_refcocog_ext \
        --out-dir results/phase1e_refcocog_external/_smoke_predictions --align-to-cache

``--align-to-cache`` keeps only rows whose image was region-extracted *and* whose
expression was text-extracted; the full run over ``cache/refcocog_external``
covers the whole cohort, so the flag is a no-op there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.external import refcocog_external as rx  # noqa: E402
from ccg.external import refcocog_manifests as rman  # noqa: E402
from ccg.features.cache import FeatureCache  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import frozen_load as fl  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402

B3_ROOT = Path("results/phase0b_independent")
B3_SEEDS: Tuple[int, ...] = (1, 2, 3)
REGIMES: Tuple[str, ...] = ("random", "same_category")
REGIME_TAG = {"random": "rand5", "same_category": "hard5"}
DEFAULT_OUT = Path("results/phase1e_refcocog_external")
SEM_CHUNK = 4096


def _log_line(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _align_rows(cohort: Any, cache_root: Path) -> List[int]:
    """Rows whose image is region-extracted *and* whose expression is text-extracted."""
    cache = FeatureCache.open(cache_root)
    try:
        region_ids = {int(i) for i in cache.region_ids()}
        sentence_ids = {int(i) for i in cache.sentence_ids()}
    finally:
        cache.close()
    return [
        i
        for i in range(len(cohort))
        if int(cohort.image_id[i]) in region_ids and int(cohort.expr_id[i]) in sentence_ids
    ]


def _semantic_stats_chunked(z_q: np.ndarray, z_i: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Frozen 16-dimensional candidate-semantic statistics (A11.7 definition)."""
    n = int(z_q.shape[0])
    out = np.empty((n, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, n, SEM_CHUNK):
        stop = min(start + SEM_CHUNK, n)
        out[start:stop] = sfeat.semantic_stats(z_q[start:stop], z_i[start:stop], scores[start:stop])
    return out


def _predict_regime(
    regime: str,
    samples: List[Any],
    batch: hscores.B3ExampleBatch,
    models: Dict[str, Any],
    bundle: fl.FrozenExternalModels,
    *,
    batch_size: int,
    out_dir: Path,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    tag = REGIME_TAG[regime]
    rows: List[Dict[str, Any]] = []
    for seed in B3_SEEDS:
        scorer = f"b3_seed{seed}"
        temperature = float(bundle.seed(scorer).temperature)
        scores = hscores.score_examples(models[scorer], batch, batch_size=batch_size)
        scores64 = scores.astype(np.float64)
        correct = np.argmax(scores, axis=1) == 0
        scalars = rfeat.scalar_confidence(scores64, temperature=temperature)
        ordered = np.sort(scores64, axis=1)[:, ::-1]
        margin = ordered[:, 0] - ordered[:, 1]
        stats17 = np.asarray(rfeat.stat_features(scores64, temperature=temperature), dtype=np.float64)
        sem16 = _semantic_stats_chunked(batch.z_q, batch.z_i, scores64)
        stats_conf, e1b_conf = bundle.predict(scorer, stats17, sem16)
        target = out_dir / f"external__{scorer}__{tag}.npz"
        np.savez(
            target,
            sentence_id=np.asarray([int(s.sentence_id) for s in samples], dtype=np.int64),
            ref_id=np.asarray([int(s.ref_id) for s in samples], dtype=np.int64),
            image_id=np.asarray([int(s.image_id) for s in samples], dtype=np.int64),
            regime=np.asarray([regime] * len(samples)),
            correct=np.asarray(correct, dtype=bool),
            scores=np.asarray(scores, dtype=np.float32),
            conf_msp=np.asarray(scalars["msp"], dtype=np.float64),
            conf_stats=np.asarray(stats_conf, dtype=np.float64),
            conf_e1b=np.asarray(e1b_conf, dtype=np.float64),
            margin=np.asarray(margin, dtype=np.float64),
            entropy=np.asarray(-scalars["neg_entropy"], dtype=np.float64),
            stats17=stats17,
            sem16=sem16,
        )
        log(
            f"[a11-infer] {scorer}/{tag}: n={len(samples)} acc={float(correct.mean()):.4f} "
            f"-> {target.name}"
        )
        rows.append(
            {
                "scorer": scorer,
                "regime": regime,
                "n": int(len(samples)),
                "acc": float(correct.mean()),
                "mean_margin": float(margin.mean()),
                "mean_entropy": float(-scalars["neg_entropy"].mean()),
                "temperature": temperature,
                "file": target.name,
            }
        )
    return rows


def run_extraction(  # noqa: D401 - named for the driver symmetry
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    cohort, image_sizes, report = rx.load_refcocog_cohort(persist_dir=None, log=log)
    if not (report["a10_pin"]["ok"] and report["matched_pair"]["ok"]):
        raise AssertionError("A11 cohort pin failed - refusing to run inference")

    rows = _align_rows(cohort, args.cache_root) if args.align_to_cache else list(range(len(cohort)))
    if args.limit_rows and args.limit_rows > 0:
        rows = rows[: int(args.limit_rows)]
    if not rows:
        raise AssertionError("no cohort rows to score")
    sub = cohort.subset(rows)
    log(f"[a11-infer] scoring {len(rows)} cohort rows over {REGIMES} x {len(B3_SEEDS)} seeds")

    bundle = fl.load_models(args.frozen_models)
    missing = [f"b3_seed{s}" for s in B3_SEEDS if f"b3_seed{s}" not in bundle.seeds]
    if missing:
        raise AssertionError(f"bundle is missing requested seeds: {missing}")

    models = hscores.load_frozen_scorers(b3_root=args.b3_root, seeds=B3_SEEDS, device=args.device)
    corpus = rx.RefCOCOGFeatureCorpus(
        cache_root=args.cache_root,
        bank_paths=rx.DEFAULT_BANK_PATHS,
        image_sizes=image_sizes,
    )

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    try:
        for regime in REGIMES:
            samples = sub.samples(regime)
            batch = hscores.materialise_examples(corpus, samples, rman.A11_K)
            results.extend(
                _predict_regime(
                    regime, samples, batch, models, bundle,
                    batch_size=args.batch_size, out_dir=out_dir, log=log,
                )
            )
            del batch
    finally:
        corpus.close()

    summary: Dict[str, Any] = {
        "protocol": "A11.6/A11.7/A11.36",
        "n_rows": len(rows),
        "n_images": int(sub.n_images),
        "n_refs": int(sub.n_refs),
        "regimes": list(REGIMES),
        "seeds": [f"b3_seed{s}" for s in B3_SEEDS],
        "align_to_cache": bool(args.align_to_cache),
        "inference_seconds": round(time.perf_counter() - started, 2),
        "rows": results,
    }
    if torch.cuda.is_available():
        summary["peak_vram_mb"] = round(float(torch.cuda.max_memory_allocated()) / (1024 ** 2), 1)
        summary["device_name"] = torch.cuda.get_device_name(0)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", type=Path, default=rx.EXTERNAL_CACHE_ROOT)
    parser.add_argument("--b3-root", type=Path, default=B3_ROOT)
    parser.add_argument("--frozen-models", type=Path, default=DEFAULT_OUT / "frozen_models")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "predictions")
    parser.add_argument("--out-report", type=Path, default=DEFAULT_OUT / "inference_report.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--align-to-cache", action="store_true",
                        help="smoke: only rows present in the (partial) cache; no-op for the full run")
    parser.add_argument("--limit-rows", type=int, default=0, help="smoke: cap the number of cohort rows (0 = all)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = _log_line
    summary = run_extraction(args, log)
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(
        f"[a11-infer] done: {summary['n_rows']} rows x {len(summary['seeds'])} seeds x "
        f"2 regimes in {summary['inference_seconds']}s -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
