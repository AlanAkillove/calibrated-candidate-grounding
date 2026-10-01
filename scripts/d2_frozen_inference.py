"""V2-D2 Phase 3b - frozen inference on the RefCOCO language cohort.

Runs the *frozen* V1 reliability stack over the re-encoded RefCOCO expressions and
writes per-row raw predictions, so every downstream statistic is computed from these
files and nothing later can re-touch a model.  Nothing is fitted:

  * scorer  - the three Phase-0B B3 seeds, loaded read-only (``results/phase0b_independent``);
  * reliability heads - the no-refit closed-form bundle R1 (Stats Logistic) and
    E1b (Stats+Semantic Logistic) + the frozen per-seed temperature
    (``results/phase1e_refcocog_external/frozen_models``);
  * features - regions / bank / image sizes reused from the RefCOCO+ main caches, text
    from ``cache/refcoco_lang`` (only the language distribution moved).

Jobs (each = regime x K, over the cohort rows eligible for that K):

  * ``random_k5 / _k10 / _k20 / _k50`` - C1 cardinality stress (``0 new parameters``);
  * ``hard_k5``                         - C4 same-category competitor set;
  * ``random_k5`` doubles as the C4 matched control (scored on the common cohort).

    conda activate deepminer
    python scripts/d2_frozen_inference.py                 # full
    python scripts/d2_frozen_inference.py --limit-rows 64 # smoke

Runtime and peak VRAM are recorded, mirroring the A11 external inference driver.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.external import refcoco_lang as L  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import frozen_load as fl  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

PHASE1_COHORT = Path("results/v2_d2_refcoco_lang/phase1/cohort.csv")
DEFAULT_OUT = Path("results/v2_d2_refcoco_lang/predictions")
B3_SEEDS: Tuple[int, ...] = (1, 2, 3)
SEM_CHUNK = 4096
#: The 14 backbone-neutral V2 semantic features (the C4 manipulation-check source).
V2_SEM_NAMES = tuple(v2feat.V2_PRIMARY_SEMANTIC_NAMES)

#: (name, regime, K) - C1 sweeps random over the cardinality set; C4 adds the hard arm.
JOBS: Tuple[Tuple[str, str, int], ...] = (
    ("random_k5", "random", 5),
    ("random_k10", "random", 10),
    ("random_k20", "random", 20),
    ("random_k50", "random", 50),
    ("hard_k5", "same_category", 5),
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def encoded_rows(cohort: L.LangCohort) -> np.ndarray:
    """The cohort rows whose RefCOCO text was actually re-encoded (Phase 3a).

    Exactly the union of the C1 common cohort (``K = 50`` eligible) and the C4
    matched-hard cohort - the same set ``extract_d2_text_features.score_rows``
    encodes.  Every scored row is therefore guaranteed to have a text embedding;
    rows outside this union are never materialised (their ``C_K`` would demand a
    query vector that does not exist in ``cache/refcoco_lang``).
    """
    keep = set(int(r) for r in L.common_cohort_rows(cohort))
    keep |= set(int(r) for r in L.matched_hard_rows(cohort))
    return np.fromiter(sorted(keep), dtype=np.int64)


def job_rows(cohort: L.LangCohort, regime: str, k: int) -> np.ndarray:
    """Rows eligible for ``(regime, K)`` and re-encoded: target present is guaranteed.

    ``random`` needs ``n_valid >= K-1``; ``same_category`` additionally needs
    ``n_same_category >= K-1`` (a real hard competitor set, never padded).  The
    result is then intersected with the encoded union so the analysis cohorts
    (C1 common / C4 matched-hard) are covered while no missing text is touched.
    """
    need = k - 1
    mask = cohort.n_valid_distractors >= need
    if regime == "same_category":
        mask = mask & (cohort.n_same_category >= need)
    rows = np.nonzero(mask)[0].astype(np.int64)
    return np.intersect1d(rows, encoded_rows(cohort))


def _semantic_stats_chunked(z_q: np.ndarray, z_i: np.ndarray, scores: np.ndarray) -> np.ndarray:
    n = int(z_q.shape[0])
    out = np.empty((n, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, n, SEM_CHUNK):
        stop = min(start + SEM_CHUNK, n)
        out[start:stop] = sfeat.semantic_stats(z_q[start:stop], z_i[start:stop], scores[start:stop])
    return out


def _v2_semantic_chunked(z_q: np.ndarray, z_i: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The 14-d V2 primary semantic features, computed in the same chunks as ``sem16``.

    Only the C4 manipulation check reads these three columns
    (``winner_competitor_max_cos`` / ``winner_top2_cos`` / ``q_margin12``); storing
    the full vector keeps the check a pure read of the frozen prediction file.
    """
    n = int(z_q.shape[0])
    out = np.empty((n, len(V2_SEM_NAMES)), dtype=np.float64)
    for start in range(0, n, SEM_CHUNK):
        stop = min(start + SEM_CHUNK, n)
        out[start:stop] = v2feat.v2_primary_semantic_stats(
            z_q[start:stop], z_i[start:stop], scores[start:stop]
        )
    return out


def score_job(
    name: str,
    regime: str,
    k: int,
    cohort: L.LangCohort,
    rows: np.ndarray,
    corpus: L.RefCOCOLangFeatureCorpus,
    models: Dict[str, Any],
    bundle: fl.FrozenExternalModels,
    *,
    batch_size: int,
    out_dir: Path,
    text_cache: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    samples = cohort.samples_for(regime, rows)
    batch = hscores.materialise_examples(corpus, samples, k, text_cache=text_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: List[Dict[str, Any]] = []
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
        sem14 = _v2_semantic_chunked(batch.z_q, batch.z_i, scores64)
        stats_conf, e1b_conf = bundle.predict(scorer, stats17, sem16)
        target = out_dir / f"d2__{name}__{scorer}.npz"
        np.savez(
            target,
            sentence_id=np.asarray([int(s.sentence_id) for s in samples], dtype=np.int64),
            ref_id=np.asarray([int(s.ref_id) for s in samples], dtype=np.int64),
            image_id=np.asarray([int(s.image_id) for s in samples], dtype=np.int64),
            regime=np.asarray([regime] * len(samples)),
            k=np.asarray([k] * len(samples), dtype=np.int64),
            correct=np.asarray(correct, dtype=bool),
            scores=np.asarray(scores, dtype=np.float32),
            conf_msp=np.asarray(scalars["msp"], dtype=np.float64),
            conf_stats=np.asarray(stats_conf, dtype=np.float64),
            conf_e1b=np.asarray(e1b_conf, dtype=np.float64),
            margin=np.asarray(margin, dtype=np.float64),
            entropy=np.asarray(-scalars["neg_entropy"], dtype=np.float64),
            stats17=stats17,
            sem16=sem16,
            sem14=sem14,
        )
        _log(f"[d2-infer] {name}/{scorer}: n={len(samples)} acc={float(correct.mean()):.4f} -> {target.name}")
        result_rows.append({
            "job": name, "regime": regime, "k": k, "scorer": scorer,
            "n": int(len(samples)), "acc": float(correct.mean()),
            "mean_margin": float(margin.mean()), "temperature": temperature,
            "file": target.name,
        })
    del batch
    return result_rows


def run_inference(args: argparse.Namespace) -> Dict[str, Any]:
    cohort = L.load_cohort_csv(Path(args.cohort_csv))
    sizes = L.load_image_sizes_map()
    image_ids = sorted({int(i) for i in cohort.image_id})
    sizes_sub = {i: sizes[i] for i in image_ids if i in sizes}
    corpus = L.RefCOCOLangFeatureCorpus(
        region_root=L.MAIN_FEATURES_ROOT, text_root=L.LANG_CACHE_ROOT,
        bank_paths={"frozen_plus": L.FROZEN_BANK}, image_sizes=sizes_sub,
    )
    bundle = fl.load_models(args.frozen_models)
    missing = [f"b3_seed{s}" for s in B3_SEEDS if f"b3_seed{s}" not in bundle.seeds]
    if missing:
        raise AssertionError(f"bundle missing seeds: {missing}")
    models = hscores.load_frozen_scorers(b3_root=L.B3_ROOT, seeds=B3_SEEDS, device=args.device)

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    text_cache: Dict[int, np.ndarray] = {}
    out_dir = Path(args.out_dir)
    started = time.perf_counter()
    all_rows: List[Dict[str, Any]] = []
    try:
        for name, regime, k in JOBS:
            rows = job_rows(cohort, regime, k)
            if args.limit_rows and args.limit_rows > 0:
                rows = rows[: int(args.limit_rows)]
            if rows.size == 0:
                raise AssertionError(f"job {name}: no eligible rows")
            all_rows.extend(score_job(
                name, regime, k, cohort, rows, corpus, models, bundle,
                batch_size=args.batch_size, out_dir=out_dir, text_cache=text_cache,
            ))
    finally:
        corpus.close()

    summary: Dict[str, Any] = {
        "protocol": "V2-D2 Phase 3b (frozen inference, 0 new parameters)",
        "cohort_rows": len(cohort),
        "jobs": [j[0] for j in JOBS],
        "seeds": [f"b3_seed{s}" for s in B3_SEEDS],
        "semantic_stat_names": list(sfeat.SEMANTIC_STAT_NAMES),
        "v2_sem_names": list(V2_SEM_NAMES),
        "inference_seconds": round(time.perf_counter() - started, 2),
        "rows": all_rows,
    }
    if torch.cuda.is_available():
        summary["peak_vram_mb"] = round(float(torch.cuda.max_memory_allocated()) / (1024 ** 2), 1)
        summary["device_name"] = torch.cuda.get_device_name(0)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cohort-csv", type=Path, default=PHASE1_COHORT)
    p.add_argument("--frozen-models", type=Path, default=L.FROZEN_MODELS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-report", type=Path, default=DEFAULT_OUT.parent / "inference_report.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit-rows", type=int, default=0, help="smoke: cap rows per job (0 = all)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_inference(args)
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"[d2-infer] done: {len(summary['rows'])} job x seed cells -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
