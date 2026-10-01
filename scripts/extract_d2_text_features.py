"""V2-D2 Phase 3a - re-encode the RefCOCO expressions with the frozen B0 text tower.

D2 changes exactly one thing: the referring-expression *language* (RefCOCO keeps the
spatial words RefCOCO+ deleted).  The images, regions, proposal bank and every model
are frozen and reused, so the only quantity that must be recomputed is the OpenCLIP
ViT-B/32 **text** embedding of each RefCOCO expression.  This writes an independent
namespace and never touches the shared ``cache/features`` region/global caches:

    cache/refcoco_lang/text_features.h5     (one query per expression, keyed by expr_id)
    cache/refcoco_lang/text_index.csv
    cache/refcoco_lang/metadata.json

The encoder is the *same* frozen checkpoint A11 pinned (identity + sha256 verified
before a single query is encoded); nothing is trained.  Only the rows the downstream
stages actually score are encoded - the union of the C1 common cohort (``K = 50``
eligible) and the C4 matched-hard cohort (``>= 4`` same-category distractors) - so
the 149 dropped and non-hard rows never enter a forward pass.

    conda activate deepminer
    # smoke (validate the path + round-trip only):
    python scripts/extract_d2_text_features.py --limit-expressions 16
    # full:
    python scripts/extract_d2_text_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.external import frozen_identity as fi  # noqa: E402
from ccg.external import refcoco_lang as L  # noqa: E402
from ccg.features import extract_text  # noqa: E402
from ccg.features.cache import CACHE_VERSION, read_metadata, update_metadata  # noqa: E402
from ccg.features.clip_encoder import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, build_encoder  # noqa: E402

PHASE1_COHORT = Path("results/v2_d2_refcoco_lang/phase1/cohort.csv")
DEFAULT_OUT = L.LANG_CACHE_ROOT
STATS_PATH = Path("results/v2_d2_refcoco_lang/feature_extraction_stats.json")


def score_rows(cohort: L.LangCohort) -> List[int]:
    """Union of C1 common rows and C4 matched-hard rows (the rows a model scores)."""
    keep = set(int(r) for r in L.common_cohort_rows(cohort))
    keep |= set(int(r) for r in L.matched_hard_rows(cohort))
    return sorted(keep)


def build_corpus(cohort: L.LangCohort, rows: List[int], limit: int) -> List[dict]:
    rows = rows[:limit] if limit and limit > 0 else rows
    corpus: List[dict] = []
    seen: set[int] = set()
    for row in rows:
        expr_id = int(cohort.expr_id[row])
        if expr_id in seen:
            continue
        seen.add(expr_id)
        text = str(cohort.text[row]).strip()
        if not text:
            raise AssertionError(f"expr {expr_id}: empty expression text")
        corpus.append({
            "sentence_id": expr_id, "ref_id": int(cohort.ref_id[row]), "sent_id": 0,
            "image_id": int(cohort.image_id[row]), "split": "test", "text": text,
        })
    return corpus


def run_extraction(args: argparse.Namespace, log) -> dict:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cohort = L.load_cohort_csv(Path(args.cohort_csv))
    rows = score_rows(cohort)
    corpus = build_corpus(cohort, rows, args.limit_expressions)
    log(f"[d2-text] cohort rows={len(cohort)} to-score={len(rows)} to-encode={len(corpus)}")

    checkpoint = None
    if not args.skip_identity_check:
        identity = fi.assert_openclip_identity(args.features_root)
        checkpoint = Path(identity["checkpoint_path"])
        log(f"[d2-text] OpenCLIP identity verified: {identity['frozen']['model_name']}/"
            f"{identity['frozen']['pretrained']} sha256 {identity['measured_checkpoint_sha256'][:16]}...")
    else:
        log("[d2-text] WARNING: identity check skipped (tests only)")

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    encoder = build_encoder(
        device=args.device, precision=args.precision, cache_dir=args.hf_cache_dir,
        model_name=DEFAULT_MODEL_NAME, pretrained=DEFAULT_PRETRAINED,
        checkpoint_path=checkpoint, batch_size=args.batch_size,
    )
    measured = getattr(encoder.config, "checkpoint_sha256", None)
    if not args.skip_identity_check and measured != fi.FROZEN_OPENCLIP["checkpoint_sha256"]:
        raise AssertionError(f"built encoder sha256 {measured} != frozen {fi.FROZEN_OPENCLIP['checkpoint_sha256']}")

    started = time.perf_counter()
    stats = extract_text.run_text_extraction(
        encoder, corpus, out_root, batch_size=args.batch_size, resume=args.resume, log=log,
    )
    stats["encoder"] = {"model_name": DEFAULT_MODEL_NAME, "pretrained": DEFAULT_PRETRAINED,
                        "checkpoint_sha256": measured, "device": str(encoder.config.device)}
    stats["n_score_rows"] = len(rows)
    stats["cohort_csv"] = str(args.cohort_csv)
    stats["text_seconds"] = round(time.perf_counter() - started, 2)
    if torch.cuda.is_available():
        stats["peak_vram_mb"] = round(float(torch.cuda.max_memory_allocated()) / (1024 ** 2), 1)

    try:
        current = read_metadata(out_root)
    except FileNotFoundError:
        current = {}
    meta = {"backbone": encoder.metadata(), "native_logit_scale": encoder.logit_scale,
            "n_sentences": int(stats.get("n_sentences") or 0), "dataset": "refcoco_lang_d2",
            "cohort_rows": len(corpus),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if "created_utc" not in current:
        meta["created_utc"] = meta["updated_utc"]
    if "cache_version" not in current:
        meta["cache_version"] = CACHE_VERSION
    update_metadata(out_root, meta)

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(__import__("json").dumps(stats, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    log(f"[d2-text] encoded {stats.get('n_sentences')} queries -> {out_root}")
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cohort-csv", type=Path, default=PHASE1_COHORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-report", type=Path, default=STATS_PATH)
    p.add_argument("--features-root", type=Path, default=fi.DEFAULT_FEATURES_ROOT)
    p.add_argument("--hf-cache-dir", type=Path, default=Path("cache/hf_hub"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit-expressions", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-identity-check", action="store_true", help="tests only")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    run_extraction(args, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
