"""Text (query) feature extraction with the frozen OpenCLIP encoder (protocol §4/§5).

Reads the UNC ``refs(unc).p`` corpus (49,856 references / 141,564 sentences /
19,992 images), orders it deterministically - references by ascending numeric
``ref_id``, sentences inside a reference by ascending ``sent_id`` - and assigns
the sequential ``sentence_id`` that indexes the cached joint embedding::

    sentence_id = global counter over (ref_id asc, sent_id asc)

The text cache is written **atomically** through :mod:`ccg.features.cache`: one
run either replaces ``text_features.h5`` + ``text_index.csv`` completely or
leaves the previous state untouched.  ``--resume`` skips the forward pass only
when the existing cache already covers the whole corpus; a partial text cache
is *rebuilt from scratch*, never silently extended (text encoding is cheap on
the frozen GPU - redoing it is cheaper than reasoning about partial state).

CLI::

    python -m ccg.features.extract_text \
        --refs "data/raw/refcoco+/refcoco+/refs(unc).p" \
        --out cache/features --batch-size 128 --precision fp16 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence

from tqdm import tqdm

from ..data.refcoco import load_refs_pickle
from ..utils.logging import utc_now_iso
from .cache import (
    FeatureCache,
    merge_extraction_stats,
    update_metadata,
    write_text_cache,
    write_text_index,
)
from .clip_encoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    ClipEncoder,
    build_encoder,
)

__all__ = [
    "SENTENCE_TEXT_KEYS",
    "sentence_text",
    "build_text_corpus",
    "text_cache_covers",
    "run_text_extraction",
    "build_parser",
    "main",
]

#: Sentence text preference: the cleaned ``sent`` first, raw second, tokens last.
SENTENCE_TEXT_KEYS = ("sent", "raw", "text")


def sentence_text(sentence, *, ref_id: int, sent_id: int) -> str:
    """The encoding text of one sentence record (never silently empty)."""
    if isinstance(sentence, str):
        text = sentence.strip()
    elif isinstance(sentence, Mapping):
        text = ""
        for key in SENTENCE_TEXT_KEYS:
            value = sentence.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        else:
            tokens = sentence.get("tokens")
            if tokens is not None:
                text = " ".join(str(token) for token in tokens).strip()
    else:
        raise ValueError(
            f"ref {ref_id} sentence {sent_id}: unsupported sentence record type "
            f"{type(sentence).__name__} (expected mapping or str)"
        )
    if not text:
        raise ValueError(
            f"ref {ref_id} sentence {sent_id}: no usable text among {SENTENCE_TEXT_KEYS!r} "
            f"or 'tokens'; refusing to encode an empty query"
        )
    return text


def build_text_corpus(refs_path: str | Path) -> List[dict]:
    """The frozen sentence ordering as rows for ``text_index.csv``.

    Every row: ``sentence_id, ref_id, sent_id, image_id, split, text``.
    References are sorted by ascending numeric ``ref_id``, sentences by
    ascending ``sent_id`` (mapping keys when ``sentences`` is a dict).  A list
    of sentence records without ``sent_id`` falls back to the in-reference
    position, which is counted and reported by the caller instead of being
    applied silently.
    """
    refs = load_refs_pickle(refs_path)
    ordered = sorted(refs, key=lambda record: int(record["ref_id"]))
    rows: List[dict] = []
    seen_ref_ids = set()
    sentence_id = 0
    for record in ordered:
        ref_id = int(record["ref_id"])
        if ref_id in seen_ref_ids:
            raise ValueError(f"duplicate ref_id {ref_id} in {refs_path}")
        seen_ref_ids.add(ref_id)
        image_id = int(record["image_id"])
        split = str(record.get("split") or "")
        sentences = record["sentences"]
        if isinstance(sentences, Mapping):
            items = [(int(key), value) for key, value in sentences.items()]
        else:
            items = []
            for position, item in enumerate(sentences):
                if isinstance(item, Mapping) and item.get("sent_id") is not None:
                    items.append((int(item["sent_id"]), item))
                else:
                    items.append((position, item))
        seen_sent_ids = set()
        for sent_id, sentence in sorted(items, key=lambda pair: pair[0]):
            if sent_id in seen_sent_ids:
                raise ValueError(f"ref {ref_id}: duplicate sent_id {sent_id}")
            seen_sent_ids.add(sent_id)
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "ref_id": ref_id,
                    "sent_id": sent_id,
                    "image_id": image_id,
                    "split": split,
                    "text": sentence_text(sentence, ref_id=ref_id, sent_id=sent_id),
                }
            )
            sentence_id += 1
    return rows


def text_cache_covers(root: str | Path, sentence_ids: Sequence[int]) -> bool:
    """True when the existing text cache already holds every ``sentence_id``."""
    try:
        with FeatureCache.open(root) as cache:
            cached = set(int(i) for i in cache.sentence_ids().tolist())
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return {int(i) for i in sentence_ids} <= cached


def run_text_extraction(
    encoder: ClipEncoder,
    corpus: Sequence[Mapping],
    out_root: str | Path,
    *,
    batch_size: Optional[int] = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
    metadata_updates: Optional[dict] = None,
) -> dict:
    """Encode ``corpus`` and atomically write ``text_features.h5`` + index.

    Returns the extraction statistics of this phase (``skipped`` is True when
    ``resume`` found a complete cache and nothing was re-encoded).  The encode
    pass runs under a ``text`` progress bar (hidden on non-TTY streams).
    """
    out_root = Path(out_root)
    rows = [dict(row) for row in corpus]
    for key in ("sentence_id", "ref_id", "sent_id", "image_id", "split", "text"):
        for row in rows:
            if key not in row:
                raise ValueError(f"corpus row misses {key!r}: {row}")

    if resume and text_cache_covers(out_root, [row["sentence_id"] for row in rows]):
        log(f"text cache already covers all {len(rows)} sentences - skipping")
        return {
            "n_sentences": len(rows),
            "text_seconds": 0.0,
            "texts_per_second": 0.0,
            "skipped": True,
        }

    texts = [str(row["text"]) for row in rows]
    started = time.perf_counter()
    bar = tqdm(total=len(texts), desc="text", unit="sent", disable=None)
    tokenize = getattr(encoder, "tokenize", None)
    try:
        if callable(tokenize):
            # ``encode_texts`` slices the sentences into encoder batches itself;
            # wrapping its tokenizer surfaces that batch loop on the bar without
            # touching the single encode call (batching/values stay identical).
            def _tokenize_with_progress(batch, *args, **kwargs):
                tokens = tokenize(batch, *args, **kwargs)
                bar.update(len(batch))
                bar.set_postfix(
                    sent_per_s=f"{bar.n / max(time.perf_counter() - started, 1e-9):.0f}",
                    refresh=False,
                )
                return tokens

            encoder.tokenize = _tokenize_with_progress
            try:
                features = encoder.encode_texts(texts, batch_size=batch_size)
            finally:
                encoder.tokenize = tokenize
        else:
            features = encoder.encode_texts(texts, batch_size=batch_size)
    finally:
        bar.close()
    if features.shape != (len(rows), encoder.feature_dim):
        raise RuntimeError(
            f"encoder returned {features.shape} for {len(rows)} texts; expected "
            f"({len(rows)}, {encoder.feature_dim})"
        )
    seconds = time.perf_counter() - started

    vector_map = {int(rows[i]["sentence_id"]): features[i] for i in range(len(rows))}
    write_text_cache(out_root, vector_map, texts=texts)
    write_text_index(out_root, rows)

    token_counts = encoder.token_counts(texts)
    n_truncated = int((token_counts > encoder.context_length).sum())
    if n_truncated:
        log(
            f"WARNING: {n_truncated} sentence(s) exceed the {encoder.context_length}-token "
            "context of this backbone's tokenizer and were encoded truncated"
        )
    stats = {
        "n_sentences": len(rows),
        "text_seconds": round(seconds, 3),
        "texts_per_second": round(len(rows) / seconds, 1) if seconds > 0 else 0.0,
        "skipped": False,
        "max_sentence_tokens": int(token_counts.max()) if token_counts.size else 0,
        "n_sentences_truncated": n_truncated,
        "text_encoded_utc": utc_now_iso(),
    }
    if metadata_updates:
        merge_extraction_stats(out_root, {"text": stats})
        update_metadata(out_root, metadata_updates)
    log(
        f"encoded {len(rows)} sentences in {seconds:.1f}s "
        f"({stats['texts_per_second']}/s) -> {out_root}"
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_text",
        description="Extract frozen OpenCLIP query embeddings for the RefCOCO+ corpus.",
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=Path("data/raw/refcoco+/refcoco+/refs(unc).p"),
        help="UNC referring pickle refs(unc).p",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("cache/features"), help="feature cache root"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--checkpoint", type=Path, default=None, help="local weights (skip download)")
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path("cache/hf_hub"),
        help="huggingface cache holding the frozen checkpoint",
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="encode only the first N sentences (smoke runs; 0 = all)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip when the cache already covers the corpus"
    )
    parser.add_argument("--download-attempts", type=int, default=10)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = print
    corpus = build_text_corpus(args.refs)
    log(f"corpus: {len(corpus)} sentences from {args.refs}")
    if args.max_sentences and args.max_sentences > 0:
        corpus = corpus[: args.max_sentences]
        log(f"truncated corpus to {len(corpus)} sentences (--max-sentences)")

    sentence_ids = [row["sentence_id"] for row in corpus]
    if args.resume and text_cache_covers(args.out, sentence_ids):
        log(f"text cache already covers all {len(corpus)} sentences - nothing to do")
        return 0

    encoder = build_encoder(
        device=args.device,
        precision=args.precision,
        cache_dir=args.hf_cache_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        download_attempts=args.download_attempts,
    )
    stats = run_text_extraction(
        encoder,
        corpus,
        args.out,
        batch_size=args.batch_size,
        resume=args.resume,
        log=log,
        metadata_updates={
            "backbone": encoder.metadata(),
            "native_logit_scale": encoder.logit_scale,
        },
    )
    if not stats.get("skipped"):
        merge_extraction_stats(args.out, {"text": stats})
    return 0


if __name__ == "__main__":
    sys.exit(main())
