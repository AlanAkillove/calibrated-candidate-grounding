"""Feature-cache integrity audit for the frozen OpenCLIP caches (protocol §9).

Two layers:

1. **structure** (pure numpy / h5py, no GPU): NaN / Inf / zero-norm row counts
   over every stored feature block, offset-table consistency (``image_offsets``
   / ``sentence_offsets`` strictly contiguous, rows == arange), dimension-512
   and dtype checks, crop total == proposal-bank total (per image and overall),
   invalid-crop CSV rows == zero-norm rows == ``metadata.invalid_crop_count``,
   sentence/ref counts vs ``text_index.csv`` and metadata;
2. **refit** (GPU, ``--sample``): re-encode random region crops / sentences /
   whole images with the very same frozen encoder and compare against the
   cached rows - ``max_abs_embedding_diff`` and ``max_cosine_diff`` are reported
   per part (fp16 tolerance: cosine deviation should stay <= 2e-3).

The JSON report is written to ``<features>/integrity_report.json``; the CLI
exits non-zero when any check fails.

Both layers show a live tqdm progress bar on an interactive terminal: one
``structure`` bar (4 parts) and one ``refit`` bar (sampled rows; ``disable=None``
keeps both silent when stderr is not a TTY).

Usage::

    python scripts/audit_cache.py --features cache/features \
        --bank cache/proposals.h5 --sample 100
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.bank import iter_bank  # noqa: E402  (pure module)
from ccg.features.cache import (  # noqa: E402
    CACHE_VERSION,
    GLOBAL_FILENAME,
    METADATA_FILENAME,
    REGION_FILENAME,
    TEXT_FILENAME,
    FeatureCache,
    feature_health,
    read_metadata,
    read_text_index,
    write_json_atomic,
)
from ccg.features.clip_encoder import FEATURE_DIM  # noqa: E402
from ccg.features.extract_regions import read_invalid_crop_rows  # noqa: E402
from ccg.utils.logging import utc_now_iso  # noqa: E402

__all__ = ["audit_structure", "audit_refit", "run_audit", "build_parser", "main"]

REPORT_FILENAME = "integrity_report.json"
INVALID_CROPS_FILENAME = "invalid_crops.csv"

#: fp16 re-forward tolerance: |1 - cos(cached, refit)| must stay below this.
COSINE_TOL = 2e-3
#: rows scanned per h5 chunk in the health sweep.
SCAN_CHUNK = 1 << 16


def _h5py():
    import h5py

    return h5py


def _scan_health(features, chunk: int = SCAN_CHUNK) -> dict:
    """Accumulate :func:`feature_health` over a (possibly huge) feature dataset."""
    totals = {"total_rows": 0, "nan_rows": 0, "inf_rows": 0, "zero_norm_rows": 0}
    n_rows = int(features.shape[0])
    for start in range(0, n_rows, chunk):
        part = feature_health(features[start : start + chunk])
        for key in totals:
            totals[key] += int(part[key])
    return totals


def _cosine_diff_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``1 - cos(a_i, b_i)`` per row (float32); caller excludes zero-norm rows."""
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a32, axis=1)
    norm_b = np.linalg.norm(b32, axis=1)
    prod = np.clip(norm_a * norm_b, 1e-12, None)
    return 1.0 - np.einsum("ij,ij->i", a32, b32) / prod


# ---------------------------------------------------------------------------
# layer 1 - structure (no GPU)
# ---------------------------------------------------------------------------
def _audit_region(root: Path, meta: dict, bank_path: Optional[Path], report: dict,
                  failures: List[str], checks: List[dict], log: Callable[[str], None]) -> None:
    region_path = root / REGION_FILENAME

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)

    if not region_path.exists():
        check("region.file", False, f"{region_path} is missing")
        return

    h5py = _h5py()
    with h5py.File(str(region_path), "r") as handle:
        feats = handle["features"]
        offsets = np.asarray(handle["image_offsets"][...], dtype=np.int64).reshape(-1, 3)
        n_crops = int(feats.shape[0])
        dim = int(feats.shape[1])
        dtype = str(feats.dtype)
        counts = offsets[:, 2] if offsets.size else np.zeros(0, dtype=np.int64)
        starts = offsets[:, 1] if offsets.size else np.zeros(0, dtype=np.int64)
        expected_starts = (
            np.concatenate([[0], np.cumsum(counts)[:-1]]) if counts.size else np.zeros(0, np.int64)
        )
        offsets_ok = bool(np.array_equal(starts, expected_starts))
        rows_ok = int(counts.sum()) == n_crops
        ids = offsets[:, 0].tolist() if offsets.size else []
        health = _scan_health(feats)

    info: dict = {
        "n_images": len(ids),
        "n_crops": n_crops,
        "dim": dim,
        "dtype": dtype,
        "offsets_contiguous": offsets_ok,
        "counts_total_matches_rows": rows_ok,
        "duplicate_image_ids": len(set(ids)) != len(ids),
        **health,
    }
    check("region.dim", dim == FEATURE_DIM, f"dim={dim}, expected {FEATURE_DIM}")
    check("region.dtype", dtype == "float16", f"dtype={dtype}")
    check("region.offsets_contiguous", offsets_ok, "start != cumsum(counts)")
    check("region.counts_total_matches_rows", rows_ok, f"sum(counts)={int(counts.sum())} rows={n_crops}")
    check("region.no_duplicate_image_ids", len(set(ids)) == len(ids))
    check("region.nan_rows", health["nan_rows"] == 0, f"nan_rows={health['nan_rows']}")
    check("region.inf_rows", health["inf_rows"] == 0, f"inf_rows={health['inf_rows']}")

    invalid_path = root / INVALID_CROPS_FILENAME
    if invalid_path.exists():
        invalid_rows = read_invalid_crop_rows(invalid_path)
        info["invalid_csv_rows"] = len(invalid_rows)
        check(
            "region.zero_norm_equals_invalid_csv",
            health["zero_norm_rows"] == len(invalid_rows),
            f"zero rows {health['zero_norm_rows']} vs csv rows {len(invalid_rows)}",
        )
    else:
        info["invalid_csv_rows"] = None
        check(
            "region.zero_norm_without_csv",
            health["zero_norm_rows"] == 0,
            f"{health['zero_norm_rows']} zero rows but no {INVALID_CROPS_FILENAME}",
        )

    if meta.get("invalid_crop_count") is not None:
        check(
            "region.metadata_invalid_count",
            int(meta["invalid_crop_count"]) == health["zero_norm_rows"],
            f"metadata {meta['invalid_crop_count']} vs zero rows {health['zero_norm_rows']}",
        )
    if meta.get("n_crops") is not None:
        check("region.metadata_n_crops", int(meta["n_crops"]) == n_crops, f"{meta['n_crops']} vs {n_crops}")
    if meta.get("n_images") is not None:
        check("region.metadata_n_images", int(meta["n_images"]) == len(ids), f"{meta['n_images']} vs {len(ids)}")

    if bank_path is not None:
        bank_counts: Dict[int, int] = {}
        for image_id, boxes, _objectness in iter_bank(bank_path):
            bank_counts[int(image_id)] = int(np.asarray(boxes).reshape(-1, 4).shape[0])
        bank_crops = int(sum(bank_counts.values()))
        cached_counts = {int(i): int(c) for i, c in zip(ids, counts.tolist())}
        missing = sorted(set(bank_counts) - set(cached_counts))
        extra = sorted(set(cached_counts) - set(bank_counts))
        mismatches = [
            (i, cached_counts[i], bank_counts[i])
            for i in sorted(set(bank_counts) & set(cached_counts))
            if cached_counts[i] != bank_counts[i]
        ]
        info.update(
            bank_n_images=len(bank_counts),
            bank_n_crops=bank_crops,
            bank_missing_images=missing[:20],
            bank_missing_images_total=len(missing),
            bank_extra_images=extra[:20],
            bank_extra_images_total=len(extra),
            bank_count_mismatches=mismatches[:20],
            bank_count_mismatches_total=len(mismatches),
        )
        check("region.crop_total_equals_bank", n_crops == bank_crops, f"cache {n_crops} vs bank {bank_crops}")
        check(
            "region.image_ids_equal_bank",
            not missing and not extra,
            f"{len(missing)} bank ids missing from cache, {len(extra)} extra ids",
        )
        check(
            "region.per_image_counts_equal_bank",
            not mismatches,
            f"{len(mismatches)} images with a differing box count",
        )
    report["region"] = info


def _audit_text(root: Path, meta: dict, report: dict, failures: List[str],
                checks: List[dict], log: Callable[[str], None]) -> None:
    text_path = root / TEXT_FILENAME

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)

    if not text_path.exists():
        check("text.file", False, f"{text_path} is missing")
        return

    h5py = _h5py()
    with h5py.File(str(text_path), "r") as handle:
        feats = handle["features"]
        offsets = np.asarray(handle["sentence_offsets"][...], dtype=np.int64).reshape(-1, 2)
        n_sentences = int(feats.shape[0])
        dim = int(feats.shape[1])
        dtype = str(feats.dtype)
        ids = offsets[:, 0].tolist()
        rows = offsets[:, 1]
        rows_ok = bool(np.array_equal(rows, np.arange(len(ids), dtype=np.int64)))
        n_texts = int(handle["texts"].shape[0]) if "texts" in handle else -1
        health = _scan_health(feats)

    info: dict = {
        "n_sentences": n_sentences,
        "dim": dim,
        "dtype": dtype,
        "offsets_rows_are_arange": rows_ok,
        "duplicate_sentence_ids": len(set(ids)) != len(ids),
        "texts_dataset_rows": n_texts,
        **health,
    }
    check("text.dim", dim == FEATURE_DIM, f"dim={dim}")
    check("text.dtype", dtype == "float16", f"dtype={dtype}")
    check("text.offsets_rows_are_arange", rows_ok and len(ids) == n_sentences,
          f"{len(ids)} offset rows for {n_sentences} feature rows")
    check("text.no_duplicate_sentence_ids", len(set(ids)) == len(ids))
    check("text.texts_dataset_rows", n_texts == n_sentences, f"texts={n_texts} vs features={n_sentences}")
    check("text.nan_rows", health["nan_rows"] == 0, f"nan_rows={health['nan_rows']}")
    check("text.inf_rows", health["inf_rows"] == 0, f"inf_rows={health['inf_rows']}")
    check("text.zero_norm_rows", health["zero_norm_rows"] == 0, f"zero_norm_rows={health['zero_norm_rows']}")

    if meta.get("n_sentences") is not None:
        check("text.metadata_n_sentences", int(meta["n_sentences"]) == n_sentences,
              f"{meta['n_sentences']} vs {n_sentences}")

    index_path = root / "text_index.csv"
    if index_path.exists():
        index_rows = read_text_index(root)
        index_ids = {int(row["sentence_id"]) for row in index_rows}
        info["text_index_rows"] = len(index_rows)
        info["text_index_n_refs"] = len({int(row["ref_id"]) for row in index_rows})
        check("text.index_rows_match", len(index_rows) == n_sentences,
              f"index {len(index_rows)} vs features {n_sentences}")
        check("text.index_ids_match", index_ids == set(ids),
              f"{len(index_ids ^ set(ids))} sentence ids differ between index and cache")
    else:
        info["text_index_rows"] = None
    report["text"] = info


def _audit_global(root: Path, meta: dict, report: dict, failures: List[str],
                  checks: List[dict], log: Callable[[str], None]) -> None:
    global_path = root / GLOBAL_FILENAME

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)

    if not global_path.exists():
        check("global.file", False, f"{global_path} is missing")
        return

    h5py = _h5py()
    with h5py.File(str(global_path), "r") as handle:
        feats = handle["features"]
        ids = np.asarray(handle["image_ids"][...], dtype=np.int64).reshape(-1)
        n_images = int(feats.shape[0])
        dim = int(feats.shape[1])
        dtype = str(feats.dtype)
        sorted_ids = bool(ids.size == 0 or np.all(np.diff(ids) > 0))
        health = _scan_health(feats)

    info: dict = {
        "n_images": n_images,
        "dim": dim,
        "dtype": dtype,
        "ids_rows_match": int(ids.size) == n_images,
        "ids_strictly_ascending": sorted_ids,
        **health,
    }
    check("global.dim", dim == FEATURE_DIM, f"dim={dim}")
    check("global.dtype", dtype == "float16", f"dtype={dtype}")
    check("global.ids_rows_match", int(ids.size) == n_images, f"ids={ids.size} rows={n_images}")
    check("global.ids_strictly_ascending", sorted_ids, "image_ids not strictly ascending")
    check("global.nan_rows", health["nan_rows"] == 0, f"nan_rows={health['nan_rows']}")
    check("global.inf_rows", health["inf_rows"] == 0, f"inf_rows={health['inf_rows']}")
    check("global.zero_norm_rows", health["zero_norm_rows"] == 0, f"zero_norm_rows={health['zero_norm_rows']}")
    report["global"] = info


def audit_structure(root: str | Path, *, bank_path: Optional[str | Path] = None,
                    log: Callable[[str], None] = print) -> dict:
    """Structural audit of one cache root (no GPU); returns the report fragment."""
    root = Path(root)
    checks: List[dict] = []
    failures: List[str] = []
    report: dict = {"root": str(root.resolve())}

    # four structure parts: metadata / region / text / global (protocol §9)
    with tqdm(total=4, desc="structure", unit="part", disable=None) as bar:
        bar.set_postfix_str("metadata")
        meta = read_metadata(root)
        report["cache_version"] = meta.get("cache_version")
        check_ok = meta.get("cache_version") == CACHE_VERSION
        checks.append({"name": "metadata.cache_version", "ok": bool(check_ok),
                       "detail": f"got {meta.get('cache_version')!r}, expected {CACHE_VERSION!r}"})
        if not check_ok:
            failures.append(f"metadata.cache_version: got {meta.get('cache_version')!r}")
        bar.update(1)

        bank = Path(bank_path) if bank_path is not None else None
        bar.set_postfix_str("region")
        _audit_region(root, meta, bank, report, failures, checks, log)
        bar.update(1)
        bar.set_postfix_str("text")
        _audit_text(root, meta, report, failures, checks, log)
        bar.update(1)
        bar.set_postfix_str("global")
        _audit_global(root, meta, report, failures, checks, log)
        bar.update(1)

    report["summary_counts"] = {
        "n_images_region": (report.get("region") or {}).get("n_images"),
        "n_crops": (report.get("region") or {}).get("n_crops"),
        "n_sentences": (report.get("text") or {}).get("n_sentences"),
        "n_images_global": (report.get("global") or {}).get("n_images"),
    }
    report["checks"] = checks
    report["failures"] = failures
    report["passed"] = not failures
    return report


# ---------------------------------------------------------------------------
# layer 2 - re-forward refit sampling (GPU)
# ---------------------------------------------------------------------------
def audit_refit(
    root: str | Path,
    sample: int,
    *,
    bank_path: Optional[str | Path] = None,
    images_root: Optional[str | Path] = None,
    manifest: Optional[Dict[int, Tuple[str, str]]] = None,
    device: str = "cuda",
    precision: str = "fp16",
    hf_cache_dir: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    batch_size: int = 128,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict:
    """Re-encode a random sample of cached rows and diff against the cache."""
    from ccg.features.clip_encoder import build_encoder, load_rgb_image
    from ccg.features.extract_regions import resolve_image_path

    root = Path(root)
    sample = int(sample)
    rng = random.Random(seed)
    out: dict = {"sample": sample, "seed": seed}
    cache = FeatureCache.open(root)
    encoder = build_encoder(
        device=device,
        precision=precision,
        cache_dir=Path(hf_cache_dir) if hf_cache_dir else None,
        checkpoint_path=Path(checkpoint) if checkpoint else None,
        batch_size=batch_size,
    )
    out["encoder"] = {
        "device": str(encoder.config.device),
        "checkpoint_sha256": encoder.config.checkpoint_sha256,
        "native_logit_scale": encoder.logit_scale,
    }
    log(f"[refit] encoder ready on {encoder.config.device}, checkpoint sha256={encoder.config.checkpoint_sha256}")

    # One shared "refit" bar for the whole re-forward layer: its total is the
    # number of sampled rows of the parts that will actually run (skipped parts
    # contribute 0) and every sampled row ticks it exactly once.
    n_region = int(cache.region_ids().size)
    n_sentence = int(cache.sentence_ids().size)
    n_global = int(cache.global_ids().size)
    planned = 0
    if n_region and bank_path is not None and images_root is not None:
        planned += min(sample, n_region)
    if n_sentence:
        planned += min(sample, n_sentence)
    if n_global and images_root is not None:
        planned += min(sample, n_global)
    bar = tqdm(total=planned, desc="refit", unit="row", disable=None)
    try:
        # ---- region crops --------------------------------------------------
        region_ids = cache.region_ids()
        if region_ids.size == 0:
            out["region"] = {"skipped": "the cache holds no region features"}
        elif bank_path is None:
            out["region"] = {"skipped": "no --bank given; region refit needs the proposal boxes"}
        elif images_root is None:
            out["region"] = {"skipped": "no --images-root given"}
        else:
            bar.set_postfix_str("region")
            take = min(sample, int(region_ids.size))
            selected = rng.sample([int(i) for i in region_ids], take)
            wanted = set(selected)
            boxes_by_id: Dict[int, np.ndarray] = {}
            for image_id, boxes, _objectness in iter_bank(bank_path):
                image_id = int(image_id)
                if image_id in wanted:
                    boxes_by_id[image_id] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            absent = [i for i in selected if i not in boxes_by_id]
            if absent:
                raise RuntimeError(f"sampled image ids absent from the bank: {absent[:5]} (+{max(0, len(absent) - 5)})")

            max_abs = 0.0
            max_cos = 0.0
            n_compared = 0
            n_invalid = 0
            mask_mismatch = 0
            shape_mismatches: List[int] = []
            for position, image_id in enumerate(selected, start=1):
                cached = np.asarray(cache.region_features(image_id), dtype=np.float32)
                boxes = boxes_by_id[image_id]
                path = resolve_image_path(images_root, image_id, manifest=manifest)
                image = load_rgb_image(path)
                refit, valid = encoder.encode_crops(image, boxes)
                refit = np.asarray(refit, dtype=np.float32)
                if refit.shape != cached.shape:
                    shape_mismatches.append(image_id)
                    bar.update(1)
                    continue
                zero_rows = np.linalg.norm(cached, axis=1) == 0
                cmp_mask = valid & ~zero_rows
                mask_mismatch += int(np.sum(valid != ~zero_rows))
                n_invalid += int((~valid).sum())
                if cmp_mask.any():
                    a = cached[cmp_mask]
                    b = refit[cmp_mask]
                    max_abs = max(max_abs, float(np.abs(a - b).max()))
                    max_cos = max(max_cos, float(_cosine_diff_rows(a, b).max()))
                    n_compared += int(cmp_mask.sum())
                if position % 25 == 0:
                    bar.clear()  # keep the log line and the live bar from clobbering
                    log(f"[refit] region {position}/{len(selected)} images sampled")
                    bar.refresh()
                bar.update(1)
            out["region"] = {
                "n_images_sampled": len(selected),
                "n_rows_compared": n_compared,
                "n_invalid_rows": n_invalid,
                "mask_mismatch_rows": mask_mismatch,
                "shape_mismatches": shape_mismatches,
                "max_abs_embedding_diff": round(max_abs, 8),
                "max_cosine_diff": round(max_cos, 8),
            }

        # ---- sentences -----------------------------------------------------
        sentence_ids = cache.sentence_ids()
        if sentence_ids.size == 0:
            out["text"] = {"skipped": "the cache holds no text features"}
        else:
            bar.set_postfix_str("text")
            index_text = None
            try:
                index_text = {int(row["sentence_id"]): row["text"] for row in read_text_index(root)}
            except (FileNotFoundError, ValueError):
                index_text = None
            take = min(sample, int(sentence_ids.size))
            selected = rng.sample([int(i) for i in sentence_ids], take)
            texts: List[str] = []
            usable: List[int] = []
            for sentence_id in selected:
                text = ""
                try:
                    text = cache.text(sentence_id)
                except (KeyError, ValueError):
                    text = ""
                if not text and index_text is not None:
                    text = index_text.get(sentence_id, "")
                if text:
                    texts.append(text)
                    usable.append(sentence_id)
            bar.update(len(selected) - len(usable))  # sampled sentences with no stored text
            refit = encoder.encode_texts(texts) if texts else np.zeros((0, FEATURE_DIM), np.float16)
            max_abs = 0.0
            max_cos = 0.0
            n_compared = 0
            zero_rows = 0
            for row, sentence_id in enumerate(usable):
                cached = np.asarray(cache.text_features(sentence_id), dtype=np.float32)
                if float(np.linalg.norm(cached)) == 0.0:
                    zero_rows += 1
                    bar.update(1)
                    continue
                b = np.asarray(refit[row], dtype=np.float32)
                max_abs = max(max_abs, float(np.abs(cached - b).max()))
                max_cos = max(max_cos, float(_cosine_diff_rows(cached[None, :], b[None, :])[0]))
                n_compared += 1
                bar.update(1)
            out["text"] = {
                "n_sentences_sampled": len(selected),
                "n_rows_compared": n_compared,
                "n_without_text": len(selected) - len(usable),
                "zero_norm_rows": zero_rows,
                "max_abs_embedding_diff": round(max_abs, 8),
                "max_cosine_diff": round(max_cos, 8),
            }

        # ---- whole images --------------------------------------------------
        global_ids = cache.global_ids()
        if global_ids.size == 0:
            out["global"] = {"skipped": "the cache holds no global features"}
        elif images_root is None:
            out["global"] = {"skipped": "no --images-root given"}
        else:
            bar.set_postfix_str("global")
            take = min(sample, int(global_ids.size))
            selected = rng.sample([int(i) for i in global_ids], take)
            images = [
                load_rgb_image(resolve_image_path(images_root, i, manifest=manifest)) for i in selected
            ]
            refit = np.zeros((len(selected), FEATURE_DIM), dtype=np.float16)
            for start in range(0, len(images), batch_size):
                chunk = images[start : start + batch_size]
                refit[start : start + batch_size] = encoder.encode_images(
                    chunk, batch_size=batch_size
                )
                bar.update(len(chunk))  # the encode batches are the sampled rows' GPU work
            max_abs = 0.0
            max_cos = 0.0
            n_compared = 0
            zero_rows = 0
            for row, image_id in enumerate(selected):
                cached = np.asarray(cache.global_feature(image_id), dtype=np.float32)
                if float(np.linalg.norm(cached)) == 0.0:
                    zero_rows += 1
                    continue
                b = np.asarray(refit[row], dtype=np.float32)
                max_abs = max(max_abs, float(np.abs(cached - b).max()))
                max_cos = max(max_cos, float(_cosine_diff_rows(cached[None, :], b[None, :])[0]))
                n_compared += 1
            out["global"] = {
                "n_images_sampled": len(selected),
                "n_rows_compared": n_compared,
                "zero_norm_rows": zero_rows,
                "max_abs_embedding_diff": round(max_abs, 8),
                "max_cosine_diff": round(max_cos, 8),
            }
    finally:
        bar.close()

    cache.close()
    return out


def _refit_failures(refit: dict, tol: float = COSINE_TOL) -> List[str]:
    failures: List[str] = []
    for part in ("region", "text", "global"):
        info = refit.get(part)
        if not isinstance(info, dict) or "skipped" in info:
            continue
        if info.get("mask_mismatch_rows"):
            failures.append(f"refit.{part}: {info['mask_mismatch_rows']} rows where the invalid-crop mask disagrees with the cache")
        if info.get("shape_mismatches"):
            failures.append(f"refit.{part}: {len(info['shape_mismatches'])} images with a shape mismatch")
        if info.get("zero_norm_rows"):
            failures.append(f"refit.{part}: {info['zero_norm_rows']} sampled rows are all-zero")
        if info.get("n_without_text"):
            failures.append(f"refit.{part}: {info['n_without_text']} sampled sentences have no stored text")
        max_cos = info.get("max_cosine_diff")
        if max_cos is not None and float(max_cos) > tol:
            failures.append(f"refit.{part}: max_cosine_diff {max_cos} > {tol}")
    return failures


# ---------------------------------------------------------------------------
# orchestration + CLI
# ---------------------------------------------------------------------------
def run_audit(
    root: str | Path,
    *,
    bank_path: Optional[str | Path] = None,
    sample: int = 100,
    images_root: Optional[str | Path] = None,
    manifest: Optional[Dict[int, Tuple[str, str]]] = None,
    device: str = "cuda",
    precision: str = "fp16",
    hf_cache_dir: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    batch_size: int = 128,
    seed: int = 0,
    report_path: Optional[str | Path] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run both audit layers, write the JSON report, return it."""
    root = Path(root)
    started = time.perf_counter()
    failures: List[str] = []
    report: dict = {
        "created_utc": utc_now_iso(),
        "features_root": str(root.resolve()),
        "bank": str(bank_path) if bank_path is not None else None,
    }

    log("[audit] structure sweep (NaN/Inf/zero-norm, offsets, counts) ...")
    report["structure"] = audit_structure(root, bank_path=bank_path, log=log)
    failures.extend(report["structure"]["failures"])

    if int(sample) > 0:
        log(f"[audit] refit sampling ({sample} region / text / global) ...")
        try:
            refit = audit_refit(
                root,
                sample,
                bank_path=bank_path,
                images_root=images_root,
                manifest=manifest,
                device=device,
                precision=precision,
                hf_cache_dir=hf_cache_dir,
                checkpoint=checkpoint,
                batch_size=batch_size,
                seed=seed,
                log=log,
            )
            report["refit"] = refit
            failures.extend(_refit_failures(refit))
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}
            failures.append(f"refit: {type(exc).__name__}: {exc}")
    else:
        report["refit"] = {"skipped": "--sample 0"}

    report["cosine_tolerance"] = COSINE_TOL
    report["failures"] = failures
    report["passed"] = not failures
    report["seconds"] = round(time.perf_counter() - started, 3)

    if report_path is None:
        report_path = root / REPORT_FILENAME
    write_json_atomic(report_path, report)

    log("")
    log("[audit] summary")
    counts = report["structure"].get("summary_counts", {})
    for key, value in counts.items():
        log(f"  {key}: {value}")
    refit = report.get("refit") or {}
    for part in ("region", "text", "global"):
        info = refit.get(part)
        if isinstance(info, dict) and "max_cosine_diff" in info:
            log(
                f"  refit.{part}: max_abs={info['max_abs_embedding_diff']} "
                f"max_cos_diff={info['max_cosine_diff']} (rows={info.get('n_rows_compared')})"
            )
        elif isinstance(info, dict) and "skipped" in info:
            log(f"  refit.{part}: skipped ({info['skipped']})")
    log(f"  passed: {report['passed']}")
    if failures:
        for failure in failures[:20]:
            log(f"  FAIL: {failure}")
        if len(failures) > 20:
            log(f"  ... and {len(failures) - 20} more")
    log(f"  report: {Path(report_path).resolve()}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_cache",
        description="Integrity audit of a frozen OpenCLIP feature cache (§9).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--features", type=Path, default=Path("cache/features"))
    parser.add_argument("--bank", type=Path, default=Path("cache/proposals.h5"))
    parser.add_argument("--sample", type=int, default=100, help="rows re-encoded per part (0 = structure only)")
    parser.add_argument("--images-root", type=Path, default=Path("data/raw/mscoco"))
    parser.add_argument(
        "--manifest", type=Path, default=None, help="optional data/full_image_manifest.csv"
    )
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("cache/hf_hub"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="local weights (skip download)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None, help="report path (default <features>/" + REPORT_FILENAME + ")")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    features = Path(args.features)
    if not features.exists():
        print(f"ERROR: feature root {features} does not exist")
        return 2
    bank = Path(args.bank) if args.bank is not None else None
    if bank is not None and not bank.exists():
        print(f"WARNING: bank {bank} not found - skipping every bank cross-check")
        bank = None
    manifest = None
    if args.manifest is not None:
        from ccg.features.extract_regions import load_image_manifest

        manifest = load_image_manifest(args.manifest)
    report = run_audit(
        features,
        bank_path=bank,
        sample=args.sample,
        images_root=args.images_root,
        manifest=manifest,
        device=args.device,
        precision=args.precision,
        hf_cache_dir=args.hf_cache_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        seed=args.seed,
        report_path=args.report,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
