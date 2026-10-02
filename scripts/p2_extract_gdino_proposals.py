"""Query-independent Grounding DINO Proposal-C bank writer (V2-P2 Phase B).

Mirrors :mod:`scripts.p1_extract_detr_proposals` (the Proposal-B DETR-R50 writer)
exactly - same ``bank-v1`` layout, same atomic / resumable append, same corruption
handling, same stats artefact - but the generator is the frozen, class-prompt
query-independent Grounding DINO (``ccg.data.gdino``) instead of DETR-R50.

    Proposal-A  cache/proposals.h5             (frozen COCO RPN, never re-run here)
    Proposal-B  cache/proposals_detr_r50.h5     (frozen DETR-R50, never re-run here)
    Proposal-C  cache/proposals_gdino.h5        (this script)

Covering the full RefCOCO+ experiment image universe (``data/full_image_manifest
.csv``, 19992 train2014 images), so the C1 replication cohort is drawn from the
same image set the frozen RPN / DETR banks cover.

The detector is run **one image per forward call** with a FIXED COCO-thing class
prompt (the referring expression never enters the detector), so the bank is a
class-prompt, query-independent proposal family - the same conditioning form as
the frozen DETR-R50.  The bank is never padded and a proposal is never dropped by
a score threshold; sanitisation / top-N share the DETR code path with a row-exact
cross-check (``ccg.data.gdino``).

    conda activate deepminer
    python scripts/p2_extract_gdino_proposals.py --out cache/proposals_gdino.h5 \
        --resume --prefetch-workers 8
    python scripts/p2_extract_gdino_proposals.py --max-images 20 ...   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg.data import gdino  # noqa: E402
from ccg.data.bank import (  # noqa: E402
    BANK_SCHEMA_VERSION,
    bank_attrs,
    finalize_bank,
    find_corrupt_images,
    image_ids,
    repair_bank,
    write_bank_entry,
)
from extract_proposals import (  # noqa: E402
    load_manifest,
    load_image_rgb,
    resolve_image_path,
)

__all__ = ["DEFAULT_TOP_N", "PROGRESS_EVERY", "build_parser", "run", "main"]

DEFAULT_TOP_N = gdino.DEFAULT_TOP_N
PROGRESS_EVERY = 200
DEFAULT_OUT = _ROOT / "cache" / "proposals_gdino.h5"
DEFAULT_MANIFEST = _ROOT / "data" / "full_image_manifest.csv"
DEFAULT_IMAGES_ROOT = _ROOT / "data" / "raw" / "mscoco"
DEFAULT_ANNOTATIONS = _ROOT / "data" / "raw" / "annotations" / "instances_train2014.json"
DEFAULT_MODEL_DIR = _ROOT / "cache" / "hf_local" / "IDEA-Research" / "grounding-dino-base"
IDENTITY_PATH = _ROOT / "results" / "v2_proposal_robustness" / "p2_gdino_probe" / "checkpoint_identity.json"


def _load_identity(path: Path) -> Dict[str, Any]:
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def _identity_weights_sha(identity: Dict[str, Any]) -> Optional[str]:
    files = identity.get("files") or {}
    model = files.get("model.safetensors") or {}
    return model.get("sha256")


def _identity_revision(identity: Dict[str, Any]) -> Optional[str]:
    return identity.get("revision_sha") or identity.get("revision")


def _versions() -> Dict[str, Any]:
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    images_root = Path(args.images_root)
    annotation_file = Path(args.annotations)
    top_n = int(args.top_n)
    workers = int(args.prefetch_workers)
    resume = bool(args.resume)
    identity = _load_identity(Path(args.identity))

    print(f"[gdino] manifest    : {manifest_path}")
    rows, duplicates = load_manifest(manifest_path)
    if duplicates:
        print(f"[gdino] WARNING     : {duplicates} duplicate image_id row(s) dropped")
    print(f"[gdino] manifest    : {len(rows)} images")

    # --- integrity of an existing GDINO bank -------------------------------
    existing_ids: set = set()
    if out_path.exists():
        corrupt = find_corrupt_images(out_path)
        if corrupt:
            if args.repair:
                removed = repair_bank(out_path)
                print(f"[gdino] repair      : removed {len(removed)} corrupt group(s) e.g. {removed[:5]}")
            else:
                head = ", ".join(str(i) for i in corrupt[:10])
                raise RuntimeError(
                    f"{len(corrupt)} corrupt group(s) in {out_path}: {head}; re-run with --repair"
                )
        existing_ids = set(int(i) for i in image_ids(out_path).tolist())
        attrs = bank_attrs(out_path)
        if attrs and attrs.get("top_n") is not None and int(attrs["top_n"]) != top_n:
            raise RuntimeError(
                f"{out_path} built with top_n={attrs['top_n']} but --top-n is {top_n}; use --out"
            )
        if attrs and attrs.get("model_name") and str(attrs["model_name"]) != gdino.MODEL_NAME:
            raise RuntimeError(
                f"{out_path} is not a GDINO bank (model_name={attrs['model_name']}); refusing to mix"
            )
        print(f"[gdino] bank        : {out_path} already holds {len(existing_ids)} image(s)")

    todo: List[Dict[str, Any]] = []
    skipped_resume = 0
    for row in rows:
        if resume and row["image_id"] in existing_ids:
            skipped_resume += 1
            continue
        todo.append(row)
    if int(args.max_images) > 0:
        todo = todo[: int(args.max_images)]
    print(f"[gdino] to process  : {len(todo)} images (resume-skip={skipped_resume}, out={out_path}, N={top_n})")

    bundle = None
    device_name = str(args.device)
    torch_mod = None
    if todo:
        print(f"[gdino] loading     : {args.model_dir} on {args.device} ...")
        bundle = gdino.build_gdino_bundle(
            model_path=str(args.model_dir),
            annotation_file=annotation_file,
            device=str(args.device),
            revision=_identity_revision(identity),
        )
        print(f"[gdino] prompt      : {len(bundle.class_names)} classes, "
              f"{bundle.n_text_tokens} tokens, "
              f"{sum(len(c) for c in bundle.class_token_cols)} class cols")
        if str(args.device).startswith("cuda"):
            import torch

            torch_mod = torch
            torch.cuda.reset_peak_memory_stats()
            device_name = torch.cuda.get_device_name(torch.device(args.device))

    failures: List[Dict[str, Any]] = []
    n_done = n_failed = n_crops = 0
    processed_t0: Optional[float] = None
    bar = tqdm(total=len(todo), desc="gdino", unit="img", disable=None) if todo else None

    def submit(executor: ThreadPoolExecutor, it: Iterator[Dict[str, Any]], pending) -> None:
        while len(pending) < workers:
            try:
                row = next(it)
            except StopIteration:
                return
            path = resolve_image_path(images_root, row["file_name"])
            pending.append((row, path, executor.submit(
                (lambda p: load_image_rgb(p) if p is not None else None), path
            )))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import h5py

    with h5py.File(out_path, "a") as h5:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: deque = deque()
            todo_iter = iter(todo)
            submit(executor, todo_iter, pending)
            while pending:
                row, path, future = pending.popleft()
                image_id = int(row["image_id"])
                if processed_t0 is None:
                    processed_t0 = time.perf_counter()
                try:
                    if path is None:
                        raise FileNotFoundError(f"image file not found under {images_root}: {row['file_name']}")
                    image = future.result()
                    prop = gdino.extract_gdino_proposals(
                        image, bundle, top_n=top_n, autocast=bool(args.autocast)
                    )
                    n_raw = int(prop.meta["sanitize"]["n_kept"])
                    write_bank_entry(
                        h5, image_id, prop.boxes, prop.objectness,
                        {"n_raw_post_nms": max(n_raw, int(prop.boxes.shape[0]))},
                        overwrite=not resume,
                    )
                except Exception as exc:
                    n_failed += 1
                    failures.append({"image_id": image_id, "reason": f"{type(exc).__name__}: {exc}"})
                    if n_failed <= 10:
                        tqdm.write(f"[gdino] FAILED      : image {image_id}: {exc}")
                else:
                    n_done += 1
                    n_crops += int(prop.boxes.shape[0])
                    if n_done % PROGRESS_EVERY == 0:
                        h5.flush()
                        el = time.perf_counter() - processed_t0
                        tqdm.write(f"[gdino] progress    : {n_done}/{len(todo)} ({n_done / el:.2f} img/s) failed={n_failed}")
                # keep the in-flight queue saturated with newly prefetched rows
                submit(executor, todo_iter, pending)
                if bar is not None:
                    el = time.perf_counter() - processed_t0 if processed_t0 else 0.0
                    pf = {"imgs/s": f"{n_done / el:.2f}" if el > 0 else "0.00"}
                    if torch_mod is not None and str(args.device).startswith("cuda"):
                        pf["peak_vram"] = f"{torch_mod.cuda.max_memory_allocated() / (1024.0 ** 2):.0f} MB"
                    bar.set_postfix(pf, refresh=False)
                    bar.update(1)
            h5.flush()
    if bar is not None:
        bar.close()
    process_elapsed = (time.perf_counter() - processed_t0) if processed_t0 else 0.0

    peak_vram_mb = None
    if torch_mod is not None:
        peak_vram_mb = float(torch_mod.cuda.max_memory_allocated() / (1024.0 ** 2))

    finalize_bank(
        out_path,
        {
            "model_name": gdino.MODEL_NAME,
            "weights": gdino.MODEL_ID,
            "proposal_type": gdino.PROPOSAL_TYPE,
            "top_n": top_n,
            "transformers_version": _versions()["transformers"],
            "checkpoint_id": str(args.model_dir),
            "checkpoint_revision": _identity_revision(identity) or (bundle.revision if bundle else None),
            "checkpoint_sha256": _identity_weights_sha(identity),
            "score_definition": "top-N by max class-token sigmoid prob ([CLS] global col excluded; no threshold)",
            "query_independent": True,
            "prompt_conditioning": "fixed COCO-thing class prompt (expression never enters detector)",
            "n_classes": int(len(bundle.class_names)) if bundle else None,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "images_root": str(images_root),
        },
    )

    all_ids = [int(i) for i in image_ids(out_path).tolist()]
    k_values: List[int] = []
    with h5py.File(out_path, "r") as h5r:
        for image_id in all_ids:
            k_values.append(int(h5r[f"image_{image_id}"]["boxes"].shape[0]))
    hist: Dict[str, int] = {}
    for k in k_values:
        hist[str(k)] = hist.get(str(k), 0) + 1

    runtime = time.perf_counter() - started
    summary: Dict[str, Any] = {
        "task": "p2_extract_gdino_proposals",
        "config": {
            "images_root": str(images_root), "manifest": str(manifest_path), "out": str(out_path),
            "annotations": str(annotation_file), "model_dir": str(args.model_dir),
            "top_n": top_n, "device": str(args.device), "max_images": int(args.max_images),
            "resume": resume, "repair": bool(args.repair), "prefetch_workers": workers,
            "autocast": bool(args.autocast), "schema_version": BANK_SCHEMA_VERSION,
            "checkpoint_id": gdino.MODEL_ID,
        },
        "versions": _versions(),
        "n_images": len(all_ids),
        "n_crops_total": int(sum(k_values)),
        "per_image_K": {
            "mean": (float(np.mean(k_values)) if k_values else None),
            "min": (int(min(k_values)) if k_values else None),
            "max": (int(max(k_values)) if k_values else None),
            "histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        },
        "runtime_seconds": float(runtime),
        "images_per_second": float(n_done / process_elapsed) if process_elapsed > 0 else 0.0,
        "peak_vram_mb": peak_vram_mb,
        "device_name": device_name,
        "failures": failures,
        "n_images_this_run": n_done,
        "n_images_skipped_resume": skipped_resume,
        "n_images_failed": n_failed,
        "process_seconds": float(process_elapsed),
        "notes": [
            "Proposal-C Grounding DINO class-prompt, query-independent, frozen; written to a bank-v1 "
            "file distinct from the RPN / DETR banks (neither is re-run here).",
            "one image per forward call; normalised boxes scale by each image's original (W,H).",
            "sanitize/top-N shared with ccg.data.detr with a row-exact cross-check; labels carried aligned.",
            "failures counted, never silent; resume skips stored images; --repair drops half-written groups.",
        ],
    }
    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(
        f"[gdino] images      : bank={summary['n_images']} this_run={n_done} "
        f"resume-skip={skipped_resume} failed={n_failed}"
    )
    print(f"[gdino] runtime     : {runtime:.1f}s ({summary['images_per_second']:.2f} img/s, peak {peak_vram_mb} MB)")
    print(f"[gdino] stats       : {stats_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--repair", action="store_true")
    p.add_argument("--autocast", action="store_true", help="FP16 (default fp32 for determinism)")
    p.add_argument("--prefetch-workers", type=int, default=8)
    p.add_argument("--identity", type=Path, default=IDENTITY_PATH)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_images) < 0 or int(args.top_n) < 1 or int(args.prefetch_workers) < 1:
        print("[gdino] bad --max-images/--top-n/--prefetch-workers")
        return 2
    summary = run(args)
    if summary["failures"] and summary["n_images"] == 0:
        print("[gdino] WARNING: every image failed; bank is empty")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
