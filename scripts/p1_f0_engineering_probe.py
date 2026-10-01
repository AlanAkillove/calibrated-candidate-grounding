"""V2-P1 P1-F0: DETR-R50 engineering probe on ~300 RefCOCO+ images.

This is a *feasibility* pass, not a result experiment: it never writes the big
proposal bank.  It loads the frozen DETR bundle, runs one image per forward call
over a size-stratified sample of the RefCOCO+ image universe and reports:

  * checkpoint load + identity (re-read from the frozen artifacts);
  * peak VRAM (gate: must stay <= 7 GB on the RTX 4060 8 GB);
  * throughput (img/s) and wall time;
  * box-coordinate-conversion self-check against torchvision ``box_convert``;
  * score extraction sanity (foreground prob in [0,1], descending order);
  * ``N = 64`` availability, invalid / degenerate / exact-duplicate counts, and
    near-duplicate redundancy statistics.

If peak VRAM exceeds the budget the probe exits non-zero with ``STOP`` - the
model is never swapped automatically (protocol sections 4 / 25).

    conda activate deepminer
    python scripts/p1_f0_engineering_probe.py --num-images 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg.data import detr  # noqa: E402
from extract_proposals import load_manifest, resolve_image_path, load_image_rgb  # noqa: E402

DEFAULT_MANIFEST = _ROOT / "data" / "full_image_manifest.csv"
DEFAULT_IMAGES_ROOT = _ROOT / "data" / "raw" / "mscoco"
DEFAULT_CACHE = str(_ROOT / "cache" / "hf_hub")
OUT_DIR = _ROOT / "results" / "v2_proposal_robustness" / "p1_detr_r50"
PEAK_VRAM_GB_LIMIT = 7.0


def stratified_sample(rows: List[Dict[str, Any]], num: int) -> List[Dict[str, Any]]:
    """Every k-th manifest row so the probe sees a spread of image sizes."""
    if num <= 0 or num >= len(rows):
        return list(rows)
    step = len(rows) / float(num)
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(num)]


def box_conversion_selfcheck(bundle, image) -> float:
    """Max abs pixel diff between our converter and torchvision ``box_convert``."""
    import torch
    from torchvision.ops import box_convert

    pil = detr._as_pil(image)
    w, h = pil.size
    enc = bundle.processor(images=pil, return_tensors="pt")
    dev = torch.device(bundle.device)
    with torch.no_grad():
        out = bundle.model(pixel_values=enc["pixel_values"].to(dev),
                           pixel_mask=enc.get("pixel_mask").to(dev) if enc.get("pixel_mask") is not None else None)
    cxcywh = out.pred_boxes[0].float().cpu()
    ref = box_convert(cxcywh, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    ref = ref * np.array([w, h, w, h], dtype=np.float64)  # xyxy -> [x1*W, y1*H, x2*W, y2*H]
    ours = detr.cxcywh_to_xyxy_pixels(cxcywh.numpy(), w, h)
    return float(np.max(np.abs(ref - ours))) if ref.size else 0.0


def run(num_images: int, device: str, autocast: bool, batch_size: int = 16) -> Dict[str, Any]:
    started = time.perf_counter()
    rows, _ = load_manifest(DEFAULT_MANIFEST)
    sample = stratified_sample(rows, num_images)

    bundle = detr.build_detr_bundle(device=device, cache_dir=DEFAULT_CACHE)
    torch_mod = None
    if str(device).startswith("cuda"):
        import torch
        torch_mod = torch
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(torch.device(device))
    else:
        device_name = "cpu"

    bank_sizes: List[int] = []
    n64_full = [0]
    invalid = {"invalid_finite": 0, "invalid_small": 0, "exact_duplicates": 0}
    redundancy: List[float] = []
    max_score_seen = 0.0
    order_ok = True
    failures: List[Dict[str, Any]] = []
    gpu_forward_seconds = 0.0

    # decode on a bounded thread pool so throughput reflects the GPU path, the
    # way the full-bank extraction (P1-F1) will run.
    from concurrent.futures import ThreadPoolExecutor

    def _load(row):
        path = resolve_image_path(DEFAULT_IMAGES_ROOT, row["file_name"])
        if path is None:
            return row, None
        return row, load_image_rgb(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(_load, sample))

    ready = [(row, img) for row, img in loaded if img is not None]
    for row, img in loaded:
        if img is None:
            failures.append({"image_id": row["image_id"], "reason": "file missing"})

    def _tally(prop) -> None:
        nonlocal max_score_seen, order_ok
        boxes, scores, meta = prop.boxes, prop.objectness, prop.meta
        k = int(boxes.shape[0])
        bank_sizes.append(k)
        n64_full[0] += int(k >= detr.DEFAULT_TOP_N)
        for key in invalid:
            invalid[key] += int(meta["sanitize"].get(key, 0))
        redundancy.append(detr.near_duplicate_stats(boxes, iou_high=0.9)["frac_pairs_gt_high"])
        if scores.size:
            max_score_seen = max(max_score_seen, float(scores.max()))
            if np.any(np.diff(scores) > 1e-6):
                order_ok = False

    for start in range(0, len(ready), batch_size):
        chunk = ready[start:start + batch_size]
        imgs = [img for _row, img in chunk]
        sizes = [img.size for img in imgs]  # (width, height)
        t0 = time.perf_counter()
        try:
            props = detr.extract_detr_proposals_batch(
                imgs, sizes, bundle, top_n=detr.DEFAULT_TOP_N, autocast=autocast
            )
            gpu_forward_seconds += time.perf_counter() - t0
            for prop in props:
                _tally(prop)
        except Exception as exc:  # one bad batch: fall back to per-image
            gpu_forward_seconds += time.perf_counter() - t0
            for row, img in chunk:
                try:
                    _tally(detr.extract_detr_proposals(img, bundle, top_n=detr.DEFAULT_TOP_N, autocast=autocast))
                except Exception as exc2:
                    failures.append({"image_id": row["image_id"],
                                     "reason": f"{type(exc2).__name__}: {exc2}"})

    peak_gb = None
    if torch_mod is not None:
        peak_gb = float(torch_mod.cuda.max_memory_allocated() / (1024.0 ** 3))

    conv_err = box_conversion_selfcheck(bundle, load_image_rgb(resolve_image_path(DEFAULT_IMAGES_ROOT, sample[0]["file_name"])))

    elapsed = time.perf_counter() - started
    n_ok = len(bank_sizes)
    report = {
        "task": "p1_f0_engineering_probe",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_id": bundle.checkpoint_id,
        "revision": bundle.revision,
        "device": str(device),
        "device_name": device_name,
        "autocast": bool(autocast),
        "batch_size": int(batch_size),
        "num_images_sampled": len(sample),
        "num_images_ok": n_ok,
        "num_images_failed": len(failures),
        "wall_seconds": float(elapsed),
        "images_per_second": float(n_ok / elapsed) if elapsed > 0 else 0.0,
        "gpu_forward_images_per_second": float(n_ok / gpu_forward_seconds) if gpu_forward_seconds > 0 else 0.0,
        "peak_vram_gb": peak_gb,
        "peak_vram_limit_gb": PEAK_VRAM_GB_LIMIT,
        "vram_within_budget": bool(peak_gb is None or peak_gb <= PEAK_VRAM_GB_LIMIT),
        "box_conversion_max_abs_pixel_error": conv_err,
        "bank_size_summary": {
            "mean": float(np.mean(bank_sizes)) if bank_sizes else None,
            "min": int(np.min(bank_sizes)) if bank_sizes else None,
            "max": int(np.max(bank_sizes)) if bank_sizes else None,
            "share_exactly_64": float(np.mean([k == detr.DEFAULT_TOP_N for k in bank_sizes])) if bank_sizes else None,
            "share_full_N64_available": float(n64_full[0] / n_ok) if n_ok else None,
        },
        "sanitize_counts_total": invalid,
        "near_duplicate_frac_pairs_gt_0p9_mean": float(np.mean(redundancy)) if redundancy else None,
        "max_foreground_score_seen": float(max_score_seen),
        "scores_descending_ok": bool(order_ok),
        "projected_full_bank_seconds_at_19992": float((19992 / (n_ok / elapsed)) if n_ok else float("nan")),
        "failures": failures[:20],
        "verdict": None,
    }
    if peak_gb is not None and peak_gb > PEAK_VRAM_GB_LIMIT:
        report["verdict"] = "STOP: peak VRAM over budget"
    elif n_ok == 0:
        report["verdict"] = "STOP: no image extracted"
    elif conv_err > 1.0:
        report["verdict"] = "STOP: box conversion mismatch"
    else:
        report["verdict"] = "OK"
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-images", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--autocast", action="store_true", help="FP16 autocast (default off: fp32 for stability)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)

    report = run(int(args.num_images), str(args.device), bool(args.autocast), int(args.batch_size))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "engineering_probe.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[p1-f0] probe        : {path}")
    print(
        f"[p1-f0] images       : ok={report['num_images_ok']} failed={report['num_images_failed']} "
        f"of {report['num_images_sampled']}"
    )
    print(
        f"[p1-f0] throughput   : {report['images_per_second']:.2f} img/s "
        f"-> projected full 19992 in {report['projected_full_bank_seconds_at_19992'] / 60:.1f} min"
    )
    print(
        f"[p1-f0] peak VRAM    : {report['peak_vram_gb']} GB (limit {PEAK_VRAM_GB_LIMIT})"
    )
    print(
        f"[p1-f0] N=64 avail   : {report['bank_size_summary']['share_full_N64_available']:.4f} "
        f"mean bank {report['bank_size_summary']['mean']}"
    )
    print(f"[p1-f0] box conv err : {report['box_conversion_max_abs_pixel_error']:.3f} px")
    print(f"[p1-f0] scores desc  : {report['scores_descending_ok']} (max {report['max_foreground_score_seen']:.3f})")
    print(f"[p1-f0] verdict      : {report['verdict']}")
    return 0 if report["verdict"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
