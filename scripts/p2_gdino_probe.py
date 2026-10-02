"""V2-P2 Phase B: Grounding DINO E1 engineering probe (feasibility gates only).

Answers the five frozen probe gates from
``results/v2_proposal_robustness/phase_b_feasibility_audit.json`` BEFORE any
full-night bank extraction:

  1. >= 98% of probed images yield >= 64 sanitized proposals (nested-K supply)
  2. ref-target recall @ 0.5 (N = 64) >= 0.95
  3. same-category supply (>= 4 distractors, N = 64) >= 0.85
  4. candidate availability K50 (audit-subset rows) >= 0.90
  5. peak VRAM <= 6 GB

The measurement is byte-for-byte the frozen V1 audit pipeline
(:mod:`scripts.run_proposal_audit` primitives + :mod:`ccg.data.audit_report`
aggregation, identical to ``p1_f2_proposal_audit.py``): the only change is the
proposal source - GDINO class-prompt extraction runs live on ~100 images from
the frozen ``data/audit_subset.csv`` (the CSV's own stratified order, first
``--max-images`` rows; deterministic, no new seed).  No bank is written, no
scoring happens, nothing is trained. detector labels are recorded per image
(diagnostic histogram) but never used: same-category follows the frozen V1 rule
(highest-IoU COCO GT object, IoU >= 0.5).

    conda activate deepminer
    python scripts/p2_gdino_probe.py            # full probe (~100 images)
    python scripts/p2_gdino_probe.py --max-images 3   # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# frozen V1 pipeline pieces - never re-implemented here
import run_proposal_audit as v1  # noqa: E402
from ccg.data.audit_report import (  # noqa: E402
    DEFAULT_K_LEVELS,
    DEFAULT_SAME_CATEGORY_LEVELS,
    aggregate_audit,
    group_refs_by_image,
)
from ccg.data.gdino import build_gdino_bundle, extract_gdino_proposals  # noqa: E402
from extract_proposals import resolve_image_path  # noqa: E402  (flat/train2014 layout)

__all__ = ["DEFAULT_N", "DEFAULT_MAX_IMAGES", "run", "build_parser", "main"]

DEFAULT_N = 64
DEFAULT_MAX_IMAGES = 100
GATES_PATH = _ROOT / "results" / "v2_proposal_robustness" / "phase_b_feasibility_audit.json"
SUBSET = _ROOT / "data" / "audit_subset.csv"
MODEL_DIR = _ROOT / "cache" / "hf_local" / "IDEA-Research" / "grounding-dino-base"
IDENTITY = _ROOT / "results" / "v2_proposal_robustness" / "p2_gdino_probe" / "checkpoint_identity.json"
OUT_DIR = _ROOT / "results" / "v2_proposal_robustness" / "p2_gdino_probe"
_LEVELS: tuple = (DEFAULT_N,)


def _load_gates(path: Path) -> Dict[str, float]:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(audit["probe_gates_before_full_run"])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    started = time.perf_counter()
    gates = _load_gates(Path(args.gates))
    rows = v1.load_subset_rows(Path(args.subset))
    rows = rows[: int(args.max_images)]
    ids = [int(r["image_id"]) for r in rows]
    print(f"[p2-probe] images   : {len(ids)} (first rows of the frozen audit subset)")

    refs = v1.load_refcoco_refs(Path(args.refcoco_pickle), Path(args.refcoco_instances))
    refs_by_image = group_refs_by_image(refs)
    gt_index = v1.load_gt_index(Path(args.coco_annotations), ids)
    print(f"[p2-probe] refs     : {len(refs)} regions; gt on {len(gt_index)} images")

    print(f"[p2-probe] loading  : {args.model_dir}")
    bundle = build_gdino_bundle(str(args.model_dir), Path(args.coco_annotations), device=args.device)
    print(f"[p2-probe] prompt   : {len(bundle.class_names)} classes, "
          f"{bundle.n_text_tokens} tokens, {sum(len(c) for c in bundle.class_token_cols)} class cols")

    records: List[Dict[str, Any]] = []
    per_image: List[Dict[str, Any]] = []
    class_hist: Dict[str, int] = {}
    n_failed = 0
    for i, row in enumerate(rows):
        image_id = int(row["image_id"])
        path = resolve_image_path(args.images_root, row["file_name"])
        if path is None:
            n_failed += 1
            print(f"[p2-probe] image {image_id} MISSING: {row['file_name']}")
            continue
        try:
            prop = extract_gdino_proposals(path, bundle, top_n=args.n)
        except RuntimeError as exc:  # zero-propose image: counted, never hidden
            n_failed += 1
            print(f"[p2-probe] image {image_id} FAILED: {exc}")
            continue
        gt = gt_index.objects(image_id).non_crowd()
        records.append(
            v1.audit_level(
                image_id=image_id,
                bank_boxes=prop.boxes,
                bank_scores=prop.objectness,
                gt_boxes=gt.boxes,
                gt_categories=gt.categories,
                refs=refs_by_image.get(image_id, []),
                n_cap=args.n,
                iou_thresh=args.iou_thresh,
                k_levels=DEFAULT_K_LEVELS,
            )
        )
        for cid in prop.class_ids.tolist():
            name = bundle.class_names[cid]
            class_hist[name] = class_hist.get(name, 0) + 1
        per_image.append({
            "image_id": image_id,
            "n_kept_post_sanitize": int(prop.meta["n_kept_post_sanitize"]),
            "n_returned": int(prop.meta["num_returned"]),
            "truncated": bool(prop.meta["truncated"]),
            "elapsed_ms": float(prop.meta["elapsed_ms"]),
            "top1_score": float(prop.objectness[0]) if prop.objectness.size else 0.0,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"[p2-probe] progress  : {i + 1}/{len(rows)}", flush=True)

    if not records:
        raise RuntimeError("no image could be probed; GDINO extraction failed entirely")

    agg = aggregate_audit(
        records,
        k_levels=DEFAULT_K_LEVELS,
        same_category_levels=DEFAULT_SAME_CATEGORY_LEVELS,
        iou_thresh=args.iou_thresh,
    )
    level = agg["per_N"][str(args.n)]
    ref = level["recall"]["ref_target"]["by_threshold"]
    recall05 = float(ref["0.5"]["recall"])
    avail50 = float(level["candidate_availability"].get("50", {}).get("frac_available", float("nan")))
    same4 = float(level["same_category"].get("4", {}).get("frac", float("nan")))
    supply = per_image
    boxes_ok_share = float(np.mean([r["n_kept_post_sanitize"] >= args.n for r in supply]))
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else float("nan")
    mean_elapsed = float(np.mean([r["elapsed_ms"] for r in supply]))

    results = {
        "boxes_ge_64_share": boxes_ok_share,
        "ref_target_recall_05": recall05,
        "same_category_ge4_frac": same4,
        "candidate_availability_K50": avail50,
        "peak_vram_gb": peak_vram_gb,
    }
    thresholds = {
        "boxes_ge_64_share": gates["boxes_ge_64_share_min"],
        "ref_target_recall_05": gates["ref_target_recall_at_05_min"],
        "same_category_ge4_frac": gates["same_category_supply_ge4_min"],
        "candidate_availability_K50": gates["k50_availability_min"],
        "peak_vram_gb": -gates["peak_vram_gb_max"],  # negative = upper bound comparison
    }
    per_gate = {
        key: {
            "measured": results[key],
            "threshold": (gates["peak_vram_gb_max"] if key == "peak_vram_gb" else thresholds[key]),
            "pass": bool(results[key] >= thresholds[key]) if key != "peak_vram_gb"
            else bool(np.isfinite(peak_vram_gb) and peak_vram_gb <= gates["peak_vram_gb_max"]),
        }
        for key in results
    }
    all_pass = all(v["pass"] for v in per_gate.values())

    print(f"[p2-probe] boxes>=64  : {boxes_ok_share:.4f} (gate {gates['boxes_ge_64_share_min']})")
    print(f"[p2-probe] recall@0.5 : {recall05:.4f} (gate {gates['ref_target_recall_at_05_min']})")
    print(f"[p2-probe] same-cat>=4: {same4:.4f} (gate {gates['same_category_supply_ge4_min']})")
    print(f"[p2-probe] K50 avail  : {avail50:.4f} (gate {gates['k50_availability_min']})")
    print(f"[p2-probe] peak VRAM  : {peak_vram_gb:.2f} GB (gate {gates['peak_vram_gb_max']})")
    print(f"[p2-probe] VERDICT    : {'PASS' if all_pass else 'FAIL'}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "probe_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        for row in per_image:
            writer.writerow(row)

    identity = {}
    if Path(IDENTITY).exists():
        identity = json.loads(Path(IDENTITY).read_text(encoding="utf-8"))
    report = {
        "task": "p2_gdino_e1_probe",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "V2-P2-B0 probe gates (phase_b_feasibility_audit.json)",
        "config": {
            "subset": str(args.subset), "max_images": int(args.max_images),
            "n": int(args.n), "iou_thresh": float(args.iou_thresh),
            "device": args.device, "autocast": bool(args.autocast),
            "model_dir": str(args.model_dir),
            "target_criterion": "proposal IoU with highest-IoU GT target >= 0.5 (V1 rule)",
            "same_category_assignment": "proposal -> highest-IoU COCO GT object, IoU >= 0.5 (V1 rule; detector labels unused)",
            "images_probed": len(records), "images_failed": n_failed,
        },
        "checkpoint_identity": {
            "repo": identity.get("repo"), "revision_sha": identity.get("revision_sha"),
            "model_safetensors_sha256": (identity.get("files", {}).get("model.safetensors", {}) or {}).get("sha256"),
        },
        "gates": per_gate,
        "all_pass": bool(all_pass),
        "throughput": {
            "mean_extract_ms": mean_elapsed,
            "projected_hours_full_bank": float(mean_elapsed * 19992 / 3.6e6),
        },
        "class_label_histogram_diagnostic": class_hist,
        "per_N_detail": {str(args.n): level},
        "runtime_seconds": float(time.perf_counter() - started),
        "verdict": "PASS" if all_pass else "FAIL",
    }
    if args.autocast:
        report["config"]["autocast_requested"] = True
    path = out_dir / "probe_report.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[p2-probe] report     : {path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = v1.build_parser().parse_args([])
    p.add_argument("--subset", type=Path, default=SUBSET)
    p.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--autocast", action="store_true")
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--gates", type=Path, default=GATES_PATH)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--images-root", type=Path, default=_ROOT / "data" / "raw" / "mscoco")
    p.add_argument("--coco-annotations", type=Path, default=defaults.coco_annotations)
    p.add_argument("--refcoco-pickle", type=Path, default=defaults.refcoco_pickle)
    p.add_argument("--refcoco-instances", type=Path, default=defaults.refcoco_instances)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
