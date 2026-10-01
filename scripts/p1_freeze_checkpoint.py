"""V2-P1 step 0: freeze the DETR-R50 Proposal-B checkpoint identity + protocol.

Run this BEFORE any feasibility or result experiment.  It pins, in committed
artifacts that later stages re-read and never re-derive:

    results/v2_proposal_robustness/p1_detr_r50/checkpoint_identity.json
    results/v2_proposal_robustness/p1_detr_r50/protocol.json

Protocol section 1 forbids swapping the checkpoint after seeing C1/C4; freezing
identity/hash/rule here is what makes that enforceable.

    conda activate deepminer
    python scripts/p1_freeze_checkpoint.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ccg.data import detr  # noqa: E402

DEFAULT_OUT = _ROOT / "results" / "v2_proposal_robustness" / "p1_detr_r50"
DEFAULT_CACHE = str(_ROOT / "cache" / "hf_hub")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_snapshot(cache_dir: str, model_id: str) -> Optional[Path]:
    owner, _, name = model_id.partition("/")
    repo = Path(cache_dir) / f"models--{owner}--{name}"
    refs_main = repo / "refs" / "main"
    revision = refs_main.read_text(encoding="utf-8").strip() if refs_main.exists() else None
    if revision:
        snap = repo / "snapshots" / revision
        if snap.exists():
            return snap
    snaps = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").exists() else []
    return snaps[-1] if snaps else None


def _license(model_id: str) -> str:
    """Model-card license (offline-safe: falls back to the documented value)."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        card_data = (info.card_data or {}) if hasattr(info, "card_data") else {}
        lic = card_data.get("license")
        if lic:
            return str(lic)
    except Exception:
        pass
    return "apache-2.0 (facebookresearch/detr; HF model card license)"


def build(cache_dir: str, model_id: str) -> Dict[str, Any]:
    from transformers import DetrConfig, DetrImageProcessor

    cfg = DetrConfig.from_pretrained(model_id, cache_dir=cache_dir)
    proc = DetrImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    snapshot = _find_snapshot(cache_dir, model_id)
    weights = snapshot / "model.safetensors" if snapshot else None
    sha = _sha256(weights) if weights and weights.exists() else None

    identity = {
        "checkpoint_id": model_id,
        "detected_family": "DETR ResNet-50 (transformer set-prediction, query-independent)",
        "source": "huggingface hub (facebook/detr-resnet-50), COCO-pretrained",
        "revision": snapshot.name if snapshot else "main",
        "weights_file": weights.name if weights else "model.safetensors",
        "weights_sha256": sha,
        "weights_bytes": int(weights.stat().st_size) if weights and weights.exists() else None,
        "license": _license(model_id),
        "query_independent": True,
        "frozen": True,
        "trained_by_us": False,
    }

    protocol = {
        "phase": "V2-P1 - Second Proposal Family Robustness",
        "proposal_family": "Proposal-B = frozen DETR-R50 (Proposal-A stays the frozen COCO RPN)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_proposals": detr.DEFAULT_TOP_N,
        "num_queries": int(getattr(cfg, "num_queries", 100)),
        "num_labels": int(cfg.num_labels),
        "no_object_index": int(cfg.num_labels),
        "id2label_0_placeholder": dict(cfg.id2label).get(0),
        "preprocessing": {
            "image_processor": type(proc).__name__,
            "do_resize": bool(getattr(proc, "do_resize", True)),
            "do_normalize": bool(getattr(proc, "do_normalize", True)),
            "size": getattr(proc, "size", None),
            "shortest_edge": (getattr(proc, "size", {}) or {}).get("shortest_edge"),
            "longest_edge": (getattr(proc, "size", {}) or {}).get("longest_edge"),
            "image_mean": list(getattr(proc, "image_mean", [])),
            "image_std": list(getattr(proc, "image_std", [])),
            "aspect_preserving": True,
            "batch_size_for_extraction": 1,
            "batch_note": (
                "one image per forward call: the processor is aspect-preserving and, "
                "at batch=1, adds no padding, so normalised pred_boxes scale exactly to "
                "original pixels (inverse of post_process_object_detection)."
            ),
        },
        "score_definition": {
            "per_query": "max foreground class probability of the softmax over class logits",
            "excludes": "the trailing no-object column (index num_labels)",
            "foreground_columns": "0 .. num_labels-1 (index 0 is a COCO-id alignment placeholder with negligible probability)",
            "ranking": "descending foreground score, stable ties, truncated to N=64",
            "threshold": None,
            "shortfall_policy": "fewer than 64 surviving sanitized boxes keeps the real count; never padded",
            "never_uses": ["referring expression", "target category", "ground-truth boxes"],
        },
        "box_selection_rule": {
            "coordinate_space": "absolute xyxy pixels in the original image",
            "conversion": "cxcywh(normalised) -> xyxy -> scale by (W, H) -> clip to [0,W]x[0,H]",
            "drop": ["non-finite", "clipped width <= 1", "clipped height <= 1", "exact duplicate xyxy (first/highest-score kept)"],
            "near_duplicate_handling": "recorded only (redundancy stats); no result-driven NMS in the first version",
            "iou_target_criterion": "proposal IoU with GT target >= 0.5 (unchanged from V1)",
            "same_category_assignment": "proposal -> highest-IoU COCO GT object only if IoU >= 0.5 (unchanged from V1)",
        },
        "proposal_a_reference": {
            "bank": "cache/proposals.h5",
            "model": "fasterrcnn_resnet50_fpn (COCO_V1) RPN stage, N=64, frozen, NOT regenerated",
        },
        "no_result_driven_tuning": (
            "N, threshold, checkpoint, NMS, target-matching and same-category rules are "
            "frozen here; a gate failure is reported, never answered by tuning (sections 10/25)."
        ),
    }
    return {"identity": identity, "protocol": protocol}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--model-id", default=detr.MODEL_ID)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = build(str(args.cache_dir), str(args.model_id))

    id_path = out / "checkpoint_identity.json"
    pr_path = out / "protocol.json"
    id_path.write_text(json.dumps(result["identity"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pr_path.write_text(json.dumps(result["protocol"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[p1-freeze] checkpoint_identity: {id_path}")
    print(f"[p1-freeze] protocol          : {pr_path}")
    ident = result["identity"]
    print(
        f"[p1-freeze] {ident['checkpoint_id']} revision={ident['revision']} "
        f"sha256={(ident['weights_sha256'] or 'UNKNOWN')[:16]}... "
        f"N={result['protocol']['n_proposals']} queries={result['protocol']['num_queries']}"
    )
    if not ident["weights_sha256"]:
        print("[p1-freeze] WARNING: could not hash the snapshot weights (offline cache?)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
