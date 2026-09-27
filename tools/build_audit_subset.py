"""Deterministic image-level audit subsset sampler (ONE-OFF canonical generator).

Frozen algorithm (seed 20260927):
  pool_train  = sorted unique image_ids with split == 'train' in refs(unc).p
  val_pool    = sorted unique image_ids with split == 'val'
  rng_v       = default_rng(seed+1); val_select = first half of rng_v.permutation(len(val_pool))
  rng         = default_rng(seed); train_sample = 1000 draws (no replace), then val_sample = 500
  image_id -> (file_name, coco_split) resolved via COCO instances_val2014/train2014 images[]

Output: data/audit_subset.csv  (frozen reference for scripts/build_audit_subset.py)
"""

import csv
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 20260927
N_TRAIN, N_VAL = 1000, 500


def main() -> None:
    with open(RAW / "refcoco+" / "refcoco+" / "refs(unc).p", "rb") as fh:
        refs = pickle.load(fh, encoding="latin1")

    def uniq(split):
        return sorted({int(r["image_id"]) for r in refs if r["split"] == split})

    train_pool, val_pool = uniq("train"), uniq("val")
    rng_v = np.random.default_rng(SEED + 1)
    perm = rng_v.permutation(len(val_pool))
    val_select = sorted(val_pool[i] for i in perm[: len(val_pool) // 2])
    print(f"pools: train={len(train_pool)} val={len(val_pool)} val_select={len(val_select)}")

    rng = np.random.default_rng(SEED)
    tr = [train_pool[i] for i in rng.choice(len(train_pool), size=N_TRAIN, replace=False)]
    va = [val_select[i] for i in rng.choice(len(val_select), size=N_VAL, replace=False)]

    ref_count = {}
    for r in refs:
        ref_count[int(r["image_id"])] = ref_count.get(int(r["image_id"]), 0) + 1

    manifest = {}
    for split, json_name in (("val2014", "instances_val2014.json"), ("train2014", "instances_train2014.json")):
        inst = json.load(open(RAW / "annotations" / json_name, encoding="utf-8"))
        for im in inst["images"]:
            manifest[int(im["id"])] = (im["file_name"], split)
        del inst

    rows = []
    for image_id, refcoco_split in [(i, "train") for i in tr] + [(i, "val_select") for i in va]:
        fname, coco_split = manifest[image_id]
        rows.append(
            {
                "image_id": image_id,
                "file_name": fname,
                "coco_split": coco_split,
                "refcoco_split": refcoco_split,
                "ref_count": ref_count.get(image_id, 0),
            }
        )
    rows.sort(key=lambda r: (r["coco_split"], r["image_id"]))

    out = ROOT / "data" / "audit_subset.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "file_name", "coco_split", "refcoco_split", "ref_count"])
        writer.writeheader()
        writer.writerows(rows)
    n_ref = sum(r["ref_count"] > 0 for r in rows)
    print(f"wrote {out}: {len(rows)} images (val={sum(r['coco_split']=='val2014' for r in rows)}, "
          f"train={sum(r['coco_split']=='train2014' for r in rows)}), images with RefCOCO+ refs: {n_ref}")


if __name__ == "__main__":
    main()
