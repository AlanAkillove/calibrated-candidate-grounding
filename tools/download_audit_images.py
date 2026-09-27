"""Download COCO images for the frozen audit subset (data/audit_subset.csv).

Empirical note: all RefCOCO+ (UNC) images live in COCO train2014 (validated 0/1500
val-split images exist in instances_val2014; every subset id resolves in train2014).
URLs: http://images.cocodataset.org/train2014/<file_name>

Usage:
    python tools/download_audit_images.py [--workers 16] [--max-images N]
"""

import argparse
import csv
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEST = RAW / "mscoco"
BASE = {"train2014": "http://images.cocodataset.org/train2014", "val2014": "http://images.cocodataset.org/val2014"}


def fetch_one(row: dict, retries: int = 3) -> tuple[int, bool, str]:
    image_id = int(row["image_id"])
    src = f"{BASE[row['coco_split']]}/{row['file_name']}"
    dest = DEST / row["coco_split"] / row["file_name"]
    if dest.exists() and dest.stat().st_size > 1024:
        return image_id, True, "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data[:2] == b"\xff\xd8":  # JPEG SOI
                return image_id, False, "not-jpeg"
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return image_id, True, "ok"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}:{str(exc)[:80]}"
            if attempt == retries - 1:
                return image_id, False, err
    return image_id, False, "unreachable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data" / "audit_subset.csv", encoding="utf-8")))
    if args.max_images:
        rows = rows[: args.max_images]
    print(f"downloading {len(rows)} images with {args.workers} workers ...", flush=True)

    ok = fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            image_id, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                failures.append({"image_id": image_id, "reason": msg})
            if i % 200 == 0:
                print(f"  {i}/{len(rows)} done (ok={ok} fail={fail})", flush=True)

    summary = {"total": len(rows), "ok": ok, "failed": fail, "failures": failures[:50]}
    (ROOT / "data" / "audit_images_download_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
