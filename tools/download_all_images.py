"""Download ALL RefCOCO+ (UNC) images (19,992, COCO train2014) into data/raw/mscoco/train2014/.

Reuses the retry/verify logic of download_audit_images.py; skips files already present
(audit subset images are already there). Writes data/full_images_download_summary.json
plus data/full_image_manifest.csv (image_id,file_name,coco_split,ref_count).
"""

import argparse
import csv
import json
import pickle
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEST = RAW / "mscoco"
BASE = "http://images.cocodataset.org/train2014"


def fetch_one(item: tuple[int, str], retries: int = 3) -> tuple[int, bool, str]:
    image_id, file_name = item
    dest = DEST / "train2014" / file_name
    if dest.exists() and dest.stat().st_size > 1024:
        return image_id, True, "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{BASE}/{file_name}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if data[:2] != b"\xff\xd8":
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
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    with open(RAW / "refcoco+" / "refcoco+" / "refs(unc).p", "rb") as fh:
        refs = pickle.load(fh, encoding="latin1")
    ref_count: dict[int, int] = {}
    for r in refs:
        ref_count[int(r["image_id"])] = ref_count.get(int(r["image_id"]), 0) + 1

    inst = json.load(open(RAW / "annotations" / "instances_train2014.json", encoding="utf-8"))
    id_to_name = {int(im["id"]): im["file_name"] for im in inst["images"]}
    del inst

    items = [(iid, id_to_name[iid]) for iid in sorted(ref_count)]
    missing = [iid for iid, _ in items if iid not in id_to_name]  # impossible by construction
    print(f"images to ensure: {len(items)}, missing mapping: {len(missing)}", flush=True)

    ok = fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, it) for it in items]
        for i, fut in enumerate(as_completed(futures), 1):
            image_id, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                failures.append({"image_id": image_id, "reason": msg})
            if i % 2000 == 0:
                print(f"  {i}/{len(items)} (ok={ok} fail={fail})", flush=True)

    with open(ROOT / "data" / "full_image_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "file_name", "coco_split", "ref_count"])
        for iid, fname in items:
            writer.writerow([iid, fname, "train2014", ref_count[iid]])

    summary = {"total": len(items), "ok": ok, "failed": fail, "failures": failures[:100]}
    (ROOT / "data" / "full_images_download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
