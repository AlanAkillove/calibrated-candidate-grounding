"""One-off data fetch helper (diagnostic/setup tool, not part of the research pipeline).

Downloads:
  1. COCO 2014 annotations_trainval2014.zip  -> data/raw/annotations/
  2. RefCOCO+ annotation zip (mirror list)    -> data/raw/refcoco+/

Usage:
    python tools/download_data.py annos
    python tools/download_data.py refcoco_plus
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

COCO_ANNOS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

REFCOCO_PLUS_URLS = [
    # original UNC server has SSL/availability issues (refer issue #14);
    # wayback 'id_' variant serves the archived raw zip bytes
    "https://web.archive.org/web/20220413011656id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
]


def fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already present: {dest} ({dest.stat().st_size} bytes)")
        return True
    print(f"downloading {url} -> {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total and done - (done % (20 << 20)) != done - ((done - len(chunk)) % (20 << 20)):
                    print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
        print(f"done: {dest} ({dest.stat().st_size} bytes)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {url}: {type(exc).__name__}: {str(exc)[:200]}")
        dest.unlink(missing_ok=True)
        return False


def unzip_if_ok(zip_path: Path, out_dir: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                print(f"corrupt entry in {zip_path}: {bad}")
                return False
            zf.extractall(out_dir)
        print(f"extracted {zip_path} -> {out_dir}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"unzip FAILED {zip_path}: {exc}")
        return False


def main(which: str) -> int:
    if which == "annos":
        dest = RAW / "annotations_trainval2014.zip"
        if not fetch(COCO_ANNOS_URL, dest):
            return 1
        return 0 if unzip_if_ok(dest, RAW) else 1
    if which == "refcoco_plus":
        out_dir = RAW / "refcoco+"
        for url in REFCOCO_PLUS_URLS:
            dest = RAW / "refcoco+.zip"
            if fetch(url, dest) and unzip_if_ok(dest, out_dir):
                return 0
        return 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
