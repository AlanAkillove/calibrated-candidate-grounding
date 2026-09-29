"""One-off FineCops-Ref fetch helper for the Phase 1E feasibility audit (F0).

Not part of the research pipeline: it only *fetches and records* the public
annotation files, so that every later stage reads a local, hash-pinned copy.

Scope discipline (Amendment A9, drafted before any FineCops result was seen):

* Only **test** annotations are fetched.  The train/val files exist on the same
  figshare article and are deliberately *not* downloaded - Phase 1E is a frozen
  external evaluation and must not even have the train/val labels on disk
  (they may not enter training, tuning or calibration).
* ``neg_images.tgz`` (567 MB of inpainted negative images) is not fetched: the
  negative branch is deferred to a future abstention phase (instruction §2/§23).
* ``dataset_card.json`` records the license, source URLs, byte sizes and
  sha256 of every file actually downloaded.

Usage:
    python tools/download_finecops.py            # fetch + verify + write card
    python tools/download_finecops.py --list     # only print the fetch plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

#: figshare article 26048050 ("FineCops-Ref ...", Junzhuo Liu), license CC BY 4.0.
FIGSHARE_ARTICLE = "https://figshare.com/articles/dataset/FineCops-Ref_A_new_Dataset_and_Task_for_Fine-Grained_Compositional_Referring_Expression_Comprehension/26048050"
FIGSHARE_API = "https://api.figshare.com/v2/articles/26048050"

#: test-split annotations only (file name -> ndownloader file id, expected bytes)
TEST_FILES: List[Dict[str, Any]] = [
    {"name": "test_expression_pos.json", "file_id": 47091958, "size": 3_075_530},
    {"name": "test_expression_pos_coco_format.json", "file_id": 47091964, "size": 5_298_930},
    {"name": "test_expression_all.json", "file_id": 47136715, "size": 12_127_597},
    {"name": "test_expression_all_coco_format.json", "file_id": 47136718, "size": 21_099_434},
]

#: GQA scene graphs: the only public source of per-image object *names*,
#: which is what the same-category candidate construction has to be judged on.
GQA_SCENE_GRAPHS = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"

UA = {"User-Agent": "Mozilla/5.0 (ccg-phase1e-feasibility)"}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: int = 0) -> Dict[str, Any]:
    """Download ``url`` to ``dest`` (resumed from a ``.part`` file if present)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not expected or dest.stat().st_size == expected):
        print(f"already present: {dest.name} ({dest.stat().st_size} bytes)", flush=True)
    else:
        print(f"downloading {url} -> {dest}", flush=True)
        part = dest.with_suffix(dest.suffix + ".part")
        done = part.stat().st_size if part.exists() else 0
        headers = dict(UA)
        if done:
            headers["Range"] = f"bytes={done}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp, open(part, "ab") as fh:
            total = done + int(resp.headers.get("Content-Length") or 0)
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total and done // (200 << 20) != (done - len(chunk)) // (200 << 20):
                    print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
        if expected and done != expected:
            raise OSError(f"{dest.name}: got {done} bytes, expected {expected}")
        part.replace(dest)
        print(f"done: {dest.name} ({dest.stat().st_size} bytes)", flush=True)
    return {
        "file": dest.name,
        "path": str(dest.relative_to(ROOT)),
        "bytes": dest.stat().st_size,
        "sha256": sha256_of(dest),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the fetch plan and exit")
    args = ap.parse_args(argv)

    plan = [
        {
            "name": f["name"],
            "url": f"https://ndownloader.figshare.com/files/{f['file_id']}",
            "expected_bytes": f["size"],
            "dest": str((RAW / "finecops" / f["name"]).relative_to(ROOT)),
        }
        for f in TEST_FILES
    ] + [
        {
            "name": "gqa_sceneGraphs.zip",
            "url": GQA_SCENE_GRAPHS,
            "expected_bytes": 0,
            "dest": str((RAW / "gqa" / "sceneGraphs.zip").relative_to(ROOT)),
        }
    ]
    if args.list:
        print(json.dumps(plan, indent=2))
        return 0

    records: List[Dict[str, Any]] = []
    for f in TEST_FILES:
        records.append(
            fetch(
                f"https://ndownloader.figshare.com/files/{f['file_id']}",
                RAW / "finecops" / f["name"],
                f["size"],
            )
        )
    zip_rec = fetch(GQA_SCENE_GRAPHS, RAW / "gqa" / "sceneGraphs.zip")
    out_dir = RAW / "gqa"
    with zipfile.ZipFile(RAW / "gqa" / "sceneGraphs.zip") as zf:
        zf.extractall(out_dir)
        zip_rec["entries"] = sorted(n for n in zf.namelist() if n.endswith(".json"))
    records.append(zip_rec)

    card = {
        "source_article": FIGSHARE_ARTICLE,
        "source_api": FIGSHARE_API,
        "license": "CC BY 4.0 (figshare article 26048050, Junzhuo Liu)",
        "image_source": "GQA / Visual Genome images - https://cs.stanford.edu/people/dorarad/gqa/download.html",
        "fetched_files": records,
        "deliberately_not_fetched": {
            "train_val_annotations": "Phase 1E is frozen external evaluation; train/val labels must not enter training, tuning or calibration (instruction section 3)",
            "neg_images.tgz": "negative images deferred to the future abstention phase (instruction sections 2 and 23)",
        },
        "bytes_total": sum(r["bytes"] for r in records),
    }
    card_path = RAW / "finecops" / "dataset_card.json"
    tmp = card_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(card, indent=2), encoding="utf-8")
    tmp.replace(card_path)
    print(json.dumps(card, indent=2)[:2000])
    print(f"OK -> {card_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
