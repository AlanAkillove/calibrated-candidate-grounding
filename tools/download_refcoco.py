"""One-off RefCOCO (original, "strict") annotation fetch helper for D2 (V2-D).

Not part of the research pipeline: it only *fetches, verifies and unpacks* the
public annotation archive so every later D2 stage reads a local, hash-pinned
copy of it.  It mirrors :mod:`tools.download_refcocog` exactly (same referit
distribution, same Internet-Archive ``id_`` serving of the untouched bytes).

Why this archive and this URL
-----------------------------
RefCOCO is distributed by UNC / Licheng Yu's ``referit`` project, one zip per
dataset (``refcoco.zip`` / ``refcoco+.zip`` / ``refcocog.zip``).  The official
direct link is broken for us too (SSL / 502 - the same failure already hit for
``refcoco+.zip`` and ``refcocog.zip``), so the fetch goes through the Internet
Archive copy of the *same* official URL (crawl 2022-04-13, the identical crawl
that serves refcoco+ and refcocog here), with the ``id_`` modifier that returns
the raw original bytes rather than a rewritten page.

The archive's refs pickle is ``refs.p`` for RefCOCO (vs ``refs(unc).p`` for
RefCOCO+); ``refer.py`` special-cases the RefCOCO file name.  Because that
spelling is the one thing that differs between the sibling datasets, the member
list is *discovered* rather than assumed and recorded in ``dataset_card.json``.

Scope discipline (D2 protocol, Phase 1)
---------------------------------------
Only the annotation archive is fetched.  Nothing is extracted from RefCOCO's
train / val splits as modelling signal - the later strict-split audit reads
their *image id lists* (to prove image-disjointness against every RefCOCO+
exposure) and expression counts (dataset integrity).  0 new parameters are ever
fitted.  Images are NOT fetched here (D2 Phase 2 fetches only the images the
frozen COCO RPN needs for the audited disjoint subset).

Usage:
    python tools/download_refcoco.py            # fetch + verify + unpack
    python tools/download_refcoco.py --list      # only print the fetch plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEST_DIR = RAW / "refcoco"

#: official referit distribution URL, kept as the provenance key.
OFFICIAL_URL = "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
#: Internet Archive copy of that URL (CDX statuscode=200 verified this session).
SNAPSHOT_TS = "20220413011718"
ARCHIVE_URL = f"https://web.archive.org/web/{SNAPSHOT_TS}id_/{OFFICIAL_URL.replace('https://', 'http://')}"
#: The Wayback CDX ``length`` field counts archived HTTP headers, so it is only a
#: lower-bound sanity floor; completeness is decided by Content-Length + ZipFile
#: .testzip() in :func:`fetch` / :func:`verify_archive`.
MIN_BYTES = 50_000_000

#: COCO instances are always required (the refs pickle carries no boxes).  The
#: refs-pickle member name is discovered among these candidates.
REQUIRED_JSON = ("instances.json",)
REFS_PICKLE_CANDIDATES = ("refs.p", "refs(unc).p", "refs(refcoco).p")

UA = {"User-Agent": "Mozilla/5.0 (ccg-d2-refcoco)"}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, min_bytes: int = 0, attempts: int = 5) -> Dict[str, Any]:
    """Download ``url`` to ``dest``; completeness = response Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= max(min_bytes, 1):
        print(f"already present: {dest.name} ({dest.stat().st_size} bytes)", flush=True)
    else:
        part = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(1, attempts + 1):
            part.unlink(missing_ok=True)
            print(f"downloading (attempt {attempt}/{attempts}) {url} -> {part.name}", flush=True)
            try:
                req = urllib.request.Request(url, headers=dict(UA))
                with urllib.request.urlopen(req, timeout=180) as resp:
                    declared = int(resp.headers.get("Content-Length") or 0)
                    with open(part, "wb") as fh:
                        done = 0
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            fh.write(chunk)
                            done += len(chunk)
                            if done // (10 << 20) != (done - len(chunk)) // (10 << 20):
                                print(f"  {done / 1e6:.0f}/{(declared or done) / 1e6:.0f} MB", flush=True)
                if declared and done != declared:
                    print(f"  stream closed at {done} of {declared} declared bytes", flush=True)
                    continue
            except (urllib.error.URLError, OSError) as exc:
                print(f"  attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            if min_bytes and done < min_bytes:
                print(f"  {done} bytes is below the {min_bytes} byte sanity floor", flush=True)
                continue
            break
        else:
            have = part.stat().st_size if part.exists() else 0
            raise OSError(f"{dest.name}: no complete transfer in {attempts} attempts (last {have} bytes)")
        part.replace(dest)
        print(f"done: {dest.name} ({dest.stat().st_size} bytes)", flush=True)
    return {
        "file": dest.name,
        "path": str(dest.relative_to(ROOT)),
        "bytes": dest.stat().st_size,
        "sha256": sha256_of(dest),
        "source_url": url,
        "official_url": OFFICIAL_URL,
        "snapshot_timestamp": SNAPSHOT_TS,
    }


def verify_archive(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise OSError(f"{zip_path.name}: CRC check failed on member {bad}")
        return [n for n in zf.namelist() if not n.endswith("/")]


def discover_refs_pickle(basenames: List[str]) -> str:
    for name in REFS_PICKLE_CANDIDATES:
        if name in basenames:
            return name
    others = [n for n in basenames if n.startswith("refs") and n.endswith(".p")]
    if len(others) == 1:
        return others[0]
    raise ValueError(
        f"could not identify the RefCOCO refs pickle among members {sorted(basenames)} "
        f"(looked for {REFS_PICKLE_CANDIDATES})"
    )


def unpack(zip_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
    members: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = verify_archive(zip_path)
        basenames = {Path(n).name for n in names}
        missing = [m for m in REQUIRED_JSON if m not in basenames]
        if missing:
            raise ValueError(f"archive lacks required members {missing} (has {sorted(basenames)})")
        refs_pickle = discover_refs_pickle(sorted(basenames))
        print(f"refs pickle discovered: {refs_pickle}", flush=True)
        # members may be nested under a ``refcoco/`` folder; identify the pickle
        # by basename so the recorded flag survives the folder prefix
        for name in names:
            data = zf.read(name)
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() and target.stat().st_size == len(data)):
                target.write_bytes(data)
            members.append(
                {
                    "member": name,
                    "path": str(target.relative_to(ROOT)),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "is_refs_pickle": Path(name).name == refs_pickle,
                }
            )
    return sorted(members, key=lambda m: m["member"])


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the fetch plan and exit")
    args = ap.parse_args(argv)

    plan = {
        "archive": {"url": ARCHIVE_URL, "official_url": OFFICIAL_URL, "snapshot": SNAPSHOT_TS},
        "dest_zip": str((DEST_DIR / "refcoco.zip").relative_to(ROOT)),
        "extract_to": str(DEST_DIR.relative_to(ROOT)),
        "required_json": list(REQUIRED_JSON),
        "refs_pickle_candidates": list(REFS_PICKLE_CANDIDATES),
        "images": "not fetched here (D2 Phase 2 fetches only the disjoint-subset COCO images)",
    }
    if args.list:
        print(json.dumps(plan, indent=2), flush=True)
        return 0

    card_path = DEST_DIR / "dataset_card.json"
    record: Dict[str, Any] = {"plan": plan}
    if card_path.exists():
        record = json.loads(card_path.read_text(encoding="utf-8"))
    zf = DEST_DIR / "refcoco.zip"
    record["downloaded"] = fetch(ARCHIVE_URL, zf, MIN_BYTES)
    record["members"] = unpack(zf, DEST_DIR)
    record["refs_pickle"] = next(m["member"] for m in record["members"] if m["is_refs_pickle"])
    record["unpacked_total_bytes"] = sum(int(m["bytes"]) for m in record["members"])
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"REFCOCO_FETCH_OK members={len(record['members'])} refs={record['refs_pickle']} "
        f"unpacked={record['unpacked_total_bytes']} sha256={record['downloaded']['sha256'][:16]}",
        flush=True,
    )
    for m in record["members"]:
        print(f"  {m['member']:26s} {m['bytes']:>12,d} B  {m['sha256'][:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
