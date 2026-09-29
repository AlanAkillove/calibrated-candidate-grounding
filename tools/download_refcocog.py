"""One-off RefCOCOg fetch helper for the A10 external feasibility audit (G0).

Not part of the research pipeline: it only *fetches, verifies and unpacks* the
public annotation archive, so that every later stage reads a local, hash-pinned
copy of it.

Why this archive and this URL
-----------------------------
RefCOCOg is distributed by UNC/Licheng Yu's ``referit`` project, one zip per
dataset (``refcoco.zip`` / ``refcoco+.zip`` / ``refcocog.zip``).  ``refer.py``
loads the split actually used from ``<dataset>/refs(<splitBy>).p``, so
``splitBy='umd'`` - the Mao et al. image-level split this study needs - is read
from ``refs(umd).p`` inside this archive.  The official direct link is broken
(SSL failure, ``refer`` issue #14; the same failure we already hit for
``refcoco+.zip``), so the fetch goes through the Internet Archive copy of the
*same* official URL, with the ``id_`` modifier that serves the untouched
original bytes.  Images are **not** fetched here: RefCOCOg uses COCO 2014
images and the driver downloads only the images the audited subset needs.

Scope discipline (Amendment A10, drafted before any RefCOCOg result was seen)
-----------------------------------------------------------------------------
Only the annotation archive is fetched.  Nothing is extracted from the train /
val splits for modelling purposes - the feasibility audit reads their *image
id lists* (to prove disjointness) and the expression counts (dataset integrity),
never their labels as training signal.  0 new parameters are ever fitted.

Usage:
    python tools/download_refcocog.py            # fetch + verify + unpack
    python tools/download_refcocog.py --list     # only print the fetch plan
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
DEST_DIR = RAW / "refcocog"

#: official referit distribution URL, kept as the provenance key.
OFFICIAL_URL = "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"
#: Internet Archive copy of that URL (snapshot verified 200 with this length).
SNAPSHOT_TS = "20220413012904"
ARCHIVE_URL = f"https://web.archive.org/web/{SNAPSHOT_TS}id_/{OFFICIAL_URL.replace('https://', 'http://')}"
#: CAUTION: the Wayback CDX ``length`` field is the size of the *archived HTTP
#: response including its headers*, not the payload size (it overshoots the real
#: file by ~2.3 KB here).  It is therefore only a lower-bound sanity check; the
#: authoritative completeness test is the response's own Content-Length plus
#: :func:`zipfile.ZipFile.testzip` on the result.
CDX_RESPONSE_LENGTH = 56_715_268
MIN_BYTES = 50_000_000

#: what the archive must contain for a UMD-split audit to be possible.  Measured
#: member list of this archive: ``instances.json``, ``refs(google).p``,
#: ``refs(umd).p`` - there is no ``cats.txt`` here (unlike some mirrors), so the
#: COCO category names come from the official COCO annotation files instead.
REQUIRED_MEMBERS = ("refs(umd).p", "instances.json")

UA = {"User-Agent": "Mozilla/5.0 (ccg-a10-feasibility)"}


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
    """Download ``url`` to ``dest``.

    Completeness is decided by the response's own ``Content-Length`` (a stream
    that closes early is retried), never by a hard-coded expected size, and each
    attempt starts from an empty ``.part`` because the Internet Archive answers a
    ``Range`` request on some snapshots with ``404`` instead of ``206``.
    """
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
        "cdx_response_length_including_headers": CDX_RESPONSE_LENGTH,
    }


def verify_archive(zip_path: Path) -> List[str]:
    """Return the member names of a CRC-verified archive, or raise."""
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise OSError(f"{zip_path.name}: CRC check failed on member {bad}")
        return [n for n in zf.namelist() if not n.endswith("/")]


def unpack(zip_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """Extract every member of a CRC-verified archive and hash it."""
    members: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = verify_archive(zip_path)
        basenames = {Path(n).name for n in names}
        missing = [m for m in REQUIRED_MEMBERS if m not in basenames]
        if missing:
            raise ValueError(f"archive lacks required UMD members: {missing} (has {sorted(basenames)})")
        for name in names:
            data = zf.read(name)
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() and target.stat().st_size == len(data)):
                # identical size is treated as identical content (deflate output is
                # deterministic here), so a re-run never rewrites a verified file
                target.write_bytes(data)
            members.append(
                {
                    "member": name,
                    "path": str(target.relative_to(ROOT)),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return sorted(members, key=lambda m: m["member"])


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the fetch plan and exit")
    args = ap.parse_args(argv)

    plan = {
        "archive": {"url": ARCHIVE_URL, "official_url": OFFICIAL_URL, "cdx_response_length": CDX_RESPONSE_LENGTH},
        "dest_zip": str((DEST_DIR / "refcocog.zip").relative_to(ROOT)),
        "extract_to": str(DEST_DIR.relative_to(ROOT)),
        "required_members": list(REQUIRED_MEMBERS),
        "images": "not fetched here (COCO 2014 images; driver fetches only the audited subset)",
    }
    if args.list:
        print(json.dumps(plan, indent=2), flush=True)
        return 0

    card_path = DEST_DIR / "dataset_card.json"
    record: Dict[str, Any] = {"plan": plan}
    if card_path.exists():
        record = json.loads(card_path.read_text(encoding="utf-8"))
    zf = DEST_DIR / "refcocog.zip"
    record["downloaded"] = fetch(ARCHIVE_URL, zf, MIN_BYTES)
    record["members"] = unpack(zf, DEST_DIR)
    record["unpacked_total_bytes"] = sum(int(m["bytes"]) for m in record["members"])
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"REFCOCOG_FETCH_OK members={len(record['members'])} "
        f"unpacked={record['unpacked_total_bytes']} sha256={record['downloaded']['sha256'][:16]}",
        flush=True,
    )
    for m in record["members"]:
        print(f"  {m['member']:26s} {m['bytes']:>12,d} B  {m['sha256'][:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
