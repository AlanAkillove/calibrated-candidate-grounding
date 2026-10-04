"""Restore byte-identical large repair JSON/CSV evidence from Git text archives.

Run after cloning, before reading the full statistics summary. This restores only
the listed text artifacts; model weights, raw bootstrap arrays and data caches
remain separate local prerequisites. Existing different files are never replaced.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/research_repair_v1/text_artifacts/manifest.json"


def checked_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def restore(*, verify_only: bool = False) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    restored = existing = 0
    for row in manifest["artifacts"]:
        source, destination = checked_path(row["archive"]), checked_path(row["original"])
        if digest(source) != row["archive_sha256"]:
            raise ValueError(f"Archive SHA mismatch: {source}")
        sha = hashlib.sha256()
        size = 0
        with gzip.open(source, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
                size += len(block)
        if size != row["original_bytes"] or sha.hexdigest() != row["original_sha256"]:
            raise ValueError(f"Uncompressed identity mismatch: {source}")
        if destination.exists():
            if digest(destination) != row["original_sha256"]:
                raise ValueError(f"Existing original differs; refusing replacement: {destination}")
            existing += 1
            continue
        if verify_only:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".restore.tmp", delete=False) as output:
                temporary = Path(output.name)
                with gzip.open(source, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(block)
            if digest(temporary) != row["original_sha256"]:
                raise ValueError(f"Restored identity mismatch: {destination}")
            # A concurrent writer must not be silently overwritten.
            with destination.open("xb") as output, temporary.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            restored += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"status": "PASS", "archives_verified": len(manifest["artifacts"]),
            "restored": restored, "existing_identical": existing, "verify_only": verify_only}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    print(json.dumps(restore(verify_only=parser.parse_args().verify_only), indent=2))
