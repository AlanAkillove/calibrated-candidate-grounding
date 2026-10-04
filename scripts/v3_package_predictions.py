"""Package/restore saved predictions losslessly without any model evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_final_validation"
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.protocol import sha256


def inside(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(OUT):
        raise ValueError("Only V3 evidence paths can be packaged/restored")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("package", "restore"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "package":
        source = inside(args.path)
        if source.name != "predictions.jsonl":
            raise ValueError("Package expects a saved predictions.jsonl")
        packed = source.with_suffix(".jsonl.gz")
        temporary = packed.with_suffix(".gz.tmp")
        with source.open("rb") as original, temporary.open("wb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
                shutil.copyfileobj(original, compressed)
        temporary.replace(packed)
        # Verify the uncompressed bytes before publishing the manifest.
        import hashlib
        digest = hashlib.sha256()
        with gzip.open(packed, "rb") as restored:
            for block in iter(lambda: restored.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != sha256(source):
            raise ValueError("Compressed predictions are not lossless")
        record = {"schema": "v3-lossless-predictions-v1", "original": source.relative_to(ROOT).as_posix(),
                  "original_sha256": digest.hexdigest(), "original_size_bytes": source.stat().st_size,
                  "compressed": packed.relative_to(ROOT).as_posix(), "compressed_sha256": sha256(packed),
                  "compressed_size_bytes": packed.stat().st_size,
                  "restore": f"python scripts/v3_package_predictions.py restore {source.parent.relative_to(ROOT).as_posix()}/predictions_package.json"}
        (source.parent / "predictions_package.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record))
    else:
        manifest = inside(args.path)
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("schema") != "v3-lossless-predictions-v1":
            raise ValueError("Unsupported predictions package")
        source, packed = inside(ROOT / record["original"]), inside(ROOT / record["compressed"])
        if sha256(packed) != record["compressed_sha256"]:
            raise ValueError("Compressed evidence has changed")
        if source.exists():
            if sha256(source) != record["original_sha256"]:
                raise ValueError("Existing original differs; refusing overwrite")
            print("Original already present and verified")
            return
        temporary = source.with_suffix(".restore.tmp")
        with gzip.open(packed, "rb") as compressed, temporary.open("xb") as destination:
            shutil.copyfileobj(compressed, destination)
        if sha256(temporary) != record["original_sha256"] or temporary.stat().st_size != record["original_size_bytes"]:
            raise ValueError("Restored bytes failed verification; original not replaced")
        temporary.replace(source)
        print("Saved predictions restored and verified; no model ran")


if __name__ == "__main__":
    main()
