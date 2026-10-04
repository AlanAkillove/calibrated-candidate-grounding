"""Preserve fitted source model bytes for replay; large public encoders stay external."""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.inference import frozen_model_paths
from ccg.v3.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("package", "verify", "restore"))
    args = parser.parse_args()
    folder = ROOT / "results/v3_final_validation/model_replay"
    archive, manifest = folder / "fitted_source_models.zip", folder / "manifest.json"
    if args.mode == "package":
        folder.mkdir(parents=True, exist_ok=True)
        inputs = []
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in frozen_model_paths():
                relative = path.resolve().relative_to(ROOT).as_posix()
                included = path.stat().st_size < 10_000_000
                record = {"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size,
                          "included": included}
                inputs.append(record)
                if included:
                    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    bundle.writestr(info, path.read_bytes())
        payload = {"schema": "v3-model-replay-v1", "archive": archive.relative_to(ROOT).as_posix(),
                   "archive_sha256": sha256(archive), "inputs": inputs,
                   "scope": "Exact fitted scorer/reliability bytes and metadata; public encoder/RPN weights omitted, identified by SHA. No training or inference."}
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != "v3-model-replay-v1" or sha256(archive) != payload["archive_sha256"]:
        raise ValueError("Model package identity changed")
    import hashlib
    with zipfile.ZipFile(archive) as bundle:
        expected = {item["path"] for item in payload["inputs"] if item["included"]}
        if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(expected):
            raise ValueError("Model archive membership differs from manifest")
        for record in payload["inputs"]:
            if not record["included"]:
                continue
            destination = (ROOT / record["path"]).resolve()
            if not destination.is_relative_to(ROOT):
                raise ValueError("Unsafe model destination")
            data = bundle.read(record["path"])
            if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Archived source model differs")
            if args.mode == "restore":
                if destination.exists():
                    if sha256(destination) != record["sha256"]:
                        raise ValueError(f"Refusing to replace existing changed model: {record['path']}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as handle:
                        handle.write(data)
    print(json.dumps({"status": "PASS", "mode": args.mode, "fitted_files": len(expected),
                      "archive_size_bytes": archive.stat().st_size, "public_weights_omitted": len(payload["inputs"]) - len(expected)}))


if __name__ == "__main__":
    main()
