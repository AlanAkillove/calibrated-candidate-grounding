"""Prepare/verify the read-only baseline for Research Repair v1.

This script never changes legacy results or caches and runs no experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "research_repair_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_paths() -> list[Path]:
    paths = {
        p for p in (ROOT / "results").rglob("*")
        if p.is_file() and OUT not in p.parents
    }
    for name in (
        "cache/features", "cache/semantic_phase1", "cache/manifests",
        "cache/manifests_detr", "cache/refcoco_lang", "cache/refcocog_external",
        "cache/v2_backbones", "cache/v2_semantic",
        "data/raw/refcoco+/refcoco+", "data/raw/refcoco/refcoco",
        "data/raw/refcocog/refcocog",
    ):
        directory = ROOT / name
        if directory.exists():
            paths.update(p for p in directory.rglob("*") if p.is_file())
    for name in (
        "cache/proposals.h5", "cache/proposals_detr_r50.h5", "cache/proposals_gdino.h5",
        "cache/image_sizes.npz",
        "data/raw/mscoco/annotations/instances_train2014.json",
    ):
        path = ROOT / name
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare() -> None:
    manifest_path = OUT / "input_manifest.json"
    if manifest_path.exists():
        raise SystemExit("Baseline already exists. Use --verify; refusing to replace it.")
    required = (
        "results/phase05_score_sufficiency/split_manifest.json",
        "results/phase0b_independent/seed_1/raw_scores/K50.npz",
        "results/phase1_semantic_sufficiency/semantic_features/e1_stats_b3_seed1.npz",
        "results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz",
        "cache/proposals.h5", "cache/features/region_features.h5",
        "data/raw/refcoco+/refcoco+/instances.json",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")
    paths = baseline_paths()
    records = []
    total_bytes = sum(path.stat().st_size for path in paths)
    print(f"Hashing {len(paths)} immutable inputs ({total_bytes / 1024**3:.2f} GiB)", flush=True)
    for index, path in enumerate(paths, start=1):
        records.append({"path": path.relative_to(ROOT).as_posix(),
                        "size_bytes": path.stat().st_size, "sha256": sha256(path)})
        if index % 100 == 0:
            print(f"Hashed {index}/{len(paths)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in ("statistics", "information", "candidates", "mechanism", "registry", "logs"):
        (OUT / directory).mkdir(exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    atomic_json(manifest_path, {
        "schema": "research-repair-input-manifest-v1", "base_commit": head,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "POST_RESULT_REPAIR_BASELINE",
        "protocol": "reviews/repair_protocol.md",
        "protocol_sha256": sha256(ROOT / "reviews/repair_protocol.md"),
        "n_files": len(records), "total_bytes": total_bytes, "inputs": records,
    })
    atomic_json(OUT / "STATUS.json", {
        "status": "PREPARED_DISPATCH_BLOCKED", "completed": False,
        "reason": "Specified gpt-6-luna/max subagent creation rejected: agent thread limit reached",
        "agent_model": "gpt-6-luna", "reasoning_effort": "max",
        "stages": {name: "NOT_STARTED" for name in
                   ("statistics", "information", "candidates", "mechanism", "integration", "acceptance")},
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Prepared {manifest_path.relative_to(ROOT)}; experiments NOT STARTED", flush=True)


def verify() -> None:
    manifest = json.loads((OUT / "input_manifest.json").read_text(encoding="utf-8"))
    changed = []
    for record in manifest["inputs"]:
        path = ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            changed.append(record["path"])
    if changed:
        raise SystemExit("Immutable input changed/missing: " + json.dumps(changed, ensure_ascii=False))
    print(f"PASS: {len(manifest['inputs'])} baseline inputs unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify, without replacing, the baseline")
    args = parser.parse_args()
    verify() if args.verify else prepare()
