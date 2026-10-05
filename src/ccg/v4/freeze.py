"""Read-only checks of sealed V4 stage files before running new analyses."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v4_targeted_strengthening'

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_freeze(stage: str, *, root: Path = ROOT) -> dict:
    base = Path(root).resolve()
    path = base / 'results/v4_targeted_strengthening' / f'{stage}_freeze.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('stage') != stage or payload.get('schema') != 'v4-stage-freeze-v1':
        raise ValueError('freeze schema/stage mismatch')
    if not payload.get('files'):
        raise ValueError('empty freeze')
    seen = set()
    for row in payload['files']:
        name = row['path']
        source = (base / name).resolve()
        if not source.is_relative_to(base) or name in seen:
            raise ValueError('unsafe/duplicate frozen path')
        seen.add(name)
        if not source.is_file() or source.stat().st_size != row['size_bytes'] or sha256_file(source) != row['sha256']:
            raise AssertionError(f'sealed input changed: {name}')
    return dict(stage=stage, files=len(seen), freeze_sha256=sha256_file(path))
