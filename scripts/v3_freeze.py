"""Seal or verify V3's actual acceptance reports and input identities."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.protocol import seal_freeze, verify_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--spec", type=Path, help="Seal from this prevalidated JSON specification")
    args = parser.parse_args()
    if args.spec:
        payload = json.loads(args.spec.read_text(encoding="utf-8"))
        print(seal_freeze(args.freeze, payload, root=ROOT))
    else:
        payload = verify_freeze(args.freeze, root=ROOT)
        print(f"PASS: {len(payload['inputs'])} bound inputs")


if __name__ == "__main__":
    main()
