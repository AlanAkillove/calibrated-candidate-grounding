#!/usr/bin/env python
"""Fit and evaluate the frozen B16 S/Q/V reliability-feature replication."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.v3.replication import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
