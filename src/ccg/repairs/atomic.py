"""Atomic output replacement with bounded retries for transient Windows sharing locks."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Sequence

REPLACE_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)
TRANSIENT_WINDOWS_LOCK_ERRORS = (32, 33)


class AtomicReplaceError(RuntimeError):
    """Atomic replacement failed; the complete temporary file is retained for recovery."""

    def __init__(self, source: Path, target: Path, cause: OSError, attempts: int) -> None:
        self.source = Path(source)
        self.target = Path(target)
        self.cause = cause
        self.attempts = int(attempts)
        self.errno = getattr(cause, "errno", None)
        self.winerror = getattr(cause, "winerror", None)
        super().__init__(
            f"atomic replace failed after {self.attempts} attempt(s); "
            f"temporary artifact retained at {self.source}; target={self.target}; "
            f"winerror={self.winerror}; cause={cause}"
        )


def replace_with_retry(
    source: Path,
    target: Path,
    *,
    retry_delays: Sequence[float] = REPLACE_RETRY_DELAYS,
    replace: Callable[[Path, Path], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Replace a target atomically, retrying only Windows sharing/lock violations.

    The source is a complete, flushed temporary artifact. Exhaustion preserves it
    and reports both paths through ``AtomicReplaceError`` so a caller can record
    recovery details without rerunning the computation.
    """
    do_replace = replace or os.replace
    do_sleep = sleep or time.sleep
    delays = tuple(float(delay) for delay in retry_delays)
    for index in range(len(delays) + 1):
        try:
            do_replace(source, target)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            transient_lock = winerror in TRANSIENT_WINDOWS_LOCK_ERRORS
            if not transient_lock or index >= len(delays):
                raise AtomicReplaceError(source, target, exc, attempts=index + 1) from exc
            do_sleep(delays[index])


__all__ = ["AtomicReplaceError", "REPLACE_RETRY_DELAYS", "replace_with_retry"]
