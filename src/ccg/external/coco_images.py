"""COCO 2014 single-image downloads for the A10 RefCOCOg external audit.

Why this module exists
----------------------
``data/raw/mscoco/train2014/`` holds exactly the **19,992** COCO images RefCOCO+
uses, not the full 82,783-image train2014 split.  The image-disjoint RefCOCOg
external subset is disjoint *by construction*, so almost all of its images are
missing locally and have to be fetched.  They come from the same public host the
RefCOCO+ images came from (``images.cocodataset.org``, see
:func:`ccg.data.coco.coco_image_url` and ``tools/download_all_images.py``), so
the pixel source of the two datasets stays identical - which is the whole point
of a *same-image-domain* external test (instruction section 24).

Deliberate separation of storage roots
--------------------------------------
Images are written to their own tree (``data/raw/refcocog/images/train2014/``)
rather than into ``data/raw/mscoco/``.  The frozen RefCOCO+ image root is the
input of every committed RefCOCO+ artifact; an external round must not be able
to change what is in it, and "we did not touch the development image tree" is a
claim that should be checkable by a path, not by a promise.

Failure policy
--------------
Per-image: verify the JPEG SOI marker, write via ``.part`` + atomic replace,
retry with exponential backoff on 429/5xx, and *report* failures rather than
silently shrinking the audit set - the caller decides whether a hole is
acceptable and says so in its own artifact.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..data.coco import coco_image_url

__all__ = [
    "DEFAULT_WORKERS",
    "RETRYABLE_STATUS",
    "ImageFetchReport",
    "local_image_path",
    "decode_dims",
    "read_rgb",
    "fetch_coco_images",
]

#: COCO's static image host tolerates moderate parallelism; 16 is what the
#: original 19,992-image download used without rate-limiting.
DEFAULT_WORKERS = 16
RETRYABLE_STATUS: Tuple[int, ...] = (429, 500, 502, 503, 504)
_JPEG_SOI = b"\xff\xd8"
_USER_AGENT = "Mozilla/5.0 (ccg-phase1e-refcocog)"
_MIN_BYTES = 1024


@dataclass
class ImageFetchReport:
    """What a fetch run did - counts and per-file reasons, never prose."""

    requested: int = 0
    fetched: int = 0
    skipped_existing: int = 0
    failed: List[Tuple[str, str]] = field(default_factory=list)
    bytes_written: int = 0

    @property
    def present(self) -> int:
        return self.fetched + self.skipped_existing

    def to_dict(self) -> Dict[str, Any]:
        reasons: Dict[str, int] = {}
        for _name, reason in self.failed:
            reasons[reason.split(":", 1)[0]] = reasons.get(reason.split(":", 1)[0], 0) + 1
        return {
            "requested": self.requested,
            "fetched": self.fetched,
            "skipped_existing": self.skipped_existing,
            "present": self.present,
            "n_failed": len(self.failed),
            "failure_reasons": reasons,
            "failed_sample": [name for name, _r in self.failed[:20]],
            "bytes_written": int(self.bytes_written),
        }


def local_image_path(root: str | Path, file_name: str) -> Path:
    """``root/train2014/COCO_train2014_000000380440.jpg`` (COCO 2014 layout)."""
    name = Path(str(file_name)).name
    folder = name.split("_")[1] if name.startswith("COCO_") and "_" in name else ""
    if not folder:
        raise ValueError(f"{file_name!r} is not a COCO 2014 file name")
    return Path(root) / folder / name


def decode_dims(path: str | Path) -> Tuple[int, int]:
    """``(width, height)`` of the JPEG as stored on disk (source of truth)."""
    from PIL import Image

    with Image.open(path) as im:
        return int(im.width), int(im.height)


def read_rgb(path: str | Path) -> Any:
    """Decoded ``HxWx3 uint8`` array, the input :func:`ccg.data.rpn` expects."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _fetch_one(
    file_name: str, root: Path, *, timeout: float, retries: int
) -> Tuple[str, bool, str, int]:
    dest = local_image_path(root, file_name)
    if dest.exists() and dest.stat().st_size > _MIN_BYTES:
        return file_name, True, "exists", 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = coco_image_url(file_name)
    last = "unreachable"
    for attempt in range(int(retries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed host
                data = resp.read()
            if data[:2] != _JPEG_SOI or len(data) <= _MIN_BYTES:
                return file_name, False, "not-jpeg", 0
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return file_name, True, "ok", len(data)
        except urllib.error.HTTPError as exc:
            last = f"http-{exc.code}"
            if exc.code not in RETRYABLE_STATUS:
                return file_name, False, last, 0
        except Exception as exc:  # noqa: BLE001 - one socket must not kill the run
            last = f"{type(exc).__name__}"
        if attempt < int(retries) - 1:
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    return file_name, False, last, 0


def fetch_coco_images(
    file_names: Iterable[str],
    root: str | Path,
    *,
    workers: int = DEFAULT_WORKERS,
    timeout: float = 30.0,
    retries: int = 4,
    log: Optional[Any] = None,
) -> ImageFetchReport:
    """Ensure every COCO ``file_name`` exists under ``root``; report the holes."""
    names = sorted({str(Path(str(n)).name) for n in file_names})
    root = Path(root)
    report = ImageFetchReport(requested=len(names))
    if not names:
        return report
    done = 0
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [
            pool.submit(_fetch_one, n, root, timeout=timeout, retries=retries) for n in names
        ]
        for fut in as_completed(futures):
            name, ok, reason, nbytes = fut.result()
            done += 1
            if ok:
                if reason == "exists":
                    report.skipped_existing += 1
                else:
                    report.fetched += 1
                    report.bytes_written += int(nbytes)
            else:
                report.failed.append((name, reason))
            if log is not None and (done % 100 == 0 or done == len(names)):
                log(
                    f"images {done}/{len(names)} fetched={report.fetched} "
                    f"existing={report.skipped_existing} failed={len(report.failed)}"
                )
    report.failed.sort()  # deterministic artifact regardless of thread order
    return report
