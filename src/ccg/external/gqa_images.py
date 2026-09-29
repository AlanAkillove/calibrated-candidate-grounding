"""On-demand retrieval of GQA images out of the official 20 GB zip (Phase 1E F2).

FineCops-Ref ships no images: its positive test rows reference GQA image ids,
and GQA publishes *all* images in one 21.8 GB archive
(``https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip``, 148,855
members).  The feasibility audit needs a few thousand of them, so downloading
the archive would be a 20 GB round trip for a 300 MB payload.

The Stanford download host answers ``Range`` requests, and a zip stores its
index in the *tail* of the file, so Python's :mod:`zipfile` can be pointed at a
seekable HTTP view of the archive and asked for individual members.  That is
exactly what :func:`open_remote_zip` does.

:func:`fetch_images` does *not* read member data through :mod:`zipfile`, though.
Profiling the archive on 2026-09-29 showed that path costs ~3.5 MB of transferred
bytes per ~150 KB JPEG (the buffered reader refills its 1 MB block for every
member) and about 20 s per image from this host, which throttles aggressively.
Instead the central directory is read **once** to get each member's
``header_offset``, and every image is then fetched with a single
``Range`` request that covers its local header plus compressed data
(:class:`RangeConnection` + :func:`fetch_member`), verified against the stored
size and CRC.  Same bytes out, ~20x fewer bytes in, on one kept-alive
connection per worker.

Failure policy: a member that cannot be fetched is reported, never faked.  The
caller decides whether the miss rate is acceptable (the audit refuses to
extrapolate recall over silently missing images).
"""

from __future__ import annotations

import http.client
import io
import ssl
import struct
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

__all__ = [
    "GQA_IMAGES_ZIP_URL",
    "IMAGE_ENTRY_TEMPLATE",
    "HttpRangeReader",
    "open_remote_zip",
    "MemberLocation",
    "member_locations",
    "RangeConnection",
    "decode_member_span",
    "fetch_member",
    "ImageFetchReport",
    "member_name",
    "local_image_path",
    "fetch_images",
    "decode_dims",
    "local_dims",
]

#: Official GQA image archive (see https://cs.stanford.edu/people/dorarad/gqa/download.html).
GQA_IMAGES_ZIP_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
#: Entry layout inside the archive.
IMAGE_ENTRY_TEMPLATE = "images/{gqa_image_id}.jpg"

_USER_AGENT = "Mozilla/5.0 (ccg-phase1e-feasibility)"
#: One HTTP block per range read.  The archive's central directory is ~10 MB,
#: so a small buffer would turn every worker start-up into hundreds of requests.
_CHUNK = 1 << 20
#: The download host answers ``503`` when one client opens too many parallel
#: range streams (each stream re-reads the ~10 MB central directory), so the
#: default is deliberately small and every retry backs off exponentially.
DEFAULT_WORKERS = 3
_BACKOFF_BASE = 3.0
_BACKOFF_CAP = 60.0
#: Requests that mean "slow down" rather than "give up".
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# --- zip member layout -----------------------------------------------------
#: ``PK\x03\x04`` - signature of a zip local file header.
LOCAL_HEADER_SIGNATURE = 0x04034B50
#: sig, version, flags, method, mod time, mod date, crc32, csize, usize, name, extra
_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
#: The local extra field may be longer than the central directory's, so the first
#: range request guesses a bit of slack and re-requests the exact span if needed.
_LOCAL_EXTRA_PAD = 2048


class HttpRangeReader(io.RawIOBase):
    """Seekable, read-only view of a remote file served with HTTP ``Range``.

    Only the last fetched block is cached, which is enough for ``zipfile``'s
    access pattern (one tail read for the central directory, then one forward
    read per member).
    """

    def __init__(self, url: str, timeout: float = 120.0, ua: str = _USER_AGENT) -> None:
        super().__init__()
        self.url = url
        self.timeout = float(timeout)
        self.headers = {"User-Agent": ua}
        self._pos = 0
        self._cache = b""
        self._cache_start = 0
        self.length = 0
        self.range_supported = False
        self._probe()

    # -- size discovery ------------------------------------------------------
    def _probe(self) -> None:
        req = urllib.request.Request(self.url, headers={**self.headers, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read()
            self.range_supported = resp.status == 206 and len(body) == 1
            total = resp.headers.get("Content-Range")
            if total and "/" in total:
                self.length = int(total.rsplit("/", 1)[1])
            else:
                self.length = int(resp.headers.get("Content-Length") or 0)

    # -- io contract ---------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.length}.get(whence)
        if base is None:
            raise ValueError(f"unsupported whence {whence}")
        self._pos = max(0, base + int(offset))
        return self._pos

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            size = self.length - self._pos
        size = min(int(size), self.length - self._pos)
        if size <= 0:
            return b""
        end = self._pos + size - 1
        if (
            self._cache
            and self._cache_start <= self._pos
            and end < self._cache_start + len(self._cache)
        ):
            out = self._cache[self._pos - self._cache_start : end - self._cache_start + 1]
            self._pos += len(out)
            return out
        req = urllib.request.Request(
            self.url, headers={**self.headers, "Range": f"bytes={self._pos}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            buf = resp.read()
        if not buf:
            raise OSError(f"{self.url}: empty range response for bytes {self._pos}-{end}")
        self._cache = buf
        self._cache_start = self._pos
        self._pos += len(buf)
        return buf

    def readinto(self, view: Any) -> int:  # type: ignore[override]
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)


def open_remote_zip(url: str = GQA_IMAGES_ZIP_URL, *, timeout: float = 120.0) -> zipfile.ZipFile:
    """Open the remote archive by range requests (no local copy is written).

    Raises
    ------
    RuntimeError
        if the host ignores ``Range`` - a partial read would then return the
        whole file, and :mod:`zipfile` would silently mis-seek.
    """
    reader = HttpRangeReader(url, timeout=timeout)
    if not reader.range_supported:
        raise RuntimeError(
            f"{url} does not honour HTTP range requests; fetch the archive instead"
        )
    return zipfile.ZipFile(io.BufferedReader(reader, buffer_size=_CHUNK))


def member_name(gqa_image_id: str | int) -> str:
    return IMAGE_ENTRY_TEMPLATE.format(gqa_image_id=str(gqa_image_id))


class _ShortSpan(ValueError):
    """The fetched span was too short to hold the member; carries the true need."""

    def __init__(self, need: int) -> None:
        super().__init__(f"need {need} bytes from the local header onwards")
        self.need = int(need)


@dataclass(frozen=True)
class MemberLocation:
    """Where one member lives inside the archive (from the central directory)."""

    name: str
    header_offset: int
    compress_size: int
    file_size: int
    compress_type: int
    crc: int
    name_len: int

    def span(self, pad: int = _LOCAL_EXTRA_PAD) -> int:
        """Bytes to fetch: local header + name + guessed extra + compressed data."""
        return _LOCAL_HEADER.size + self.name_len + self.compress_size + int(pad)


def member_locations(
    url: str = GQA_IMAGES_ZIP_URL,
    *,
    timeout: float = 120.0,
    only: Optional[Iterable[str]] = None,
) -> Dict[str, MemberLocation]:
    """Read the archive's central directory once and index its members.

    ``only`` restricts the returned table to the given entry names; the transfer
    cost is the same either way (the directory is read as a whole), but the
    resident table stays small.
    """
    wanted = None if only is None else {str(n) for n in only}
    zf = open_remote_zip(url, timeout=timeout)
    try:
        table: Dict[str, MemberLocation] = {}
        for info in zf.infolist():
            if wanted is not None and info.filename not in wanted:
                continue
            table[info.filename] = MemberLocation(
                name=info.filename,
                header_offset=int(info.header_offset),
                compress_size=int(info.compress_size),
                file_size=int(info.file_size),
                compress_type=int(info.compress_type),
                crc=int(info.CRC),
                name_len=len(info.filename.encode("utf-8")),
            )
    finally:
        zf.close()
    return table


class RangeConnection:
    """One kept-alive HTTPS connection that serves byte-range reads.

    Deliberately not thread-safe: every worker thread owns its own instance, and
    a fresh TLS connection per request is expensive on this host (measured ~4 s).
    """

    def __init__(self, url: str, *, timeout: float = 120.0, ua: str = _USER_AGENT) -> None:
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise ValueError(f"{url!r} is not an https url")
        self.url = url
        self.path = parts.path or "/"
        self.timeout = float(timeout)
        self.headers = {"User-Agent": ua}
        self._parts = parts
        self._conn: Optional[http.client.HTTPSConnection] = None
        self.n_requests = 0
        self.n_bytes = 0

    def _client(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(
                self._parts.netloc, timeout=self.timeout, context=ssl.create_default_context()
            )
        return self._conn

    def reset(self) -> None:
        """Drop the socket so the next read reconnects (keep-alive went stale)."""
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - closing a dead socket is not a failure
                pass

    def get_range(self, start: int, end: int) -> bytes:
        """Bytes ``[start, end]`` inclusive; raises unless the host sent ``206``."""
        try:
            conn = self._client()
            conn.request(
                "GET", self.path, headers={**self.headers, "Range": f"bytes={start}-{end}"}
            )
            resp = conn.getresponse()
            body = resp.read()
        except (http.client.HTTPException, OSError):
            self.reset()
            raise
        if resp.status == 200:
            self.reset()
            raise RuntimeError(f"{self.url}: host ignored Range for bytes {start}-{end}")
        if resp.status != 206:
            self.reset()
            raise urllib.error.HTTPError(
                self.url, int(resp.status), resp.reason or "error", resp.headers or {}, None
            )
        self.n_requests += 1
        self.n_bytes += len(body)
        return body

    def close(self) -> None:
        self.reset()


def decode_member_span(body: bytes, location: MemberLocation) -> bytes:
    """Verify and unpack the bytes fetched for ``location``'s header offset.

    The stored size and CRC from the central directory are checked against the
    decoded member, so a wrong offset or a truncated response cannot quietly
    produce a half-written JPEG.
    """
    if len(body) < _LOCAL_HEADER.size:
        raise ValueError(f"{location.name}: response shorter than a local header")
    sig, _ver, _flags, method, _mt, _md, crc, csize, usize, name_len, extra_len = (
        _LOCAL_HEADER.unpack(body[: _LOCAL_HEADER.size])
    )
    if sig != LOCAL_HEADER_SIGNATURE:
        raise ValueError(
            f"{location.name}: bad local header signature 0x{sig:08x} "
            f"at offset {location.header_offset}"
        )
    if method != location.compress_type:
        raise ValueError(
            f"{location.name}: local method {method} != indexed method {location.compress_type}"
        )
    start = _LOCAL_HEADER.size + int(name_len) + int(extra_len)
    # A data descriptor (flag bit 3) zeroes the local sizes; trust the index then.
    csize = int(csize) or location.compress_size
    usize = int(usize) or location.file_size
    crc = int(crc) or location.crc
    need = start + csize
    if len(body) < need:
        raise _ShortSpan(need)
    data = body[start:need]
    if location.compress_type == zipfile.ZIP_DEFLATED:
        # zip carries raw deflate (RFC 1951): no zlib wrapper, hence wbits=-15.
        out = zlib.decompressobj(-15).decompress(data)
    elif location.compress_type == zipfile.ZIP_STORED:
        out = data
    else:
        raise ValueError(f"{location.name}: unsupported zip method {location.compress_type}")
    if len(out) != usize:
        raise ValueError(f"{location.name}: decoded {len(out)} bytes, index says {usize}")
    if crc and (zlib.crc32(out) & 0xFFFFFFFF) != crc:
        raise ValueError(f"{location.name}: CRC mismatch")
    return out


def fetch_member(
    conn: RangeConnection, location: MemberLocation, *, pad: int = _LOCAL_EXTRA_PAD,
    attempts: int = 3,
) -> bytes:
    """Fetch one member with one range request, growing the span if it came back short.

    The local extra field is not known before the header is read, so the first
    span is an estimate; each short answer enlarges it and costs one more request.
    """
    span = location.span(pad)
    last: Optional[_ShortSpan] = None
    for _ in range(max(1, int(attempts))):
        body = conn.get_range(location.header_offset, location.header_offset + span - 1)
        try:
            return decode_member_span(body, location)
        except _ShortSpan as short:
            # grow past both the reported need and the previous guess
            last = short
            span = max(short.need, span) + pad
    raise last if last is not None else ValueError(f"{location.name}: unreadable member")


def local_image_path(out_dir: Path, gqa_image_id: str | int) -> Path:
    return Path(out_dir) / f"{gqa_image_id}.jpg"


@dataclass
class ImageFetchReport:
    """Outcome of :func:`fetch_images` (serialisable, never lossy)."""

    requested: int = 0
    fetched: int = 0
    skipped_existing: int = 0
    missing_in_archive: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str, str]] = field(default_factory=list)
    bytes_written: int = 0
    bytes_transferred: int = 0
    requests: int = 0
    seconds: float = 0.0

    @property
    def present(self) -> int:
        return self.fetched + self.skipped_existing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "fetched": self.fetched,
            "skipped_existing": self.skipped_existing,
            "present_after_run": self.present,
            "n_missing_in_archive": len(self.missing_in_archive),
            "missing_in_archive": self.missing_in_archive[:50],
            "n_failed": len(self.failed),
            "failed": [[i, f"{kind}: {detail}"[:160]] for i, kind, detail in self.failed[:50]],
            "bytes_written": self.bytes_written,
            "bytes_transferred": self.bytes_transferred,
            "requests": self.requests,
            "seconds": round(self.seconds, 2),
        }


def fetch_images(
    gqa_image_ids: Sequence[str | int],
    out_dir: Path,
    *,
    url: str = GQA_IMAGES_ZIP_URL,
    workers: int = DEFAULT_WORKERS,
    retries: int = 5,
    timeout: float = 120.0,
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> ImageFetchReport:
    """Extract the requested JPEGs from the remote archive into ``out_dir``.

    The archive index is read once (``member_locations``); each worker thread then
    owns a single kept-alive :class:`RangeConnection` and pulls one range per
    image.  Files already on disk are skipped unless ``overwrite``.

    Throttling policy: the host returns ``503`` under parallel load, so a failed
    request sleeps ``_BACKOFF_BASE * 2 ** attempt`` (capped, ``Retry-After``
    honoured) before rebuilding its connection.  Parallelism stays low on purpose -
    a feasibility audit may not depend on an unthrottled mirror.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = ImageFetchReport(requested=len(gqa_image_ids))
    todo: List[str] = []
    for gid in gqa_image_ids:
        gid = str(gid)
        if not overwrite and local_image_path(out_dir, gid).exists():
            report.skipped_existing += 1
        else:
            todo.append(gid)
    if not todo:
        report.seconds = time.time() - t0
        return report

    def _wait(exc: Exception, attempt: int) -> float:
        """Sleep after a failed attempt; ``Retry-After`` wins over the backoff."""
        delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** int(attempt)))
        if isinstance(exc, urllib.error.HTTPError):
            hdr = exc.headers.get("Retry-After") if exc.headers else None
            if hdr and str(hdr).strip().isdigit():
                delay = max(delay, min(_BACKOFF_CAP, float(hdr.strip())))
        time.sleep(delay)
        return delay

    names = {member_name(gid) for gid in todo}
    locations: Dict[str, MemberLocation] = {}
    for attempt in range(int(retries)):
        try:
            locations = member_locations(url, timeout=timeout, only=names)
            break
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            if attempt == int(retries) - 1:
                raise RuntimeError(
                    f"cannot read the archive index after {retries} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            _wait(exc, attempt)

    pending: List[Tuple[str, MemberLocation]] = []
    for gid in todo:
        found = locations.get(member_name(gid))
        if found is None:
            report.missing_in_archive.append(gid)
        else:
            pending.append((gid, found))

    local = threading.local()
    lock = threading.Lock()

    def _conn() -> RangeConnection:
        conn = getattr(local, "conn", None)
        if conn is None:
            conn = RangeConnection(url, timeout=timeout)
            local.conn = conn
        return conn

    def _one(item: Tuple[str, MemberLocation]) -> Tuple[str, Optional[bytes], Optional[str]]:
        gid, location = item
        last: Optional[str] = None
        for attempt in range(int(retries)):
            conn = _conn()
            before = (conn.n_requests, conn.n_bytes)
            try:
                payload = fetch_member(conn, location)
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                last = f"{type(exc).__name__}: {exc}"
                conn.reset()
                # A layout/CRC disagreement is deterministic; retrying cannot help.
                # A short span is a server answer, so it does get retried.
                if isinstance(exc, ValueError) and not isinstance(exc, _ShortSpan):
                    return gid, None, last
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in _RETRYABLE_STATUS:
                    return gid, None, last
                _wait(exc, attempt)
                continue
            with lock:
                report.requests += conn.n_requests - before[0]
                report.bytes_transferred += conn.n_bytes - before[1]
            return gid, payload, None
        return gid, None, last or "unknown error"

    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_one, item) for item in pending]
        done = 0
        for fut in as_completed(futures):
            gid, payload, err = fut.result()
            done += 1
            if payload is None:
                report.failed.append((gid, "range_fetch", str(err)))
            else:
                dest = local_image_path(out_dir, gid)
                tmp = dest.with_suffix(".jpg.part")
                tmp.write_bytes(payload)
                tmp.replace(dest)
                report.fetched += 1
                report.bytes_written += len(payload)
            if log is not None and (done % 100 == 0 or done == len(futures)):
                log(
                    f"[images] {done}/{len(futures)} fetched={report.fetched} "
                    f"failed={len(report.failed)} missing={len(report.missing_in_archive)} "
                    f"({report.bytes_written / 1e6:.0f} MB out, "
                    f"{report.bytes_transferred / 1e6:.0f} MB in over "
                    f"{report.requests} requests)"
                )
    report.seconds = time.time() - t0
    return report


# ---------------------------------------------------------------------------
# decode sanity
# ---------------------------------------------------------------------------
def decode_dims(path: Path | str) -> Optional[Tuple[int, int]]:
    """``(width, height)`` of a readable JPEG, or ``None`` when it cannot decode."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hard dependency here
        return None
    try:
        with Image.open(str(path)) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001 - truncated or corrupt file
        return None


def local_dims(image_dir: Path, ids: Optional[Iterable[int]] = None) -> Dict[int, Tuple[int, int]]:
    """Dimensions of every ``<id>.jpg`` in ``image_dir`` (``ids`` filters by GQA id)."""
    image_dir = Path(image_dir)
    wanted = None if ids is None else {int(i) for i in ids}
    out: Dict[int, Tuple[int, int]] = {}
    for p in sorted(image_dir.glob("*.jpg")):
        stem = p.stem
        if not stem.isdigit():
            continue
        iid = int(stem)
        if wanted is not None and iid not in wanted:
            continue
        dims = decode_dims(p)
        if dims is not None:
            out[iid] = dims
    return out
