"""Deterministic image fingerprints and duplicate matching for V3 exposure audits.

The audit rule is frozen before V3 confirmation scores are computed:

* exact duplicate: SHA-256 of EXIF-oriented, decoded RGB pixels including dimensions;
* near duplicate: 64-bit horizontal difference hash (dHash), Hamming distance <= 4.

File-byte hashes are retained for provenance but are not used as image identity.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageOps

AUDIT_RULE_VERSION = "v3-image-exposure-v1"
DHASH_DISTANCE_MAX = 4
DHASH_ALGORITHM = "Pillow EXIF-transposed RGB -> L -> 9x8 LANCZOS -> adjacent horizontal left>right bits"


@dataclass(frozen=True)
class ImageFingerprint:
    path: str
    file_sha256: str
    pixel_sha256: str
    dhash64: str
    width: int
    height: int


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> Tuple[str, int, int]:
    """Hash decoded, EXIF-oriented RGB pixels and dimensions."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    digest = hashlib.sha256()
    digest.update(b"ccg-v3-rgb-pixels-v1\0")
    digest.update(struct.pack("<II", width, height))
    digest.update(rgb.tobytes())
    return digest.hexdigest(), width, height


def dhash64(image: Image.Image) -> str:
    """Return the frozen 64-bit horizontal difference hash as 16 hex digits."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    gray = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(gray.getdata())
    bits = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            bits = (bits << 1) | int(values[offset + col] > values[offset + col + 1])
    return f"{bits:016x}"


def fingerprint_image(path: Path) -> ImageFingerprint:
    """Hash source bytes and decoded pixels from one local image."""
    path = Path(path)
    file_hash = sha256_file(path)
    with Image.open(path) as image:
        pixel_hash, width, height = pixel_sha256(image)
        perceptual_hash = dhash64(image)
    return ImageFingerprint(
        path=str(path),
        file_sha256=file_hash,
        pixel_sha256=pixel_hash,
        dhash64=perceptual_hash,
        width=width,
        height=height,
    )


def hamming64(left: str, right: str) -> int:
    """Hamming distance for two 64-bit hashes written as hexadecimal."""
    if len(left) != 16 or len(right) != 16:
        raise ValueError("dHash values must be 16 hexadecimal characters")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("dHash values must be hexadecimal") from exc


class _BKNode:
    __slots__ = ("hash_value", "keys", "children")

    def __init__(self, hash_value: str, key: str) -> None:
        self.hash_value = hash_value
        self.keys = [key]
        self.children: Dict[int, _BKNode] = {}


class BKTree:
    """Small exact-metric index for finding every hash within a Hamming radius."""

    def __init__(self, entries: Iterable[Tuple[str, str]] = ()) -> None:
        self.root: _BKNode | None = None
        for hash_value, key in entries:
            self.add(hash_value, key)

    def add(self, hash_value: str, key: str) -> None:
        int(hash_value, 16)
        if len(hash_value) != 16:
            raise ValueError("dHash values must be 16 hexadecimal characters")
        if self.root is None:
            self.root = _BKNode(hash_value, key)
            return
        node = self.root
        while True:
            distance = hamming64(hash_value, node.hash_value)
            if distance == 0:
                node.keys.append(key)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(hash_value, key)
                return
            node = child

    def query(self, hash_value: str, max_distance: int = DHASH_DISTANCE_MAX) -> List[Tuple[str, int]]:
        """Return all stored keys within the inclusive Hamming radius."""
        int(hash_value, 16)
        if len(hash_value) != 16:
            raise ValueError("dHash values must be 16 hexadecimal characters")
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")
        matches: List[Tuple[str, int]] = []
        if self.root is None:
            return matches
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = hamming64(hash_value, node.hash_value)
            if distance <= max_distance:
                matches.extend((key, distance) for key in node.keys)
            lower = distance - max_distance
            upper = distance + max_distance
            pending.extend(
                child for edge_distance, child in node.children.items()
                if lower <= edge_distance <= upper
            )
        return sorted(matches, key=lambda item: (item[1], item[0]))


def build_dhash_index(rows: Sequence[Mapping[str, str]], hash_field: str = "dhash64") -> BKTree:
    """Build an exact BK-tree from row mappings with a stable row identifier."""
    entries = []
    for index, row in enumerate(rows):
        hash_value = str(row.get(hash_field, ""))
        if not hash_value:
            continue
        key = str(row.get("record_id", index))
        entries.append((hash_value, key))
    return BKTree(entries)

