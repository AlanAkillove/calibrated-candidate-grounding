"""Artifact-identity primitives shared by the A11 external run (protocol A11.5 / A11.8).

Amendment A11 freezes the *inputs* as tightly as the *models*: before any
RefCOCOg prediction is produced the run must be able to show, file by file, that

* the OpenCLIP checkpoint it is about to run is the same checkpoint the frozen
  RefCOCO+ cache was built with (A11.5 - "a mismatch stops the run"), and
* every frozen artifact the reliability coefficients were exported from is
  pinned by content hash (A11.8 - "save and check the checksums before the
  formal forward pass").

Neither question has an existing answer in the repository: :mod:`ccg.features.cache`
records ``checkpoint_sha256`` inside ``cache/features/metadata.json`` but nothing
ever *verifies* it against the file on disk, and the encoder factory accepts any
``pretrained`` string.  This module is that missing verification layer - pure
hashing and comparison, no model loading, no I/O beyond reads, so it is safe to
call on the production path and trivial to unit-test.

The frozen identity itself is taken from protocol amendment A11.5, which quotes
``cache/features/metadata.json`` - :data:`FROZEN_OPENCLIP` therefore holds the
literals rather than re-reading them from the file being checked (a self-fulfilling
verification would prove nothing).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "DEFAULT_FEATURES_ROOT",
    "FROZEN_OPENCLIP",
    "REPO_ROOT",
    "IdentityMismatch",
    "ArtifactIdentity",
    "sha256_file",
    "sha256_text",
    "file_identity",
    "checkpoint_identity",
    "assert_openclip_identity",
    "checksum_manifest",
    "verify_checksum_manifest",
]

#: The frozen encoder identity (A11.5): model, weights, checkpoint hash, precision,
#: embedding dimensionality and tokenizer.  Every field is compared, not assumed.
FROZEN_OPENCLIP: Dict[str, Any] = {
    "model_name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "library": "open_clip_torch",
    "precision": "fp16",
    "embedding_dim": 512,
    "resolution": 224,
    "tokenizer": "SimpleTokenizer",
    "context_length": 77,
    "cache_version": "phase0a-v1",
    "checkpoint_sha256": (
        "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad"
    ),
}

#: Default root of the frozen RefCOCO+ feature cache that carries the identity.
DEFAULT_FEATURES_ROOT = Path("cache/features")
#: Repository root, so a ``checkpoint_path`` recorded relative to the repo resolves
#: whatever the current working directory happens to be.
REPO_ROOT = Path(__file__).resolve().parents[3]


class IdentityMismatch(AssertionError):
    """Raised when an artifact is not the frozen one (the A11.5/A11.8 STOP)."""


def sha256_file(path: Path | str, *, block: int = 1 << 20) -> str:
    """Streaming ``sha256`` of a file's bytes (the repo's provenance digest)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """``sha256`` of a UTF-8 string (json payloads written from memory)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactIdentity:
    """One pinned file: path, content digest, size and modification time."""

    path: str
    sha256: str
    bytes: int
    mtime_utc: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "mtime_utc": self.mtime_utc,
        }


def file_identity(path: Path | str, *, root: Optional[Path | str] = None) -> ArtifactIdentity:
    """Hash one existing file; ``root`` only shortens the recorded path."""
    target = Path(path)
    if not target.exists():
        raise IdentityMismatch(f"{target}: frozen artifact is missing")
    stat = target.stat()
    label = str(target)
    if root is not None:
        try:
            label = str(target.resolve().relative_to(Path(root).resolve()))
        except ValueError:  # pragma: no cover - outside the repo
            label = str(target)
    return ArtifactIdentity(
        path=label.replace("\\", "/"),
        sha256=sha256_file(target),
        bytes=int(stat.st_size),
        mtime_utc=float(stat.st_mtime),
    )


def checkpoint_identity(
    features_root: Path | str = DEFAULT_FEATURES_ROOT,
) -> Dict[str, Any]:
    """What ``features_root/metadata.json`` *claims* the frozen cache was built with.

    Returns the backbone block plus the recorded checkpoint path; raises
    :class:`IdentityMismatch` when the metadata is missing or has no checkpoint
    hash, because then nothing can be verified and A11 must stop.
    """
    metadata_path = Path(features_root) / "metadata.json"
    if not metadata_path.exists():
        raise IdentityMismatch(
            f"{metadata_path} not found - the frozen OpenCLIP identity cannot be read"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    backbone = dict(payload.get("backbone") or {})
    recorded = str(backbone.get("checkpoint_sha256") or "")
    if not recorded:
        raise IdentityMismatch(
            f"{metadata_path}: backbone.checkpoint_sha256 is absent - refusing to extract "
            "external features against an unidentifiable encoder"
        )
    return {
        "metadata_path": str(metadata_path).replace("\\", "/"),
        "recorded_checkpoint_sha256": recorded,
        "checkpoint_path": str(backbone.get("checkpoint_path") or ""),
        "model_name": str(backbone.get("model_name") or ""),
        "pretrained": str(backbone.get("pretrained") or ""),
        "library": str(backbone.get("library") or ""),
        "library_version": str(backbone.get("library_version") or ""),
        "precision": str(backbone.get("precision") or ""),
        "embedding_dim": int(backbone.get("embedding_dim") or 0),
        "resolution": int(backbone.get("resolution") or 0),
        "tokenizer": str((backbone.get("tokenizer") or {}).get("name") or ""),
        "context_length": int((backbone.get("tokenizer") or {}).get("context_length") or 0),
        "cache_version": str(payload.get("cache_version") or ""),
        "preprocessing": dict((backbone.get("preprocessing") or {})),
    }


def assert_openclip_identity(
    features_root: Path | str = DEFAULT_FEATURES_ROOT,
    *,
    checkpoint_path: Optional[Path | str] = None,
    expected: Mapping[str, Any] = FROZEN_OPENCLIP,
) -> Dict[str, Any]:
    """Pin the encoder A11 will extract RefCOCOg features with (A11.5).

    Three separate questions, all fatal on failure:

    1. every identity field of ``metadata.json`` equals the frozen literal
       (:data:`FROZEN_OPENCLIP`) - a *different* recorded checkpoint is a mismatch
       even if the file hashes correctly;
    2. the checkpoint file on disk still hashes to the recorded ``checkpoint_sha256``
       - the metadata alone is not trusted;
    3. the file exists and is non-empty.

    ``checkpoint_path`` overrides the (repository-relative) path recorded in the
    metadata, which is how a run on another machine passes its own copy.  Returns
    the verified identity plus the measured digest; raises
    :class:`IdentityMismatch` listing *every* divergence at once.
    """
    recorded = checkpoint_identity(features_root)
    problems: List[str] = []

    for field, want in expected.items():
        if field == "checkpoint_sha256":
            continue
        got = recorded.get(field)
        if got != want:
            problems.append(f"{field}: metadata says {got!r}, A11.5 froze {want!r}")

    digest = recorded["recorded_checkpoint_sha256"]
    if digest != expected["checkpoint_sha256"]:
        problems.append(
            f"checkpoint_sha256: metadata says {digest}, A11.5 froze "
            f"{expected['checkpoint_sha256']}"
        )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
    else:
        recorded_path = Path(recorded["checkpoint_path"])
        path = recorded_path if recorded_path.is_absolute() else REPO_ROOT / recorded_path
    if not path.exists():
        problems.append(f"checkpoint file {path} is not on disk")
    else:
        if path.stat().st_size == 0:
            problems.append(f"checkpoint file {path} is empty")
        else:
            measured = sha256_file(path)
            if measured != expected["checkpoint_sha256"]:
                problems.append(
                    f"checkpoint file {path}: sha256 {measured} != frozen "
                    f"{expected['checkpoint_sha256']} - the encoder is not the frozen one"
                )

    if problems:
        raise IdentityMismatch(
            "OPENCLIP_IDENTITY_FAILURE: " + "; ".join(problems)
        )
    return {
        "ok": True,
        "checkpoint_path": str(path).replace("\\", "/"),
        "measured_checkpoint_sha256": expected["checkpoint_sha256"],
        "metadata": recorded,
        "frozen": dict(expected),
    }


def checksum_manifest(
    paths: Iterable[Path | str], *, root: Path | str = REPO_ROOT
) -> Dict[str, ArtifactIdentity]:
    """``path -> identity`` for a batch of frozen inputs (A11.8)."""
    out: Dict[str, ArtifactIdentity] = {}
    for raw in paths:
        identity = file_identity(raw, root=root)
        if identity.path in out and out[identity.path].sha256 != identity.sha256:
            raise IdentityMismatch(f"{identity.path}: recorded twice with different content")
        out[identity.path] = identity
    return out


def verify_checksum_manifest(
    manifest: Mapping[str, Mapping[str, Any]],
    *,
    root: Path | str = REPO_ROOT,
    required: Sequence[str] = (),
) -> Dict[str, Any]:
    """Re-hash every entry of a stored manifest and report the drift.

    ``required`` names *labels* (manifest keys) that must be present, so a
    manifest that quietly dropped the E1b coefficient table fails rather than
    passing.  Each entry's file is located through its recorded ``path`` field
    (falling back to the key when the entry stores no path), resolved under
    ``root``; a missing file is a mismatch, never a skip.
    """
    mismatches: List[str] = []
    missing_required: List[str] = [str(name) for name in required if str(name) not in manifest]
    checked = 0
    for label, entry in manifest.items():
        want = str(entry.get("sha256") or "")
        target = Path(root) / str(entry.get("path") or label)
        if not target.exists():
            mismatches.append(f"{label}: file {entry.get('path') or label} is missing")
            continue
        measured = sha256_file(target)
        checked += 1
        if measured != want:
            mismatches.append(f"{label}: sha256 {measured[:16]}... != recorded {want[:16]}...")
    if missing_required:
        mismatches.append(f"manifest is missing required entries: {missing_required}")
    return {
        "ok": not mismatches,
        "n_entries": len(manifest),
        "n_checked": checked,
        "n_missing_required": len(missing_required),
        "mismatches": mismatches,
    }
