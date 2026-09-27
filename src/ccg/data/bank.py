"""Frozen HDF5 proposal bank schema (``bank-v1``) - the shared cache contract.

The full-data proposal bank is written **once** by ``scripts/extract_proposals.py``
and read by every parallel work stream (candidate builders, features, presence
models, natural-omission counting).  Its on-disk layout is therefore a frozen
contract:

h5 layout (frozen)
------------------
* one top-level group per image, ``image_{image_id}`` (decimal, e.g.
  ``image_1234``);
* datasets inside the group:

  - ``boxes``      ``[K, 4]`` float32, ``xyxy`` in original-image pixels,
  - ``objectness`` ``[K]`` float32, descending (row ``i`` belongs to row ``i``);

* group attr ``n_raw_post_nms`` (int): how many proposals the RPN produced
  *before* the top-N truncation (``n_raw_post_nms >= K`` always holds);
* root attrs written by :func:`finalize_bank`:
  ``schema_version`` (``"bank-v1"``), ``model_name``, ``weights``,
  ``proposal_type``, ``top_n``, ``torchvision_version``, ``n_images``,
  ``created_utc``, ``images_root``.

Write semantics (atomic / resumable)
------------------------------------
The bank is **appended to in place** (``h5py`` mode ``"a"``), one image at a
time, so a killed process loses at most the group being written:

* :func:`write_bank_entry` skips an image whose group already exists
  (``--resume`` semantics); pass ``overwrite=True`` to delete-and-rewrite;
* a half-written group (missing ``boxes``/``objectness``, wrong shape, or
  empty) is detected by :func:`find_corrupt_images` and removed by
  :func:`repair_bank` (``--repair`` downstream); nothing is repaired silently.

Read semantics
--------------
:func:`iter_bank` yields images in ascending ``image_id`` order;
:func:`read_bank_image` is the random-access read.  A corrupt group makes both
raise with the image id and the repair hint - silent skipping is forbidden by
the project's data-integrity rules.

The ten public names below form the frozen interface (multi-agent brief,
2026-09-27); additional helpers are additive and must not replace them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "BANK_SCHEMA_VERSION",
    "BANK_GROUP_PREFIX",
    "ROOT_ATTR_KEYS",
    "bank_group_name",
    "write_bank_entry",
    "iter_bank",
    "read_bank_image",
    "bank_attrs",
    "image_ids",
    "bank_has_image",
    "finalize_bank",
    "find_corrupt_images",
    "repair_bank",
    "delete_bank_entry",
]

#: Frozen schema version - bumping it invalidates every consumer.
BANK_SCHEMA_VERSION = "bank-v1"
#: Top-level group name template; ``image_{int(image_id)}``.
BANK_GROUP_PREFIX = "image_"
#: Root attributes the frozen contract requires (``finalize_bank``); values
#: that must always be present after a finished run.  ``n_images`` is derived
#: from the file itself, never from the caller.
ROOT_ATTR_KEYS: Tuple[str, ...] = (
    "schema_version",
    "model_name",
    "weights",
    "proposal_type",
    "top_n",
    "torchvision_version",
    "n_images",
    "created_utc",
    "images_root",
)

_BOXES = "boxes"
_OBJECTNESS = "objectness"
_N_RAW = "n_raw_post_nms"


# ---------------------------------------------------------------------------
# naming helpers
# ---------------------------------------------------------------------------
def bank_group_name(image_id: Any) -> str:
    """``image_<decimal image_id>`` - the frozen group name."""
    try:
        value = int(np.asarray(image_id).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image_id must be integer-like, got {image_id!r}") from exc
    if value < 0:
        raise ValueError(f"image_id must be >= 0, got {value}")
    return f"{BANK_GROUP_PREFIX}{value}"


def _sorted_image_ids(handle: Any) -> List[int]:
    """Ascending image ids of every ``image_*`` group in an open h5 handle.

    Groups whose name does not match ``image_<digits>`` are ignored (they are
    outside the frozen schema and never produced by this module).
    """
    found: List[int] = []
    for key in handle.keys():
        if not isinstance(key, str) or not key.startswith(BANK_GROUP_PREFIX):
            continue
        suffix = key[len(BANK_GROUP_PREFIX) :]
        if suffix.isdigit():
            found.append(int(suffix))
    found.sort()
    return found


# ---------------------------------------------------------------------------
# validation / attribute conversion
# ---------------------------------------------------------------------------
def _validated_arrays(boxes: Any, objectness: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the frozen dataset contract; return float32 ``(boxes, objectness)``.

    Raises :class:`ValueError` for anything that would silently corrupt the
    bank: non-numeric dtypes, wrong shapes, misaligned lengths, empty banks and
    non-finite values.
    """
    raw_boxes = np.asarray(boxes)
    if raw_boxes.dtype.kind not in "fiu":
        raise ValueError(
            f"boxes must be numeric, got dtype {raw_boxes.dtype} (kind {raw_boxes.dtype.kind!r})"
        )
    if raw_boxes.ndim != 2 or raw_boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape [K,4], got {raw_boxes.shape}")
    k = int(raw_boxes.shape[0])
    if k < 1:
        raise ValueError("boxes must hold at least one row (empty banks are forbidden)")
    boxes32 = np.ascontiguousarray(raw_boxes, dtype=np.float32)
    if not np.all(np.isfinite(boxes32)):
        raise ValueError("boxes contain non-finite values (nan/inf)")

    raw_objectness = np.asarray(objectness)
    if raw_objectness.dtype.kind not in "fiu":
        raise ValueError(
            f"objectness must be numeric, got dtype {raw_objectness.dtype} "
            f"(kind {raw_objectness.dtype.kind!r})"
        )
    if raw_objectness.ndim != 1:
        raise ValueError(f"objectness must have shape [K], got {raw_objectness.shape}")
    if int(raw_objectness.shape[0]) != k:
        raise ValueError(
            f"boxes and objectness disagree on K: {k} != {int(raw_objectness.shape[0])}"
        )
    objectness32 = np.ascontiguousarray(raw_objectness, dtype=np.float32)
    if not np.all(np.isfinite(objectness32)):
        raise ValueError("objectness contains non-finite values (nan/inf)")
    return boxes32, objectness32


def _serialize_attr(key: str, value: Any) -> Any:
    """Convert one group/root attribute value to an h5py-storable scalar."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise ValueError(
        f"attribute {key!r} has unsupported type {type(value).__name__}; "
        "bank attrs must be str/int/float/bool scalars"
    )


def _attr_to_python(value: Any) -> Any:
    """h5py attribute -> plain python scalar (bytes decoded as utf-8)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_attr_to_python(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_bank_entry(
    h5: Any,
    image_id: Any,
    boxes: Any,
    objectness: Any,
    attrs_extra: Optional[Dict[str, Any]] = None,
    *,
    overwrite: bool = False,
) -> None:
    """Append one image's proposal bank to an **open** h5 object (mode ``"a"``).

    ``h5`` is an open ``h5py.File`` (or ``h5py.Group``) the caller owns; this
    function never opens or closes it, which is what makes the write sequence
    cheap enough for the whole-image loop in ``scripts/extract_proposals.py``.

    The group is written datasets-first, attributes-last; a process killed
    mid-write leaves a group missing a dataset, which
    :func:`find_corrupt_images` detects and :func:`repair_bank` removes.

    Parameters
    ----------
    h5:
        Open writable ``h5py`` file/group.
    image_id:
        Integer-like COCO image id (the group is ``image_{image_id}``).
    boxes, objectness:
        ``[K,4]`` ``xyxy`` and ``[K]`` arrays (any numeric dtype; stored as
        float32).  Validated: wrong shape / non-numeric dtype / non-finite
        values raise :class:`ValueError`.
    attrs_extra:
        Extra group attributes.  ``n_raw_post_nms`` is read from here when
        present (must satisfy ``>= K``); otherwise it defaults to ``K``.
        Any other key lands on the group verbatim (scalars only, nothing
        silent about a rejected value).
    overwrite:
        ``False`` (default) skips the write when ``image_{image_id}`` already
        exists - the resume semantics.  ``True`` deletes the existing group
        first (re-extraction).
    """
    name = bank_group_name(image_id)
    boxes32, objectness32 = _validated_arrays(boxes, objectness)
    k = int(boxes32.shape[0])

    extras: Dict[str, Any] = {}
    n_raw = k
    if attrs_extra is not None:
        for key, value in dict(attrs_extra).items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"attrs_extra keys must be non-empty strings, got {key!r}")
            if key == _N_RAW:
                try:
                    n_raw = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"n_raw_post_nms must be an int, got {value!r}") from exc
            else:
                # serialise *before* any group is created: a rejected attr
                # value must never leave a half-written group behind
                extras[key] = _serialize_attr(key, value)
    if n_raw < k:
        raise ValueError(
            f"n_raw_post_nms ({n_raw}) must be >= the stored bank size K ({k}); "
            "the top-N truncation can only remove rows"
        )

    if name in h5:
        if not overwrite:
            return
        del h5[name]
    group = h5.create_group(name)
    group.create_dataset(_BOXES, data=boxes32)
    group.create_dataset(_OBJECTNESS, data=objectness32)
    group.attrs[_N_RAW] = int(n_raw)
    for key, value in extras.items():
        group.attrs[key] = _serialize_attr(key, value)


def delete_bank_entry(path: Any, image_id: Any) -> bool:
    """Remove one image group from the bank file; ``True`` when it existed."""
    import h5py

    name = bank_group_name(image_id)
    with h5py.File(Path(path), "a") as handle:
        if name not in handle:
            return False
        del handle[name]
        handle.flush()
    return True


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _read_group(group: Any, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(boxes, objectness)`` float32 from one open h5 group; corrupt => KeyError."""
    for dataset in (_BOXES, _OBJECTNESS):
        if dataset not in group:
            raise KeyError(
                f"bank group image_{image_id} is corrupt: dataset {dataset!r} missing "
                "(half-written group); run find_corrupt_images/repair_bank (--repair) first"
            )
    boxes_node = group[_BOXES]
    objectness_node = group[_OBJECTNESS]
    boxes_shape = getattr(boxes_node, "shape", None)
    objectness_shape = getattr(objectness_node, "shape", None)
    if (
        boxes_shape is None
        or objectness_shape is None
        or len(boxes_shape) != 2
        or boxes_shape[1] != 4
        or len(objectness_shape) != 1
        or boxes_shape[0] < 1
        or boxes_shape[0] != objectness_shape[0]
    ):
        raise KeyError(
            f"bank group image_{image_id} is corrupt: boxes shape {boxes_shape}, "
            f"objectness shape {objectness_shape}; run repair_bank (--repair) first"
        )
    boxes = np.asarray(boxes_node, dtype=np.float32).reshape(-1, 4)
    objectness = np.asarray(objectness_node, dtype=np.float32).reshape(-1)
    return boxes, objectness


def iter_bank(path: Any) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Stream every ``(image_id, boxes, objectness)`` in ascending image order.

    The file stays open while the generator is consumed.  A corrupt group
    raises :class:`KeyError` (with the repair hint) instead of being skipped.
    """
    import h5py

    with h5py.File(Path(path), "r") as handle:
        for image_id in _sorted_image_ids(handle):
            boxes, objectness = _read_group(handle[f"{BANK_GROUP_PREFIX}{image_id}"], image_id)
            yield image_id, boxes, objectness


def read_bank_image(path: Any, image_id: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Random-access ``(boxes, objectness)`` for one image.

    Raises :class:`KeyError` when the image is absent or its group is corrupt
    (missing dataset / length mismatch) - never returns an empty placeholder.
    """
    import h5py

    value = int(np.asarray(image_id).item())
    name = bank_group_name(value)
    with h5py.File(Path(path), "r") as handle:
        if name not in handle:
            raise KeyError(f"image {value} not in bank {path}")
        return _read_group(handle[name], value)


def image_ids(path: Any) -> np.ndarray:
    """Sorted ``int64`` array of every image id present in the bank."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        return np.asarray(_sorted_image_ids(handle), dtype=np.int64)


def bank_has_image(path: Any, image_id: Any) -> bool:
    """``True`` when the image group exists (existence only - not integrity).

    Pair with :func:`find_corrupt_images` when a broken group must count as
    "missing" (that is what ``--repair`` in the extraction script does).
    """
    import h5py

    name = bank_group_name(image_id)
    with h5py.File(Path(path), "r") as handle:
        return name in handle


def bank_attrs(path: Any) -> dict:
    """Root attributes as a plain python dict (``{}`` before ``finalize_bank``)."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        return {str(key): _attr_to_python(value) for key, value in handle.attrs.items()}


# ---------------------------------------------------------------------------
# integrity: detection + repair
# ---------------------------------------------------------------------------
def _is_corrupt_group(group: Any) -> bool:
    """Structural check of one open group (metadata only, no data read)."""
    for dataset in (_BOXES, _OBJECTNESS):
        if dataset not in group:
            return True
    boxes = group[_BOXES]
    objectness = group[_OBJECTNESS]
    try:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            return True
        if objectness.ndim != 1:
            return True
        if boxes.shape[0] < 1 or boxes.shape[0] != objectness.shape[0]:
            return True
    except AttributeError:  # not a dataset (a nested group in its place)
        return True
    return False


def _find_corrupt(handle: Any) -> List[int]:
    return [
        image_id
        for image_id in _sorted_image_ids(handle)
        if _is_corrupt_group(handle[f"{BANK_GROUP_PREFIX}{image_id}"])
    ]


def find_corrupt_images(path: Any) -> List[int]:
    """Image ids whose group violates the frozen layout (half-written etc.).

    Checks, in order: both datasets present, ``boxes`` ``[K,4]``, ``objectness``
    ``[K]``, ``K >= 1`` and matching row counts.  Sorted ascending.
    """
    import h5py

    with h5py.File(Path(path), "r") as handle:
        return _find_corrupt(handle)


def repair_bank(path: Any, *, dry_run: bool = False) -> List[int]:
    """Delete every corrupt group; returns the removed image ids (sorted).

    ``dry_run=True`` reports without deleting.  Only groups failing the
    :func:`find_corrupt_images` structural check are touched - valid entries
    are never rewritten.
    """
    import h5py

    with h5py.File(Path(path), "a") as handle:
        corrupt = _find_corrupt(handle)
        if not dry_run:
            for image_id in corrupt:
                del handle[f"{BANK_GROUP_PREFIX}{image_id}"]
            handle.flush()
        return corrupt


# ---------------------------------------------------------------------------
# finalisation
# ---------------------------------------------------------------------------
def finalize_bank(path: Any, attrs: Dict[str, Any]) -> None:
    """Write the frozen root attributes after the last image was appended.

    ``attrs`` supplies ``model_name`` / ``weights`` / ``proposal_type`` /
    ``top_n`` / ``torchvision_version`` / ``images_root`` (+ optionally
    ``created_utc``; generated when absent).  Two values are always forced by
    this function instead of trusting the caller:

    * ``schema_version`` = :data:`BANK_SCHEMA_VERSION` (a bank can never claim
      a different schema than its writer), and
    * ``n_images`` = the actual number of ``image_*`` groups in the file, so a
      finished bank cannot lie about its size.
    """
    import h5py

    if not isinstance(attrs, dict):
        raise TypeError(f"attrs must be a dict, got {type(attrs).__name__}")
    payload = {str(key): value for key, value in attrs.items() if key != "schema_version"}
    payload.pop("n_images", None)
    payload.setdefault("created_utc", _utc_now())

    with h5py.File(Path(path), "a") as handle:
        handle.attrs["schema_version"] = BANK_SCHEMA_VERSION
        handle.attrs["n_images"] = int(len(_sorted_image_ids(handle)))
        for key, value in payload.items():
            handle.attrs[key] = _serialize_attr(key, value)
        handle.flush()
