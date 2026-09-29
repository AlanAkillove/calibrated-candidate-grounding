"""RefCOCOg external feature corpus + cohort loading for A11 (protocol A11.5/A11.11).

This module is the bridge between the *index-level* candidate cohort built by
:mod:`ccg.external.refcocog_manifests` (which rows, which bank indices) and the
*tensor-level* frozen B3 scorer, which reads CLIP embeddings through the
:class:`ccg.models.b3_data.B3Corpus` duck interface
(``_region`` / ``_text`` / ``_bank`` / ``image_size``).  Nothing here fits or
re-derives a model; it only materialises the frozen candidate sets from the
extracted RefCOCOg feature cache so the loaded bundle can predict on them.

Two responsibilities:

* :func:`load_refcocog_cohort` - rebuild the exact A10-froze ``1,890 / 755 / 978``
  cohort (reusing A10's own expression / GT loaders through :mod:`run_phase1e_refcocog_feasibility`
  so the pixel provenance is identical), enrich each row with its RefCOCOg
  *expression text* (which ``audit_expressions.csv`` stores in the ``expression``
  column and the reconstruction helper does not set), and pin it back to A10.
* :class:`RefCOCOGFeatureCorpus` - the read-only feature store the frozen scorer
  walks, over ``cache/refcocog_external`` (regions) and the same cache's text file
  (one query per expression, keyed by ``expr_id``).

The encoder itself is never built or verified here - that is the extraction
driver's job (A11.5 pins the checkpoint before any crop is encoded); this module
only consumes the cache it wrote.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data.bank import read_bank_image
from ..features.cache import FeatureCache
from ..models import b3_data
from . import refcocog as rg
from . import refcocog_manifests as rman

__all__ = [
    "A10_FEASIBILITY_ROOT",
    "COHORT_ARTIFACTS",
    "DEFAULT_BANK_PATHS",
    "EXTERNAL_CACHE_ROOT",
    "IMAGE_ROOT",
    "RefCOCOGFeatureCorpus",
    "build_image_sizes_map",
    "load_refcocog_cohort",
    "resolve_refcocog_image_path",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
_A10_DRIVER = REPO_ROOT / "scripts" / "run_phase1e_refcocog_feasibility.py"

#: A10's frozen feasibility output - the single source of truth for the row list.
A10_FEASIBILITY_ROOT = Path("results/phase1e_refcocog_feasibility")
#: Where A11's own external feature cache lives (A11.26 independent namespace).
EXTERNAL_CACHE_ROOT = Path("cache/refcocog_external")
#: Cohort sidecar written once by the manifests stage, re-read by every later stage.
COHORT_ARTIFACTS = Path("results/phase1e_refcocog_external/cohort")
#: RefCOCOg images (downloaded by A10's images stage, COCO train2014 naming).
IMAGE_ROOT = Path("data/raw/refcocog/images")
#: Official COCO GT - the category basis A10 froze (``non_crowd()``).
COCO_GT = Path("data/raw/annotations/instances_train2014.json")
REFCOCOG_ANNOTATION_DIR = Path("data/raw/refcocog/refcocog")
#: The proposal banks A11 reads (RefCOCOg strict bank + the frozen RefCOCO+ bank).
DEFAULT_BANK_PATHS: Dict[str, Path] = {
    "refcocog": Path("cache/phase1e_refcocog/proposals_refcocog.h5"),
    "frozen_plus": Path("cache/proposals.h5"),
}


def _load_a10_driver() -> Any:
    """Import ``run_phase1e_refcocog_feasibility.py`` (guarded, so import is safe)."""
    spec = importlib.util.spec_from_file_location("a11_a10_driver", _A10_DRIVER)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {_A10_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_refcocog_image_path(image_id: int, images_root: Path = IMAGE_ROOT) -> Path:
    """COCO-convention path of one RefCOCOg image under a flat/``train2014`` root."""
    from ..features.extract_regions import resolve_image_path

    return resolve_image_path(images_root, int(image_id))


def build_image_sizes_map(
    image_ids: Sequence[int], *, images_root: Path = IMAGE_ROOT
) -> Dict[int, Tuple[int, int]]:
    """``{image_id: (W, H)}`` decoded from the JPEG headers (no silent default)."""
    out: Dict[int, Tuple[int, int]] = {}
    for image_id in sorted({int(i) for i in image_ids}):
        path = resolve_refcocog_image_path(image_id, images_root)
        out[image_id] = b3_data._read_pil_size(path)  # noqa: SLF001 - the header reader is the point
    return out


def _expressions_with_text(
    driver: Any, inventory: Sequence[Mapping[str, str]], by_ref: Mapping[Tuple[int, int], Any], adir: Path
) -> List[rg.RefCOCOGExpression]:
    """A10's reconstruction, plus the ``expression`` text the scorer must encode."""
    expressions = driver._expressions_from_inventory(inventory, by_ref, adir)
    text_by_expr = {int(r["expr_id"]): str(r.get("expression") or "") for r in inventory}
    for expr in expressions:
        text = text_by_expr.get(int(expr.expr_id), "")
        object.__setattr__(expr, "text", text)
        object.__setattr__(expr, "tokens", tuple(rg.whitespace_tokens(text)))
    return expressions


def load_refcocog_cohort(
    *,
    feasibility_root: Path = A10_FEASIBILITY_ROOT,
    annotation_dir: Path = REFCOCOG_ANNOTATION_DIR,
    bank_paths: Optional[Mapping[str, Path]] = None,
    gt_file: Path = COCO_GT,
    images_root: Path = IMAGE_ROOT,
    persist_dir: Optional[Path] = COHORT_ARTIFACTS,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple["rman.RefCOCOGCohort", Dict[int, Tuple[int, int]], Dict[str, Any]]:
    """Rebuild and pin the A10-froze cohort, with expression text + image sizes.

    Returns ``(cohort, image_sizes, report)``.  ``report`` records the A10 pin
    (row set / supply / IoU agreement) and the matched-pair failure counters, so
    the caller can assert the cohort is exactly the one A10 froze before any
    prediction.  A11 evaluation must abort on any non-zero failure.  When
    ``persist_dir`` is given the two regime manifests, the ``cohort.csv``
    sidecar, the image sizes and the pin report are written there so later
    stages re-read the frozen cohort instead of re-deriving it.
    """
    emit = log or (lambda message: None)
    bank_paths = dict(DEFAULT_BANK_PATHS if bank_paths is None else bank_paths)
    driver = _load_a10_driver()
    adir = Path(annotation_dir)

    inventory = driver._read_csv(Path(feasibility_root) / "audit_expressions.csv")
    if not inventory:
        raise FileNotFoundError(f"{feasibility_root}/audit_expressions.csv missing or empty")
    regions = driver.load_regions(adir)
    by_ref = {(int(r.image_id), int(r.ann_id)): r for r in regions}
    expressions = _expressions_with_text(driver, inventory, by_ref, adir)
    emit(f"[a11-cohort] inventory {len(inventory)} / rebuilt expressions {len(expressions)}")

    a10_rows = driver._read_csv(Path(feasibility_root) / "hard_cohort.csv")
    strict_exprs = [e for e in expressions if str(e.subset) == rg.SUBSET_STRICT]
    cohort_expr_ids = {int(r["expr_id"]) for r in a10_rows}
    selected = [e for e in strict_exprs if int(e.expr_id) in cohort_expr_ids]
    emit(f"[a11-cohort] strict {len(strict_exprs)} / A10 cohort {len(cohort_expr_ids)} / selected {len(selected)}")

    rman.configure_bank_source(bank_paths)
    image_ids = sorted({int(e.image_id) for e in selected})
    gt_index = driver.official_gt(image_ids, Path(gt_file))

    common = dict(bank_paths=bank_paths, gt_index=gt_index)
    rand_manifest = rman.build_expression_manifest(selected, regime="random", **common)
    hard_manifest = rman.build_expression_manifest(selected, regime="same_category", **common)
    cohort = rman.cohort_from_manifests(rand_manifest, hard_manifest, selected, gt_index=gt_index)

    pin = rman.verify_against_a10_cohort(cohort, a10_rows)
    matched = rman.verify_matched_pair(cohort, gt_index=gt_index)
    report = {
        "a10_pin": pin,
        "matched_pair": matched,
        "cohort": {
            "n_rows": len(cohort),
            "n_images": cohort.n_images,
            "n_refs": cohort.n_refs,
            "K": rman.A11_K,
        },
    }
    if not (pin["ok"] and matched["ok"]):
        raise AssertionError(f"A11 cohort pin failed: pin_ok={pin['ok']} matched_ok={matched['ok']}")
    if len(cohort) != len(cohort_expr_ids):
        raise AssertionError(
            f"cohort rows {len(cohort)} != A10 hard_cohort rows {len(cohort_expr_ids)}"
        )

    image_sizes = build_image_sizes_map(cohort.image_id, images_root=images_root)
    missing = [int(i) for i in cohort.image_id if int(i) not in image_sizes]
    if missing:
        raise AssertionError(f"{len(set(missing))} cohort images have no measured size")
    if persist_dir is not None:
        target = Path(persist_dir)
        target.mkdir(parents=True, exist_ok=True)
        provenance = rman.save_cohort_manifests(
            target, {"random": rand_manifest, "same_category": hard_manifest}, cohort
        )
        (target / "image_sizes.json").write_text(
            json.dumps({str(k): list(v) for k, v in sorted(image_sizes.items())}, indent=2) + "\n",
            encoding="utf-8",
        )
        report["persist"] = provenance
        (target / "cohort_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    emit(
        f"[a11-cohort] pinned {len(cohort)} rows / {cohort.n_images} images / "
        f"{cohort.n_refs} refs; A10 pin ok, matched-pair ok"
    )
    return cohort, image_sizes, report


# ---------------------------------------------------------------------------
# the read-only feature corpus the frozen scorer walks
# ---------------------------------------------------------------------------
@dataclass
class RefCOCOGFeatureCorpus:
    """``B3Corpus`` duck-type over the extracted RefCOCOg cache (read-only).

    ``materialise_examples`` reads ``_region`` / ``_text`` / ``_bank`` /
    ``image_size`` and ``B3Corpus.candidate_bank_indices`` reads the sample's
    ``target_index`` / ``distractor_order``; this class supplies exactly those.
    Region / text blocks are widened to float32 the same way :class:`B3Corpus`
    does, so the frozen forward sees the identical dtype contract as on RefCOCO+.
    """

    cache_root: Path
    bank_paths: Mapping[str, Path]
    image_sizes: Mapping[int, Tuple[int, int]]
    ks: Tuple[int, ...] = (rman.A11_K,)
    regime: str = "random"

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)
        self.bank_paths = {str(k): Path(v) for k, v in dict(self.bank_paths).items()}
        self.image_sizes = {int(k): (int(v[0]), int(v[1])) for k, v in dict(self.image_sizes).items()}
        self._cache: Optional[FeatureCache] = None
        self._bank_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # -- handles -------------------------------------------------------------
    def _features(self) -> FeatureCache:
        if self._cache is None:
            self._cache = FeatureCache.open(self.cache_root)
        return self._cache

    # -- B3Corpus duck interface --------------------------------------------
    def _region(self, image_id: int) -> np.ndarray:
        return np.asarray(self._features().region_features(int(image_id)), dtype=np.float32)

    def _text(self, sentence_id: int) -> np.ndarray:
        return np.asarray(self._features().text_features(int(sentence_id)), dtype=np.float32)

    def _bank(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        image_id = int(image_id)
        hit = self._bank_cache.get(image_id)
        if hit is None:
            last: Optional[Exception] = None
            for path in self.bank_paths.values():
                try:
                    hit = read_bank_image(path, image_id)
                    break
                except KeyError as exc:
                    last = exc
            if hit is None:
                raise KeyError(f"image {image_id} in no bank; last error {last}")
            hit = (
                np.ascontiguousarray(hit[0], dtype=np.float32),
                np.ascontiguousarray(hit[1], dtype=np.float32),
            )
            self._bank_cache[image_id] = hit
        return hit

    def image_size(self, image_id: int) -> Tuple[int, int]:
        image_id = int(image_id)
        if image_id not in self.image_sizes:
            raise KeyError(f"image {image_id} has no measured size in the RefCOCOg corpus")
        return self.image_sizes[image_id]

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> "RefCOCOGFeatureCorpus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
