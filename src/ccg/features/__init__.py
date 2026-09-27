"""Feature extraction layer: frozen OpenCLIP encoder + crash-resumable cache.

The package stays importable without torch, open_clip, GPU or weights: all
heavy imports live inside functions (see :mod:`ccg.features.clip_encoder`).

Frozen multi-agent interface (2026-09-27):

* :class:`~ccg.features.cache.FeatureCache` / :func:`~ccg.features.cache.write_feature_cache`
  - the O(1) random-access reader and the one-shot writer of the cache layout;
* :class:`~ccg.features.clip_encoder.ClipEncoder` / :func:`~ccg.features.clip_encoder.build_encoder`
  - the frozen ViT-B/32 (laion2b_s34b_b79k) encoder with the exact-crop policy;
* :mod:`ccg.features.extract_regions` / :mod:`ccg.features.extract_text` /
  :mod:`ccg.features.extract_image` - the real extraction CLIs (orchestrated by
  ``scripts/extract_features.py``).
"""

from __future__ import annotations

from .cache import (
    CACHE_VERSION,
    GLOBAL_FILENAME,
    METADATA_FILENAME,
    REGION_FILENAME,
    TEXT_FILENAME,
    TEXT_INDEX_COLUMNS,
    TEXT_INDEX_FILENAME,
    FeatureCache,
    StreamingGlobalWriter,
    StreamingRegionWriter,
    feature_health,
    merge_extraction_stats,
    read_metadata,
    read_text_index,
    update_metadata,
    write_csv_atomic,
    write_feature_cache,
    write_global_cache,
    write_json_atomic,
    write_region_cache,
    write_text_cache,
    write_text_index,
)
from .clip_encoder import (
    CROP_POLICY,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    FEATURE_DIM,
    HF_MIRROR_ENDPOINT,
    MIN_CROP_PX,
    ClipEncoder,
    ClipEncoderConfig,
    build_encoder,
    crop_image_at_boxes,
    describe_preprocessing,
    l2_normalize,
    load_rgb_image,
    proposal_crop_boxes,
    resolve_hf_endpoint,
)

__all__ = [
    # storage (frozen)
    "FeatureCache",
    "write_feature_cache",
    "CACHE_VERSION",
    "REGION_FILENAME",
    "TEXT_FILENAME",
    "GLOBAL_FILENAME",
    "METADATA_FILENAME",
    "TEXT_INDEX_FILENAME",
    "TEXT_INDEX_COLUMNS",
    "write_region_cache",
    "write_text_cache",
    "write_global_cache",
    "write_text_index",
    "read_text_index",
    "read_metadata",
    "update_metadata",
    "merge_extraction_stats",
    "feature_health",
    "write_json_atomic",
    "write_csv_atomic",
    "StreamingRegionWriter",
    "StreamingGlobalWriter",
    # encoder (frozen)
    "FEATURE_DIM",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PRETRAINED",
    "MIN_CROP_PX",
    "CROP_POLICY",
    "HF_MIRROR_ENDPOINT",
    "ClipEncoder",
    "ClipEncoderConfig",
    "build_encoder",
    "l2_normalize",
    "resolve_hf_endpoint",
    "proposal_crop_boxes",
    "crop_image_at_boxes",
    "describe_preprocessing",
    "load_rgb_image",
]
