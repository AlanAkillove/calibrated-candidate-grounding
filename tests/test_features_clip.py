"""GPU / slow tests for the *real* frozen OpenCLIP encoder (no mocks).

The frozen backbone is ViT-B/32 ``laion2b_s34b_b79k``; the verified 605 MB
checkpoint already lives in ``cache/hf_hub`` (lfs sha256
``1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad``), so the
encoder is discovered entirely offline through :func:`build_encoder`.  On a
machine without CUDA these tests skip; on this machine (RTX 4060 Laptop) they
must genuinely run - nothing here is mocked.

Checks: output shape/dtype + L2 normalisation, the frozen crop policy
(zero-width box -> invalid, all-zero row, valid mask), determinism of a second
forward pass (< 2e-3 fp16), checkpoint sha256 provenance and a positive
``native_logit_scale``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.features.clip_encoder import (  # noqa: E402
    CROP_POLICY,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    FEATURE_DIM,
    build_encoder,
)

HF_CACHE = _REPO_ROOT / "cache" / "hf_hub"
EXPECTED_SHA256 = "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad"
NORM_TOL = 2e-2
REPEAT_TOL = 2e-3

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


requires_gpu = pytest.mark.skipif(not _cuda_available(), reason="CUDA device + real weights required")


def _synth_image(width=224, height=224, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _l2_norms(features: np.ndarray) -> np.ndarray:
    return np.linalg.norm(features.astype(np.float32), axis=1)


@pytest.fixture(scope="module")
def encoder():
    if not _cuda_available():
        pytest.skip("CUDA device required for the real-weights encoder tests")
    started = time.perf_counter()
    built = build_encoder(device="cuda", precision="fp16", cache_dir=HF_CACHE)
    loaded_seconds = time.perf_counter() - started
    print(f"\n[clip] encoder ready in {loaded_seconds:.1f}s: {built.config.checkpoint_path}")
    return built


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
@requires_gpu
def test_checkpoint_provenance_and_logit_scale(encoder):
    metadata = encoder.metadata()
    assert metadata["library"] == "open_clip_torch"
    assert metadata["library_version"]
    assert metadata["model_name"] == DEFAULT_MODEL_NAME == "ViT-B-32"
    assert metadata["pretrained"] == DEFAULT_PRETRAINED == "laion2b_s34b_b79k"
    assert metadata["embedding_dim"] == FEATURE_DIM == 512
    assert metadata["precision"] == "fp16"
    assert metadata["resolution"] == 224

    assert metadata["checkpoint_sha256"], "real checkpoint sha256 must be recorded"
    assert metadata["checkpoint_sha256"] == EXPECTED_SHA256
    assert Path(metadata["checkpoint_path"]).exists()
    assert Path(metadata["checkpoint_path"]).stat().st_size == 605219813

    assert metadata["tokenizer"]["context_length"] == 77
    assert metadata["preprocessing"]["crop_policy"] == CROP_POLICY
    assert len(metadata["preprocessing"].get("mean", [])) == 3
    assert len(metadata["preprocessing"].get("std", [])) == 3

    native_scale = encoder.logit_scale
    assert native_scale > 0.0, f"native_logit_scale must be positive, got {native_scale}"
    assert 1.0 < native_scale < 1000.0


# ---------------------------------------------------------------------------
# whole images
# ---------------------------------------------------------------------------
@requires_gpu
def test_encode_images_shape_dtype_norm(encoder):
    images = [_synth_image(seed=i) for i in range(3)]
    features = encoder.encode_images(images)
    assert features.shape == (3, FEATURE_DIM)
    assert features.dtype == np.float16
    norms = _l2_norms(features)
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), norms
    assert np.abs(features.astype(np.float32)).max() <= 1.0 + NORM_TOL


@requires_gpu
def test_encode_images_second_pass_is_stable(encoder):
    images = [_synth_image(seed=11), _synth_image(seed=12)]
    first = encoder.encode_images(images)
    second = encoder.encode_images(images)
    diff = np.abs(first.astype(np.float32) - second.astype(np.float32)).max()
    assert diff < REPEAT_TOL, f"repeat forward diff {diff} >= {REPEAT_TOL}"


# ---------------------------------------------------------------------------
# crops (frozen policy)
# ---------------------------------------------------------------------------
@requires_gpu
def test_encode_crops_invalid_mask_and_zero_rows(encoder):
    image = _synth_image(width=200, height=120, seed=3)
    boxes = np.asarray(
        [
            [10.0, 20.0, 110.0, 100.0],   # valid interior crop
            [50.0, 20.0, 50.0, 100.0],    # zero width -> invalid
            [10.0, 60.0, 110.0, 60.0],    # zero height -> invalid
            [-30.0, -30.0, 500.0, 500.0],  # clamps to the full image -> valid
            [10.4, 20.4, 11.6, 21.6],     # rounds to a 2x2 px crop -> valid
            [30.0, 30.0, 31.0, 90.0],     # 1 px thin (== min_crop_px boundary) -> valid
            [40.0, 20.0, 40.4, 100.0],    # sub-pixel width rounds to 0 -> invalid
        ],
        dtype=np.float32,
    )
    features, valid = encoder.encode_crops(image, boxes)
    assert features.shape == (7, FEATURE_DIM)
    assert features.dtype == np.float16
    assert valid.dtype == np.bool_
    assert valid.tolist() == [True, False, False, True, True, True, False]

    # invalid crops must be all-zero rows - never silently dropped or expanded
    assert float(np.abs(features[1]).max()) == 0.0
    assert float(np.abs(features[2]).max()) == 0.0
    assert float(np.abs(features[6]).max()) == 0.0
    # valid crops are L2-normalised
    norms = _l2_norms(features[valid])
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), norms
    # the clamped box equals whole-image content -> same embedding as encode_images
    whole = encoder.encode_images([image])[0].astype(np.float32)
    clamped = features[3].astype(np.float32)
    assert float(np.abs(whole - clamped).max()) <= REPEAT_TOL


@requires_gpu
def test_encode_crops_second_pass_is_stable(encoder):
    image = _synth_image(width=160, height=160, seed=5)
    boxes = np.asarray([[5.0, 5.0, 80.0, 120.0], [80.0, 5.0, 155.0, 155.0]], dtype=np.float32)
    first, mask_a = encoder.encode_crops(image, boxes)
    second, mask_b = encoder.encode_crops(image, boxes)
    assert mask_a.tolist() == mask_b.tolist() == [True, True]
    diff = np.abs(first.astype(np.float32) - second.astype(np.float32)).max()
    assert diff < REPEAT_TOL, f"repeat crop diff {diff} >= {REPEAT_TOL}"


# ---------------------------------------------------------------------------
# texts
# ---------------------------------------------------------------------------
@requires_gpu
def test_encode_texts_shape_dtype_norm_and_stability(encoder):
    texts = ["a photo of a cat", "two dogs running on the grass"]
    features = encoder.encode_texts(texts)
    assert features.shape == (2, FEATURE_DIM)
    assert features.dtype == np.float16
    norms = _l2_norms(features)
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), norms

    again = encoder.encode_texts(texts)
    diff = np.abs(features.astype(np.float32) - again.astype(np.float32)).max()
    assert diff < REPEAT_TOL, f"repeat text diff {diff} >= {REPEAT_TOL}"

    # different texts must produce different embeddings
    assert float(np.abs(features[0].astype(np.float32) - features[1].astype(np.float32)).max()) > 1e-3


@requires_gpu
def test_token_counts_bounded_by_context_length(encoder):
    counts = encoder.token_counts(["hello world", "x " * 100])
    assert counts.dtype == np.int64
    assert counts[0] > 2
    assert counts[1] > encoder.context_length == 77
