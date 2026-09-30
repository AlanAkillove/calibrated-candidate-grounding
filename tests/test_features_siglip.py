"""Real-weights tests for the SigLIP (transformers) backbone backend.

Nothing here is mocked: the encoder loads ``cache/hf_local/siglip_b16_224``
offline via :func:`ccg.features.backbone.build_siglip_encoder`.  The tests run
on CUDA when available and otherwise fall back to CPU (the preprocessing path
under test is device-independent), so they guard the regression even without a
GPU.  Marked ``slow`` because a real ~775 MB model is loaded once per module.

Motivating regression (V2-G G1): the *slow* SigLIP image processor mis-infers
the channel dimension of a batch that contains a degenerate crop - a proposal
whose width or height rounds to ``1``, or a ``3 x W`` thin crop whose numpy
array is ``(3, W, 3)`` - and raises ``TypeError: Cannot handle this data
type: (1, 1, <W>), |u1``.  ``encode_images`` now pins ``channels_last``; every
real crop is HWC, so this is behaviour-preserving for the unambiguous images
that already streamed successfully.
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

from ccg.features.backbone import BACKEND_SIGLIP, build_siglip_encoder  # noqa: E402

SIGLIP_LOCAL = _REPO_ROOT / "cache" / "hf_local" / "siglip_b16_224"
EMBED_DIM = 768
NORM_TOL = 2e-2

pytestmark = [pytest.mark.slow]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


requires_weights = pytest.mark.skipif(
    not SIGLIP_LOCAL.exists() or not _torch_available(),
    reason="SigLIP local weights (cache/hf_local/siglip_b16_224) + torch required",
)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _crop(width: int, height: int):
    """A real PIL crop (as produced by the frozen crop policy) of exact size."""
    from PIL import Image

    base = Image.new("RGB", (max(width, 1) + 8, max(height, 1) + 8))
    return base.crop((0, 0, width, height))


def _l2_norms(features: np.ndarray) -> np.ndarray:
    return np.linalg.norm(features.astype(np.float32), axis=1)


@pytest.fixture(scope="module")
def encoder():
    if not SIGLIP_LOCAL.exists():
        pytest.skip(f"SigLIP weights not present at {SIGLIP_LOCAL}")
    started = time.perf_counter()
    built = build_siglip_encoder(
        device=_device(), precision="fp32", model_path=str(SIGLIP_LOCAL), compute_sha256=False
    )
    print(f"\n[siglip] encoder ready in {time.perf_counter() - started:.1f}s on {built.config.device}")
    return built


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
@requires_weights
def test_siglip_provenance(encoder):
    md = encoder.metadata()
    assert md["library"] == "transformers"
    assert md["backend"] == BACKEND_SIGLIP
    assert md["embedding_dim"] == encoder.feature_dim == EMBED_DIM
    assert md["objective"] == "sigmoid"
    assert md["l2_normalized"] is True
    assert md["resolution"] == 224
    assert md["tokenizer"]["context_length"] == encoder.context_length == 64
    # SigLIP carries a learned logit bias (absent for a CLIP-style backbone)
    assert md["native_logit_scale"] > 0.0
    assert np.isfinite(md["native_logit_bias"])
    assert Path(md["checkpoint_path"]).exists()


# ---------------------------------------------------------------------------
# the regression: degenerate crops must not crash the slow processor
# ---------------------------------------------------------------------------
@requires_weights
def test_encode_images_degenerate_crops_do_not_crash(encoder):
    images = [
        _crop(224, 224),
        _crop(5, 3),      # numpy (3, 5, 3): h == c ambiguity
        _crop(3, 5),
        _crop(1, 217),    # width-1 sliver -> (1, 1, 217) under channels-first
        _crop(217, 1),
        _crop(3, 3),
        _crop(1, 1),
    ]
    features = encoder.encode_images(images, batch_size=len(images))
    assert features.shape == (len(images), EMBED_DIM)
    assert features.dtype == np.float16
    assert np.all(np.isfinite(features.astype(np.float32)))
    norms = _l2_norms(features)
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), norms


@requires_weights
def test_encode_crops_thin_boxes_are_valid_rows(encoder):
    """A 1-px and a 3-px-tall proposal go through the frozen crop policy and
    return encoded (non-zero) rows rather than raising."""
    image = _crop(300, 200)
    boxes = np.asarray(
        [
            [10.0, 20.0, 110.0, 100.0],  # ordinary interior crop
            [40.0, 30.0, 41.0, 150.0],   # 1 px wide sliver -> valid, thin
            [60.0, 40.0, 200.0, 43.0],   # 3 px tall -> (3, W, 3) ambiguity
        ],
        dtype=np.float32,
    )
    features, valid = encoder.encode_crops(image, boxes)
    assert features.shape == (3, EMBED_DIM)
    assert valid.all()
    for row in features:
        assert float(np.abs(row).max()) > 0.0
    norms = _l2_norms(features)
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), norms


# ---------------------------------------------------------------------------
# texts + truncation provenance
# ---------------------------------------------------------------------------
@requires_weights
def test_encode_texts_and_truncation(encoder):
    short = "a photo of a cat"
    long = " ".join(["word"] * 200)
    features = encoder.encode_texts([short, long])
    assert features.shape == (2, EMBED_DIM)
    norms = _l2_norms(features)
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), norms

    counts = encoder.token_counts([short, long])
    assert counts.dtype == np.int64
    # the long text exceeds the 64-token context -> the stored embedding is
    # truncated (recorded as provenance), while the raw tokenizer count is high
    assert counts[0] <= encoder.context_length
    assert counts[1] > encoder.context_length


@requires_weights
def test_encode_images_second_pass_is_stable(encoder):
    images = [_crop(224, 224), _crop(5, 3)]
    first = encoder.encode_images(images)
    second = encoder.encode_images(images)
    diff = np.abs(first.astype(np.float32) - second.astype(np.float32)).max()
    assert diff < 2e-2, f"repeat forward diff {diff} too large"
