"""RPN proposal extraction: ranking, coordinate mapping and live-model smoke.

Three tiers:

* pure numpy / torchvision arithmetic (fast, offline): ``_sort_and_truncate``
  tie stability and truncation; ``inverse_resized_boxes`` known values and
  bit-for-bit agreement with torchvision's own ``resize_boxes`` /
  ``GeneralizedRCNNTransform``;
* CPU integration with randomly-initialised weights (fast, offline): the
  frozen API contract, ``meta`` bookkeeping, determinism, clipping and
  box-for-box agreement with the stock ``model.rpn(...)`` call;
* CUDA integration with the COCO_V1 checkpoint (skipped when no GPU): the
  480x640 synthetic-image smoke and a real-image smoke from ``data/raw``
  once one has been downloaded.  GPU determinism is pinned at
  ``atol=1e-4`` - with fp16 autocast two calls are bit-identical in practice
  but only guaranteed at ULP level (see the ``ccg.data.rpn`` docstring).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes

from ccg.data.rpn import (
    RPNProposals,
    _sort_and_truncate,
    as_image_tensor,
    build_rpn_model,
    extract_proposal_bank,
    extract_proposals,
    inverse_resized_boxes,
    resized_boxes_to_original,
)

HAS_CUDA = torch.cuda.is_available()
NEEDS_CUDA = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

FROZEN_META_KEYS = (
    "original_hw",
    "resized_hw",
    "n_raw_post_nms",
    "nms_thresh",
    "pre_nms_top_n",
    "post_nms_top_n",
    "torchvision_version",
    "truncated_to",
    "model_name",
)


# ---------------------------------------------------------------------------
# helpers and fixtures
# ---------------------------------------------------------------------------
def synthetic_image(height: int = 240, width: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def lexsorted_rows(boxes: np.ndarray) -> np.ndarray:
    """Row-order-free comparison of two box sets."""
    return boxes[np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))]


def downloaded_images(limit: int = 1):
    root = REPO_ROOT / "data" / "raw"
    if not root.is_dir():
        return []
    found = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return found[:limit]


def device_of(module) -> torch.device:
    return next(module.parameters()).device


@pytest.fixture(scope="module")
def cpu_model():
    """Random-weights detector with a small resize window: fast and offline."""
    return build_rpn_model(device="cpu", weights=None, min_size=256, max_size=512)


@pytest.fixture(scope="module")
def cuda_model():
    """COCO_V1 checkpoint on CUDA (downloads ~160 MB into TORCH_HOME on first use)."""
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    return build_rpn_model(device="cuda", weights="COCO_V1")


# ---------------------------------------------------------------------------
# coordinate mapping (pure arithmetic)
# ---------------------------------------------------------------------------
def test_inverse_resized_boxes_identity():
    boxes = np.array([[0.0, 0.0, 10.0, 20.0], [3.5, 4.5, 100.25, 200.0]], dtype=np.float32)
    out = inverse_resized_boxes(boxes, (480, 640), (480, 640))
    np.testing.assert_array_equal(out, boxes)
    assert out.dtype == np.float32
    assert out is not boxes  # never aliases the caller's array


def test_inverse_resized_boxes_min_size_scaling_exact():
    # 400x600 scaled by 2.0 to min_size=800 -> 800x1200; the inverse divides by 2
    boxes = np.array([[0.0, 0.0, 40.0, 40.0], [100.0, 125.0, 1000.0, 775.0]], dtype=np.float32)
    out = inverse_resized_boxes(boxes, (400, 600), (800, 1200))
    np.testing.assert_array_equal(out, boxes * np.float32(0.5))


def test_inverse_resized_boxes_max_size_non_square_known_values():
    # 600x1800 cannot reach min_size=800 without busting max_size=1333: the
    # long edge is capped and the transform emits (444, 1333) - asserted
    # against the live transform in test_generalized_rcnn_transform_sizes.
    o_hw, r_hw = (600, 1800), (444, 1333)
    boxes = np.array([[0.0, 0.0, 1333.0, 444.0], [10.0, 20.0, 1332.5, 443.5]], dtype=np.float32)
    out = inverse_resized_boxes(boxes, o_hw, r_hw)
    expected = boxes.astype(np.float64)
    expected[:, [0, 2]] *= 1800.0 / 1333.0
    expected[:, [1, 3]] *= 600.0 / 444.0
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-4)


def test_inverse_resized_boxes_one_pixel_boundary():
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
    out = inverse_resized_boxes(boxes, (100, 200), (50, 100))
    expected = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "o_hw, r_hw",
    [
        ((400, 600), (800, 1200)),
        ((480, 640), (800, 1066)),
        ((600, 1800), (444, 1333)),
        ((100, 200), (50, 100)),
    ],
)
def test_inverse_resized_boxes_matches_torchvision_resize_boxes(o_hw, r_hw):
    # torchvision's postprocess maps resized -> original with exactly this
    # call: resize_boxes(boxes, original_size=resized, new_size=original).
    rng = np.random.RandomState(0)
    boxes = rng.rand(7, 4).astype(np.float32) * np.array(
        [r_hw[1], r_hw[0], r_hw[1], r_hw[0]], dtype=np.float32
    )
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    ours = inverse_resized_boxes(boxes, o_hw, r_hw)
    tv = resize_boxes(torch.from_numpy(boxes), list(r_hw), list(o_hw)).numpy()
    # same IEEE float32 multiplications in the same order - must be bit-identical
    np.testing.assert_array_equal(ours, tv)


def test_resized_boxes_to_original_is_the_forward_direction():
    o_hw, r_hw = (480, 640), (800, 1066)
    rng = np.random.RandomState(3)
    boxes = rng.rand(5, 4).astype(np.float32) * 500.0
    resized = resized_boxes_to_original(boxes, o_hw, r_hw)
    np.testing.assert_array_equal(
        resized, resize_boxes(torch.from_numpy(boxes), list(o_hw), list(r_hw)).numpy()
    )
    back = inverse_resized_boxes(resized, o_hw, r_hw)
    np.testing.assert_allclose(back, boxes, rtol=0.0, atol=1e-4)


def test_inverse_resized_boxes_rejects_bad_shapes():
    with pytest.raises(ValueError):
        inverse_resized_boxes(np.zeros((3, 3), dtype=np.float32), (10, 10), (5, 5))
    with pytest.raises(ValueError):
        inverse_resized_boxes(np.zeros((2, 4), dtype=np.float32), (0, 10), (5, 5))


def test_generalized_rcnn_transform_sizes():
    """The resize behaviour inverse_resized_boxes assumes (live torchvision)."""
    transform = GeneralizedRCNNTransform(
        min_size=800, max_size=1333, image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0]
    )
    # min_size scaling, exact ratio
    small, _ = transform([torch.zeros(3, 400, 600)], None)
    assert tuple(small.image_sizes[0]) == (800, 1200)
    # non-square, min_size on the short side (floor of 640 * 800/480)
    mid, _ = transform([torch.zeros(3, 480, 640)], None)
    assert tuple(mid.image_sizes[0]) == (800, 1066)
    # max_size caps the long edge instead
    tall, _ = transform([torch.zeros(3, 600, 1800)], None)
    assert tuple(tall.image_sizes[0]) == (444, 1333)


# ---------------------------------------------------------------------------
# ranking / truncation (pure numpy)
# ---------------------------------------------------------------------------
def test_sort_and_truncate_stable_ties():
    scores = np.array([0.4, 0.9, 0.4, 0.9, 0.1], dtype=np.float32)
    boxes = np.arange(20, dtype=np.float32).reshape(5, 4)
    out, sc, truncated = _sort_and_truncate(boxes, scores, None)
    np.testing.assert_array_equal(sc, np.array([0.9, 0.9, 0.4, 0.4, 0.1], dtype=np.float32))
    # ties keep the input row order: indices [1, 3] then [0, 2] then [4]
    np.testing.assert_array_equal(out[:, 0], np.array([4.0, 12.0, 0.0, 8.0, 16.0]))
    assert truncated is False


def test_sort_and_truncate_top_n_64_and_128():
    rng = np.random.RandomState(7)
    scores = rng.rand(200).astype(np.float32)
    boxes = rng.rand(200, 4).astype(np.float32) * 100.0
    boxes[:, 0] = scores  # row identity travels with the box
    for n in (64, 128):
        out, sc, truncated = _sort_and_truncate(boxes, scores, n)
        assert out.shape == (n, 4) and sc.shape == (n,)
        assert truncated is True
        assert np.all(np.diff(sc) <= 0)
        # rows stay paired with their scores after the reorder
        np.testing.assert_array_equal(out[:, 0], sc)

    out_all, sc_all, truncated_all = _sort_and_truncate(boxes, scores, None)
    assert out_all.shape == (200, 4) and truncated_all is False
    assert np.all(np.diff(sc_all) <= 0)
    # the truncated ranking is exactly the prefix of the full ranking
    np.testing.assert_array_equal(sc_all[:128], _sort_and_truncate(boxes, scores, 128)[1])
    assert out_all.dtype == np.float32 and sc_all.dtype == np.float32


def test_sort_and_truncate_short_input_not_padded():
    scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    boxes = np.zeros((3, 4), dtype=np.float32)
    out, sc, truncated = _sort_and_truncate(boxes, scores, 64)
    # fewer rows than top_n: returned as-is (sorted), never padded
    assert out.shape == (3, 4) and sc.shape == (3,)
    assert truncated is False
    np.testing.assert_array_equal(sc, np.array([0.9, 0.5, 0.2], dtype=np.float32))


def test_sort_and_truncate_row_mismatch_raises():
    with pytest.raises(ValueError):
        _sort_and_truncate(np.zeros((3, 4), np.float32), np.zeros(2, np.float32), 2)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def test_build_rpn_model_defaults_are_frozen(cpu_model):
    assert cpu_model.training is False
    assert tuple(cpu_model.transform.min_size) == (256,)
    assert int(cpu_model.transform.max_size) == 512
    # COCO_V1 testing defaults (faster_rcnn.py L179-188 in torchvision 0.20.1)
    assert cpu_model.rpn.pre_nms_top_n() == 1000
    assert cpu_model.rpn.post_nms_top_n() == 1000
    assert cpu_model.rpn.nms_thresh == pytest.approx(0.7)
    assert cpu_model.rpn.score_thresh == pytest.approx(0.0)
    assert cpu_model.ccg_model_name == "fasterrcnn_resnet50_fpn(random)"


def test_build_rpn_model_overrides_propagate_to_extraction():
    model = build_rpn_model(
        "cpu", None, 256, 512, rpn_nms_thresh=0.5, pre_nms_top_n=200, post_nms_top_n=40
    )
    # the private dicts are rebuilt through the public constructor, not patched
    assert model.rpn.pre_nms_top_n() == 200
    assert model.rpn.post_nms_top_n() == 40
    assert model.rpn.nms_thresh == pytest.approx(0.5)

    out = extract_proposals(synthetic_image(160, 200, seed=5), model, top_n=64)
    assert out.meta["pre_nms_top_n"] == 200
    assert out.meta["post_nms_top_n"] == 40
    assert out.meta["nms_thresh"] == pytest.approx(0.5)
    assert out.meta["n_raw_post_nms"] <= 40
    # top_n=64 > the 40 post-NMS cap: everything is returned, nothing padded
    assert out.boxes.shape[0] == out.meta["n_raw_post_nms"]
    assert out.meta["truncated"] is False


def test_build_rpn_model_rejects_unknown_weights():
    with pytest.raises(ValueError):
        build_rpn_model(device="cpu", weights="no-such-weights")


# ---------------------------------------------------------------------------
# CPU integration (random weights, no download)
# ---------------------------------------------------------------------------
def test_extract_cpu_contract_determinism_and_nesting(cpu_model):
    image = synthetic_image(240, 320, seed=0)
    a = extract_proposals(image, cpu_model, 64)
    b = extract_proposals(image, cpu_model, top_n=64)
    c = extract_proposals(image, cpu_model, top_n=16)

    assert isinstance(a, RPNProposals)
    boxes, objectness, meta = a.boxes, a.objectness, a.meta
    assert boxes.dtype == np.float32 and boxes.ndim == 2 and boxes.shape[1] == 4
    k = boxes.shape[0]
    assert k >= 1
    assert k == min(64, meta["n_raw_post_nms"])
    assert objectness.shape == (k,) and objectness.dtype == np.float32
    assert np.all(objectness >= 0.0) and np.all(objectness <= 1.0)
    assert np.all(np.diff(objectness) <= 0.0)

    h, w = 240, 320
    assert np.all(boxes[:, 0] >= 0.0) and np.all(boxes[:, 2] <= float(w))
    assert np.all(boxes[:, 1] >= 0.0) and np.all(boxes[:, 3] <= float(h))
    assert np.all(boxes[:, 2] >= boxes[:, 0]) and np.all(boxes[:, 3] >= boxes[:, 1])

    for key in FROZEN_META_KEYS:
        assert key in meta
    assert meta["original_hw"] == (h, w)
    assert meta["resized_hw"] != (h, w)  # min_size scaling really happened
    # the recorded resized size is what the live transform produced
    with torch.no_grad():
        images, _ = cpu_model.transform([as_image_tensor(image)], None)
    assert meta["resized_hw"] == tuple(images.image_sizes[0])
    assert meta["truncated_to"] == 64
    assert meta["truncated"] == (meta["n_raw_post_nms"] > 64)
    assert meta["num_returned"] == k
    assert meta["torchvision_version"] == torchvision.__version__
    assert meta["nms_thresh"] == pytest.approx(0.7)
    assert meta["pre_nms_top_n"] == 1000 and meta["post_nms_top_n"] == 1000
    assert meta["model_name"] == "fasterrcnn_resnet50_fpn(random)"
    assert meta["autocast_effective"] is False  # no-op on CPU

    # determinism: identical input, identical bits (CPU fp32)
    np.testing.assert_array_equal(a.boxes, b.boxes)
    np.testing.assert_array_equal(a.objectness, b.objectness)

    # truncation is a prefix of the (deterministic) full ranking
    np.testing.assert_array_equal(c.boxes, a.boxes[:16])
    np.testing.assert_array_equal(c.objectness, a.objectness[:16])
    assert c.meta["truncated_to"] == 16


def test_extract_proposal_bank_attaches_gt_ious(cpu_model):
    gt = np.array([[0.0, 0.0, 160.0, 120.0], [200.0, 100.0, 300.0, 220.0]], dtype=np.float32)
    bank, meta = extract_proposal_bank(
        synthetic_image(240, 320, seed=4),
        cpu_model,
        42,
        top_n=8,
        gt_boxes=gt,
        gt_object_ids=np.array([7, 9]),
    )
    assert bank.image_id == 42
    assert bank.boxes.shape == (8, 4) and bank.boxes.dtype == np.float32
    assert bank.objectness.shape == (8,)
    assert bank.gt_ious.shape == (8,) and bank.gt_assignment.shape == (8,)
    assert np.all(bank.gt_ious >= 0.0) and np.all(bank.gt_ious <= 1.0)
    assert set(np.unique(bank.gt_assignment)) <= {7, 9, -1}
    assert meta["truncated_to"] == 8


def test_extract_matches_official_rpn_cpu(cpu_model):
    """The public-submodule replay reproduces the stock model.rpn(...) call."""
    image = synthetic_image(240, 320, seed=2)
    result = extract_proposals(image, cpu_model, top_n=None)

    tensor = as_image_tensor(image)
    with torch.no_grad():
        images, _ = cpu_model.transform([tensor], None)
        features = cpu_model.backbone(images.tensors)
        official, losses = cpu_model.rpn(images, features)
    assert losses == {}  # eval mode
    official_boxes = official[0].detach().cpu().numpy()

    assert result.meta["truncated_to"] is None
    assert result.meta["n_raw_post_nms"] == official_boxes.shape[0]
    assert result.boxes.shape[0] == official_boxes.shape[0]

    mine_in_resized = resized_boxes_to_original(
        result.boxes, result.meta["original_hw"], result.meta["resized_hw"]
    )
    np.testing.assert_allclose(
        lexsorted_rows(mine_in_resized), lexsorted_rows(official_boxes), atol=1e-3
    )


# ---------------------------------------------------------------------------
# CUDA integration (COCO_V1 weights)
# ---------------------------------------------------------------------------
@NEEDS_CUDA
def test_extract_cuda_contract_and_determinism(cuda_model):
    image = synthetic_image(480, 640, seed=0)
    a = extract_proposals(image, cuda_model)  # frozen defaults: top_n=128, autocast=True
    b = extract_proposals(image, cuda_model)
    c = extract_proposals(image, cuda_model, top_n=32)

    boxes, objectness, meta = a.boxes, a.objectness, a.meta
    k = boxes.shape[0]
    assert k == min(128, meta["n_raw_post_nms"]) and k >= 1
    assert np.all(np.diff(objectness) <= 0.0)
    h, w = 480, 640
    assert np.all(boxes[:, 0] >= 0.0) and np.all(boxes[:, 2] <= float(w))
    assert np.all(boxes[:, 1] >= 0.0) and np.all(boxes[:, 3] <= float(h))
    assert meta["original_hw"] == (h, w)
    assert meta["truncated_to"] == 128
    assert meta["autocast_effective"] is True
    assert meta["device"].startswith("cuda")
    assert meta["model_name"] == "fasterrcnn_resnet50_fpn(COCO_V1)"
    assert meta["torchvision_version"] == torchvision.__version__
    for key in FROZEN_META_KEYS:
        assert key in meta

    with torch.no_grad():
        images, _ = cuda_model.transform([as_image_tensor(image).to(device_of(cuda_model))], None)
    assert meta["resized_hw"] == tuple(images.image_sizes[0])

    # fp16 autocast on GPU: bit-identical in practice, pinned at ULP level
    np.testing.assert_allclose(a.boxes, b.boxes, atol=1e-4)
    np.testing.assert_allclose(a.objectness, b.objectness, atol=1e-4)
    assert a.meta["n_raw_post_nms"] == b.meta["n_raw_post_nms"]
    np.testing.assert_allclose(c.boxes, a.boxes[:32], atol=1e-4)
    np.testing.assert_allclose(c.objectness, a.objectness[:32], atol=1e-4)

    print(
        f"\n[RPN smoke] synthetic 480x640: K={k}, n_raw={meta['n_raw_post_nms']}, "
        f"elapsed_ms={meta['elapsed_ms']:.1f}, top5={np.round(objectness[:5], 4).tolist()}"
    )


@NEEDS_CUDA
def test_extract_real_image_smoke(cuda_model):
    images = downloaded_images()
    if not images:
        pytest.skip("no image downloaded under data/raw yet")
    path = images[0]

    out = extract_proposals(path, cuda_model, top_n=64)
    assert out.boxes.shape[0] >= 1
    h, w = out.meta["original_hw"]
    assert np.all(out.boxes[:, [0, 2]] >= 0.0) and np.all(out.boxes[:, [0, 2]] <= float(w))
    assert np.all(out.boxes[:, [1, 3]] >= 0.0) and np.all(out.boxes[:, [1, 3]] <= float(h))

    print(
        f"\n[RPN smoke] {path.name} (WxH={w}x{h}): K={out.boxes.shape[0]}, "
        f"n_raw={out.meta['n_raw_post_nms']}, elapsed_ms={out.meta['elapsed_ms']:.1f}, "
        f"top5={np.round(out.objectness[:5], 4).tolist()}"
    )
