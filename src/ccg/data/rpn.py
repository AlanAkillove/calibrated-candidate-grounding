"""Class-agnostic RPN proposal extraction wrapper (torchvision Faster R-CNN).

The perception layer of the protocol (section 4) is *one* frozen,
query-independent detector producing a bank of ``N`` proposals per image.
This module wraps ``torchvision``'s ``fasterrcnn_resnet50_fpn`` and uses only
its **RPN stage** (``transform`` -> ``backbone`` -> proposal network), which
is class-agnostic by construction: the RPN head predicts a single objectness
logit per anchor and never sees a class label.

Coordinate space (source-verified against torchvision 0.20.1)
-------------------------------------------------------------
:class:`RPNProposals` boxes are delivered in **original-image pixel
coordinates**, but the raw RPN output is *not*: it lives in the **resized**
(post-transform) space.  Evidence from the installed source:

* ``GeneralizedRCNN.forward``
  (``torchvision/models/detection/generalized_rcnn.py`` L104-106) feeds
  ``self.rpn(images, features, targets)`` straight into
  ``self.roi_heads(features, proposals, images.image_sizes, targets)`` and
  maps *only* the roi-head detections back, via
  ``self.transform.postprocess(detections, images.image_sizes,
  original_image_sizes)`` (L106).  The RPN boxes are never post-processed.
* ``RegionProposalNetwork.filter_proposals``
  (``torchvision/models/detection/rpn.py`` L242-297) clips the boxes with
  ``clip_boxes_to_image(boxes, img_shape)`` (L277) where ``img_shape`` is the
  *resized* image size from ``ImageList.image_sizes``.  That list is filled
  after resizing (``transform.py`` L147) and excludes the ``size_divisible``
  padding, which ``batch_images`` writes only to the bottom/right
  (``transform.py`` L148, L237-255, esp. L253) - so padding can never shift a
  box coordinate, and only the per-axis rescale matters.
* The resize uses ``scale = min(min_size / min(h, w), max_size / max(h, w))``
  (``transform.py`` L59-63) and the inverse of the mapping is exactly what
  ``postprocess`` -> ``resize_boxes`` applies in the forward direction
  (``transform.py`` L257-274 and L306-319), i.e. per-axis ratio
  ``original / resized`` of the *actual* ``ImageList`` sizes.  Using the
  actual post-resize sizes (instead of re-deriving the scale from
  min/max_size) also folds in the ``floor`` behaviour of the interpolate
  kernel, so non-square images and the ``max_size`` cap are handled
  identically.

torchvision 0.20.1 has **no** ``GeneralizedRCNNTransform.box_trans``
attribute (it is introduced in later releases; a full-package search finds no
occurrence), so on this version the only coordinate mapping is the
``postprocess`` call above.  :func:`inverse_resized_boxes` implements its
inverse, and :func:`extract_proposals` additionally clips the mapped boxes to
``[0, W] x [0, H]``.

The objectness scores: 0.20.1's ``RegionProposalNetwork.forward`` ends with
``return boxes, losses`` (rpn.py L388) - the sigmoid objectness computed in
``filter_proposals`` (L272) dies inside the module.  To recover the scores
**without forward hooks and without patching private attributes**, this
module replays the last steps of ``forward`` through the RPN's public parts
(``head``, ``anchor_generator``, ``box_coder``, ``filter_proposals`` and the
module-level ``concat_box_prediction_layers``), exactly as rpn.py L359-373
do; ``filter_proposals`` still performs the per-level top-k, clipping, small
box removal, ``score_thresh`` cut and ``batched_nms``.  The unit tests pin
the resulting boxes against the stock ``model.rpn(...)`` output row-for-row.

Version dependency
------------------
* Behaviour measured against ``torchvision==0.20.1+cu121`` / ``torch==2.5.1``
  (the RPN module lives at ``torchvision/models/detection/rpn.py`` in this
  release; it was ``.../detection/rpn.py`` for a long time, but older/newer
  releases may move or alter it).
* The COCO_V1 inference hyper-parameters
  (``rpn_pre_nms_top_n_test = rpn_post_nms_top_n_test = 1000``,
  ``rpn_nms_thresh = 0.7``, ``rpn_score_thresh = 0.0``) are the constructor
  defaults of ``faster_rcnn.py`` L179-188.  **These defaults may differ in
  other torchvision versions**; every ``meta`` records the values actually
  used, and cached banks must not be compared across versions without
  checking them (the ``torchvision_version`` key is there for exactly this).
* If a future torchvision changes the RPN API (module layout,
  ``filter_proposals`` signature, forward return values), the replay fails
  loudly (error / failing consistency test) instead of silently producing
  wrong coordinates.

Determinism: ``eval()`` + ``torch.no_grad()``, one image per call, float32
outputs, and a stable descending-objectness sort with ties broken by the
detector's own row order.  With ``autocast=True`` on CUDA the
transform/backbone run in fp16, so two calls on the same image are
bit-identical in practice but only *guaranteed* to ``atol=1e-4``; the tests
pin exactly that bound.  Fewer raw proposals than ``top_n`` are returned
as-is: the bank is never padded and never silently filtered.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_MIN_SIZE",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_NMS_THRESH",
    "DEFAULT_SCORE_THRESH",
    "DEFAULT_TOP_N",
    "MODEL_NAME",
    "RPNProposals",
    "RPNProposalCountError",
    "build_rpn_model",
    "as_image_tensor",
    "extract_proposals",
    "extract_proposal_bank",
    "inverse_resized_boxes",
    "resized_boxes_to_original",
]

#: FROZEN inference defaults of ``fasterrcnn_resnet50_fpn`` (COCO_V1) as set by
#: the constructor defaults in ``torchvision/models/detection/faster_rcnn.py``
#: L179-188 (torchvision 0.20.1): pre/post-NMS top-n (testing) = 1000, NMS
#: threshold = 0.7, score threshold = 0.0, transform 800/1333.  Other versions
#: may pick different values - see the module docstring.
DEFAULT_MIN_SIZE = 800
DEFAULT_MAX_SIZE = 1333
DEFAULT_NMS_THRESH = 0.7
DEFAULT_SCORE_THRESH = 0.0
#: Proposal-bank size N used by the audit (truncation of the objectness ranking).
DEFAULT_TOP_N = 128
#: Identifier of the detector family wrapped here (recorded in ``meta``).
MODEL_NAME = "fasterrcnn_resnet50_fpn"


@dataclass
class RPNProposals:
    """Output contract of :func:`extract_proposals` (frozen).

    Attributes
    ----------
    boxes:
        ``[K, 4]`` float32 ``xyxy`` in **original image pixel coordinates**,
        clipped to ``[0, W] x [0, H]``, ordered by descending ``objectness``.
    objectness:
        ``[K]`` float32 sigmoid objectness in ``[0, 1]``, same order.  ``K``
        may be smaller than the requested ``top_n`` when the RPN produced
        fewer proposals - the bank is never padded.
    meta:
        Bookkeeping dict.  Frozen keys: ``original_hw``, ``resized_hw``,
        ``n_raw_post_nms``, ``nms_thresh``, ``pre_nms_top_n``,
        ``post_nms_top_n``, ``torchvision_version``, ``truncated_to``,
        ``model_name``; plus extras documented in :func:`extract_proposals`.
    """

    boxes: np.ndarray
    objectness: np.ndarray
    meta: Dict[str, Any]


class RPNProposalCountError(RuntimeError):
    """The RPN returned zero proposals - a run-level error, not a filterable row."""


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def build_rpn_model(
    device: str = "cuda",
    weights: Any = "COCO_V1",
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    rpn_nms_thresh: float = DEFAULT_NMS_THRESH,
    rpn_score_thresh: float = DEFAULT_SCORE_THRESH,
    pre_nms_top_n: Optional[int] = None,
    post_nms_top_n: Optional[int] = None,
) -> Any:
    """Load ``fasterrcnn_resnet50_fpn`` and pin its RPN inference behaviour.

    Parameters
    ----------
    device:
        ``"cuda"`` (default - the frozen protocol runs the detector on GPU) or
        ``"cpu"`` / any ``torch.device``-compatible string.  No silent
        fallback: an unavailable device raises, as it should for an
        experiment.
    weights:
        ``"COCO_V1"`` (default, the frozen COCO 2017 detection checkpoint),
        ``None`` (random initialisation - used by fast unit tests, never by
        an experiment), or a ``FasterRCNN_ResNet50_FPN_Weights`` member.
        First use downloads ~160 MB into ``TORCH_HOME``.
    min_size, max_size:
        ``GeneralizedRCNNTransform`` resize targets; they define the resized
        coordinate space the RPN works in (see :func:`inverse_resized_boxes`).
    rpn_nms_thresh, rpn_score_thresh:
        ``RegionProposalNetwork`` inference thresholds.  Both are **public**
        attributes in torchvision 0.20.1 and are adjusted in place.
        ``rpn_score_thresh > 0`` removes proposals - forbidden by the
        protocol's "no silent filtering" rule, so it must stay ``0.0`` in
        real runs.
    pre_nms_top_n, post_nms_top_n:
        Top-k per FPN level before NMS and top-k after NMS (``None`` = keep
        the COCO_V1 testing defaults, 1000/1000).  In 0.20.1 these live in
        the private dicts ``_pre_nms_top_n`` / ``_post_nms_top_n``, which must
        **not** be monkeypatched; when an override actually changes a value
        the RPN is rebuilt through its public constructor reusing the loaded
        ``anchor_generator`` and ``head`` (no weights move - the RPN module
        itself owns no parameters).  The training-side constructor arguments
        of that rebuild are pinned to the COCO_V1 training defaults (they are
        never read in eval mode).

    Returns
    -------
    torch.nn.Module
        The ``GeneralizedRCNN`` instance in ``eval()`` mode with
        ``transform`` / ``backbone`` / ``rpn`` ready for inference; its
        ``ccg_model_name`` attribute labels it for ``meta``.  ``roi_heads``
        is kept untouched and never called.
    """
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.rpn import RegionProposalNetwork

    resolved_weights = _resolve_weights(weights)
    model = fasterrcnn_resnet50_fpn(weights=resolved_weights, weights_backbone=None)

    # eval() first: the RPN accessors below return the *testing* values only
    # once the module is out of training mode.
    model.eval()

    model.transform.min_size = (int(min_size),)
    model.transform.max_size = int(max_size)

    rpn = model.rpn
    default_pre = int(rpn.pre_nms_top_n())
    default_post = int(rpn.post_nms_top_n())
    pre = default_pre if pre_nms_top_n is None else int(pre_nms_top_n)
    post = default_post if post_nms_top_n is None else int(post_nms_top_n)

    # nms_thresh / score_thresh are public attributes (set in RPN.__init__).
    rpn.nms_thresh = float(rpn_nms_thresh)
    rpn.score_thresh = float(rpn_score_thresh)

    if pre != default_pre or post != default_post:
        # _pre_nms_top_n / _post_nms_top_n are private and must not be
        # patched; rebuild instead and re-attach the *loaded* sub-modules.
        model.rpn = RegionProposalNetwork(
            rpn.anchor_generator,
            rpn.head,
            fg_iou_thresh=0.7,  # training-only args, pinned to COCO_V1 defaults
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": pre, "testing": pre},
            post_nms_top_n={"training": post, "testing": post},
            nms_thresh=float(rpn_nms_thresh),
            score_thresh=float(rpn_score_thresh),
        )

    model.to(torch.device(device))
    model.eval()

    label = "random" if resolved_weights is None else resolved_weights.name
    model.ccg_model_name = f"{MODEL_NAME}({label})"
    return model


def _resolve_weights(weights: Any) -> Any:
    """Accept ``None`` / ``"COCO_V1"`` / an enum member for the weights arg."""
    if weights is None:
        return None
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as _W

    if isinstance(weights, _W):
        return weights
    key = str(weights).strip().upper()
    if key in {"DEFAULT", "COCO", "COCO_V1"}:
        return _W.COCO_V1
    try:
        return _W[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown torchvision weights {weights!r}; use None, 'COCO_V1' or a "
            "torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights member"
        ) from exc


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def as_image_tensor(image: Any) -> Any:
    """Coerce path / PIL image / ndarray / tensor to a ``[C,H,W]`` float32 CHW.

    Accepted inputs (all values end up in ``[0, 1]``):

    * ``str`` / ``os.PathLike`` - decoded with PIL (JPEG/PNG).
    * ``PIL.Image.Image`` (any mode, converted to RGB).
    * ``numpy`` ``[H,W,3]`` uint8 (``/255``) or float (assumed ``[0,1]``);
      ``[H,W]`` grey is expanded to 3 channels.
    * ``torch.Tensor`` ``[C,H,W]`` or ``[1,C,H,W]`` (uint8 -> ``/255``).
    """
    import torch
    import torchvision

    if isinstance(image, torch.Tensor):
        t = image.detach()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"expected a [C,H,W] tensor, got shape {tuple(t.shape)}")
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        return t.float().contiguous()
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(str(image)) as handle:
            return (
                torchvision.transforms.functional.pil_to_tensor(handle.convert("RGB")).float()
                / 255.0
            )
    if hasattr(image, "convert") and hasattr(image, "size"):  # PIL image duck-type
        from PIL import Image  # noqa: F401  (keeps the dependency explicit)

        return torchvision.transforms.functional.pil_to_tensor(image.convert("RGB")).float() / 255.0
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3):
        raise ValueError(f"expected an [H,W,3] image, got shape {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    as_float = arr.astype(np.float32, copy=False)
    if arr.dtype == np.uint8:
        as_float = as_float / 255.0
    elif float(as_float.max()) > 1.0:
        raise ValueError(
            f"float images must already be in [0,1]; got max={float(as_float.max()):.3f}"
        )
    chw = np.ascontiguousarray(np.moveaxis(as_float, 2, 0), dtype=np.float32)
    return torch.from_numpy(chw)


# ---------------------------------------------------------------------------
# coordinate space
# ---------------------------------------------------------------------------
def inverse_resized_boxes(
    boxes: np.ndarray,
    original_hw: Sequence[int],
    resized_hw: Sequence[int],
) -> np.ndarray:
    """Map ``xyxy`` boxes from the RPN's resized space back to original pixels.

    The RPN works entirely in the space produced by
    ``GeneralizedRCNNTransform`` - resize such that ``min(h, w) -> min_size``
    unless that would push the long edge past ``max_size`` (then the long
    edge is capped), no coordinate-shifting padding (it is written to the
    bottom/right only).  The inverse is exactly the per-axis rescaling that
    ``transform.postprocess`` applies to detections via ``resize_boxes``:

        ``x_orig = x_resized * (W_orig / W_resized)``,
        ``y_orig = y_resized * (H_orig / H_resized)``

    Pass the **actual** resized ``(h, w)`` from ``ImageList.image_sizes``
    (``meta["resized_hw"]``); those already fold in ``min_size`` / ``max_size``
    and the integrally-rounded output size of the resize kernel, which is why
    no scale is re-derived here.  The unit test pins equality against
    torchvision's own ``resize_boxes``.

    Parameters
    ----------
    boxes:
        ``[N, 4]`` (or ``[4]``) float boxes in resized coordinates.
    original_hw, resized_hw:
        ``(height, width)`` of the original and the resized image.

    Returns
    -------
    numpy.ndarray
        ``[N, 4]`` float32 in original coordinates (not yet clipped).
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected [N,4] boxes, got {arr.shape}")
    o_h, o_w = (int(original_hw[0]), int(original_hw[1]))
    r_h, r_w = (int(resized_hw[0]), int(resized_hw[1]))
    if min(o_h, o_w, r_h, r_w) <= 0:
        raise ValueError(f"degenerate image size: original={o_h}x{o_w}, resized={r_h}x{r_w}")
    ratio_x = np.float32(o_w / r_w)
    ratio_y = np.float32(o_h / r_h)
    out = arr.copy()
    out[:, [0, 2]] *= ratio_x
    out[:, [1, 3]] *= ratio_y
    return out


def resized_boxes_to_original(
    boxes: np.ndarray,
    original_hw: Sequence[int],
    resized_hw: Sequence[int],
) -> np.ndarray:
    """Forward direction (original -> resized); exists for round-trips / tests."""
    arr = np.asarray(boxes, dtype=np.float32)
    single = arr.ndim == 1
    arr = arr.reshape(1, 4) if single else arr
    o_h, o_w = (int(original_hw[0]), int(original_hw[1]))
    r_h, r_w = (int(resized_hw[0]), int(resized_hw[1]))
    out = arr.copy()
    out[:, [0, 2]] *= np.float32(r_w / o_w)
    out[:, [1, 3]] *= np.float32(r_h / o_h)
    return out.reshape(4) if single else out


# ---------------------------------------------------------------------------
# ranking / truncation
# ---------------------------------------------------------------------------
def _sort_and_truncate(
    boxes: np.ndarray,
    scores: np.ndarray,
    top_n: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Stable descending-score sort + top-n truncation (pure numpy).

    Ties keep the input row order (``np.argsort(kind="stable")`` on the
    negated scores), which makes the result independent of the sort
    implementation - the frozen determinism requirement.  A shorter input is
    returned untouched (never padded); ``truncated`` reports whether rows
    were actually dropped.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"boxes and scores disagree on the number of rows: "
            f"{boxes.shape[0]} != {scores.shape[0]}"
        )
    order = np.argsort(-scores, kind="stable")
    truncated = top_n is not None and order.size > int(top_n)
    if top_n is not None:
        order = order[: int(top_n)]
    return boxes[order], scores[order], bool(truncated)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def _rpn_forward_with_scores(images: Any, features: Any, rpn: Any) -> Tuple[List[Any], List[Any]]:
    """``RegionProposalNetwork.forward`` replayed so the objectness survives.

    Mirrors torchvision 0.20.1's ``rpn.py`` L359-373 line by line, calling
    only public parts (``head``, ``anchor_generator``, ``box_coder``,
    ``filter_proposals`` and the module-level ``concat_box_prediction_layers``
    helper).  No forward hooks, no private patching.  ``filter_proposals``
    itself still performs the per-level top-k, ``clip_boxes_to_image``,
    ``remove_small_boxes``, the ``score_thresh`` cut and ``batched_nms``
    exactly as the stock detector does - it is simply also given back the
    scores that ``forward`` (rpn.py L388) throws away.
    """
    from torchvision.models.detection.rpn import concat_box_prediction_layers

    features_list = list(features.values())
    objectness, pred_bbox_deltas = rpn.head(features_list)
    anchors = rpn.anchor_generator(images, features_list)

    num_images = len(anchors)
    num_anchors_per_level = [o[0].shape[0] * o[0].shape[1] * o[0].shape[2] for o in objectness]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # proposals are not differentiable in Faster R-CNN inference - detach, as
    # upstream does (rpn.py L371).
    proposals = rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = rpn.filter_proposals(
        proposals, objectness, images.image_sizes, num_anchors_per_level
    )
    return boxes, scores


def extract_proposals(
    image: Any,
    model: Any,
    top_n: Optional[int] = DEFAULT_TOP_N,
    autocast: bool = True,
) -> RPNProposals:
    """Extract class-agnostic RPN proposals for one image.

    Runs the detector through its public sub-modules:
    ``model.transform([x]) -> model.backbone(images.tensors) -> RPN``.  The
    RPN stage is replayed through its public parts because 0.20.1's
    ``RegionProposalNetwork.forward`` returns only ``(boxes, losses)`` and
    throws the objectness scores away - see the module docstring.

    Parameters
    ----------
    image:
        PIL image (RGB) / ``np.ndarray`` ``[H,W,3]`` uint8 (or float
        ``[0,1]``) / path / ``[C,H,W]`` tensor.
    model:
        A model built by :func:`build_rpn_model`.
    top_n:
        Bank size ``N``.  The objectness ranking is truncated to ``top_n``
        (ties keep the detector's row order); if the RPN produced fewer
        proposals the actual, smaller count is returned as-is - padding and
        silent filtering are both forbidden by the protocol.  ``None`` keeps
        everything.
    autocast:
        FP16 autocast over the transform + backbone stages when the model is
        on CUDA (no-op on CPU).  The RPN head, box decode and NMS always run
        in fp32: the NMS kernel rejects half precision, and all returned
        arrays are float32 regardless.

    Returns
    -------
    RPNProposals
        ``boxes`` (``[K,4]`` float32 xyxy, original-image coordinates, clipped
        to the image) and ``objectness`` (``[K]`` float32), both ordered by
        descending objectness, plus the ``meta`` dict.  Besides the frozen
        keys, ``meta`` carries ``truncated`` (bool), ``num_returned``,
        ``padded_hw``, ``torch_version``, ``device``, ``autocast_effective``
        and ``elapsed_ms`` for the run log.  Raises
        :class:`RPNProposalCountError` when the detector returned zero
        proposals (a run-level failure, not an empty bank).
    """
    import torch
    import torchvision

    model_device = next(model.parameters()).device
    tensor = as_image_tensor(image).to(model_device)
    original_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))

    model.eval()
    use_autocast = bool(autocast) and model_device.type == "cuda"

    started = time.perf_counter()
    with torch.no_grad():
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True) if use_autocast else nullcontext()
        )
        with autocast_ctx:  # fp16 covers transform + backbone only
            images, _ = model.transform([tensor], None)
            features = model.backbone(images.tensors)
        # The RPN runs in fp32: the NMS kernel is float32-only, and keeping
        # the arithmetic in fp32 is what makes the returned values stable.
        features = {k: v.float() for k, v in features.items()}
        resized_hw = (int(images.image_sizes[0][0]), int(images.image_sizes[0][1]))
        boxes_list, scores_list = _rpn_forward_with_scores(images, features, model.rpn)
    if model_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3

    if len(boxes_list) != 1:
        raise RuntimeError(f"expected a single-image proposal list, got {len(boxes_list)}")
    raw_boxes = boxes_list[0].detach().to("cpu", torch.float32).numpy()
    raw_scores = scores_list[0].detach().to("cpu", torch.float32).numpy()
    n_raw = int(raw_boxes.shape[0])
    if n_raw == 0:
        raise RPNProposalCountError(
            f"the RPN produced no proposals for an image of size {original_hw}; "
            "check score_thresh (must be 0.0) and the input value range"
        )

    # descending objectness, stable => ties keep the detector's own row order
    boxes_sel, scores_sel, truncated = _sort_and_truncate(raw_boxes, raw_scores, top_n)

    boxes = inverse_resized_boxes(boxes_sel, original_hw, resized_hw)
    height, width = original_hw
    np.clip(
        boxes,
        0.0,
        np.array([width, height, width, height], dtype=np.float32),
        out=boxes,
    )

    meta: Dict[str, Any] = {
        # -- frozen contract keys ---------------------------------------------
        "original_hw": original_hw,
        "resized_hw": resized_hw,
        "n_raw_post_nms": n_raw,
        "nms_thresh": float(model.rpn.nms_thresh),
        "pre_nms_top_n": int(model.rpn.pre_nms_top_n()),
        "post_nms_top_n": int(model.rpn.post_nms_top_n()),
        "torchvision_version": torchvision.__version__,
        "truncated_to": None if top_n is None else int(top_n),
        "model_name": getattr(model, "ccg_model_name", type(model).__name__),
        # -- extras for the run log -------------------------------------------
        "truncated": bool(truncated),
        "num_returned": int(boxes.shape[0]),
        "padded_hw": (int(images.tensors.shape[-2]), int(images.tensors.shape[-1])),
        "torch_version": torch.__version__,
        "device": str(model_device),
        "autocast_effective": bool(use_autocast),
        "elapsed_ms": float(elapsed_ms),
    }
    return RPNProposals(
        boxes=np.ascontiguousarray(boxes, dtype=np.float32),
        objectness=np.ascontiguousarray(scores_sel, dtype=np.float32),
        meta=meta,
    )


def extract_proposal_bank(
    image: Any,
    model: Any,
    image_id: int,
    *,
    top_n: int = DEFAULT_TOP_N,
    gt_boxes: Optional[np.ndarray] = None,
    gt_categories: Optional[np.ndarray] = None,
    gt_object_ids: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """:func:`extract_proposals` wrapped into a :class:`~ccg.data.types.ProposalBank`.

    ``gt_*`` are optional COCO ground-truth rows for the same image; when
    given they are attached as ``gt_ious`` (IoU with the nearest GT object)
    and ``gt_assignment`` (that object's id, ``-1`` when nothing overlaps) -
    the two fields the candidate builders and the audit rely on.  Geometry
    only: no features are produced here (they stay in the on-disk cache
    behind ``feature_ref``).  Returns ``(bank, meta)`` so the extraction
    bookkeeping never has to be smuggled onto the dataclass.
    """
    from .proposals import iou_matrix
    from .types import ProposalBank

    proposals = extract_proposals(image, model, top_n=top_n, **kwargs)
    boxes, objectness, meta = proposals.boxes, proposals.objectness, proposals.meta
    gt_ious = gt_assignment = None
    if gt_boxes is not None:
        gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
        if gt.shape[0]:
            ious = iou_matrix(boxes, gt)
            gt_ious = ious.max(axis=1).astype(np.float32)
            best = ious.argmax(axis=1)
            ids = (
                np.asarray(gt_object_ids, dtype=np.int64).reshape(-1)
                if gt_object_ids is not None
                else np.arange(gt.shape[0], dtype=np.int64)
            )
            # "unassigned" = no GT object overlaps this proposal at all
            gt_assignment = np.where(gt_ious > 0.0, ids[best], -1).astype(np.int64)
        else:
            gt_ious = np.zeros(boxes.shape[0], dtype=np.float32)
            gt_assignment = np.full(boxes.shape[0], -1, dtype=np.int64)
    bank = ProposalBank(
        image_id=int(image_id),
        boxes=boxes,
        objectness=objectness,
        gt_ious=gt_ious,
        gt_assignment=gt_assignment,
    )
    if gt_categories is not None:
        meta["gt_categories"] = np.asarray(gt_categories, dtype=np.int64).reshape(-1).tolist()
    return bank, meta
