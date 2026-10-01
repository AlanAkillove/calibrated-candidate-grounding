"""V2-P1 DETR Proposal-B invariants (protocol section 24).

These cover the checks that are decidable *offline* (no GPU, no checkpoint
download, no dataset): the frozen ``N = 64`` budget, deterministic box
sanitisation, the query-independent / GT-independent generator contract, the
foreground score definition that excludes the no-object column, exact per-axis
coordinate conversion, bank isolation from the V1 RPN cache, and the
deterministic P1-F2 gate.  The live single-image extraction smoke is guarded by
both CUDA availability and an opt-in env flag so the default full-pytest run
never touches the network.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from ccg.data import detr

# make the scripts/ drivers importable for the isolation / gate tests
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT / "scripts"))

HAS_CUDA = False
try:  # torch may not even be importable in a stripped environment
    import torch

    HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch is a hard dependency in this repo
    pass

NEEDS_CUDA = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
RUN_LIVE = pytest.mark.skipif(
    os.environ.get("CCG_RUN_DETR_LIVE") != "1",
    reason="set CCG_RUN_DETR_LIVE=1 to run the live DETR checkpoint smoke",
)


# ---------------------------------------------------------------------------
# frozen budget / constants
# ---------------------------------------------------------------------------
def test_n_64_is_frozen():
    assert detr.DEFAULT_TOP_N == 64
    assert detr.MODEL_NAME == "detr_resnet50"
    assert detr.MODEL_ID == "facebook/detr-resnet-50"
    assert "query-independent" in detr.PROPOSAL_TYPE


def test_no_object_column_is_foreground_exclusive():
    # foreground score takes max softmax over the num_labels COCO-side columns
    # and must ignore the trailing no-object column entirely.
    num_labels = 3  # logits are [Q, num_labels + 1] = [Q, 4]; col 3 == no-object
    logits = np.array(
        [
            [0.0, 0.0, 0.0, 50.0],   # almost all mass on no-object -> foreground ~0
            [10.0, 0.0, 0.0, 0.0],   # clear foreground win on class 0
        ]
    )
    scores = detr.foreground_scores(logits, num_labels)
    assert scores.shape == (2,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert scores[1] > 0.9
    assert scores[0] < 0.2 and scores[1] > scores[0]


def test_foreground_scores_shape_guard():
    with pytest.raises(ValueError):
        detr.foreground_scores(np.zeros((5, 91)), 91)   # needs num_labels + 1
    with pytest.raises(ValueError):
        detr.foreground_scores(np.zeros((5,)), 91)      # needs 2-D


# ---------------------------------------------------------------------------
# coordinate conversion: exact per-axis scaling (== torchvision post_process)
# ---------------------------------------------------------------------------
def test_cxcywh_to_xyxy_pixels_matches_box_convert():
    from torchvision.ops import box_convert

    rng = np.random.default_rng(0)
    width, height = 640, 480
    cx = rng.uniform(0.1, 0.9, 50)
    cy = rng.uniform(0.1, 0.9, 50)
    w = rng.uniform(0.02, 0.3, 50)
    h = rng.uniform(0.02, 0.3, 50)
    cxcywh = np.stack([cx, cy, w, h], axis=1)

    ref = box_convert(
        torch.as_tensor(cxcywh, dtype=torch.float64), in_fmt="cxcywh", out_fmt="xyxy"
    ).numpy() * np.array([width, height, width, height], dtype=np.float64)
    ours = detr.cxcywh_to_xyxy_pixels(cxcywh, width, height)
    assert np.allclose(ref, ours, atol=1e-6)


def test_cxcywh_rejects_degenerate_size_and_shape():
    with pytest.raises(ValueError):
        detr.cxcywh_to_xyxy_pixels(np.zeros((3, 4)), 0, 10)
    with pytest.raises(ValueError):
        detr.cxcywh_to_xyxy_pixels(np.zeros((3, 3)), 10, 10)


# ---------------------------------------------------------------------------
# sanitisation: deterministic, clips, drops non-finite / small / exact dupes
# ---------------------------------------------------------------------------
def _sample_boxes():
    boxes = np.array(
        [
            [10.0, 10.0, 60.0, 70.0],
            [10.0, 10.0, 60.0, 70.0],     # exact duplicate of row 0
            [100.0, 50.0, 100.5, 200.0],  # width 0.5 -> dropped as small
            [np.nan, 0.0, 10.0, 10.0],    # non-finite -> dropped
            [-20.0, -30.0, 30.0, 40.0],   # clipped to [0,0,30,40]
        ]
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    return boxes, scores


def test_sanitize_boxes_counts_and_clips():
    boxes, scores = _sample_boxes()
    b, s, counts = detr.sanitize_boxes(boxes, scores, 80, 80)
    assert counts["invalid_finite"] == 1
    assert counts["invalid_small"] == 1
    assert counts["exact_duplicates"] == 1
    assert counts["n_input"] == 5
    assert counts["n_kept"] == b.shape[0] == 2
    # the out-of-bounds box is clipped to [0, 0, 30, 40]; the valid box survives
    assert np.all(b[:, :2] >= 0.0) and np.all(b[:, 2] <= 80.0) and np.all(b[:, 3] <= 80.0)
    assert np.allclose(b[0], [10, 10, 60, 70])
    assert np.allclose(b[1], [0, 0, 30, 40])
    assert b.dtype == np.float32 and s.dtype == np.float32


def test_sanitize_boxes_is_deterministic():
    boxes, scores = _sample_boxes()
    a = detr.sanitize_boxes(boxes, scores, 64, 64)
    b = detr.sanitize_boxes(boxes.copy(), scores.copy(), 64, 64)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_select_top_n_stable_descending_and_never_pads():
    rng = np.random.default_rng(1)
    boxes = rng.uniform(0, 100, (10, 4)).astype(np.float32)
    scores = rng.uniform(0, 1, 10).astype(np.float32)
    b, s, truncated = detr.select_top_n(boxes, scores, top_n=64)
    assert truncated is False and b.shape[0] == 10       # fewer than N -> real count
    assert np.all(np.diff(s) <= 1e-6)                     # descending
    b2, s2, truncated2 = detr.select_top_n(boxes, scores, top_n=5)
    assert truncated2 is True and b2.shape[0] == 5


# ---------------------------------------------------------------------------
# the generator contract is query- and GT-independent (protocol section 5/24)
# ---------------------------------------------------------------------------
_FORBIDDEN = {
    "query", "text", "expression", "caption", "phrase",
    "category", "target", "gt", "ground_truth", "prompt",
}


@pytest.mark.parametrize("fn", ["extract_detr_proposals", "extract_detr_proposals_batch"])
def test_generator_signature_has_no_query_or_gt_inputs(fn):
    params = set(inspect.signature(getattr(detr, fn)).parameters)
    overlap = {p for p in params if any(bad in p.lower() for bad in _FORBIDDEN)}
    assert not overlap, f"{fn} must not take query/GT inputs, got {overlap}"


# ---------------------------------------------------------------------------
# bank isolation: Proposal-B never touches the V1 RPN cache
# ---------------------------------------------------------------------------
def test_detr_bank_is_isolated_from_v1_rpn():
    import p1_extract_detr_proposals as f1

    assert f1.DEFAULT_OUT.name == "proposals_detr_r50.h5"
    assert f1.DEFAULT_OUT.name != "proposals.h5"
    assert str(f1.DEFAULT_MANIFEST).endswith("full_image_manifest.csv")
    # the writer stamps a DETR-specific model name used by the refuse-to-mix guard
    assert f1.detr.MODEL_NAME == "detr_resnet50"


# ---------------------------------------------------------------------------
# F2 gate: deterministic and threshold-exact, reusing the V1 primitives
# ---------------------------------------------------------------------------
def _synthetic_per_n(recall05: float, avail50: float):
    return {
        "64": {
            "recall": {
                "ref_target": {
                    "by_threshold": {
                        "0.5": {"recall": recall05, "ci_low": recall05, "ci_high": recall05},
                        "0.7": {"recall": recall05, "ci_low": recall05, "ci_high": recall05},
                    }
                }
            },
            "candidate_availability": {"50": {"frac_available": avail50}},
        }
    }


@pytest.mark.parametrize(
    "recall,avail,expected",
    [
        (0.97, 0.95, "FULL"),
        (0.95, 0.91, "FULL"),
        (0.92, 0.91, "GRAY"),
        (0.85, 0.95, "LIMITED"),
        (0.79, 0.95, "STOP"),
        (0.98, 0.89, "STOP"),   # recall fine but K50 availability sub-gate fails
    ],
)
def test_gate_verdict_bands(recall, avail, expected):
    import p1_f2_proposal_audit as f2

    gate = f2._gate_block(_synthetic_per_n(recall, avail), 64)
    assert gate["verdict"] == expected
    assert gate["candidate_availability_K50_ok"] == (avail >= 0.90)


def test_f2_reuses_v1_primitives_not_reimplemented():
    import p1_f2_proposal_audit as f2
    import run_proposal_audit as v1

    # F2 must call straight into the frozen V1 pipeline pieces
    assert f2.v1.audit_level is v1.audit_level
    assert f2.v1.load_refcoco_refs is v1.load_refcoco_refs
    assert f2.v1.load_gt_index is v1.load_gt_index
    # and must not re-implement the per-image audit locally
    assert not hasattr(f2, "audit_level")


# ---------------------------------------------------------------------------
# live single-image extraction smoke (CUDA + opt-in)
# ---------------------------------------------------------------------------
@NEEDS_CUDA
@RUN_LIVE
def test_live_extract_detr_proposals_single_image():  # pragma: no cover - needs GPU
    from PIL import Image

    bundle = detr.build_detr_bundle(device="cuda", cache_dir=str(_ROOT / "cache" / "hf_hub"))
    assert bundle.num_queries == 100
    assert bundle.no_object_index == bundle.num_labels
    img = Image.fromarray(np.full((96, 128, 3), 128, dtype=np.uint8))
    prop = detr.extract_detr_proposals(img, bundle, top_n=64)
    assert prop.boxes.ndim == 2 and prop.boxes.shape[1] == 4
    assert prop.boxes.shape[0] == prop.objectness.shape[0] <= 64
    assert np.all(np.diff(prop.objectness) <= 1e-6)
    assert prop.meta["no_object_index"] == prop.meta["num_labels"]
