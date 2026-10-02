"""V2-P2 Phase B: tests for the GDINO class-prompt generator + probe wiring.

Pure-function tests only (no GPU forward, no bank): the prompt-token mapping,
the label-aligned sanitisation cross-check against the frozen
``ccg.data.detr.sanitize_boxes``, the frozen probe-gate contract, and a static
zero-training-parameter scan of the new code path.  The real-tokenizer cases
skip when the downloaded snapshot is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg.data.detr import sanitize_boxes  # noqa: E402
from ccg.data.gdino import (  # noqa: E402
    MERGE_TOKEN,
    assert_sanitisation_matches,
    build_class_prompt,
    class_token_columns,
    coco_thing_classes,
    sanitize_boxes_with_labels,
)

ANN = _ROOT / "data" / "raw" / "annotations" / "instances_train2014.json"
MODEL_DIR = _ROOT / "cache" / "hf_local" / "IDEA-Research" / "grounding-dino-base"
AUDIT_JSON = _ROOT / "results" / "v2_proposal_robustness" / "phase_b_feasibility_audit.json"
HAS_TOK = MODEL_DIR.exists() and ANN.exists()
NEEDS_TOK = pytest.mark.skipif(not HAS_TOK, reason="GDINO snapshot or COCO annotations absent")


# ---------------------------------------------------------------------------
# class list / prompt construction
# ---------------------------------------------------------------------------
def test_coco_thing_classes_frozen_universe():
    cats = coco_thing_classes(ANN)
    assert len(cats) == 80
    ids = [i for i, _ in cats]
    assert ids == sorted(ids)
    names = [n for _, n in cats]
    assert "person" in names and "potted plant" in names
    assert not any("traffic lights" == n for n in names)  # things only, singular forms


def test_build_class_prompt_structure():
    prompt = build_class_prompt(["person", "bicycle"])
    assert prompt == "person . bicycle ."


@NEEDS_TOK
def test_class_token_columns_real_tokenizer():
    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained(str(MODEL_DIR)).tokenizer
    names = [n for _, n in coco_thing_classes(ANN)]
    cols = class_token_columns(tok, build_class_prompt(names), names)
    assert len(cols) == 80
    assert cols[0] == [1]  # first class starts right after [CLS]
    # every class phrase sits strictly inside the non-special column range
    ids = tok(build_class_prompt(names), add_special_tokens=True)["input_ids"]
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    assert ids[0] == cls_id and ids[-1] == sep_id
    for cc in cols:
        assert all(0 < c < len(ids) - 1 for c in cc)
    # multi-word classes own consecutive columns (wordpiece may split a word
    # further: "fire hydrant" -> [fire, hu, ##drant], so only check contiguity)
    for i, name in enumerate(names):
        if " " in name:
            assert cols[i] == list(range(cols[i][0], cols[i][-1] + 1))
            assert len(cols[i]) >= len(name.split(" "))
    # no column is shared between two classes
    flat = [c for cc in cols for c in cc]
    assert len(flat) == len(set(flat))


@NEEDS_TOK
def test_class_token_columns_detects_surface_drift():
    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained(str(MODEL_DIR)).tokenizer
    names = [n for _, n in coco_thing_classes(ANN)]
    bad = list(names)
    bad[0] = "persan"  # the prompt really says "person": claimed name will not match
    with pytest.raises(ValueError, match="persan"):
        class_token_columns(tok, build_class_prompt(names), bad)


@NEEDS_TOK
def test_class_token_columns_detects_phrase_count_drift():
    from transformers import AutoProcessor

    tok = AutoProcessor.from_pretrained(str(MODEL_DIR)).tokenizer
    names = [n for _, n in coco_thing_classes(ANN)]
    prompt = build_class_prompt(names[:-1])  # one phrase short of the class list
    with pytest.raises(ValueError, match="class phrases"):
        class_token_columns(tok, prompt, names)


# ---------------------------------------------------------------------------
# label-aligned sanitisation == frozen sanitize_boxes, row-exact
# ---------------------------------------------------------------------------
def _adversarial_input(seed: int = 0):
    rng = np.random.default_rng(seed)
    xyxy = rng.uniform(-20, 210, size=(500, 4))
    xyxy[:, 2] = np.maximum(xyxy[:, 2], xyxy[:, 0])
    scores = rng.uniform(0.0, 1.0, size=(500,))
    scores[3] = np.nan  # non-finite row
    labels = rng.integers(0, 80, size=(500,))
    xyxy[7] = xyxy[11]
    xyxy[20] = xyxy[21]
    xyxy[21] = xyxy[22]  # duplicate chain: first row must win
    return xyxy, scores, labels


def test_aligned_sanitisation_matches_frozen_row_exact():
    xyxy, scores, labels = _adversarial_input()
    b_ref, s_ref, c_ref = sanitize_boxes(xyxy, scores, 200, 200)
    b_got, s_got, l_got, c_got = sanitize_boxes_with_labels(xyxy, scores, labels, 200, 200)
    assert np.array_equal(b_ref, b_got)
    assert np.array_equal(s_ref, s_got)
    assert c_ref == c_got
    assert l_got.shape[0] == b_got.shape[0]
    assert_sanitisation_matches(xyxy, scores, labels, 200, 200)


def test_aligned_sanitisation_keeps_first_duplicate_label():
    xyxy = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]])
    scores = np.array([0.9, 0.5, 0.7])
    labels = np.array([3, 7, 11])
    _b, _s, l, counts = sanitize_boxes_with_labels(xyxy, scores, labels, 100, 100)
    assert counts["exact_duplicates"] == 1
    assert l[0] == 3  # first (highest-position) duplicate row wins, matching detr
    assert list(l) == [3, 11]


def test_aligned_sanitisation_length_validation():
    xyxy = np.zeros((4, 4), dtype=np.float64)
    scores = np.ones(4)
    with pytest.raises(ValueError):
        sanitize_boxes_with_labels(xyxy, scores, np.ones(3, dtype=np.int64), 10, 10)


# ---------------------------------------------------------------------------
# frozen probe-gate contract (probe reads thresholds from the audit json)
# ---------------------------------------------------------------------------
def test_probe_gates_loaded_from_frozen_audit():
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    gates = audit["probe_gates_before_full_run"]
    for key in (
        "boxes_ge_64_share_min",
        "ref_target_recall_at_05_min",
        "same_category_supply_ge4_min",
        "k50_availability_min",
        "peak_vram_gb_max",
    ):
        assert key in gates, key
    assert gates["ref_target_recall_at_05_min"] == 0.95
    assert gates["same_category_supply_ge4_min"] == 0.85
    assert audit["targets"]["E2_gdino_query_conditioned_topk"]["verdict"].startswith("NOT usable")


# ---------------------------------------------------------------------------
# zero-fit static scan of the new code path
# ---------------------------------------------------------------------------
def test_no_fit_path_in_p2_code():
    needles = (".fit(", ".fit_transform(", "fit_predict(", "LogisticRegression(", "train(")
    for rel in ("src/ccg/data/gdino.py", "scripts/p2_gdino_probe.py"):
        text = (_ROOT / rel).read_text(encoding="utf-8")
        # strip docstrings/comments roughly: scan code lines only
        code = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        for needle in needles:
            assert needle not in code, f"{rel} contains {needle!r}"


def test_merge_token_is_dot_separator():
    assert MERGE_TOKEN == "."
