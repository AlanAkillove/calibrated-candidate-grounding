"""FineCops-Ref test cohort + GQA scene-graph join (protocol A9, feasibility).

Phase 1E is a *frozen external confirmation* run: every model, feature and
threshold comes from RefCOCO+, and this module owns only the external **data**
side.  It reads the publicly released *test* annotations (and only test - see
:func:`load_test_expressions`) and joins them against the GQA scene graph,
which is the single public source of per-image object *names*.

Why the scene graph matters here
--------------------------------
Amendment A8 built the RefCOCO+ hard cohort from COCO GT categories.  FineCops
is built on GQA, so no COCO category exists (instruction section 10 forbids
inventing a mapping); the equivalent quantity is the GQA object ``name``, and
FineCops' own difficulty definition is stated in terms of it:

* level 1 - no object in the image shares the target's name;
* level 2 - a same-name object exists, one attribute/relation separates them;
* level 3 - two or more relations/attributes are needed.

So the level is *itself* a same-category-competition stratification, and the
scene graph lets us check that claim instead of trusting it:
:func:`resolve_target_object` matches the released target box against the graph
objects, and :func:`same_name_supply` counts how many objects of that name the
image really contains.

Determinism: no random state anywhere in this module.  The audit subset is
drawn with an explicit seed passed to :func:`select_audit_subset`, and every
list is ordered by expression id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..data.proposals import box_iou, iou_matrix, xywh_to_xyxy

__all__ = [
    "ANNOTATION_DIR",
    "SCENE_GRAPH_PATH",
    "IMAGE_DIR",
    "COCO_POS_FILE",
    "COCO_ALL_FILE",
    "VANILLA_POS_FILE",
    "VANILLA_ALL_FILE",
    "LEVELS",
    "TUPLE_TYPES",
    "GRAPH_IOU_MIN",
    "FineCopsExpression",
    "read_json",
    "load_test_expressions",
    "load_scene_graph",
    "resolve_target_object",
    "same_name_supply",
    "cross_check_coco_against_vanilla",
    "select_audit_subset",
    "image_dim_rows",
    "difficulty_distribution",
    "tuple_type_distribution",
    "negative_feasibility_counts",
]

#: Local, hash-pinned copies of the public files (see ``data/raw/finecops/dataset_card.json``).
ANNOTATION_DIR = Path("data/raw/finecops")
SCENE_GRAPH_PATH = Path("data/raw/gqa/val_sceneGraphs.json")
IMAGE_DIR = Path("data/raw/gqa/images")

#: Test-split annotation files.  Train/val files are deliberately unreachable
#: from this module: Phase 1E may not touch their labels at all (section 3).
COCO_POS_FILE = "test_expression_pos_coco_format.json"
COCO_ALL_FILE = "test_expression_all_coco_format.json"
VANILLA_POS_FILE = "test_expression_pos.json"
VANILLA_ALL_FILE = "test_expression_all.json"

#: Official positive difficulty levels (section 13: group as published, never re-merge).
LEVELS: Tuple[int, ...] = (1, 2, 3)
#: The six tuple types actually present in the released test split (section 22).
TUPLE_TYPES: Tuple[str, ...] = (
    "0_hop",
    "1_hop",
    "2_hop",
    "and",
    "same_attr",
    "same_attr_two_hop",
)
#: A scene-graph object counts as "the target" only above this IoU with the released box.
GRAPH_IOU_MIN = 0.9

_NUMERIC_ID = re.compile(r"^\d+$")


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class FineCopsExpression:
    """One positive test expression with its target box and graph metadata.

    ``gt_box`` is ``xyxy`` float32 in *original image pixel* coordinates,
    converted from the released ``xywh`` (section 33 test 1).
    """

    expr_id: int
    image_id: int  # numeric GQA image id, also the bootstrap cluster key
    gqa_image_id: str
    text: str
    gt_box: np.ndarray
    width: int
    height: int
    level: int
    tuple_type: str
    objects_id: Tuple[str, ...]
    attribute: Tuple[Tuple[str, ...], ...] = ()
    spatial: str = ""
    # -- filled by the scene-graph join (None when the image has no graph) --
    graph_present: bool = False
    target_object_id: Optional[str] = None
    target_name: Optional[str] = None
    target_name_source: str = "unresolved"  # graph_box | objects_id_first | unresolved
    target_box_iou: float = 0.0
    n_same_name: int = 0  # objects in the image sharing the target name (incl. target)
    n_graph_objects: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_same_name_distractors(self) -> int:
        """Same-name objects that could serve as distractors (excludes target)."""
        return max(0, self.n_same_name - 1)

    def to_row(self) -> Dict[str, Any]:
        return {
            "expr_id": self.expr_id,
            "image_id": self.image_id,
            "gqa_image_id": self.gqa_image_id,
            "level": self.level,
            "tuple_type": self.tuple_type,
            "target_name": self.target_name or "",
            "target_name_source": self.target_name_source,
            "target_box_iou": f"{self.target_box_iou:.4f}",
            "n_graph_objects": self.n_graph_objects,
            "n_same_name": self.n_same_name,
            "n_same_name_distractors": self.n_same_name_distractors,
            "n_objects_id": len(self.objects_id),
            "spatial": self.spatial or "",
            "width": self.width,
            "height": self.height,
            "gt_box_xyxy": ",".join(f"{v:.1f}" for v in np.asarray(self.gt_box).tolist()),
            "expression": self.text,
        }


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_scene_graph(path: Path = SCENE_GRAPH_PATH) -> Dict[str, Dict[str, Any]]:
    """Load one GQA scene-graph file: ``image_id -> {"width","height","objects"}``."""
    data = read_json(Path(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dict keyed by image id, got {type(data).__name__}")
    return data


def _graph_object_boxes_xyxy(objects: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    ids = sorted(objects.keys())
    boxes = np.empty((len(ids), 4), dtype=np.float32)
    for row, oid in enumerate(ids):
        o = objects[oid]
        boxes[row] = (o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"])
    return boxes, ids


def resolve_target_object(
    expr: FineCopsExpression, objects: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str], str, float]:
    """Identify the target node of ``expr`` inside its GQA scene graph.

    Returns ``(object_id, name, source, iou)``.  The released target box is
    matched against every graph object; a confident match (IoU >=
    :data:`GRAPH_IOU_MIN`) wins, because the box is what the benchmark's own
    metric uses.  Without a match we fall back to ``objects_id[0]`` - the
    documented subject of the first reasoning hop - and label the source so
    that the audit can report how often the fallback was needed.
    """
    if not objects:
        return None, None, "unresolved", 0.0
    boxes, ids = _graph_object_boxes_xyxy(objects)
    ious = iou_matrix(expr.gt_box.reshape(1, 4), boxes)[0]
    best = int(np.argmax(ious))
    if float(ious[best]) >= GRAPH_IOU_MIN:
        oid = ids[best]
        return oid, str(objects[oid]["name"]), "graph_box", float(ious[best])
    if expr.objects_id and expr.objects_id[0] in objects:
        oid = expr.objects_id[0]
        i = ids.index(oid) if oid in ids else -1
        return (
            oid,
            str(objects[oid]["name"]),
            "objects_id_first",
            float(ious[i]) if i >= 0 else 0.0,
        )
    return None, None, "unresolved", float(ious[best])


def same_name_supply(objects: Dict[str, Dict[str, Any]], name: Optional[str]) -> int:
    """How many graph objects carry exactly this ``name`` (0 when ``name`` is None).

    Exact string identity is used on purpose: that is the equivalence relation
    FineCops' own level definition is built on, and any coarser grouping would
    be an invented taxonomy (instruction section 10/11).
    """
    if not name:
        return 0
    return sum(1 for o in objects.values() if str(o.get("name")) == name)


def load_test_expressions(
    annotation_dir: Path = ANNOTATION_DIR,
    scene_graph_path: Path = SCENE_GRAPH_PATH,
    *,
    require_scene_graph: bool = True,
) -> List[FineCopsExpression]:
    """Parse the FineCops-Ref **positive test** split and join the scene graph.

    Only ``test_expression_pos_coco_format.json`` is read: it carries one
    target box and the full reasoning metadata per expression.  Negative rows
    never enter this loader (they have no ``c*`` for a
    :math:`P(\\hat c = c^*)` model; sections 2 and 23), and no train/val file
    is reachable from this function.

    Raises
    ------
    ValueError
        when a positive row references a ``neg_`` image id, a non-numeric GQA
        id, a degenerate box, or when ``require_scene_graph`` and the image has
        no public scene graph.
    """
    annotation_dir = Path(annotation_dir)
    coco = read_json(annotation_dir / COCO_POS_FILE)
    images = {im["id"]: im for im in coco["images"]}
    graph = load_scene_graph(scene_graph_path) if Path(scene_graph_path).exists() else {}

    out: List[FineCopsExpression] = []
    for ann in sorted(coco["annotations"], key=lambda a: int(a["id"])):
        im = images[ann["image_id"]]
        file_name = str(im["file_name"])
        gqa_id = file_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if gqa_id.startswith("neg_") or not _NUMERIC_ID.match(gqa_id):
            raise ValueError(
                f"expr {ann['id']}: positive row references non-numeric image id {gqa_id!r}"
            )
        box = xywh_to_xyxy(np.asarray(ann["bbox"], dtype=np.float32))
        if not (box[2] > box[0] and box[3] > box[1]):
            raise ValueError(f"expr {ann['id']}: degenerate target box {ann['bbox']}")
        expr = FineCopsExpression(
            expr_id=int(ann["id"]),
            image_id=int(gqa_id),
            gqa_image_id=gqa_id,
            text=str(ann.get("caption", im.get("caption", ""))),
            gt_box=box,
            width=int(im["width"]),
            height=int(im["height"]),
            level=int(ann["level"]),
            tuple_type=str(ann["tuple_type"]),
            objects_id=tuple(str(o) for o in ann.get("objects_id", ())),
            attribute=tuple(tuple(a or ()) for a in ann.get("attribute", ())),
            spatial=str(ann.get("spatial", "") or ""),
        )
        entry = graph.get(gqa_id)
        if entry is None:
            if require_scene_graph:
                raise ValueError(f"expr {expr.expr_id}: image {gqa_id} has no public scene graph")
        else:
            objects = entry.get("objects", {}) or {}
            oid, name, source, iou = resolve_target_object(expr, objects)
            expr.graph_present = True
            expr.target_object_id = oid
            expr.target_name = name
            expr.target_name_source = source
            expr.target_box_iou = float(iou)
            expr.n_graph_objects = len(objects)
            expr.n_same_name = same_name_supply(objects, name)
            w, h = int(entry.get("width", expr.width)), int(entry.get("height", expr.height))
            expr.extra["graph_width"], expr.extra["graph_height"] = w, h
        out.append(expr)
    return out


def cross_check_coco_against_vanilla(
    expressions: Sequence[FineCopsExpression],
    annotation_dir: Path = ANNOTATION_DIR,
) -> Dict[str, Any]:
    """Verify the COCO-format file against the vanilla file (identity, text, level)."""
    annotation_dir = Path(annotation_dir)
    vanilla = read_json(annotation_dir / VANILLA_POS_FILE)
    rows = {e.expr_id: e for e in expressions}
    same_ids = set(map(int, vanilla.keys())) == set(rows)
    text_mismatch = sum(1 for k, v in vanilla.items() if rows[int(k)].text != v["expression"])
    level_mismatch = sum(1 for k, v in vanilla.items() if rows[int(k)].level != int(v["level"]))
    tuple_mismatch = sum(
        1 for k, v in vanilla.items() if rows[int(k)].tuple_type != str(v["tuple_type"])
    )
    obj_mismatch = sum(
        1
        for k, v in vanilla.items()
        if list(rows[int(k)].objects_id) != [str(o) for o in v.get("objects_id", [])]
    )
    return {
        "n_vanilla": len(vanilla),
        "n_coco": len(rows),
        "ids_identical": bool(same_ids),
        "text_mismatch": int(text_mismatch),
        "level_mismatch": int(level_mismatch),
        "tuple_type_mismatch": int(tuple_mismatch),
        "objects_id_mismatch": int(obj_mismatch),
    }


# ---------------------------------------------------------------------------
# distributions (sections 13, 22, 24)
# ---------------------------------------------------------------------------
def difficulty_distribution(
    expressions: Sequence[FineCopsExpression],
) -> List[Dict[str, Any]]:
    """Per official ``level`` counts, with the same-name competition facts.

    The returned rows are what ``difficulty_distribution.csv`` holds: expression
    and image counts, the target-name resolution quality, and how many
    same-name objects the images really carry - i.e. whether the published
    level and the scene graph agree (the precondition for using levels as the
    external hardness axis).

    Levels are reported exactly as published (section 13): the official ones
    first, then any unexpected value as its own row, so an out-of-range level can
    never be silently merged into 1/2/3 or dropped from the ``all`` denominator.
    """
    present = sorted({int(e.level) for e in expressions})
    order: List[Any] = [level for level in LEVELS if level in present]
    order += [level for level in present if level not in LEVELS]
    rows: List[Dict[str, Any]] = []
    for level in (*order, "all"):
        sel = [
            e for e in expressions if (level == "all" or e.level == int(level))
        ]
        if not sel:
            continue
        same = [e.n_same_name for e in sel]
        rows.append(
            {
                "level": level,
                "n_expressions": len(sel),
                "n_images": len({e.image_id for e in sel}),
                "share_of_test": round(len(sel) / max(1, len(expressions)), 6),
                "graph_coverage": round(
                    sum(1 for e in sel if e.graph_present) / len(sel), 6
                ),
                "name_from_graph_box": round(
                    sum(1 for e in sel if e.target_name_source == "graph_box") / len(sel), 6
                ),
                "name_from_objects_id": round(
                    sum(1 for e in sel if e.target_name_source == "objects_id_first") / len(sel), 6
                ),
                "mean_target_box_iou": round(
                    float(np.mean([e.target_box_iou for e in sel])), 6
                ),
                "same_name_objects_mean": round(float(np.mean(same)), 4),
                "same_name_objects_median": float(np.median(same)),
                "same_name_objects_max": int(max(same)),
                "share_no_same_name": round(sum(1 for s in same if s <= 1) / len(sel), 6),
                "share_ge2_same_name": round(sum(1 for s in same if s >= 2) / len(sel), 6),
                "share_ge5_same_name": round(sum(1 for s in same if s >= 5) / len(sel), 6),
                "share_ge9_same_name": round(sum(1 for s in same if s >= 9) / len(sel), 6),
            }
        )
    return rows


def tuple_type_distribution(
    expressions: Sequence[FineCopsExpression], min_group: int = 300
) -> List[Dict[str, Any]]:
    """Per ``tuple_type`` counts (section 22), flagging groups under ``min_group``."""
    rows: List[Dict[str, Any]] = []
    for tt in (*TUPLE_TYPES, "other"):
        sel = [e for e in expressions if (tt == "other" and e.tuple_type not in TUPLE_TYPES)
               or (tt != "other" and e.tuple_type == tt)]
        if not sel:
            continue
        rows.append(
            {
                "tuple_type": tt,
                "n_expressions": len(sel),
                "n_images": len({e.image_id for e in sel}),
                "share_of_test": round(len(sel) / max(1, len(expressions)), 6),
                "level1": sum(1 for e in sel if e.level == 1),
                "level2": sum(1 for e in sel if e.level == 2),
                "level3": sum(1 for e in sel if e.level == 3),
                "reportable": bool(len(sel) >= int(min_group)),
            }
        )
    return rows


def negative_feasibility_counts(
    annotation_dir: Path = ANNOTATION_DIR,
) -> Dict[str, Any]:
    """Count the negative branch *without* ever modelling it (section 24).

    Reads the vanilla ``test_expression_all.json`` (the only file carrying
    ``negative_cate`` / ``negative_type`` / ``negative_level``) and reports the
    future-abstention feasibility numbers.  No box, no model, no gate use.
    """
    all_rows = read_json(Path(annotation_dir) / VANILLA_ALL_FILE)
    pos = [r for r in all_rows.values() if "negative_cate" not in r]
    neg_text = [r for r in all_rows.values() if r.get("negative_cate") == "text"]
    neg_img = [r for r in all_rows.values() if r.get("negative_cate") == "image"]

    def counter(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rows:
            out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
        return dict(sorted(out.items()))

    return {
        "n_all": len(all_rows),
        "n_positive": len(pos),
        "n_negative_text": len(neg_text),
        "n_negative_image": len(neg_img),
        "n_images_positive": len({r["image_id"] for r in pos}),
        "n_images_negative_image": len({r["image_id"] for r in neg_img}),
        "negative_type": counter(all_rows.values(), "negative_type"),
        "negative_level": counter(all_rows.values(), "negative_level"),
        "negative_cate": counter(all_rows.values(), "negative_cate"),
        "positive_level": counter(pos, "level"),
        "positive_tuple_type": counter(pos, "tuple_type"),
    }


# ---------------------------------------------------------------------------
# audit subset + image sanity
# ---------------------------------------------------------------------------
def select_audit_subset(
    expressions: Sequence[FineCopsExpression],
    n_images: int,
    seed: int,
) -> List[int]:
    """Deterministic uniform sample of ``n_images`` distinct image ids (F2).

    ``numpy.random.default_rng(seed)`` over the *sorted* unique image ids, so
    the subset depends only on (cohort, n_images, seed) and can be regenerated
    byte-for-byte by the tests.
    """
    images = sorted({int(e.image_id) for e in expressions})
    n = min(int(n_images), len(images))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.asarray(images, dtype=np.int64), size=n, replace=False)
    return sorted(int(c) for c in chosen)


def image_dim_rows(
    expressions: Sequence[FineCopsExpression],
    actual_dims: Dict[int, Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Compare the released ``width``/``height`` with the decoded JPEG (section 33 test 2).

    A mismatch would silently invalidate every IoU computed from the released
    boxes, so the audit refuses to proceed on a systematic offset; this helper
    only produces the rows, the driver applies the verdict.
    """
    seen: Dict[int, Dict[str, Any]] = {}
    for e in expressions:
        if e.image_id not in actual_dims:
            continue
        aw, ah = actual_dims[e.image_id]
        row = seen.setdefault(
            e.image_id,
            {
                "image_id": e.image_id,
                "gqa_image_id": e.gqa_image_id,
                "ann_width": e.width,
                "ann_height": e.height,
                "actual_width": aw,
                "actual_height": ah,
                "graph_width": e.extra.get("graph_width", ""),
                "graph_height": e.extra.get("graph_height", ""),
                "ann_matches_actual": bool(aw == e.width and ah == e.height),
                "ann_matches_graph": bool(e.width == e.extra.get("graph_width")
                                           and e.height == e.extra.get("graph_height")),
                "n_expressions": 0,
            },
        )
        row["n_expressions"] += 1
    return [seen[k] for k in sorted(seen)]


def boxes_inside_image(expressions: Sequence[FineCopsExpression]) -> Dict[str, Any]:
    """How much of the released target geometry falls outside the image frame."""
    n = len(expressions)
    outside = 0
    tiny = 0
    areas: List[float] = []
    for e in expressions:
        x1, y1, x2, y2 = (float(v) for v in e.gt_box)
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        if x1 < -1 or y1 < -1 or x2 > e.width + 1 or y2 > e.height + 1:
            outside += 1
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            tiny += 1
    arr = np.asarray(areas, dtype=np.float64)
    px = np.sqrt(arr)
    return {
        "n_expressions": n,
        "share_box_outside_frame": round(outside / max(1, n), 6),
        "share_side_lt_4px": round(tiny / max(1, n), 6),
        "box_side_px_median": float(np.median(px)),
        "box_side_px_p10": float(np.quantile(px, 0.10)),
        "box_side_px_p90": float(np.quantile(px, 0.90)),
        "image_area_px_median": float(np.median([e.width * e.height for e in expressions])),
        "box_area_share_median": float(
            np.median(arr / np.asarray([e.width * e.height for e in expressions], dtype=float))
        ),
    }


def target_pair_overlaps(
    expressions: Sequence[FineCopsExpression], iou_thresh: float = 0.5
) -> Dict[str, Any]:
    """Same-image expression pairs whose targets collide (multi-target risk).

    RefCOCO+ removed "equivalent" proposals so a candidate set holds exactly one
    valid target.  On FineCops two expressions of one image may point at the
    same object: that is harmless per se, but it decides whether the
    *expression-level* or the *object-level* row is the unit of analysis, and a
    high share of near-duplicate boxes would need reporting.
    """
    by_image: Dict[int, List[FineCopsExpression]] = {}
    for e in expressions:
        by_image.setdefault(e.image_id, []).append(e)
    pairs = 0
    identical = 0
    overlapping = 0
    for rows in by_image.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                pairs += 1
                iou = box_iou(rows[i].gt_box, rows[j].gt_box)
                if iou >= 0.99:
                    identical += 1
                if iou >= iou_thresh:
                    overlapping += 1
    return {
        "n_images": len(by_image),
        "n_same_image_pairs": pairs,
        "n_identical_target_pairs": identical,
        "n_overlapping_target_pairs": overlapping,
        "expressions_per_image_mean": round(len(expressions) / max(1, len(by_image)), 4),
    }
