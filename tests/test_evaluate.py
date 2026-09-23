"""Accuracy-evaluation plumbing: argument validation, ground-truth filtering,
per-class matching and report rendering.

No model, no weights and no COCO download -- every test here builds its own
tiny ground truth by hand, so this runs on a clean clone. What it cannot cover
is the pycocotools mAP itself; that needs the real subset and is what
`python -m src.evaluate` is for.
"""

from __future__ import annotations

import argparse

import pytest

from src.coco_classes import COCO_91_IDS, COCO_CLASSES
from src.detector import Detection
from src.evaluate import (
    ClassMetrics,
    _positive_int,
    _unit_float,
    format_report,
    iou_matrix,
    per_class_pr,
    restrict_ground_truth,
    to_coco_json,
)

import numpy as np


def _gt(*annotations: tuple[int, int, list[float]], iscrowd: int = 0) -> dict:
    """Ground truth from (image_id, category_id, xywh box) triples."""
    return {
        "images": [{"id": i} for i in sorted({a[0] for a in annotations})],
        "annotations": [
            {
                "id": n,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "iscrowd": iscrowd,
            }
            for n, (image_id, category_id, bbox) in enumerate(annotations, 1)
        ],
    }


def _det(x1, y1, x2, y2, score, class_id) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


# --- argparse types ---------------------------------------------------------


@pytest.mark.parametrize("text", ["1", "20", "300"])
def test_positive_int_accepts_counts_above_zero(text):
    assert _positive_int(text) == int(text)


@pytest.mark.parametrize("text", ["0", "-1", "-300"])
def test_positive_int_rejects_zero_and_below(text):
    # --limit 0 is the one that matters: it is falsy, so it used to fall
    # through the filter and score the whole subset.
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(text)


@pytest.mark.parametrize("text", ["0.001", "0.3", "0.999"])
def test_unit_float_accepts_thresholds_inside_the_open_interval(text):
    assert _unit_float(text) == float(text)


@pytest.mark.parametrize("text", ["0", "1", "30", "-0.5", "1.5"])
def test_unit_float_rejects_thresholds_outside_it(text):
    # "30" is the plausible typo for "0.30", and it wrote a table of zeros.
    with pytest.raises(argparse.ArgumentTypeError):
        _unit_float(text)


# --- restrict_ground_truth --------------------------------------------------
#
# pycocotools scores every image in the GT file, so an image listed but never
# run counts as pure recall loss. These pin that images and annotations are
# always dropped together.


def test_restricting_keeps_only_the_named_images():
    gt = _gt((1, 1, [0, 0, 10, 10]), (2, 1, [0, 0, 10, 10]), (3, 1, [0, 0, 10, 10]))
    out = restrict_ground_truth(gt, {1, 3})
    assert [im["id"] for im in out["images"]] == [1, 3]


def test_restricting_drops_the_annotations_of_dropped_images():
    gt = _gt((1, 1, [0, 0, 10, 10]), (2, 1, [0, 0, 10, 10]), (2, 3, [5, 5, 10, 10]))
    out = restrict_ground_truth(gt, {1})
    assert {a["image_id"] for a in out["annotations"]} == {1}
    assert len(out["annotations"]) == 1


def test_restricting_to_every_id_is_an_identity():
    # The normal case: nothing unreadable, no --limit. The published numbers
    # depend on this being a no-op.
    gt = _gt((1, 1, [0, 0, 10, 10]), (2, 3, [0, 0, 10, 10]))
    out = restrict_ground_truth(gt, {1, 2})
    assert out["images"] == gt["images"]
    assert out["annotations"] == gt["annotations"]


def test_restricting_preserves_unrelated_top_level_keys():
    gt = {**_gt((1, 1, [0, 0, 10, 10])), "categories": [{"id": 1}], "info": {"x": 1}}
    out = restrict_ground_truth(gt, {1})
    assert out["categories"] == [{"id": 1}]
    assert out["info"] == {"x": 1}


# --- the two id spaces ------------------------------------------------------


def test_to_coco_json_translates_contiguous_indices_to_sparse_category_ids():
    # The model emits 0-79; COCO annotations use sparse 91-category ids. Getting
    # this wrong scores every detection against the wrong category.
    person = to_coco_json({1: [_det(0, 0, 10, 10, 0.9, 0)]}, conf=0.5)
    assert person[0]["category_id"] == COCO_91_IDS[0] == 1

    # class 79 is the last COCO class; its sparse id is far from 79.
    last = to_coco_json({1: [_det(0, 0, 10, 10, 0.9, 79)]}, conf=0.5)
    assert last[0]["category_id"] == COCO_91_IDS[79]
    assert last[0]["category_id"] != 79


def test_to_coco_json_emits_xywh_not_xyxy():
    out = to_coco_json({7: [_det(10, 20, 40, 60, 0.9, 0)]}, conf=0.5)
    assert out[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert out[0]["image_id"] == 7


def test_to_coco_json_drops_detections_below_the_threshold():
    dets = {1: [_det(0, 0, 10, 10, 0.9, 0), _det(0, 0, 10, 10, 0.1, 0)]}
    assert len(to_coco_json(dets, conf=0.5)) == 1
    assert len(to_coco_json(dets, conf=0.05)) == 2


# --- iou_matrix -------------------------------------------------------------


def test_iou_of_identical_boxes_is_one():
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_of_half_overlapping_boxes():
    # Intersection 50, union 150.
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150, abs=1e-6)


def test_iou_matrix_is_empty_when_either_side_is():
    empty = np.empty((0, 4), dtype=np.float32)
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(empty, box).shape == (0, 1)
    assert iou_matrix(box, empty).shape == (1, 0)


# --- per_class_pr -----------------------------------------------------------


def _find(classes: list[ClassMetrics], name: str) -> ClassMetrics:
    return next(c for c in classes if c.name == name)


def test_an_overlapping_detection_of_the_right_class_is_a_true_positive():
    gt = _gt((1, COCO_91_IDS[0], [0, 0, 10, 10]))          # one person
    out = per_class_pr({1: [_det(0, 0, 10, 10, 0.9, 0)]}, gt, conf=0.3)
    person = _find(out, "person")
    assert (person.tp, person.fp, person.fn, person.support) == (1, 0, 0, 1)


def test_a_detection_where_there_is_no_ground_truth_is_a_false_positive():
    gt = _gt((1, COCO_91_IDS[0], [0, 0, 10, 10]))
    # A cup (class 41) detected on an image whose only label is a person.
    out = per_class_pr({1: [_det(0, 0, 10, 10, 0.9, 41)]}, gt, conf=0.3)
    assert _find(out, COCO_CLASSES[41]).fp == 1
    assert _find(out, "person").fn == 1


def test_a_missed_ground_truth_box_is_a_false_negative():
    gt = _gt((1, COCO_91_IDS[0], [0, 0, 10, 10]))
    out = per_class_pr({1: []}, gt, conf=0.3)
    person = _find(out, "person")
    assert (person.tp, person.fn) == (0, 1)


def test_a_detection_below_pr_conf_does_not_count_as_a_prediction():
    gt = _gt((1, COCO_91_IDS[0], [0, 0, 10, 10]))
    out = per_class_pr({1: [_det(0, 0, 10, 10, 0.1, 0)]}, gt, conf=0.3)
    person = _find(out, "person")
    assert (person.tp, person.fp, person.fn) == (0, 0, 1)


def test_a_second_detection_of_one_object_is_a_false_positive():
    # Greedy matching: the best box takes the ground truth, the duplicate
    # cannot match it again.
    gt = _gt((1, COCO_91_IDS[0], [0, 0, 10, 10]))
    dets = [_det(0, 0, 10, 10, 0.9, 0), _det(0, 0, 10, 10, 0.8, 0)]
    person = _find(per_class_pr({1: dets}, gt, conf=0.3), "person")
    assert (person.tp, person.fp) == (1, 1)


def test_crowd_regions_are_excluded_rather_than_counted_as_misses():
    # iscrowd marks "a pile of objects, not individually labelled". Counting
    # them would penalise correct behaviour. The crowd box contributes no
    # support, so person is all-zero and drops out of the table entirely.
    gt = _gt((1, COCO_91_IDS[0], [0, 0, 10, 10]), iscrowd=1)
    assert per_class_pr({1: []}, gt, conf=0.3) == []


def test_a_crowd_box_is_not_counted_even_alongside_a_real_one():
    gt = {
        **_gt((1, COCO_91_IDS[0], [0, 0, 10, 10])),
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": COCO_91_IDS[0],
             "bbox": [0, 0, 10, 10], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": COCO_91_IDS[0],
             "bbox": [50, 50, 10, 10], "iscrowd": 1},
        ],
    }
    person = _find(per_class_pr({1: []}, gt, conf=0.3), "person")
    assert (person.support, person.fn) == (1, 1)  # the crowd box is not the 2nd


def test_classes_absent_from_the_subset_are_left_out_of_the_table():
    # A 300-image subset does not contain all 80 classes; all-zero rows would
    # be noise, so per_class_pr drops them.
    gt = _gt((1, COCO_91_IDS[0], [0, 0, 10, 10]))
    names = {c.name for c in per_class_pr({1: [_det(0, 0, 10, 10, 0.9, 0)]}, gt, 0.3)}
    assert names == {"person"}
    assert len(names) < len(COCO_CLASSES)


# --- format_report ----------------------------------------------------------


def _classes() -> list[ClassMetrics]:
    return [
        ClassMetrics("cup", support=40, tp=10, fp=20, fn=30),
        ClassMetrics("person", support=40, tp=30, fp=5, fn=10),
        ClassMetrics("tv", support=5, tp=0, fp=0, fn=5),
        ClassMetrics("zebra", support=0, tp=0, fp=0, fn=0),
    ]


def _micro_row(report: str) -> str:
    return next(l for l in report.splitlines() if "micro-average" in l)


def test_micro_average_matches_the_summed_counts():
    report = format_report(["model: x"], {"mAP@0.5": 0.5}, _classes(), [], 0.30, 10)
    # tp=40 fp=25 fn=45 -> p=40/65, r=40/85
    assert "| 85 | 40 | 25 | 45 | 0.615 | 0.471 | 0.533 |" in _micro_row(report)


def test_micro_average_survives_an_all_zero_class_set():
    zeros = [ClassMetrics("zebra", support=0, tp=0, fp=0, fn=0)]
    report = format_report(["model: x"], {"mAP@0.5": 0.0}, zeros, [], 0.30, 1)
    assert "| 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |" in _micro_row(report)


def test_the_report_renders_classes_in_the_order_given():
    # format_report does not sort -- main does, once, so the .md and the .csv
    # written by the same run cannot disagree on row order.
    classes = _classes()
    report = format_report(["model: x"], {"mAP@0.5": 0.5}, classes, [], 0.30, 10)
    rendered = [
        line.split("|")[1].strip()
        for line in report.splitlines()
        if line.startswith("| ") and "---" not in line
    ]
    for c in classes:
        assert c.name in rendered
    assert rendered.index("cup") < rendered.index("person") < rendered.index("tv")


def test_the_report_states_the_image_count_it_was_given():
    report = format_report(["model: x"], {"mAP@0.5": 0.5}, _classes(), [], 0.30, 296)
    assert "N=296 labelled images" in report


def test_the_report_says_the_numbers_are_not_from_a_webcam():
    # The module's central claim: no accuracy figure is attached to webcam
    # footage, because webcam footage has no ground truth.
    report = format_report(["model: x"], {"mAP@0.5": 0.5}, _classes(), [], 0.30, 10)
    assert "not webcam measurements" in report


def test_the_report_records_that_map_used_the_low_confidence_tail():
    # mAP at 0.001 and P/R/F1 at --pr-conf are different thresholds on purpose;
    # the report has to say which is which or the numbers look inconsistent.
    report = format_report(["model: x"], {"mAP@0.5": 0.5}, _classes(), [], 0.30, 10)
    assert "conf=0.001" in report
    assert "conf=0.30" in report
