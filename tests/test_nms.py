"""NMS behaviour. Pure geometry -- no model, no weights, runs anywhere."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import batched_nms, nms


def test_empty_input_returns_empty():
    keep = nms(np.empty((0, 4), dtype=np.float32), np.empty((0,)), 0.5)
    assert keep.shape == (0,)


def test_single_box_is_kept():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    keep = nms(boxes, np.array([0.9]), 0.5)
    assert keep.tolist() == [0]


def test_heavily_overlapping_boxes_collapse_to_the_highest_score():
    # Two near-identical boxes; only the more confident one should survive.
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.6, 0.9])
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [1]


def test_disjoint_boxes_are_both_kept():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    keep = nms(boxes, np.array([0.9, 0.8]), 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_output_is_ordered_by_descending_score():
    boxes = np.array(
        [[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]], dtype=np.float32
    )
    scores = np.array([0.3, 0.9, 0.6])
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2, 0]


def test_threshold_controls_suppression():
    # IoU of these two is 1/7 ~= 0.143.
    boxes = np.array([[0, 0, 10, 10], [8, 0, 18, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8])
    iou = 20.0 / 180.0

    assert len(nms(boxes, scores, iou + 0.01)) == 2, "above IoU: both survive"
    assert len(nms(boxes, scores, iou - 0.01)) == 1, "below IoU: one suppressed"


def test_zero_area_box_does_not_divide_by_zero():
    boxes = np.array([[5, 5, 5, 5], [0, 0, 10, 10]], dtype=np.float32)
    keep = nms(boxes, np.array([0.9, 0.8]), 0.5)
    assert len(keep) == 2
    assert np.isfinite(boxes).all()


def test_batched_nms_keeps_overlapping_boxes_of_different_classes():
    """A person standing in front of a TV must keep both boxes."""
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.85])
    same_class = np.array([0, 0])
    different = np.array([0, 62])

    assert len(batched_nms(boxes, scores, same_class, 0.5)) == 1
    assert len(batched_nms(boxes, scores, different, 0.5)) == 2


def test_batched_nms_does_not_mutate_its_input():
    """The class offset must be applied to a copy -- the caller reads these
    boxes back afterwards to build Detection objects."""
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    original = boxes.copy()
    batched_nms(boxes, np.array([0.9]), np.array([5]), 0.5)
    np.testing.assert_array_equal(boxes, original)


@pytest.mark.parametrize("n", [1, 10, 200])
def test_indices_are_always_valid(n):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 500, size=(n, 2)).astype(np.float32)
    boxes = np.hstack([xy, xy + rng.uniform(5, 50, size=(n, 2)).astype(np.float32)])
    keep = nms(boxes, rng.uniform(0, 1, size=n), 0.5)
    assert len(set(keep.tolist())) == len(keep), "no duplicate indices"
    assert all(0 <= i < n for i in keep)
