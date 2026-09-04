"""Detector contract and the model's preprocessing assumption.

These need the weights. If they are missing the tests skip with the command that
fetches them rather than failing -- a clean clone should not look broken.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.coco_classes import COCO_CLASSES, resolve_class_filter
from src.config import REPO_ROOT
from src.detector import Detection, Detector, ModelNotFoundError

MODEL_PATH = REPO_ROOT / "models" / "yolox_tiny.onnx"
INT8_PATH = REPO_ROOT / "models" / "yolox_tiny_int8.onnx"
SAMPLE_PATH = REPO_ROOT / "assets" / "sample.jpg"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="weights absent - run: python models/download_weights.py",
)
needs_int8 = pytest.mark.skipif(
    not INT8_PATH.exists(),
    reason="INT8 model absent - run: python models/quantize.py",
)


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector(MODEL_PATH, conf_threshold=0.30)


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    image = cv2.imread(str(SAMPLE_PATH), cv2.IMREAD_COLOR)
    assert image is not None, f"bundled sample missing: {SAMPLE_PATH}"
    return image


def test_missing_weights_raise_a_useful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="download_weights"):
        Detector(tmp_path / "absent.onnx")


def test_missing_int8_model_explains_that_it_is_built_locally(tmp_path):
    """The INT8 model is the shipped default but is not downloadable, so its
    error has to point at the build step, not at the download script alone."""
    with pytest.raises(ModelNotFoundError, match="quantize.py") as exc:
        Detector(tmp_path / "yolox_tiny_int8.onnx")
    assert "yolox_tiny.onnx" in str(exc.value), "should offer the FP32 fallback"


@needs_int8
def test_int8_model_detects_the_same_scene(sample):
    """Quantisation costs accuracy but must not break the pipeline: the same
    obvious objects should still be found in the bundled sample."""
    quantised = Detector(INT8_PATH, conf_threshold=0.30)
    found = {d.class_name for d in quantised.detect(sample)}
    for expected in ("laptop", "keyboard", "tv"):
        assert expected in found, f"expected {expected}, got {sorted(found)}"


@needs_int8
def test_int8_model_is_substantially_smaller():
    assert INT8_PATH.stat().st_size < 0.5 * MODEL_PATH.stat().st_size


@needs_model
def test_rejects_input_size_that_is_not_a_multiple_of_32():
    with pytest.raises(ValueError, match="multiple of 32"):
        Detector(MODEL_PATH, input_size=(400, 400))


@needs_model
def test_fixed_export_rejects_a_foreign_resolution():
    """The stock release is exported at a hard 416; asking for anything else
    must point at make_dynamic.py rather than fail deep inside ORT."""
    with pytest.raises(ValueError, match="make_dynamic"):
        Detector(MODEL_PATH, input_size=(320, 320))


@needs_model
def test_preprocess_produces_the_expected_blob(detector, sample):
    blob = detector.preprocess(sample)
    assert blob.shape == (1, 3, 416, 416)
    assert blob.dtype == np.float32
    # Raw 0-255, not normalised. See test_normalisation_would_break_the_model.
    assert blob.max() > 1.0


@needs_model
def test_preprocess_reuses_its_buffer(detector, sample):
    """The hot path must not allocate a new blob per frame."""
    first = detector.preprocess(sample)
    second = detector.preprocess(sample)
    assert first is second


@needs_model
def test_both_preprocess_modes_give_identical_output(sample):
    """The 'naive' mode exists only for the ablation table, so it has to be
    numerically identical -- otherwise the before/after compares two detectors."""
    fast = Detector(MODEL_PATH, preprocess_mode="prealloc")
    slow = Detector(MODEL_PATH, preprocess_mode="naive")
    np.testing.assert_array_equal(fast.preprocess(sample), slow.preprocess(sample))


@needs_model
def test_detect_returns_detections_with_a_sane_shape(detector, sample):
    detections = detector.detect(sample)
    assert detections, "the bundled sample should produce detections"
    for det in detections:
        assert isinstance(det, Detection)
        assert 0.0 <= det.score <= 1.0
        assert 0 <= det.class_id < len(COCO_CLASSES)
        assert det.x2 > det.x1 and det.y2 > det.y1
        assert det.class_name in COCO_CLASSES


@needs_model
def test_detections_land_inside_the_frame(detector, sample):
    h, w = sample.shape[:2]
    # A slack of a few pixels: boxes may legitimately extend a little past the
    # edge for a partly visible object, but not by half a frame.
    for det in detector.detect(sample):
        assert -0.1 * w <= det.x1 <= 1.1 * w
        assert -0.1 * h <= det.y1 <= 1.1 * h


@needs_model
def test_known_objects_are_found_in_the_sample(detector, sample):
    """The sample is COCO val2017 #340894, labelled with laptop/keyboard/tv/
    mouse among others. Anything less than these four means the decode or the
    preprocessing has regressed."""
    found = {d.class_name for d in detector.detect(sample)}
    for expected in ("laptop", "keyboard", "tv", "mouse"):
        assert expected in found, f"expected {expected}, got {sorted(found)}"


@needs_model
def test_normalisation_would_break_the_model(detector, sample):
    """Guards the single assumption the whole preprocessing path rests on.

    YOLOX's ONNX export consumes raw 0-255 BGR. Dividing by 255 the way a
    YOLOv8-style pipeline does is silent -- no error, no warning, just an empty
    detection list. If this test ever starts passing with detections, the model
    file has changed and preprocess() needs revisiting.
    """
    blob = detector.preprocess(sample).copy()
    assert detector.postprocess(detector.infer(blob)), "raw 0-255 must detect"
    assert not detector.postprocess(detector.infer(blob / 255.0)), (
        "normalised input must not detect - the export takes raw 0-255"
    )


@needs_model
def test_raising_confidence_never_adds_detections(detector, sample):
    counts = []
    for conf in (0.10, 0.30, 0.60, 0.90):
        detector.conf_threshold = conf
        counts.append(len(detector.detect(sample)))
    detector.conf_threshold = 0.30
    assert counts == sorted(counts, reverse=True), counts


@needs_model
def test_class_filter_restricts_the_output(sample):
    wanted = resolve_class_filter(["laptop"])
    filtered = Detector(MODEL_PATH, conf_threshold=0.30, class_filter=wanted)
    names = {d.class_name for d in filtered.detect(sample)}
    assert names, "laptop should still be detected"
    assert names == {"laptop"}


@needs_model
def test_max_detections_is_respected(sample):
    capped = Detector(MODEL_PATH, conf_threshold=0.05, max_detections=3)
    assert len(capped.detect(sample)) <= 3


@needs_model
def test_blank_frame_produces_nothing(detector):
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


@needs_model
def test_non_square_frames_are_handled(detector):
    """Letterboxing has to cope with whatever aspect ratio the camera gives."""
    for shape in ((480, 640, 3), (720, 1280, 3), (640, 480, 3), (300, 300, 3)):
        assert detector.detect(np.zeros(shape, dtype=np.uint8)) == []


def test_resolve_class_filter_accepts_names_and_indices():
    assert resolve_class_filter(["person"]) == {0}
    assert resolve_class_filter(["0"]) == {0}
    assert resolve_class_filter(["cell phone", "67"]) == {67}
    assert resolve_class_filter(None) is None
    assert resolve_class_filter([]) is None


def test_resolve_class_filter_rejects_nonsense():
    with pytest.raises(ValueError, match="Unknown class"):
        resolve_class_filter(["dragon"])
    with pytest.raises(ValueError, match="Unknown class"):
        resolve_class_filter(["999"])
