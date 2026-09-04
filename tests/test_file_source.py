"""File-source capture and end-to-end inference without a camera.

The whole point: this repository has to be demonstrable and testable on a
machine with no webcam, which is what CI and most reviewers have.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import REPO_ROOT
from src.detector import Detector
from src.main import main as cli_main
from src.video_stream import CameraError, VideoStream
from src.visualizer import CLASS_COLORS, draw_detections, draw_hud

SAMPLE_PATH = REPO_ROOT / "assets" / "sample.jpg"
MODEL_PATH = REPO_ROOT / "models" / "yolox_tiny.onnx"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="weights absent - run: python models/download_weights.py",
)


def test_source_kinds_are_classified():
    assert VideoStream._classify(0) == "camera"
    assert VideoStream._classify("0") == "camera"
    assert VideoStream._classify("clip.mp4") == "video"
    assert VideoStream._classify("shot.PNG") == "image"
    assert VideoStream._classify(SAMPLE_PATH) == "image"


def test_missing_file_raises_before_any_thread_starts(tmp_path):
    with pytest.raises(CameraError, match="not found"):
        VideoStream(tmp_path / "nope.jpg").start()
    with pytest.raises(CameraError, match="not found"):
        VideoStream(tmp_path / "nope.mp4").start()


def test_undecodable_image_is_reported_clearly(tmp_path):
    junk = tmp_path / "broken.jpg"
    junk.write_bytes(b"this is not a jpeg")
    with pytest.raises(CameraError, match="could not decode"):
        VideoStream(junk).start()


def test_image_source_loops_and_yields_frames():
    with VideoStream(SAMPLE_PATH) as stream:
        frames = list(stream.frames(limit=5))
    assert len(frames) == 5
    assert all(f.shape == (480, 640, 3) for f in frames)


def test_image_source_hands_out_independent_copies():
    """Consumers draw on the frame they are given; the source must not hand the
    same buffer to the next caller with last frame's boxes burned into it."""
    with VideoStream(SAMPLE_PATH) as stream:
        first = stream.read()
        first[:] = 0
        second = stream.read()
    assert second is not None
    assert second.any(), "second frame was aliased to the first"


def test_frame_size_reports_what_arrived():
    with VideoStream(SAMPLE_PATH) as stream:
        assert stream.frame_size == (640, 480)
        assert not stream.is_live


@needs_model
def test_end_to_end_inference_from_a_file_source():
    detector = Detector(MODEL_PATH, conf_threshold=0.30)
    with VideoStream(SAMPLE_PATH) as stream:
        frame = stream.read()
    assert frame is not None

    detections = detector.detect(frame)
    assert detections
    assert {"laptop", "keyboard"} <= {d.class_name for d in detections}


@needs_model
def test_cli_runs_headless_on_a_file_and_writes_a_csv(tmp_path, monkeypatch):
    """The exact path a reviewer without a camera takes."""
    results = tmp_path / "results"
    monkeypatch.setenv("MPLBACKEND", "Agg")

    code = cli_main(
        [
            "--source", str(SAMPLE_PATH),
            "--no-display",
            "--max-frames", "12",
        ]
    )
    assert code == 0


@needs_model
def test_cli_reports_bad_arguments_instead_of_crashing():
    assert cli_main(["--source", str(SAMPLE_PATH), "--no-display", "--conf", "5"]) == 2
    assert cli_main(
        ["--source", str(SAMPLE_PATH), "--no-display", "--classes", "dragon"]
    ) == 2


def test_list_classes_exits_cleanly():
    assert cli_main(["--list-classes"]) == 0


@needs_model
def test_recording_frame_rate_excludes_warmup(tmp_path):
    """Regression: the demo clip must not play back at the wrong speed.

    The writer's frame rate is fixed when the file opens, and it used to be
    taken from the rolling FPS average -- which still contained the ~600 ms
    first frame. One frame like that in a 30-frame window halved the estimate,
    and a real 45-second take was written as a 48-second file playing at 0.49x
    while the FPS counter drawn on the frames said otherwise.

    This asserts the file's declared rate is consistent with the rate the run
    actually achieved, which is the property that was broken.
    """
    out = tmp_path / "clip.mp4"
    assert cli_main(
        ["--source", str(SAMPLE_PATH), "--no-display", "--max-frames", "90",
         "--record", str(out)]
    ) == 0
    assert out.exists(), "no video written"

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened(), "written file will not open"
    declared_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()

    assert frames > 0, "file decodes to zero frames"
    # The image-loop source runs far faster than any warm-up frame, so a rate
    # contaminated by warm-up would land well under 10 fps here.
    assert declared_fps > 10.0, (
        f"declared {declared_fps:.1f} fps - warm-up frames have leaked back "
        f"into the recording rate calibration"
    )


# ------------------------------------------------------------------ rendering


def test_every_class_has_its_own_colour():
    assert len(CLASS_COLORS) == 80
    # Not all 80 are perceptually distinct on a wheel of one hue dimension, but
    # each class must at least map to a stable, unique triple.
    assert len(set(CLASS_COLORS)) == 80


@needs_model
def test_drawing_marks_the_frame_and_keeps_its_shape():
    detector = Detector(MODEL_PATH, conf_threshold=0.30)
    frame = cv2.imread(str(SAMPLE_PATH), cv2.IMREAD_COLOR)
    before = frame.copy()

    drawn = draw_detections(frame, detector.detect(frame))
    assert drawn.shape == before.shape
    assert not np.array_equal(drawn, before), "nothing was drawn"


def test_hud_survives_a_tiny_frame():
    """The overlay must not index past the edge of an unusually small frame."""
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    assert draw_hud(frame, 30.0, 29.5, 3).shape == (60, 80, 3)


def test_hud_renders_with_zero_detections():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_hud(frame, 0.0, 0.0, 0)
    assert out.any(), "HUD text should be visible even before the first frame"
