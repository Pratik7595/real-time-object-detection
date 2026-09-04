"""Threaded frame source: webcam index, video file, or still image.

The whole point of this module is that inference is never made to wait on a
backlog. For a live camera the capture thread keeps exactly one frame -- the
newest -- and throws away anything the consumer did not get to. Queueing instead
would mean that every time inference falls behind, the picture on screen drifts
further into the past and never recovers.

For a *file* source that behaviour would be wrong: a benchmark or an evaluation
must see every frame. So file sources use a blocking handoff instead of a drop
buffer, and the producer waits for the consumer.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


class CameraError(RuntimeError):
    """Camera or file could not be opened, or produced no frames in time."""


def _backend_candidates() -> list[int]:
    """Capture backends to try, best-first for the current OS.

    This is platform-*adaptive*, not platform-specific: every entry falls back to
    cv2.CAP_ANY, so the code path is identical everywhere and nothing here is a
    hard OS dependency. The ordering matters in practice -- on Windows, MSMF can
    take 3-5 seconds to open a camera that DirectShow opens instantly.
    """
    if sys.platform.startswith("win"):
        names = ("CAP_DSHOW", "CAP_MSMF")
    elif sys.platform == "darwin":
        names = ("CAP_AVFOUNDATION",)
    else:
        names = ("CAP_V4L2",)

    backends = [getattr(cv2, n) for n in names if hasattr(cv2, n)]
    backends.append(cv2.CAP_ANY)
    return backends


class VideoStream:
    """Background-thread frame source with a one-slot buffer."""

    def __init__(
        self,
        source: str | int | Path = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        buffer_size: int = 1,
        open_timeout_s: float = 6.0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.open_timeout_s = open_timeout_s

        self.kind = self._classify(source)
        # Cameras drop stale frames (bounded latency); files must not (no data loss).
        self.drop_stale = self.kind == "camera"

        self._cap: cv2.VideoCapture | None = None
        self._still: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._stopped = threading.Event()
        self._ended = False
        self._error: BaseException | None = None
        self.frames_produced = 0
        self.frames_dropped = 0

    # ---------------------------------------------------------------- setup

    @staticmethod
    def _classify(source: str | int | Path) -> str:
        if isinstance(source, int):
            return "camera"
        text = str(source)
        if text.isdigit():
            return "camera"
        return "image" if Path(text).suffix.lower() in IMAGE_SUFFIXES else "video"

    def _open(self) -> None:
        if self.kind == "image":
            path = Path(str(self.source))
            if not path.exists():
                raise CameraError(f"Image not found: {path}")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise CameraError(
                    f"OpenCV could not decode {path}. Is it actually an image?"
                )
            self._still = image
            return

        if self.kind == "video":
            path = Path(str(self.source))
            if not path.exists():
                raise CameraError(
                    f"Video file not found: {path}\n"
                    f"Pass a path that exists, or use --source 0 for a webcam."
                )
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise CameraError(
                    f"OpenCV opened no stream for {path}. The container or codec "
                    f"may be unsupported by this OpenCV build."
                )
            self._cap = cap
            return

        # Camera: try each backend, keep the first that yields a real frame.
        index = int(self.source)
        tried: list[str] = []
        for backend in _backend_candidates():
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                tried.append(f"backend={backend} (would not open)")
                continue

            # Must be set before the first read() or some drivers ignore them.
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            if self.fps:
                cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.buffer_size))
            except cv2.error:
                pass  # Not supported by every backend; the drop buffer covers us.

            ok, frame = cap.read()
            if ok and frame is not None:
                self._cap = cap
                with self._cond:
                    self._frame = frame
                    self.frames_produced += 1
                    self._cond.notify_all()
                return

            cap.release()
            tried.append(f"backend={backend} (opened, no frames)")

        raise CameraError(
            f"Could not read from camera index {index}.\n"
            f"Tried: {'; '.join(tried) or 'no backends available'}\n"
            f"Common causes: another app is holding the camera (Teams, Zoom, the "
            f"Windows Camera app), the OS camera permission is off, or the index "
            f"is wrong -- try --source 1.\n"
            f"No camera at all? Everything here also runs on a file: "
            f"--source assets/sample.jpg"
        )

    # ---------------------------------------------------------------- thread

    def start(self) -> VideoStream:
        self._open()
        self._thread = threading.Thread(
            target=self._run, name="capture", daemon=True
        )
        self._thread.start()

        # Fail fast rather than showing a black window for ever.
        deadline = time.perf_counter() + self.open_timeout_s
        with self._cond:
            while self._frame is None and not self._ended and self._error is None:
                if not self._cond.wait(timeout=max(0.0, deadline - time.perf_counter())):
                    self.stop()
                    raise CameraError(
                        f"Source {self.source!r} opened but produced no frame within "
                        f"{self.open_timeout_s:.0f}s. If this is a webcam, another "
                        f"application is probably using it."
                    )
        if self._error is not None:
            self.stop()
            raise CameraError(f"Capture thread failed to start: {self._error}")
        return self

    def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                if self.kind == "image":
                    assert self._still is not None
                    frame = self._still.copy()  # consumers draw on it
                    ok = True
                else:
                    assert self._cap is not None
                    ok, frame = self._cap.read()

                if not ok or frame is None:
                    with self._cond:
                        self._ended = True
                        self._cond.notify_all()
                    return

                with self._cond:
                    # File sources: wait until the consumer has taken the frame,
                    # so nothing is silently skipped.
                    while (
                        not self.drop_stale
                        and self._frame is not None
                        and not self._stopped.is_set()
                    ):
                        self._cond.wait(timeout=0.5)
                    if self._stopped.is_set():
                        return
                    if self._frame is not None:
                        self.frames_dropped += 1
                    self._frame = frame
                    self.frames_produced += 1
                    self._cond.notify_all()
        except BaseException as exc:  # surfaced to the consumer via read()
            with self._cond:
                self._error = exc
                self._ended = True
                self._cond.notify_all()

    # ---------------------------------------------------------------- read

    def read(self, timeout: float = 2.0) -> np.ndarray | None:
        """Return the newest frame, or None when the source is exhausted.

        Blocks until a frame is available. That is deliberate: re-running the
        detector on a frame we have already processed would inflate the FPS
        counter without showing the user anything new.
        """
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._frame is None:
                if self._error is not None:
                    raise CameraError(f"Capture thread died: {self._error}")
                if self._ended or self._stopped.is_set():
                    return None
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not self._cond.wait(timeout=remaining):
                    return None
            frame = self._frame
            self._frame = None
            self._cond.notify_all()  # unblocks a file producer
        return frame

    def frames(self, limit: int | None = None) -> Iterator[np.ndarray]:
        """Iterate frames until the source ends or `limit` frames are yielded."""
        count = 0
        while limit is None or count < limit:
            frame = self.read()
            if frame is None:
                return
            count += 1
            yield frame

    # ---------------------------------------------------------------- teardown

    @property
    def is_live(self) -> bool:
        return self.kind == "camera"

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) actually delivered -- drivers routinely ignore the
        resolution you asked for, so always report what arrived."""
        if self._still is not None:
            h, w = self._still.shape[:2]
            return w, h
        if self._cap is not None:
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return 0, 0

    @property
    def total_frames(self) -> int:
        """Frame count for a video file; 0 when unknown or unbounded."""
        if self._cap is not None and self.kind == "video":
            return max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        return 0

    def stop(self) -> None:
        self._stopped.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._still = None

    def __enter__(self) -> VideoStream:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
