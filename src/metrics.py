"""Per-stage timing, rolling FPS, and the per-run CSV.

One module so the HUD, the benchmark and the evaluator all read the same
numbers. When timing lived inside the main loop the benchmark re-implemented it
and the two FPS figures quietly disagreed by ~8%, which is exactly the kind of
thing that makes a performance table worthless.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterator, Sequence

# Order matters: it is the CSV column order and the HUD line order.
STAGES: tuple[str, ...] = (
    "capture",      # waiting on / copying the newest frame from the capture thread
    "preprocess",   # letterbox + CHW
    "inference",    # ort session.run
    "postprocess",  # grid decode + NMS + class filter
    "render",       # drawing, HUD, imshow, optional VideoWriter
)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency so tests stay trivial."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


@dataclass
class Summary:
    frames: int
    elapsed_s: float
    fps_mean: float
    fps_median: float
    fps_p95: float
    fps_p5: float
    stage_ms_mean: dict[str, float]
    stage_ms_p95: dict[str, float]

    def as_row(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "frames": self.frames,
            "elapsed_s": round(self.elapsed_s, 3),
            "fps_mean": round(self.fps_mean, 2),
            "fps_median": round(self.fps_median, 2),
            "fps_p95": round(self.fps_p95, 2),
            "fps_p5": round(self.fps_p5, 2),
        }
        for stage in STAGES:
            row[f"{stage}_ms"] = round(self.stage_ms_mean.get(stage, 0.0), 2)
        return row


class Metrics:
    """Rolling window for the live HUD plus a full-history log for the CSV.

    The rolling window drives what you see on screen; the full history is what
    gets summarised at the end. Keeping both means a 5-minute run does not have
    its p95 dominated by the first two seconds of warm-up, and the HUD still
    reacts within a second when something changes.
    """

    def __init__(self, window: int = 120, keep_history: bool = True) -> None:
        self.window = window
        self.keep_history = keep_history
        self._rolling: dict[str, deque[float]] = {
            s: deque(maxlen=window) for s in STAGES
        }
        self._rolling_frame: deque[float] = deque(maxlen=window)
        self._history: list[dict[str, float]] = []
        self._current: dict[str, float] = {}
        self._frame_start: float | None = None
        self._last_frame_ms: float = 0.0
        self.frame_count = 0
        self.started_at = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one pipeline stage. Nested use is not supported and not needed."""
        if name not in self._rolling:
            raise KeyError(f"Unknown stage {name!r}; expected one of {STAGES}")
        if self._frame_start is None:
            self._frame_start = time.perf_counter()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            # Accumulate rather than assign: a stage may legitimately be entered
            # more than once per frame (e.g. render + record).
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._current[name] = self._current.get(name, 0.0) + elapsed_ms

    def end_frame(self) -> None:
        """Close the frame: fold the stage times into the window and history."""
        now = time.perf_counter()
        start = self._frame_start if self._frame_start is not None else now
        frame_ms = (now - start) * 1000.0

        for stage in STAGES:
            self._rolling[stage].append(self._current.get(stage, 0.0))
        self._rolling_frame.append(frame_ms)

        if self.keep_history:
            row = {s: self._current.get(s, 0.0) for s in STAGES}
            row["frame_ms"] = frame_ms
            self._history.append(row)

        self._last_frame_ms = frame_ms
        self.frame_count += 1
        self._current = {}
        self._frame_start = None

    @property
    def fps_instant(self) -> float:
        return 1000.0 / self._last_frame_ms if self._last_frame_ms > 0 else 0.0

    @property
    def fps_rolling(self) -> float:
        """Mean FPS over the window, computed from total time rather than as a
        mean of per-frame FPS values -- averaging reciprocals overstates FPS."""
        if not self._rolling_frame:
            return 0.0
        total_ms = sum(self._rolling_frame)
        return (len(self._rolling_frame) * 1000.0 / total_ms) if total_ms > 0 else 0.0

    def stage_ms(self, name: str) -> float:
        window = self._rolling[name]
        return sum(window) / len(window) if window else 0.0

    def summarize(self, skip_first: int = 0) -> Summary:
        """Summarise the run, optionally ignoring the first `skip_first` frames.

        Warm-up frames are *skipped, never deleted*: they stay in the CSV so the
        headline numbers can always be re-checked against the raw data. They are
        excluded from the summary because they measure one-off lazy
        initialisation, not steady-state throughput -- on this machine frame 0
        costs ~625 ms (first cv2.resize, first draw, ORT arena) against ~26 ms
        for every frame after it, which drags a 60-frame mean down by a third.
        """
        history = self._history[skip_first:] if skip_first else self._history
        if not history:
            empty = {s: 0.0 for s in STAGES}
            return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, empty, empty)

        frame_ms = [r["frame_ms"] for r in history]
        fps_values = [1000.0 / ms for ms in frame_ms if ms > 0]
        total_s = sum(frame_ms) / 1000.0

        return Summary(
            frames=len(history),
            elapsed_s=total_s,
            # Throughput mean: frames / total time. Not mean(per-frame FPS).
            fps_mean=(len(frame_ms) / total_s) if total_s > 0 else 0.0,
            fps_median=median(fps_values) if fps_values else 0.0,
            # p95 FPS is the *fast* tail; p5 is the stutter that users notice.
            fps_p95=percentile(fps_values, 95),
            fps_p5=percentile(fps_values, 5),
            stage_ms_mean={
                s: sum(r[s] for r in history) / len(history) for s in STAGES
            },
            stage_ms_p95={
                s: percentile([r[s] for r in history], 95) for s in STAGES
            },
        )

    def write_csv(self, path: Path) -> Path:
        """One row per frame. Written even for short runs -- a CSV nobody looks
        at costs nothing; a measurement you cannot re-check costs credibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["frame", *STAGES, "frame_ms", "fps"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for i, row in enumerate(self._history):
                frame_ms = row["frame_ms"]
                writer.writerow(
                    {
                        "frame": i,
                        **{s: round(row[s], 3) for s in STAGES},
                        "frame_ms": round(frame_ms, 3),
                        "fps": round(1000.0 / frame_ms, 2) if frame_ms > 0 else 0.0,
                    }
                )
        return path


def timestamped_path(directory: Path, prefix: str, suffix: str) -> Path:
    """`results/run_20260904-181530.csv` -- sortable, no collisions, no spaces."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return directory / f"{prefix}_{stamp}{suffix}"
