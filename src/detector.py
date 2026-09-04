"""YOLOX-Tiny inference through ONNX Runtime.

The released YOLOX ONNX is a bare backbone+head: it hands back 3549 raw
candidates of the form (dx, dy, log_w, log_h, objectness, 80 class scores) and
expects the caller to do the grid decode, thresholding and NMS. All of that
lives here, in NumPy, so the only runtime dependency is onnxruntime itself.

Two things about the model that are easy to get wrong and cost hours:

1. It takes raw 0-255 **BGR**. No /255, no mean/std, no cvtColor. Verified
   against the released weights, not assumed -- see tests/test_detector.py.
2. Objectness and class scores are already sigmoid'd inside the graph, but the
   box terms are not: xy are grid-cell offsets and wh are log-space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .coco_classes import COCO_CLASSES

# YOLOX FPN strides. The input side must be divisible by the largest of these.
STRIDES: tuple[int, ...] = (8, 16, 32)


class ModelNotFoundError(FileNotFoundError):
    """Weights are missing. Raised with the command that fetches them."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One box in the coordinate space of the *original* frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    @property
    def class_name(self) -> str:
        return COCO_CLASSES[self.class_id]

    @property
    def box_int(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_xywh(self) -> tuple[float, float, float, float]:
        """COCO-style [x, y, w, h], which is what evaluate.py has to emit."""
        return self.x1, self.y1, self.width, self.height


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, score-descending.

    Vectorised: the IoU of the current best box against all remaining boxes is
    one array operation, so the Python loop runs once per *kept* box rather than
    once per candidate. After confidence thresholding there are usually fewer
    than 30 candidates, so this comfortably beats calling into cv2.dnn.

    boxes: (N, 4) as x1, y1, x2, y2.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # +1 is deliberately absent: these are continuous coordinates, not pixel
    # indices, so a zero-width box should have zero area.
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]

        ix1 = np.maximum(x1[best], x1[rest])
        iy1 = np.maximum(y1[best], y1[rest])
        ix2 = np.minimum(x2[best], x2[rest])
        iy2 = np.minimum(y2[best], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Per-class NMS in a single pass.

    Offsetting every box by (class_id * a large constant) puts each class in its
    own region of coordinate space, so boxes of different classes can never
    overlap and one global NMS call behaves exactly like N per-class calls. A
    person standing in front of a TV keeps both boxes, which is what you want.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    # Larger than any plausible image dimension, so classes cannot collide.
    offsets = class_ids.astype(np.float32) * 100_000.0
    return nms(boxes + offsets[:, None], scores, iou_threshold)


def _make_grids(input_h: int, input_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid centres and strides for every anchor, in the model's output order.

    Built once per input size and cached -- it is pure geometry and does not
    change from frame to frame, but rebuilding it per frame costs ~0.4 ms.
    """
    grids: list[np.ndarray] = []
    strides: list[np.ndarray] = []
    for stride in STRIDES:
        gh, gw = input_h // stride, input_w // stride
        yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        strides.append(np.full((1, grid.shape[1], 1), stride))
    return (
        np.concatenate(grids, axis=1).astype(np.float32),
        np.concatenate(strides, axis=1).astype(np.float32),
    )


class Detector:
    """Loads the ONNX model once and turns frames into `Detection` lists."""

    def __init__(
        self,
        model_path: Path | str,
        input_size: tuple[int, int] = (416, 416),
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        class_filter: Iterable[int] | None = None,
        device: str = "cpu",
        intra_op_threads: int = 0,
        inter_op_threads: int = 1,
        preprocess_mode: str = "prealloc",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # The INT8 model is built locally rather than downloaded, so the
            # two cases need different instructions.
            if "_int8" in self.model_path.stem:
                hint = (
                    "This is the quantised model, which is built locally (~8s):\n"
                    "  python models/download_weights.py    # downloads FP32, then builds it\n"
                    "  python models/quantize.py            # if you already have the FP32 weights\n"
                    "Or run the FP32 model directly:\n"
                    "  python -m src.main --model models/yolox_tiny.onnx"
                )
            else:
                hint = "Fetch them with:  python models/download_weights.py"
            raise ModelNotFoundError(f"Model weights not found at {self.model_path}\n{hint}")

        self.input_h, self.input_w = int(input_size[0]), int(input_size[1])
        if self.input_h % 32 or self.input_w % 32:
            raise ValueError(
                f"input_size must be a multiple of 32, got "
                f"{self.input_h}x{self.input_w}"
            )

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.class_filter = set(class_filter) if class_filter else None
        self.preprocess_mode = preprocess_mode

        self.session = self._build_session(device, intra_op_threads, inter_op_threads)
        self.input_name = self.session.get_inputs()[0].name
        self.providers = self.session.get_providers()

        declared = self.session.get_inputs()[0].shape
        self._input_is_dynamic = not (
            isinstance(declared[2], int) and isinstance(declared[3], int)
        )
        if not self._input_is_dynamic and (
            declared[2] != self.input_h or declared[3] != self.input_w
        ):
            raise ValueError(
                f"{self.model_path.name} is exported for a fixed "
                f"{declared[2]}x{declared[3]} input but {self.input_h}x{self.input_w} "
                f"was requested.\nRun:  python models/make_dynamic.py  "
                f"to produce a variable-resolution copy."
            )

        self._grids, self._grid_strides = _make_grids(self.input_h, self.input_w)

        # Hot-path buffers. Allocated once, reused for the life of the detector.
        self._canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        self._blob = np.empty((1, 3, self.input_h, self.input_w), dtype=np.float32)
        self._resize_buf: np.ndarray | None = None
        self._resize_key: tuple[int, int] | None = None
        self._last_ratio: float = 1.0

    # ------------------------------------------------------------- session

    def _build_session(
        self, device: str, intra_op: int, inter_op: int
    ) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 0 means "leave ONNX Runtime's own heuristic alone", which is the right
        # default; the benchmark's ablation mode passes explicit values.
        if intra_op > 0:
            options.intra_op_num_threads = intra_op
        if inter_op > 0:
            options.inter_op_num_threads = inter_op
        # Nothing here benefits from spilling logs to stdout during a live demo.
        options.log_severity_level = 3

        available = ort.get_available_providers()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "--device cuda requested but no CUDAExecutionProvider is "
                    "installed. Install onnxruntime-gpu, or use --device cpu "
                    "(the default, and what this project is tuned for)."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # auto
            providers = [p for p in ("CUDAExecutionProvider",) if p in available]
            providers.append("CPUExecutionProvider")

        return ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=providers
        )

    # ---------------------------------------------------------- preprocess

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox to the network input and return an NCHW float32 blob.

        Padding goes bottom/right only (YOLOX's own convention), which means
        mapping boxes back is a single divide by the scale ratio with no offset
        term -- one less thing to get wrong.
        """
        h, w = frame.shape[:2]
        ratio = min(self.input_h / h, self.input_w / w)
        self._last_ratio = ratio
        new_w, new_h = int(w * ratio), int(h * ratio)

        if self.preprocess_mode == "naive":
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
            canvas[:new_h, :new_w] = resized
            return np.ascontiguousarray(
                canvas.transpose(2, 0, 1)[None].astype(np.float32)
            )

        # Reuse the resize target across frames. Source resolution is constant
        # for a given stream, so this allocates exactly once in practice.
        if self._resize_key != (new_h, new_w):
            self._resize_buf = np.empty((new_h, new_w, 3), dtype=np.uint8)
            self._resize_key = (new_h, new_w)
            self._canvas[:] = 114  # only the padding region needs re-filling

        cv2.resize(
            frame, (new_w, new_h), dst=self._resize_buf, interpolation=cv2.INTER_LINEAR
        )
        self._canvas[:new_h, :new_w] = self._resize_buf
        # transpose gives a view; the assignment does the uint8->float32 cast and
        # the copy in one pass, straight into the buffer ORT will read.
        self._blob[0] = self._canvas.transpose(2, 0, 1)
        return self._blob

    # ------------------------------------------------------------ inference

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Raw network output, shape (1, num_anchors, 85)."""
        return self.session.run(None, {self.input_name: blob})[0]

    # ----------------------------------------------------------- postprocess

    def postprocess(self, raw: np.ndarray, ratio: float | None = None) -> list[Detection]:
        """Grid decode -> threshold -> NMS -> original-frame coordinates."""
        ratio = self._last_ratio if ratio is None else ratio
        predictions = raw[0]

        # Score first, decode second: thresholding on ~3.5k rows is cheap, and
        # decoding boxes we are about to throw away is not.
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = scores >= self.conf_threshold
        if self.class_filter is not None:
            keep &= np.isin(class_ids, list(self.class_filter))
        if not keep.any():
            return []

        boxes_raw = predictions[keep, :4]
        scores = scores[keep]
        class_ids = class_ids[keep]
        grids = self._grids[0][keep]
        strides = self._grid_strides[0][keep]

        centers = (boxes_raw[:, :2] + grids) * strides
        sizes = np.exp(boxes_raw[:, 2:4]) * strides

        half = sizes * 0.5
        boxes = np.empty((boxes_raw.shape[0], 4), dtype=np.float32)
        boxes[:, :2] = centers - half
        boxes[:, 2:] = centers + half
        boxes /= ratio  # letterbox padding was bottom/right, so no offset

        kept = batched_nms(boxes, scores, class_ids, self.iou_threshold)
        if kept.size > self.max_detections:
            kept = kept[: self.max_detections]

        return [
            Detection(
                x1=float(boxes[i, 0]),
                y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]),
                y2=float(boxes[i, 3]),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
            )
            for i in kept
        ]

    # ------------------------------------------------------------------ api

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """The one call most users need. Stages are public for instrumentation."""
        blob = self.preprocess(frame)
        raw = self.infer(blob)
        return self.postprocess(raw)

    def warmup(self, rounds: int = 3) -> None:
        """Burn the first few runs.

        ORT's first run() allocates its arena and spins up the thread pool, and
        comes in 5-10x slower than steady state. Doing it here keeps that cost
        out of both the HUD and the benchmark.
        """
        dummy = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.detect(dummy)

    def describe(self) -> dict[str, object]:
        """Provenance for the CSV/README so a number can be traced to a config."""
        return {
            "model": self.model_path.name,
            "input_size": f"{self.input_h}x{self.input_w}",
            "providers": ",".join(self.providers),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "preprocess_mode": self.preprocess_mode,
        }


def set_opencv_threads(n: int) -> None:
    """Cap OpenCV's thread pool. 0 leaves OpenCV's own default in place.

    Capping is *not* the default, despite the obvious-sounding argument that
    OpenCV and ONNX Runtime both grabbing 8 threads must oversubscribe a 4-core
    chip. Measured on the development laptop over three interleaved rounds,
    capping to 2 cost about 5% (20.1 -> 19.2 FPS). Preprocessing is roughly
    1.1 ms of a 50 ms frame, so there was very little to reclaim. The knob stays
    because it is the right lever on a busier machine -- it is just not a win
    here. Numbers in docs/PERFORMANCE_ANALYSIS.md.
    """
    if n > 0:
        cv2.setNumThreads(n)


def cpu_label() -> str:
    """Best-effort CPU name for report headers. Never fails the run."""
    try:
        import platform

        return platform.processor() or f"{os.cpu_count()} logical cores"
    except Exception:
        return "unknown CPU"


def summarize_detections(detections: Sequence[Detection]) -> str:
    """`2x person, 1x laptop` -- used in console output and the demo checklist."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return ", ".join(f"{n}x {name}" for name, n in sorted(counts.items())) or "nothing"
