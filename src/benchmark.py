"""Fixed-length throughput benchmark: FPS, per-stage latency, CPU and RAM.

    python -m src.benchmark                     # model-input sweep, 320/416/512
    python -m src.benchmark --ablation          # optimisation before/after table
    python -m src.benchmark --threads-sweep     # ONNX Runtime thread scaling
    python -m src.benchmark --capture-sweep --source 0   # capture-resolution sweep

Defaults to `assets/sample.jpg` rather than a camera, on purpose. A webcam caps
throughput at its own frame rate (30 fps here) and varies with exposure and
lighting, so it measures the camera, not the detector. The image loop is
deterministic and reproducible on any machine with no downloads -- which also
means the capture-stage timing it reports is near-zero and not representative of
a live camera. Both numbers are in docs/PERFORMANCE_ANALYSIS.md, labelled.

Every table this prints is also written to results/ as CSV and Markdown.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, load_config
from .detector import Detector, ModelNotFoundError, set_opencv_threads
from .metrics import STAGES, Metrics, timestamped_path
from .video_stream import CameraError, VideoStream
from .visualizer import draw_detections

DEFAULT_SOURCE = REPO_ROOT / "assets" / "sample.jpg"
DEFAULT_FRAMES = 300
DEFAULT_WARMUP = 30


class ResourceSampler:
    """Polls process CPU% and RSS on a background thread during a run.

    Sampled rather than measured at the endpoints because RSS peaks mid-run (ORT
    grows its arena on the first few inferences) and an endpoint reading would
    miss it entirely.
    """

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.cpu_percent = 0.0
        self.rss_peak_mb = 0.0
        self.available = False

    def __enter__(self) -> ResourceSampler:
        try:
            import psutil
        except ImportError:
            print("note: psutil not installed; CPU/RAM columns will read 0", file=sys.stderr)
            return self

        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # prime: the first call always returns 0.0
        self.available = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.rss_peak_mb = max(
                    self.rss_peak_mb, self._proc.memory_info().rss / 1e6
                )
            except Exception:
                return

    def __exit__(self, *exc: object) -> None:
        if not self.available:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # cpu_percent over the whole window, relative to one core -- so 400% on
        # this 4-core i5 means all four cores saturated.
        self.cpu_percent = self._proc.cpu_percent(None)
        self.rss_peak_mb = max(self.rss_peak_mb, self._proc.memory_info().rss / 1e6)


@dataclass
class BenchResult:
    label: str
    model: str
    imgsz: int
    capture: str
    preprocess: str
    ort_threads: int
    cv_threads: int
    infer_every: int
    frames: int
    fps_mean: float
    fps_median: float
    fps_p95: float
    fps_p5: float
    cpu_percent: float
    rss_peak_mb: float
    detections_mean: float
    stage_ms: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, object]:
        row = {k: v for k, v in asdict(self).items() if k != "stage_ms"}
        row.update({f"{s}_ms": round(self.stage_ms.get(s, 0.0), 3) for s in STAGES})
        return row


def run_case(
    label: str,
    source: str | int,
    frames: int,
    warmup: int,
    model_path: Path,
    imgsz: int,
    preprocess_mode: str = "prealloc",
    ort_threads: int = 0,
    cv_threads: int = 2,
    infer_every: int = 1,
    conf: float = 0.30,
    iou: float = 0.45,
    capture_size: tuple[int, int] | None = None,
    draw: bool = True,
) -> BenchResult:
    """Run one configuration end to end and summarise it.

    Mirrors main.py's loop exactly -- including the drawing pass -- minus the
    imshow call. If the benchmark skipped rendering it would report a frame rate
    the live app can never actually hit.
    """
    set_opencv_threads(cv_threads)

    detector = Detector(
        model_path=model_path,
        input_size=(imgsz, imgsz),
        conf_threshold=conf,
        iou_threshold=iou,
        intra_op_threads=ort_threads,
        preprocess_mode=preprocess_mode,
    )
    detector.warmup(3)

    width, height = capture_size or (None, None)
    stream = VideoStream(source=source, width=width, height=height).start()
    actual_w, actual_h = stream.frame_size

    metrics = Metrics(window=frames + warmup, keep_history=True)
    detections: list = []
    total_detections = 0
    index = 0

    try:
        with ResourceSampler() as sampler:
            while index < frames + warmup:
                with metrics.stage("capture"):
                    frame = stream.read()
                if frame is None:
                    break

                if index % infer_every == 0:
                    with metrics.stage("preprocess"):
                        blob = detector.preprocess(frame)
                    with metrics.stage("inference"):
                        raw = detector.infer(blob)
                    with metrics.stage("postprocess"):
                        detections = detector.postprocess(raw)

                with metrics.stage("render"):
                    if draw:
                        draw_detections(frame, detections)

                total_detections += len(detections)
                metrics.end_frame()
                index += 1
    finally:
        stream.stop()

    summary = metrics.summarize(skip_first=warmup)
    counted = max(1, index - warmup)
    return BenchResult(
        label=label,
        model=model_path.name,
        imgsz=imgsz,
        capture=f"{actual_w}x{actual_h}",
        preprocess=preprocess_mode,
        ort_threads=ort_threads,
        cv_threads=cv_threads,
        infer_every=infer_every,
        frames=summary.frames,
        fps_mean=round(summary.fps_mean, 2),
        fps_median=round(summary.fps_median, 2),
        fps_p95=round(summary.fps_p95, 2),
        fps_p5=round(summary.fps_p5, 2),
        cpu_percent=round(sampler.cpu_percent, 1),
        rss_peak_mb=round(sampler.rss_peak_mb, 1),
        detections_mean=round(total_detections / counted, 2),
        stage_ms={s: round(v, 3) for s, v in summary.stage_ms_mean.items()},
    )


def settle(model_path: Path, imgsz: int, seconds: float) -> None:
    """Drive the CPU to a steady thermal state before anything is measured.

    This is not optional book-keeping -- it is the difference between a table
    that means something and one that does not. On the development laptop
    (i5-1135G7, thin chassis) four identical back-to-back 150-frame runs came out
    at 24.3, 21.0, 20.4 and 19.4 FPS: a 20% decline caused purely by the package
    dropping off its boost clock. Without a settle step the first configuration
    in any sweep is flattered and the last is penalised, and the "effect" you
    measure is the order you happened to run them in.

    So: burn the boost budget first, then measure everything warm. The numbers
    are lower than a cold run and they are the ones a user actually lives with.
    """
    if seconds <= 0:
        return
    print(f"settling : {seconds:.0f}s of load so every config is measured warm...")
    detector = Detector(model_path=model_path, input_size=(imgsz, imgsz))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        detector.detect(frame)


# --------------------------------------------------------------------- sweeps


def _dynamic_model(base: Path) -> Path:
    """Return a variable-resolution copy of `base`, building it if needed."""
    dynamic = base.with_name(f"{base.stem}_dynamic.onnx")
    if dynamic.exists():
        return dynamic
    print(f"building {dynamic.name} (needed for non-native resolutions)...")
    sys.path.insert(0, str(REPO_ROOT / "models"))
    from make_dynamic import make_dynamic  # noqa: E402

    return make_dynamic(base, dynamic)


def sweep_resolutions(args: argparse.Namespace, model: Path) -> list[BenchResult]:
    sizes = args.sizes
    results = []
    for size in sizes:
        # 416 is the model's native size, so it needs no rewritten graph.
        path = model if size == 416 else _dynamic_model(model)
        results.append(
            run_case(
                label=f"imgsz {size}",
                source=args.source,
                frames=args.frames,
                warmup=args.warmup,
                model_path=path,
                imgsz=size,
                conf=args.conf,
            )
        )
    return results


def sweep_ablation(args: argparse.Namespace, model: Path) -> list[BenchResult]:
    """Cumulative optimisations: each row adds one change to the row above."""
    # Note on reading this table: rows run in order and the machine heats up as
    # it goes, so a later row is measured under slightly worse conditions than
    # an earlier one. --settle bounds that, but for a difference of a few
    # percent (row C) the ordering still matters more than the change does.
    # Anything at that magnitude was re-checked with an interleaved A/B.
    cases = [
        # (label, preprocess, ort_threads, cv_threads, infer_every)
        ("A baseline (naive preprocess, per-frame allocation)", "naive", 0, 0, 1),
        ("B + preallocated preprocess buffers  [shipped]", "prealloc", 0, 0, 1),
        ("C + OpenCV capped to 2 threads (rejected)", "prealloc", 0, 2, 1),
        ("D + --infer-every 2", "prealloc", 0, 0, 2),
    ]
    results = []
    for label, pre, ort_t, cv_t, every in cases:
        results.append(
            run_case(
                label=label,
                source=args.source,
                frames=args.frames,
                warmup=args.warmup,
                model_path=model,
                imgsz=args.imgsz,
                preprocess_mode=pre,
                ort_threads=ort_t,
                cv_threads=cv_t,
                infer_every=every,
                conf=args.conf,
            )
        )

    int8 = model.with_name(f"{model.stem}_int8.onnx")
    if int8.exists():
        results.append(
            run_case(
                # Deliberately not naming the quantisation mode: quantize.py can
                # emit either dynamic or static INT8 into the same filename.
                label="E + INT8 quantisation (models/quantize.py)",
                source=args.source,
                frames=args.frames,
                warmup=args.warmup,
                model_path=int8,
                imgsz=args.imgsz,
                conf=args.conf,
            )
        )
    else:
        print(f"note: {int8.name} not found - skipping the INT8 row.")
        print("      build it with: python models/quantize.py")
    return results


def sweep_threads(args: argparse.Namespace, model: Path) -> list[BenchResult]:
    results = []
    for threads in (0, 1, 2, 4, 8):
        label = "ORT default" if threads == 0 else f"intra_op={threads}"
        results.append(
            run_case(
                label=label,
                source=args.source,
                frames=args.frames,
                warmup=args.warmup,
                model_path=model,
                imgsz=args.imgsz,
                ort_threads=threads,
                conf=args.conf,
            )
        )
    return results


def sweep_capture(args: argparse.Namespace, model: Path) -> list[BenchResult]:
    """Capture-resolution sweep. Only meaningful on a live camera."""
    stream_kind = VideoStream._classify(args.source)
    if stream_kind != "camera":
        print(
            f"error: --capture-sweep needs a camera (--source 0); "
            f"{args.source!r} is a {stream_kind} source with a fixed resolution.",
            file=sys.stderr,
        )
        return []

    results = []
    for width, height in ((640, 480), (960, 540), (1280, 720)):
        results.append(
            run_case(
                label=f"capture {width}x{height}",
                source=args.source,
                frames=args.frames,
                warmup=args.warmup,
                model_path=model,
                imgsz=args.imgsz,
                capture_size=(width, height),
                conf=args.conf,
            )
        )
    return results


# -------------------------------------------------------------------- output


def machine_header() -> list[str]:
    lines = [
        f"platform : {platform.platform()}",
        f"python   : {platform.python_version()} ({platform.machine()})",
        f"cpu      : {platform.processor() or 'unknown'}",
    ]
    try:
        import psutil

        lines.append(
            f"cores    : {psutil.cpu_count(logical=False)} physical / "
            f"{psutil.cpu_count(logical=True)} logical"
        )
        lines.append(f"ram      : {psutil.virtual_memory().total / 1e9:.1f} GB")
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        lines.append(f"ort      : {ort.__version__}")
    except ImportError:
        pass
    return lines


def print_table(results: list[BenchResult]) -> str:
    """Render the results as a Markdown table (printed and saved verbatim)."""
    header = (
        "| Config | Model | imgsz | Capture | FPS mean | FPS median | FPS p95 | "
        "FPS p5 | Infer ms | Total ms | CPU % | Peak RSS MB | Dets/frame |"
    )
    sep = "|" + "---|" * 13
    rows = [header, sep]
    skipping = False
    for r in results:
        total = sum(r.stage_ms.get(s, 0.0) for s in STAGES)
        # With frame skipping the per-frame cost is bimodal -- skipped frames do
        # almost no work, so their "FPS" is in the thousands and the median and
        # p95 of that distribution describe nothing real. Mean is throughput
        # (frames / wall time) and stays meaningful, so only it is reported.
        if r.infer_every > 1:
            skipping = True
            median = p95 = p5 = "n/a"
        else:
            median, p95, p5 = f"{r.fps_median:.1f}", f"{r.fps_p95:.1f}", f"{r.fps_p5:.1f}"
        rows.append(
            f"| {r.label} | {r.model} | {r.imgsz} | {r.capture} | "
            f"{r.fps_mean:.1f} | {median} | {p95} | "
            f"{p5} | {r.stage_ms.get('inference', 0):.1f} | "
            f"{total:.1f} | {r.cpu_percent:.0f} | {r.rss_peak_mb:.0f} | "
            f"{r.detections_mean:.2f} |"
        )
    if skipping:
        rows.append("")
        rows.append(
            "> `n/a`: with `--infer-every > 1` the per-frame time is bimodal, so "
            "median/p95 of per-frame FPS are meaningless. Mean is throughput "
            "(frames / wall clock) and is still valid. Note also that inference "
            "ms is the average over *all* frames, including the skipped ones."
        )

    stage_header = "| Config | " + " | ".join(f"{s} ms" for s in STAGES) + " | total ms |"
    stage_rows = [stage_header, "|" + "---|" * (len(STAGES) + 2)]
    for r in results:
        total = sum(r.stage_ms.get(s, 0.0) for s in STAGES)
        cells = " | ".join(f"{r.stage_ms.get(s, 0.0):.2f}" for s in STAGES)
        stage_rows.append(f"| {r.label} | {cells} | {total:.2f} |")

    return "\n".join(rows) + "\n\n" + "\n".join(stage_rows)


def write_outputs(results: list[BenchResult], out_dir: Path, tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = timestamped_path(out_dir, f"benchmark_{tag}", ".csv")
    md_path = csv_path.with_suffix(".md")

    rows = [r.flat() for r in results]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    body = [f"# Benchmark: {tag}", "", "```"]
    body.extend(machine_header())
    body.append("```")
    body.append("")
    body.append(print_table(results))
    md_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--model", default=str(cfg.model.path))
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Frames run but excluded from the summary.",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[320, 416, 512],
        help="Model input sizes for the default sweep.",
    )
    parser.add_argument("--conf", type=float, default=cfg.detection.conf_threshold)
    parser.add_argument(
        "--settle",
        type=float,
        default=20.0,
        help="Seconds of CPU load before measuring, so every configuration is "
        "timed in the same thermal state. 0 disables it (and biases the sweep "
        "in favour of whatever runs first).",
    )
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--threads-sweep", action="store_true")
    parser.add_argument("--capture-sweep", action="store_true")
    args = parser.parse_args(argv)

    if str(args.source).isdigit():
        args.source = int(args.source)

    model = Path(args.model)
    if not model.is_absolute():
        model = REPO_ROOT / model

    if args.ablation:
        tag, sweep = "ablation", sweep_ablation
    elif args.threads_sweep:
        tag, sweep = "threads", sweep_threads
    elif args.capture_sweep:
        tag, sweep = "capture", sweep_capture
    else:
        tag, sweep = "resolution", sweep_resolutions

    print("\n".join(machine_header()))
    print(f"source   : {args.source!r}")
    print(f"frames   : {args.frames} measured (+{args.warmup} warm-up, discarded)\n")

    try:
        settle(model if model.exists() else Path(args.model), args.imgsz, args.settle)
        results = sweep(args, model)
    except (ModelNotFoundError, CameraError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not results:
        return 1

    print()
    print(print_table(results))

    csv_path, md_path = write_outputs(results, cfg.output.results_dir, tag)
    print(f"\nwrote {csv_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
