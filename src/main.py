"""CLI entry point for live detection.

    python -m src.main                          # default webcam
    python -m src.main --source assets/sample.jpg
    python -m src.main --source clip.mp4 --record results/demo.mp4
    python -m src.main --classes person cup laptop --conf 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .coco_classes import COCO_CLASSES, resolve_class_filter
from .config import REPO_ROOT, Config, ConfigError, apply_cli_overrides, load_config
from .detector import Detection, Detector, ModelNotFoundError, set_opencv_threads
from .metrics import Metrics, timestamped_path
from .video_stream import CameraError, VideoStream
from .visualizer import draw_detections, draw_footer, draw_hud, draw_recording_dot

# Give the FPS estimate time to settle before it decides the recording's
# playback rate. ~30 frames is 1-2 seconds, enough to average out the jitter.
RECORD_CALIBRATION_FRAMES = 30

# Frames excluded from the *summary* (never from the CSV). Detector.warmup()
# cannot reach the render path or OpenCV's first-call lazy init, so the first
# real frame still costs an order of magnitude more than the rest.
WARMUP_FRAMES = 5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Real-time object detection on a webcam, video file or image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Keys while running:  q / Esc = quit,  s = save a screenshot to results/\n"
            "Every flag defaults to the matching entry in config/config.yaml."
        ),
    )
    # Defaults are None on purpose: that is how apply_cli_overrides tells a flag
    # the user typed from one argparse filled in, so config.yaml is not ignored.
    p.add_argument(
        "--source",
        default=None,
        help="Webcam index (0, 1, ...), video file, or image file.",
    )
    p.add_argument("--model", default=None, help="Path to a YOLOX .onnx file.")
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Square network input, multiple of 32. Sizes other than 416 need "
        "models/make_dynamic.py first.",
    )
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    p.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help="Only show these classes, by name or index: --classes person 'cell phone'",
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="ONNX Runtime intra-op threads (0 = runtime default).",
    )
    p.add_argument(
        "--infer-every",
        type=int,
        default=None,
        metavar="N",
        help="Detect on every Nth frame and reuse the previous boxes in between. "
        "The HUD says so when this is active.",
    )
    p.add_argument("--width", type=int, default=None, help="Requested capture width.")
    p.add_argument("--height", type=int, default=None, help="Requested capture height.")
    p.add_argument(
        "--record",
        default=None,
        metavar="PATH",
        help="Write an annotated MP4 to PATH. This is how the demo video is made.",
    )
    p.add_argument(
        "--record-fps",
        type=float,
        default=None,
        help="Force the recording's frame rate. Default: measure it, so the clip "
        "plays back at real speed.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames. Useful for scripted demo takes and CI.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Headless: no window. Still records and still writes the CSV.",
    )
    p.add_argument("--config", default=None, help="Alternative config.yaml.")
    p.add_argument(
        "--list-classes",
        action="store_true",
        help="Print the 80 COCO classes with their indices and exit.",
    )
    return p


def _fourcc(code: str) -> int:
    """cv2.VideoWriter_fourcc moved to VideoWriter.fourcc in OpenCV 5."""
    fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    return int(fn(*code))


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v is the codec that is present in every stock OpenCV wheel on every
    # platform. H.264 would be smaller but is not redistributable in all builds,
    # so the README uses ffmpeg for the compression pass instead.
    writer = cv2.VideoWriter(str(path), _fourcc("mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open a video writer for {path}. Check the directory exists "
            f"and the extension is .mp4"
        )
    return writer


def run(cfg: Config, args: argparse.Namespace) -> int:
    set_opencv_threads(cfg.runtime.opencv_threads)

    try:
        class_filter = resolve_class_filter(cfg.detection.classes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        detector = Detector(
            model_path=cfg.model.path,
            input_size=cfg.model.input_size,
            conf_threshold=cfg.detection.conf_threshold,
            iou_threshold=cfg.detection.iou_threshold,
            max_detections=cfg.detection.max_detections,
            class_filter=class_filter,
            device=cfg.runtime.device,
            intra_op_threads=cfg.runtime.intra_op_threads,
            inter_op_threads=cfg.runtime.inter_op_threads,
            preprocess_mode=cfg.runtime.preprocess_mode,
        )
    except (ModelNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    detector.warmup()
    print(f"model     : {detector.model_path.name} @ {cfg.model.input_size[0]}px")
    print(f"providers : {', '.join(detector.providers)}")
    print(f"source    : {cfg.camera.source!r}")

    try:
        stream = VideoStream(
            source=cfg.camera.source,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
            buffer_size=cfg.camera.buffer_size,
        ).start()
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"capture   : {stream.frame_size[0]}x{stream.frame_size[1]} ({stream.kind})")
    if stream.is_live:
        print(
            "note      : a live camera caps FPS at its own frame rate. For detector "
            "throughput, run src/benchmark.py."
        )

    metrics = Metrics(window=120)
    writer: cv2.VideoWriter | None = None
    record_path = Path(args.record) if args.record else None
    if record_path is not None and not record_path.is_absolute():
        record_path = REPO_ROOT / record_path

    detections: list[Detection] = []
    frame_index = 0
    stale = False
    exit_code = 0
    window = cfg.display.window_name

    try:
        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            with metrics.stage("capture"):
                frame = stream.read()
            if frame is None:
                break

            run_inference = frame_index % cfg.runtime.infer_every == 0
            stale = not run_inference

            if run_inference:
                with metrics.stage("preprocess"):
                    blob = detector.preprocess(frame)
                with metrics.stage("inference"):
                    raw = detector.infer(blob)
                with metrics.stage("postprocess"):
                    detections = detector.postprocess(raw)

            with metrics.stage("render"):
                draw_detections(frame, detections, font_scale=cfg.display.font_scale)
                if cfg.display.show_hud:
                    draw_hud(
                        frame,
                        metrics.fps_instant,
                        metrics.fps_rolling,
                        len(detections),
                        extra_lines=[
                            f"infer {metrics.stage_ms('inference'):4.1f} ms  "
                            f"total {metrics.stage_ms('inference') + metrics.stage_ms('preprocess') + metrics.stage_ms('postprocess'):4.1f} ms"
                        ],
                        font_scale=cfg.display.font_scale,
                        stale=stale,
                    )

                if record_path is not None:
                    if writer is None and (
                        args.record_fps is not None
                        or frame_index >= RECORD_CALIBRATION_FRAMES
                    ):
                        fps = args.record_fps or max(1.0, min(120.0, metrics.fps_rolling))
                        writer = _open_writer(
                            record_path, fps, (frame.shape[1], frame.shape[0])
                        )
                        print(f"recording : {record_path} at {fps:.1f} fps")
                    if writer is not None:
                        draw_recording_dot(frame)
                        writer.write(frame)

                if cfg.display.show:
                    draw_footer(frame, "q quit | s screenshot")
                    cv2.imshow(window, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s"):
                        shot = timestamped_path(
                            cfg.output.results_dir, "screenshot", ".png"
                        )
                        shot.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(shot), frame)
                        print(f"saved     : {shot}")

            metrics.end_frame()
            frame_index += 1

    except KeyboardInterrupt:
        print("\ninterrupted")
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        stream.stop()
        if writer is not None:
            writer.release()
        if cfg.display.show:
            cv2.destroyAllWindows()

    _report(metrics, cfg, detector, record_path, writer is not None)
    return exit_code


def _report(
    metrics: Metrics,
    cfg: Config,
    detector: Detector,
    record_path: Path | None,
    recorded: bool,
) -> None:
    # Warm-up frames are excluded from the summary but stay in the CSV, so the
    # raw data can always be re-checked against the headline numbers.
    skip = WARMUP_FRAMES if metrics.frame_count > WARMUP_FRAMES * 2 else 0
    summary = metrics.summarize(skip_first=skip)
    if summary.frames == 0:
        print("no frames processed")
        return

    print()
    warm = f" (first {skip} excluded as warm-up)" if skip else ""
    print(f"frames    : {summary.frames} in {summary.elapsed_s:.1f}s{warm}")
    print(
        f"fps       : mean {summary.fps_mean:.1f} | median {summary.fps_median:.1f} "
        f"| p95 {summary.fps_p95:.1f} | p5 {summary.fps_p5:.1f}"
    )
    print("stage ms  : " + "  ".join(
        f"{k} {v:.2f}" for k, v in summary.stage_ms_mean.items()
    ))

    if cfg.output.write_csv:
        csv_path = metrics.write_csv(
            timestamped_path(cfg.output.results_dir, "run", ".csv")
        )
        print(f"csv       : {csv_path}")
    if record_path is not None and recorded:
        size_mb = record_path.stat().st_size / 1e6 if record_path.exists() else 0.0
        print(f"video     : {record_path} ({size_mb:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_classes:
        for i, name in enumerate(COCO_CLASSES):
            print(f"{i:2d}  {name}")
        return 0

    try:
        cfg = apply_cli_overrides(load_config(args.config), args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    return run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
