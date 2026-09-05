"""Regenerate the committed sample input -> output pair.

The repository ships one sample input (`assets/sample.jpg`) and the two outputs
the detector produces from it:

    results/sample_detection.png   what a user sees: boxes, labels, confidences
    results/sample_output.json     the same detections as machine-readable data

Both are committed so a reviewer can see exactly what this program produces
without installing anything, and can diff their own run against a known-good
result. This script rebuilds them.

    python scripts/make_sample_output.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.detector import Detector  # noqa: E402
from src.metrics import Metrics  # noqa: E402
from src.visualizer import draw_detections, draw_footer, draw_hud  # noqa: E402

SAMPLE_IN = REPO_ROOT / "assets" / "sample.jpg"
PNG_OUT = REPO_ROOT / "results" / "sample_detection.png"
JSON_OUT = REPO_ROOT / "results" / "sample_output.json"

# Enough frames for the on-screen FPS readout to reflect steady state rather
# than the one-off cost of the first frame.
WARM_FRAMES = 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="Defaults to config.yaml.")
    parser.add_argument("--conf", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    model_path = Path(args.model) if args.model else cfg.model.path
    conf = args.conf if args.conf is not None else cfg.detection.conf_threshold

    frame = cv2.imread(str(SAMPLE_IN), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"error: could not read {SAMPLE_IN}")

    detector = Detector(
        model_path=model_path,
        input_size=cfg.model.input_size,
        conf_threshold=conf,
        iou_threshold=cfg.detection.iou_threshold,
    )
    detector.warmup(3)

    metrics = Metrics(window=WARM_FRAMES * 2)
    detections: list = []
    for _ in range(WARM_FRAMES):
        with metrics.stage("capture"):
            working = frame.copy()
        with metrics.stage("preprocess"):
            blob = detector.preprocess(working)
        with metrics.stage("inference"):
            raw = detector.infer(blob)
        with metrics.stage("postprocess"):
            detections = detector.postprocess(raw)
        with metrics.stage("render"):
            pass
        metrics.end_frame()

    # --- machine-readable output -------------------------------------------
    payload = {
        "input": {
            "file": SAMPLE_IN.relative_to(REPO_ROOT).as_posix(),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "source": "COCO val2017 image 340894, CC BY 2.0 (see NOTICE)",
        },
        "model": detector.describe(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "timing_ms": {
            stage: round(metrics.stage_ms(stage), 3)
            for stage in ("preprocess", "inference", "postprocess")
        },
        "detection_count": len(detections),
        "detections": [
            {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": round(det.score, 4),
                "box_xyxy": [round(v, 1) for v in (det.x1, det.y1, det.x2, det.y2)],
            }
            for det in sorted(detections, key=lambda d: -d.score)
        ],
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- human-readable output ---------------------------------------------
    annotated = frame.copy()
    draw_detections(annotated, detections, font_scale=cfg.display.font_scale)
    draw_hud(
        annotated,
        metrics.fps_instant,
        metrics.fps_rolling,
        len(detections),
        extra_lines=[
            f"infer {metrics.stage_ms('inference'):4.1f} ms  "
            f"total {metrics.stage_ms('inference') + metrics.stage_ms('preprocess') + metrics.stage_ms('postprocess'):4.1f} ms"
        ],
        font_scale=cfg.display.font_scale,
    )
    draw_footer(annotated, "q quit | s screenshot")
    cv2.imwrite(str(PNG_OUT), annotated)

    print(f"input   : {SAMPLE_IN.relative_to(REPO_ROOT)}")
    print(f"model   : {model_path.name} @ conf {conf}")
    print(f"found   : {len(detections)} objects")
    for det in sorted(detections, key=lambda d: -d.score):
        print(f"          {det.class_name:12s} {det.score:.2f}")
    print(f"wrote   : {PNG_OUT.relative_to(REPO_ROOT)}")
    print(f"          {JSON_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
