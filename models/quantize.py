"""Optional INT8 quantisation of the YOLOX ONNX graph.

    python models/quantize.py                    # dynamic (no calibration data)
    python models/quantize.py --mode static      # static, calibrated on real images

Dynamic quantisation is the zero-setup option, but for a convolution-heavy net
it is frequently a *loss*: activations are quantised on the fly every inference,
and that overhead can outweigh the cheaper integer maths. Static quantisation
calibrates activation ranges ahead of time from sample images and is usually the
one that actually pays.

This script does not decide which is better for your machine -- benchmark.py
measures it. The result on the development laptop is in
docs/PERFORMANCE_ANALYSIS.md, including the case where INT8 lost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODELS_DIR.parent


CALIB_DIR = REPO_ROOT / "assets" / "calib"


def _calibration_images(limit: int, directory: Path | None = None) -> list[Path]:
    """The images whose activation ranges define the INT8 model.

    Defaults to the 24 images committed in assets/calib/, which exist so that a
    clean clone can build the shipped model offline and get byte-identical
    results. They were chosen by scripts/build_calibration_set.py under three
    constraints: redistributable licence, **disjoint from the 300-image
    evaluation subset**, and greedy class coverage (69 of 80 classes).

    The disjointness matters. An earlier build of this model calibrated on the
    first 64 images of the evaluation set and scored 0.3312 mAP@0.5:0.95 -- a
    number that was quietly flattered by having seen its own test data.
    """
    if directory is not None:
        return sorted(directory.glob("*.jpg"))[:limit]
    return sorted(CALIB_DIR.glob("*.jpg"))[:limit]


def build_reader(images: list[Path], input_name: str, size: int):
    """CalibrationDataReader over real frames.

    Calibrating on real images matters: activation ranges taken from noise or
    from a single image give ranges that do not reflect what the model sees, and
    the resulting INT8 model loses far more accuracy than it needs to.
    """
    import cv2
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._index = 0

        def get_next(self):
            if self._index >= len(images):
                return None
            image = cv2.imread(str(images[self._index]), cv2.IMREAD_COLOR)
            self._index += 1
            if image is None:
                return self.get_next()

            # Must match src/detector.preprocess exactly, or the calibrated
            # ranges describe a distribution the model never actually sees.
            h, w = image.shape[:2]
            ratio = min(size / h, size / w)
            nh, nw = int(h * ratio), int(w * ratio)
            canvas = np.full((size, size, 3), 114, dtype=np.uint8)
            canvas[:nh, :nw] = cv2.resize(image, (nw, nh))
            blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
            return {input_name: blob}

    return Reader()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(MODELS_DIR / "yolox_tiny.onnx"))
    parser.add_argument("--dst", default=None)
    # Static is the default because it is what ships. Dynamic quantises
    # activations on every inference; for a convolution-heavy net that overhead
    # can cancel out the cheaper integer maths.
    parser.add_argument("--mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument(
        "--calib-images", type=int, default=64, help="Static mode only."
    )
    parser.add_argument(
        "--calib-dir",
        default=None,
        help="Override the bundled calibration images (assets/calib). Pointing "
        "this at data/coco_subset/images would calibrate on the evaluation set "
        "and inflate the resulting accuracy figures -- don't.",
    )
    args = parser.parse_args(argv)

    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ImportError:
        raise SystemExit(
            "error: onnxruntime.quantization is unavailable in this build.\n"
            "  pip install -r requirements.txt"
        )

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(
            f"error: {src} not found. Run: python models/download_weights.py"
        )
    dst = Path(args.dst) if args.dst else src.with_name(f"{src.stem}_int8.onnx")

    if args.mode == "dynamic":
        # Conv is included explicitly: ORT's default op set for dynamic
        # quantisation is MatMul-centric and would leave a CNN almost untouched.
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=["Conv", "MatMul"],
        )
    else:
        import onnxruntime as ort

        directory = Path(args.calib_dir) if args.calib_dir else None
        images = _calibration_images(args.calib_images, directory)
        if not images:
            raise SystemExit(
                f"error: no calibration images found in "
                f"{directory or CALIB_DIR}.\n"
                f"The bundled set is committed at assets/calib/; if it is "
                f"missing, rebuild it with:\n"
                f"  python scripts/build_calibration_set.py --images 24"
            )
        print(f"calibrating on {len(images)} image(s) from {images[0].parent.name}/")
        session = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        reader = build_reader(images, session.get_inputs()[0].name, args.imgsz)
        quantize_static(
            model_input=str(src),
            model_output=str(dst),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )

    size_ratio = dst.stat().st_size / src.stat().st_size
    print(f"wrote {dst}")
    print(
        f"size  {src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB "
        f"({size_ratio:.0%})"
    )
    print("\nNow measure it -- do not assume it is faster:")
    print(f"  python -m src.benchmark --model {dst.relative_to(REPO_ROOT).as_posix()}")
    print(f"  python -m src.evaluate --model {dst.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
