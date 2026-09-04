"""Relax YOLOX's fixed 416x416 input to a variable one.

Megvii export the released ONNX with a hard-coded [1, 3, 416, 416] input, so
ONNX Runtime rejects any other resolution outright. That would make the
resolution sweep in benchmark.py impossible.

The network itself has no such limitation: the three head Reshape nodes target
[1, 85, -1], so the spatial dimension is already inferred rather than baked in.
Only the declared input/output shapes need widening -- no weights are touched
and no graph surgery beyond editing two dimension fields. The script verifies
that by running three resolutions and checking each returns exactly the anchor
count the FPN strides predict.

    python models/make_dynamic.py
    python models/make_dynamic.py --src models/yolox_nano.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
STRIDES = (8, 16, 32)


def expected_anchors(size: int) -> int:
    return sum((size // s) ** 2 for s in STRIDES)


def make_dynamic(src: Path, dst: Path) -> Path:
    try:
        import onnx
    except ImportError:  # pragma: no cover - dependency is pinned in requirements
        raise SystemExit(
            "error: the 'onnx' package is required for this script.\n"
            "  pip install -r requirements.txt"
        )

    if not src.exists():
        raise SystemExit(
            f"error: {src} not found. Fetch it first:\n"
            f"  python models/download_weights.py"
        )

    model = onnx.load(str(src))

    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise SystemExit(f"error: expected a 4-D NCHW input, got {len(dims)} dims")
    dims[2].dim_param = "height"
    dims[3].dim_param = "width"

    # The declared output length is tied to the input size too. Left alone it
    # still runs, but ORT logs a shape-mismatch warning on every single call.
    out_dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(out_dims) == 3:
        out_dims[1].dim_param = "anchors"

    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return dst


def verify(path: Path, sizes: tuple[int, ...] = (320, 416, 512)) -> bool:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    print(f"declared input: {session.get_inputs()[0].shape}")

    ok = True
    for size in sizes:
        blob = np.zeros((1, 3, size, size), dtype=np.float32)
        try:
            out = session.run(None, {name: blob})[0]
        except Exception as exc:
            print(f"  {size}x{size}: FAILED - {str(exc)[:120]}")
            ok = False
            continue
        want = expected_anchors(size)
        got = out.shape[1]
        status = "ok" if got == want else "WRONG"
        print(f"  {size}x{size}: {status} anchors={got} (expected {want})")
        ok &= got == want
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(MODELS_DIR / "yolox_tiny.onnx"))
    parser.add_argument(
        "--dst", default=None, help="Default: <src stem>_dynamic.onnx alongside src."
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.with_name(f"{src.stem}_dynamic.onnx")

    make_dynamic(src, dst)
    print(f"wrote {dst}")

    if not args.no_verify:
        if not verify(dst):
            print(
                "\nVerification failed: this model does not tolerate a variable "
                "input. Stick to its native resolution.",
                file=sys.stderr,
            )
            return 1
        print("verified: variable-resolution inference produces the expected grids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
