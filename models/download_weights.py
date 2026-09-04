"""Fetch YOLOX ONNX weights and verify them against a pinned SHA-256.

    python models/download_weights.py            # yolox-tiny (the default model)
    python models/download_weights.py --model nano
    python models/download_weights.py --all

Weights are deliberately not committed to git. The checksums below were computed
from the files this project was built and benchmarked against, so a corrupted or
substituted download fails loudly instead of producing quietly wrong boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
CHUNK = 1 << 16


@dataclass(frozen=True)
class Weight:
    name: str
    url: str
    sha256: str
    size_bytes: int
    note: str


# Megvii publish these as release assets on the YOLOX repo. Apache-2.0.
WEIGHTS: dict[str, Weight] = {
    "tiny": Weight(
        name="yolox_tiny.onnx",
        url=(
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            "0.1.1rc0/yolox_tiny.onnx"
        ),
        sha256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        size_bytes=20_219_662,
        note="default; 32.8 mAP published, 416x416",
    ),
    "nano": Weight(
        name="yolox_nano.onnx",
        url=(
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            "0.1.1rc0/yolox_nano.onnx"
        ),
        sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
        size_bytes=3_659_407,
        note="fallback for slower machines; 25.8 mAP published",
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(weight: Weight, dest_dir: Path, force: bool = False) -> Path:
    dest = dest_dir / weight.name
    if dest.exists() and not force:
        actual = sha256_of(dest)
        if actual == weight.sha256:
            print(f"ok       {weight.name} (already present, checksum matches)")
            return dest
        print(
            f"warning  {weight.name} exists but its checksum does not match; "
            f"re-downloading"
        )

    # Download to .part and rename only after the checksum passes, so an
    # interrupted run can never leave a half-file that looks valid.
    part = dest.with_suffix(dest.suffix + ".part")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetching {weight.name}  ({weight.size_bytes / 1e6:.1f} MB)")
    print(f"         {weight.url}")

    try:
        request = urllib.request.Request(
            weight.url, headers={"User-Agent": "real-time-object-detection/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, part.open(
            "wb"
        ) as out:
            total = int(response.headers.get("Content-Length") or weight.size_bytes)
            done = 0
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                pct = 100.0 * done / total if total else 0.0
                print(f"\r         {done / 1e6:6.1f} MB  {pct:5.1f}%", end="")
        print()
    except (urllib.error.URLError, TimeoutError) as exc:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"error: download failed ({exc}).\n"
            f"You can fetch it by hand and drop it in {dest_dir}:\n  {weight.url}"
        ) from exc

    actual = sha256_of(part)
    if actual != weight.sha256:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"error: checksum mismatch for {weight.name}\n"
            f"  expected {weight.sha256}\n  got      {actual}\n"
            f"The download was corrupted or the release asset changed. Not using it."
        )

    shutil.move(str(part), str(dest))
    print(f"ok       {weight.name} verified -> {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", choices=sorted(WEIGHTS), default="tiny", help="Which weights."
    )
    parser.add_argument("--all", action="store_true", help="Fetch every model.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if present."
    )
    parser.add_argument(
        "--dest", default=None, help="Target directory (default: models/)."
    )
    args = parser.parse_args(argv)

    dest_dir = Path(args.dest) if args.dest else MODELS_DIR
    wanted = list(WEIGHTS.values()) if args.all else [WEIGHTS[args.model]]

    for weight in wanted:
        download(weight, dest_dir, force=args.force)
        print(f"         {weight.note}")
    print("\nLicence: YOLOX weights are Apache-2.0, (c) Megvii Inc. See NOTICE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
