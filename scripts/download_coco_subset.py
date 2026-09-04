"""Fetch a small labelled slice of COCO val2017 for accuracy evaluation.

    python scripts/download_coco_subset.py                 # 300 images
    python scripts/download_coco_subset.py --images 500

Why a subset: full val2017 is 5000 images / ~780 MB, which nobody wants to pull
just to sanity-check a take-home. 300 images is enough for a stable mAP@0.5
(the run-to-run spread on a fixed model is zero -- the sampling error is against
the *full* val set, which is why the reported numbers say "300-image subset"
everywhere rather than quoting them as COCO val2017 results).

Images are always the first N by image id, sorted, so the subset is identical on
every machine and the numbers are comparable.

Everything lands in data/, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_ZIP = DATA_DIR / "annotations_trainval2017.zip"
INSTANCES_JSON = DATA_DIR / "annotations" / "instances_val2017.json"
SUBSET_DIR = DATA_DIR / "coco_subset"
IMAGES_DIR = SUBSET_DIR / "images"
SUBSET_JSON = SUBSET_DIR / "instances_subset.json"


def _download(url: str, dest: Path, label: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "real-time-object-detection/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 16):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r{label}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB "
                    f"({100 * done / total:5.1f}%)",
                    end="",
                )
    print()
    shutil.move(str(part), str(dest))
    return dest


def ensure_annotations() -> Path:
    if INSTANCES_JSON.exists():
        print(f"ok       annotations already extracted ({INSTANCES_JSON.name})")
        return INSTANCES_JSON

    if not ANNOTATIONS_ZIP.exists():
        print("fetching COCO 2017 annotations (~241 MB, one time)")
        _download(ANNOTATIONS_URL, ANNOTATIONS_ZIP, "annotations")

    print("extracting instances_val2017.json")
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as zf:
        # Extract only the file we need; the archive also holds train2017
        # instances (~470 MB) and the caption/keypoint sets.
        member = "annotations/instances_val2017.json"
        zf.extract(member, DATA_DIR)
    return INSTANCES_JSON


def build_subset(n_images: int, keep_zip: bool) -> None:
    ensure_annotations()

    print(f"reading {INSTANCES_JSON.name}")
    with INSTANCES_JSON.open("r", encoding="utf-8") as fh:
        coco = json.load(fh)

    images = sorted(coco["images"], key=lambda im: im["id"])[:n_images]
    wanted = {im["id"] for im in images}
    annotations = [a for a in coco["annotations"] if a["image_id"] in wanted]

    SUBSET_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUBSET_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "images": images,
                "annotations": annotations,
                "categories": coco["categories"],
            },
            fh,
        )
    print(
        f"wrote    {SUBSET_JSON.name}: {len(images)} images, "
        f"{len(annotations)} annotations"
    )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    missing = [im for im in images if not (IMAGES_DIR / im["file_name"]).exists()]
    if not missing:
        print(f"ok       all {len(images)} images already present")
    else:
        print(f"fetching {len(missing)} images into {IMAGES_DIR.relative_to(REPO_ROOT)}")
        _download_images(missing)

    if ANNOTATIONS_ZIP.exists() and not keep_zip:
        ANNOTATIONS_ZIP.unlink()
        print("cleaned  removed the 241 MB annotations zip (extract is kept)")

    print("\nNow run:  python -m src.evaluate")


def _download_images(images: list[dict]) -> None:
    """Eight at a time: these are ~160 KB each, so latency dominates, not bandwidth."""
    failures: list[str] = []

    def fetch(image: dict) -> str | None:
        dest = IMAGES_DIR / image["file_name"]
        try:
            request = urllib.request.Request(
                image["coco_url"], headers={"User-Agent": "real-time-object-detection/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                dest.write_bytes(response.read())
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return f"{image['file_name']}: {exc}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, im): im for im in images}
        for i, future in enumerate(as_completed(futures), 1):
            error = future.result()
            if error:
                failures.append(error)
            print(f"\r         {i}/{len(images)}", end="")
    print()

    if failures:
        print(f"warning  {len(failures)} image(s) failed:", file=sys.stderr)
        for line in failures[:5]:
            print(f"         {line}", file=sys.stderr)
        print("         Re-run this script to retry only the missing ones.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--images", type=int, default=300, help="How many val2017 images (default 300)."
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the 241 MB annotations archive after extracting.",
    )
    args = parser.parse_args(argv)

    if args.images < 1:
        raise SystemExit("error: --images must be >= 1")
    build_subset(args.images, args.keep_zip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
