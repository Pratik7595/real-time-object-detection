"""Rebuild the bundled INT8 calibration set in assets/calib/.

This is not part of anyone's normal workflow -- the resulting images are
committed, so `models/quantize.py` works offline on a clean clone. It is here so
the *choice* of calibration images is auditable rather than arbitrary.

Three constraints drive the selection, in order:

1. **Licence.** Only COCO images whose licence field is 4 (CC BY 2.0), 7 (no
   known copyright restrictions) or 8 (US Government work) can be redistributed
   inside this repository. Most of COCO is non-commercial and cannot.
2. **Disjoint from the evaluation set.** `scripts/download_coco_subset.py` takes
   the first N images by id, so calibration draws only from images *outside*
   that range. Calibrating a quantised model on the same images it is later
   scored against inflates the reported accuracy.
3. **Class coverage.** Static quantisation calibrates activation ranges from
   whatever it is shown. A greedy set-cover pass picks images that between them
   touch as many of the 80 classes as possible, so the ranges are not tuned to
   a handful of object types.

    python scripts/build_calibration_set.py --images 24
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_JSON = REPO_ROOT / "data" / "annotations" / "instances_val2017.json"
CALIB_DIR = REPO_ROOT / "assets" / "calib"
MANIFEST = CALIB_DIR / "MANIFEST.json"

REDISTRIBUTABLE_LICENCES = {4, 7, 8}
# Must match download_coco_subset.py's default, or the two sets can overlap.
EVAL_SUBSET_SIZE = 300


def select(coco: dict, n_images: int, eval_size: int) -> list[dict]:
    images = {im["id"]: im for im in coco["images"]}
    reserved = {im["id"] for im in sorted(coco["images"], key=lambda i: i["id"])[:eval_size]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    per_image: dict[int, set[str]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        per_image.setdefault(ann["image_id"], set()).add(categories[ann["category_id"]])

    candidates = [
        (image_id, classes)
        for image_id, classes in per_image.items()
        if image_id not in reserved
        and images[image_id]["license"] in REDISTRIBUTABLE_LICENCES
        # Skip the extremes: a 30-object crowd scene and a single-object
        # close-up are both poor representatives of a typical webcam frame.
        and 2 <= len(classes) <= 8
    ]
    candidates.sort(key=lambda pair: pair[0])  # deterministic tie-breaking

    chosen: list[dict] = []
    covered: set[str] = set()
    remaining = dict(candidates)

    # Greedy set cover: repeatedly take the image adding the most new classes.
    while remaining and len(chosen) < n_images:
        best_id = max(
            remaining,
            key=lambda i: (len(remaining[i] - covered), -i),
        )
        gained = remaining[best_id] - covered
        if not gained and len(covered) < len(categories):
            # Nothing new left to cover; fall back to plain id order so the
            # remainder of the set is still deterministic.
            best_id = min(remaining)
        covered |= remaining.pop(best_id)
        chosen.append(images[best_id])

    print(f"selected {len(chosen)} images covering {len(covered)} of 80 classes")
    return sorted(chosen, key=lambda im: im["id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--eval-size", type=int, default=EVAL_SUBSET_SIZE)
    args = parser.parse_args(argv)

    if not INSTANCES_JSON.exists():
        raise SystemExit(
            f"error: {INSTANCES_JSON.relative_to(REPO_ROOT)} not found.\n"
            f"  python scripts/download_coco_subset.py --images 300"
        )

    with INSTANCES_JSON.open("r", encoding="utf-8") as fh:
        coco = json.load(fh)

    licences = {lic["id"]: lic for lic in coco["licenses"]}
    chosen = select(coco, args.images, args.eval_size)

    if CALIB_DIR.exists():
        shutil.rmtree(CALIB_DIR)
    CALIB_DIR.mkdir(parents=True)

    records = []
    for i, image in enumerate(chosen, 1):
        dest = CALIB_DIR / image["file_name"]
        request = urllib.request.Request(
            image["coco_url"], headers={"User-Agent": "real-time-object-detection/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            dest.write_bytes(response.read())
        records.append(
            {
                "file_name": image["file_name"],
                "coco_image_id": image["id"],
                "license_id": image["license"],
                "license_name": licences[image["license"]]["name"],
                "license_url": licences[image["license"]]["url"],
                "flickr_url": image.get("flickr_url", ""),
            }
        )
        print(f"\r  {i}/{len(chosen)}", end="")
    print()

    MANIFEST.write_text(
        json.dumps(
            {
                "purpose": "INT8 static-quantisation calibration for models/quantize.py",
                "source": "COCO val2017",
                "disjoint_from": f"first {args.eval_size} val2017 images by id (the evaluation subset)",
                "licence_filter": sorted(REDISTRIBUTABLE_LICENCES),
                "images": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(f.stat().st_size for f in CALIB_DIR.glob("*.jpg"))
    print(f"wrote {CALIB_DIR.relative_to(REPO_ROOT)}: {len(records)} images, {total / 1e6:.1f} MB")
    print("Remember to refresh the attribution block in NOTICE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
