"""Accuracy on a labelled COCO subset. Never on webcam footage.

    python scripts/download_coco_subset.py      # once
    python -m src.evaluate
    python -m src.evaluate --conf-sweep 0.15 0.25 0.30 0.40 0.60

Every number this produces comes from COCO val2017 images with human ground
truth. A live webcam stream has no labels, so no accuracy figure can honestly be
attached to it -- the README describes live behaviour qualitatively and quotes
no percentages for it.

Two different confidence thresholds are in play here, and conflating them is the
most common way to publish a wrong mAP:

* mAP is computed from detections gathered at conf=0.001. Average precision is
  the area under the precision-recall curve, so it needs the low-confidence tail
  to trace the curve out. Evaluating at 0.30 truncates the curve and understates
  mAP by a wide margin.
* Precision / recall / F1 are operating-point metrics and only mean something at
  a stated threshold. Those use --pr-conf (default 0.30, the value the app
  actually ships with).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .coco_classes import COCO_91_IDS, COCO_CLASSES
from .config import REPO_ROOT, load_config
from .detector import Detection, Detector, ModelNotFoundError
from .metrics import timestamped_path

SUBSET_DIR = REPO_ROOT / "data" / "coco_subset"
SUBSET_JSON = SUBSET_DIR / "instances_subset.json"
IMAGES_DIR = SUBSET_DIR / "images"

# Low enough to trace the full PR curve; COCO's own eval uses 0.0 but a floor
# keeps the JSON from ballooning with thousands of near-zero boxes per image.
MAP_CONF = 0.001
COCO_MAX_DETS = 100


@dataclass
class ClassMetrics:
    name: str
    support: int   # ground-truth instances
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def require_subset() -> None:
    if SUBSET_JSON.exists() and any(IMAGES_DIR.glob("*.jpg")):
        return
    raise SystemExit(
        "error: the labelled COCO subset is not present.\n"
        "  python scripts/download_coco_subset.py --images 300\n"
        "(~50 MB of images plus a one-time 241 MB annotations download.)"
    )


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. boxes are xyxy; returns (len(a), len(b))."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def run_detections(
    detector: Detector, images: list[dict], quiet: bool = False
) -> dict[int, list[Detection]]:
    """Detect over the whole subset once, at MAP_CONF.

    Run once and threshold afterwards: re-running inference for every confidence
    level in the sweep would take 5x as long and produce identical boxes.
    """
    per_image: dict[int, list[Detection]] = {}
    for i, meta in enumerate(images, 1):
        path = IMAGES_DIR / meta["file_name"]
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"warning: could not read {path.name}, skipping", file=sys.stderr)
            continue
        per_image[meta["id"]] = detector.detect(frame)
        if not quiet and (i % 25 == 0 or i == len(images)):
            print(f"\r  detecting {i}/{len(images)}", end="")
    if not quiet:
        print()
    return per_image


def to_coco_json(per_image: dict[int, list[Detection]], conf: float) -> list[dict]:
    results = []
    for image_id, detections in per_image.items():
        for det in sorted(detections, key=lambda d: -d.score)[:COCO_MAX_DETS]:
            if det.score < conf:
                continue
            x, y, w, h = det.as_xywh()
            results.append(
                {
                    "image_id": int(image_id),
                    # COCO ground truth uses sparse 91-category ids, not the
                    # model's contiguous 0-79 indices.
                    "category_id": int(COCO_91_IDS[det.class_id]),
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "score": round(float(det.score), 5),
                }
            )
    return results


def coco_map(detections_json: list[dict], gt_path: Path) -> dict[str, float]:
    """mAP via pycocotools -- the reference implementation, not a re-derivation."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise SystemExit(
            "error: pycocotools is required for mAP.\n"
            "  pip install -r requirements-dev.txt"
        )

    if not detections_json:
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0, "mAP@0.75": 0.0}

    # pycocotools writes a lot to stdout; keep the report readable.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(detections_json)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    s = ev.stats
    return {
        "mAP@0.5:0.95": float(s[0]),
        "mAP@0.5": float(s[1]),
        "mAP@0.75": float(s[2]),
        "mAP_small": float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large": float(s[5]),
        "AR@100": float(s[8]),
    }


def per_class_pr(
    per_image: dict[int, list[Detection]],
    ground_truth: dict,
    conf: float,
    iou_threshold: float = 0.5,
) -> list[ClassMetrics]:
    """Greedy IoU matching at one operating point.

    pycocotools reports AP, not precision/recall at a threshold, so this is
    computed here. Crowd regions (iscrowd=1) are excluded entirely rather than
    counted as misses -- they mark "a pile of objects, not individually
    labelled", and scoring against them would penalise correct behaviour.
    """
    id_to_index = {cat_id: i for i, cat_id in enumerate(COCO_91_IDS)}

    gt_by_image: dict[int, dict[int, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ann in ground_truth["annotations"]:
        if ann.get("iscrowd"):
            continue
        index = id_to_index.get(ann["category_id"])
        if index is None:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_image[ann["image_id"]][index].append([x, y, x + w, y + h])

    stats = {
        i: ClassMetrics(name=COCO_CLASSES[i], support=0, tp=0, fp=0, fn=0)
        for i in range(len(COCO_CLASSES))
    }
    for per_class in gt_by_image.values():
        for index, boxes in per_class.items():
            stats[index].support += len(boxes)

    for image_id, detections in per_image.items():
        gt_classes = gt_by_image.get(image_id, {})
        by_class: dict[int, list[Detection]] = defaultdict(list)
        for det in detections:
            if det.score >= conf:
                by_class[det.class_id].append(det)

        for class_index in set(by_class) | set(gt_classes):
            preds = sorted(by_class.get(class_index, []), key=lambda d: -d.score)
            gts = np.asarray(gt_classes.get(class_index, []), dtype=np.float32)
            if gts.size == 0:
                stats[class_index].fp += len(preds)
                continue
            if not preds:
                stats[class_index].fn += len(gts)
                continue

            pred_boxes = np.asarray(
                [[d.x1, d.y1, d.x2, d.y2] for d in preds], dtype=np.float32
            )
            ious = iou_matrix(pred_boxes, gts)
            claimed = set()
            for row in range(len(preds)):
                # Highest-scoring prediction picks first; each GT box can only
                # be claimed once, so duplicates land as false positives.
                order = np.argsort(-ious[row])
                match = next(
                    (
                        int(c)
                        for c in order
                        if c not in claimed and ious[row, c] >= iou_threshold
                    ),
                    None,
                )
                if match is None:
                    stats[class_index].fp += 1
                else:
                    claimed.add(match)
                    stats[class_index].tp += 1
            stats[class_index].fn += len(gts) - len(claimed)

    # Classes with no ground truth in a 300-image subset would be all-zero rows.
    return [m for m in stats.values() if m.support or m.tp or m.fp]


def format_report(
    header: list[str],
    map_scores: dict[str, float],
    classes: list[ClassMetrics],
    sweep: list[tuple[float, dict[str, float], float]],
    pr_conf: float,
    n_images: int,
) -> str:
    out = ["# Accuracy evaluation", ""]
    out.append(
        f"**Source: COCO val2017 subset, N={n_images} labelled images. "
        f"These are not webcam measurements.**"
    )
    out += ["", "```", *header, "```", ""]

    out += ["## mAP (detections gathered at conf=0.001)", ""]
    out += ["| Metric | Value |", "|---|---|"]
    for key, value in map_scores.items():
        out.append(f"| {key} | {value:.4f} |")
    out.append("")

    if sweep:
        out += [
            "## Confidence-threshold sensitivity",
            "",
            "mAP is computed over the surviving detections at each threshold, so it "
            "falls as the low-confidence tail of the PR curve is cut off. "
            "`dets/image` is what a live viewer actually sees.",
            "",
            "| conf | mAP@0.5 | mAP@0.5:0.95 | dets/image |",
            "|---|---|---|---|",
        ]
        for conf, scores, dets in sweep:
            out.append(
                f"| {conf:.2f} | {scores['mAP@0.5']:.4f} | "
                f"{scores['mAP@0.5:0.95']:.4f} | {dets:.2f} |"
            )
        out.append("")

    out += [
        f"## Per-class precision / recall / F1 (conf={pr_conf:.2f}, IoU=0.5)",
        "",
        "| Class | GT | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(classes, key=lambda c: (-c.support, c.name)):
        out.append(
            f"| {m.name} | {m.support} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |"
        )

    total_tp = sum(m.tp for m in classes)
    total_fp = sum(m.fp for m in classes)
    total_fn = sum(m.fn for m in classes)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    out += [
        f"| **micro-average** | {sum(m.support for m in classes)} | {total_tp} | "
        f"{total_fp} | {total_fn} | {micro_p:.3f} | {micro_r:.3f} | {micro_f1:.3f} |",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--model", default=str(cfg.model.path))
    parser.add_argument("--imgsz", type=int, default=cfg.model.input_size[0])
    parser.add_argument("--iou", type=float, default=cfg.detection.iou_threshold)
    parser.add_argument(
        "--pr-conf",
        type=float,
        default=cfg.detection.conf_threshold,
        help="Operating point for the precision/recall/F1 table.",
    )
    parser.add_argument(
        "--conf-sweep",
        type=float,
        nargs="*",
        default=[0.15, 0.25, 0.30, 0.40, 0.60],
        help="Thresholds for the sensitivity table. Empty list disables it.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use fewer images.")
    args = parser.parse_args(argv)

    require_subset()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    with SUBSET_JSON.open("r", encoding="utf-8") as fh:
        ground_truth = json.load(fh)
    images = ground_truth["images"]
    if args.limit:
        images = images[: args.limit]
        wanted = {im["id"] for im in images}
        ground_truth = {
            **ground_truth,
            "images": images,
            "annotations": [
                a for a in ground_truth["annotations"] if a["image_id"] in wanted
            ],
        }

    try:
        detector = Detector(
            model_path=model_path,
            input_size=(args.imgsz, args.imgsz),
            conf_threshold=MAP_CONF,
            iou_threshold=args.iou,
            max_detections=COCO_MAX_DETS,
        )
    except (ModelNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    header = [
        f"model      : {model_path.name}",
        f"input size : {args.imgsz}x{args.imgsz}",
        f"nms iou    : {args.iou}",
        f"images     : {len(images)} (COCO val2017 subset)",
        f"providers  : {', '.join(detector.providers)}",
    ]
    print("\n".join(header))
    print()

    per_image = run_detections(detector, images)

    # A filtered GT file is needed when --limit is in play; write it either way
    # so pycocotools always scores against exactly the images we ran.
    gt_path = SUBSET_DIR / "_eval_gt.json"
    with gt_path.open("w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh)

    print("scoring mAP...")
    map_scores = coco_map(to_coco_json(per_image, MAP_CONF), gt_path)

    sweep: list[tuple[float, dict[str, float], float]] = []
    for conf in sorted(args.conf_sweep or []):
        detections = to_coco_json(per_image, conf)
        dets_per_image = len(detections) / max(1, len(per_image))
        sweep.append((conf, coco_map(detections, gt_path), dets_per_image))

    classes = per_class_pr(per_image, ground_truth, conf=args.pr_conf)
    report = format_report(
        header, map_scores, classes, sweep, args.pr_conf, len(images)
    )

    cfg.output.results_dir.mkdir(parents=True, exist_ok=True)
    md_path = timestamped_path(cfg.output.results_dir, "accuracy", ".md")
    md_path.write_text(report, encoding="utf-8")

    csv_path = md_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "support", "tp", "fp", "fn", "precision", "recall", "f1"])
        for m in sorted(classes, key=lambda c: (-c.support, c.name)):
            writer.writerow(
                [
                    m.name, m.support, m.tp, m.fp, m.fn,
                    round(m.precision, 4), round(m.recall, 4), round(m.f1, 4),
                ]
            )

    print()
    print(report)
    print(f"wrote {md_path}\n      {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
