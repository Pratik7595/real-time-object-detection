"""COCO class names and the 80 -> 91 category-id mapping.

YOLOX predicts 80 contiguous class indices. COCO's ground-truth annotations use
91 sparse category ids (the original dataset reserved ids for classes that were
later dropped). `evaluate.py` has to translate between the two or every single
detection is scored against the wrong category.
"""

from __future__ import annotations

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Index i in COCO_CLASSES corresponds to COCO_91_IDS[i] in the annotation JSON.
COCO_91_IDS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
)

assert len(COCO_CLASSES) == len(COCO_91_IDS) == 80

_NAME_TO_INDEX = {name: i for i, name in enumerate(COCO_CLASSES)}


def resolve_class_filter(tokens: list[str] | None) -> set[int] | None:
    """Turn CLI `--classes` tokens into a set of class indices.

    Accepts names ("cell phone") or indices ("67"), so users can pass whichever
    they have to hand. Returns None for "no filter" rather than the full set,
    because the detector uses None to skip the filtering branch entirely.
    """
    if not tokens:
        return None

    out: set[int] = set()
    unknown: list[str] = []
    for raw in tokens:
        token = raw.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if 0 <= idx < len(COCO_CLASSES):
                out.add(idx)
            else:
                unknown.append(raw)
        elif token in _NAME_TO_INDEX:
            out.add(_NAME_TO_INDEX[token])
        else:
            unknown.append(raw)

    if unknown:
        raise ValueError(
            f"Unknown class name(s)/index(es): {', '.join(unknown)}. "
            f"Valid names are the 80 COCO classes, e.g. 'person', 'cell phone'."
        )
    return out or None
