"""Drawing: boxes, labels with confidence, and the FPS/HUD overlay.

Readability against both light and dark scenes is the constraint that drives
most of the decisions here. Plain coloured text on a webcam frame is unreadable
about half the time -- point a camera at a window and white text vanishes. So
every label sits on a filled chip in its class colour, and the text on that chip
is black or white depending on the chip's luminance.
"""

from __future__ import annotations

import colorsys
from typing import Sequence

import cv2
import numpy as np

from .coco_classes import COCO_CLASSES
from .detector import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _build_palette(n: int) -> list[tuple[int, int, int]]:
    """One stable BGR colour per class.

    Hues are spaced by the golden-ratio conjugate rather than evenly. With even
    spacing, neighbouring class ids (car/motorcycle, cup/fork) get neighbouring
    hues and are hard to tell apart on screen; the golden-ratio walk scatters
    them while still covering the wheel evenly.
    """
    palette: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = (i * 0.61803398875) % 1.0
        # Alternating value keeps consecutive classes distinguishable even for
        # viewers who struggle to separate hues.
        value = 0.95 if i % 2 == 0 else 0.72
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, value)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))
    return palette


CLASS_COLORS: list[tuple[int, int, int]] = _build_palette(len(COCO_CLASSES))


def _text_color_for(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever survives on this background (Rec. 601 luma)."""
    b, g, r = bgr
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luma > 150 else (255, 255, 255)


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    font_scale: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """Draw every detection in place and return the frame.

    Confidence is always rendered, always to two decimals -- a box without a
    score tells the viewer nothing about how much to trust it.
    """
    h, w = frame.shape[:2]

    for det in detections:
        # Clip before drawing: boxes can extend past the frame after the
        # letterbox inverse, and OpenCV will happily draw off-canvas.
        x1 = max(0, min(w - 1, int(det.x1)))
        y1 = max(0, min(h - 1, int(det.y1)))
        x2 = max(0, min(w - 1, int(det.x2)))
        y2 = max(0, min(h - 1, int(det.y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        color = CLASS_COLORS[det.class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.class_name} {det.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, 1)
        pad = 3
        chip_h = th + baseline + pad

        # Prefer the label above the box; flip inside when it would clip the top.
        chip_top = y1 - chip_h
        if chip_top < 0:
            chip_top = y1
        chip_bottom = chip_top + chip_h
        chip_right = min(w, x1 + tw + pad * 2)

        cv2.rectangle(frame, (x1, chip_top), (chip_right, chip_bottom), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + pad, chip_bottom - baseline - 1),
            FONT,
            font_scale,
            _text_color_for(color),
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_hud(
    frame: np.ndarray,
    fps_instant: float,
    fps_rolling: float,
    detection_count: int,
    extra_lines: Sequence[str] = (),
    font_scale: float = 0.5,
    stale: bool = False,
) -> np.ndarray:
    """Translucent stats panel, top-left.

    The panel is alpha-blended rather than solid so the user can still see what
    the camera is pointed at underneath it, and so the demo video does not have
    an opaque block sitting over a corner of every frame.
    """
    lines = [
        f"FPS {fps_instant:5.1f} inst | {fps_rolling:5.1f} avg",
        f"Detections: {detection_count}",
    ]
    lines.extend(extra_lines)
    if stale:
        lines.append("boxes reused (--infer-every)")

    widths = [cv2.getTextSize(t, FONT, font_scale, 1)[0][0] for t in lines]
    line_h = int(22 * max(font_scale / 0.5, 1.0))
    pad = 8
    panel_w = max(widths) + pad * 2
    panel_h = line_h * len(lines) + pad

    panel = frame[0:panel_h, 0:panel_w]
    if panel.size:
        # 0.45 keeps the underlying scene visible while still darkening it
        # enough that white text reads against a bright window.
        cv2.addWeighted(panel, 0.45, np.zeros_like(panel), 0.0, 0.0, dst=panel)

    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - 6
        # Dark stroke under light text: readable over a white wall and a dark
        # room without needing to know which one we are looking at.
        cv2.putText(frame, text, (pad, y), FONT, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            frame, text, (pad, y), FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )

    return frame


def draw_footer(frame: np.ndarray, text: str, font_scale: float = 0.45) -> np.ndarray:
    """Bottom-left hint line (key bindings, recording state)."""
    h = frame.shape[0]
    y = h - 10
    cv2.putText(frame, text, (8, y), FONT, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_recording_dot(frame: np.ndarray) -> np.ndarray:
    """Unmissable red dot, top-right, so a demo take is never ambiguous."""
    w = frame.shape[1]
    cv2.circle(frame, (w - 22, 22), 8, (0, 0, 255), -1)
    cv2.circle(frame, (w - 22, 22), 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (w - 70, 28), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "REC", (w - 70, 28), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
