"""Dependency-free geometric primitives shared by detection, tracking, and
grid decoding. Kept deliberately small and pure so every caller can unit-test
against it without a torch/opencv dependency.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


def iou_xyxy(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_rect(point: tuple[float, float], rect: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def rect_center(rect: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def nms_xyxy(boxes: list[tuple], iou_thresh: float = 0.4, score_index: int = 4) -> list[tuple]:
    """Greedy non-max suppression over `(x0, y0, x1, y1, score, ...)` tuples.

    Keeps the tuples' extra trailing fields (e.g. a class index) intact --
    callers decode class/score from fixed positions, this only removes
    overlapping lower-scored boxes.
    """
    remaining = sorted(boxes, key=lambda b: -b[score_index])
    keep: list[tuple] = []
    while remaining:
        best = remaining.pop(0)
        keep.append(best)
        remaining = [b for b in remaining if iou_xyxy(best[:4], b[:4]) < iou_thresh]
    return keep
