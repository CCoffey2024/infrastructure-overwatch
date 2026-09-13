"""Encode/decode between (box, class) ground truth and the grid-tensor label
format `GridDetector` (see detectors/grid_cnn.py) predicts.

The source R&D notebooks defined this encode/decode/NMS logic twice -- once
for a 96x96/6x6 synthetic chip detector, once for a 640x352/40x22 real-scene
detector -- with the only real differences being image size and grid shape.
This module parametrizes over `(img_w, img_h, grid_w, grid_h, n_classes)`
instead, so both detectors share one implementation.

Label tensor layout, per grid cell: `[objectness, tx, ty, tw, th, one_hot_class...]`
where `(tx, ty)` are the box center's offset within its cell (0-1) and
`(tw, th)` are the box width/height as a fraction of the full image.
"""

from __future__ import annotations

import numpy as np

from .geometry import nms_xyxy

BoxClass = tuple[tuple[float, float, float, float], int]


def encode_grid_label(
    boxes: list[BoxClass],
    img_w: int,
    img_h: int,
    grid_w: int,
    grid_h: int,
    n_classes: int,
) -> np.ndarray:
    """`[(x0, y0, x1, y1), class_id]` pairs -> a `(grid_h, grid_w, 5+n_classes)` label array."""
    cell_w, cell_h = img_w / grid_w, img_h / grid_h
    label = np.zeros((grid_h, grid_w, 5 + n_classes), dtype=np.float32)
    for (x0, y0, x1, y1), cls in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        gx = min(max(int(cx // cell_w), 0), grid_w - 1)
        gy = min(max(int(cy // cell_h), 0), grid_h - 1)
        tx, ty = (cx - gx * cell_w) / cell_w, (cy - gy * cell_h) / cell_h
        label[gy, gx, 0] = 1.0
        label[gy, gx, 1:5] = [tx, ty, w / img_w, h / img_h]
        label[gy, gx, 5:] = 0.0
        label[gy, gx, 5 + cls] = 1.0
    return label


def decode_grid_predictions(
    pred,
    img_w: int,
    img_h: int,
    grid_w: int,
    grid_h: int,
    thresh: float = 0.5,
    nms_iou: float = 0.4,
) -> list[tuple[float, float, float, float, float, int]]:
    """A `(grid_h, grid_w, 5+n_classes)` tensor -> NMS'd `(x0, y0, x1, y1, confidence, class_id)` boxes.

    `pred` is anything indexable like a `(grid_h, grid_w, 5+n_classes)` array
    that supports `float(...)` on a scalar slice and iteration over a 1-D
    slice -- a numpy array or a CPU torch tensor both work, so this module
    itself has no torch dependency. Callers passing a torch tensor must move
    it to CPU first (`.cpu()`); a CUDA tensor cannot be converted directly.
    """
    cell_w, cell_h = img_w / grid_w, img_h / grid_h
    boxes = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            obj = float(pred[gy, gx, 0])
            if obj < thresh:
                continue
            tx, ty, tw, th = [float(v) for v in pred[gy, gx, 1:5]]
            cx, cy = gx * cell_w + tx * cell_w, gy * cell_h + ty * cell_h
            w, h = tw * img_w, th * img_h
            cls = int(np.asarray(pred[gy, gx, 5:]).argmax())
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, obj, cls))
    return nms_xyxy(boxes, iou_thresh=nms_iou)
