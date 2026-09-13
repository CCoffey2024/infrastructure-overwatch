"""A zero-model motion detector: useful as a dependency-free fallback, a
sanity check for the tracking/event layers, and a baseline to measure the
learned detectors against."""

from __future__ import annotations

import cv2
import numpy as np

from ..types import Detection
from . import DetectorAdapter


class ClassicalMotionAdapter(DetectorAdapter):
    """Background subtraction + morphology + contours.

    This detector intentionally knows only that something moved. It does not
    infer drone/dismount/vehicle semantics -- `label` is always `"motion"`,
    `class_id` is always `-1`. Downstream code that needs a class should use
    a learned detector instead; this one exists to prove the tracking/event
    machinery works even when the detector is trivial.
    """

    def __init__(
        self, min_area: float = 70, history: int = 120, var_threshold: float = 28, detect_shadows: bool = False
    ):
        self.min_area = float(min_area)
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray, frame_id: str) -> list[Detection]:
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg = cv2.dilate(fg, self.kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < self.min_area:
                continue
            conf = min(0.99, 0.25 + np.log1p(area) / 15)
            out.append(
                Detection(
                    frame_id=frame_id,
                    class_id=-1,
                    label="motion",
                    confidence=float(conf),
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    model="opencv_mog2",
                )
            )
        return out
