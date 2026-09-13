"""Detector backends, all behind one `DetectorAdapter` interface.

Every backend -- classical background subtraction, the project's own grid-CNN,
or an Ultralytics YOLO model -- returns the same `types.Detection` object.
That lets the rest of the pipeline (tracking, events, calibration) stay
backend-agnostic, and lets a reviewer swap or compare detectors without
touching anything downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection


class DetectorAdapter(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: str) -> list[Detection]: ...
