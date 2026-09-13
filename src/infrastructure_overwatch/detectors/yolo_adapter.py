"""An Ultralytics YOLO backend behind the same `DetectorAdapter` interface.

`vehicle_of_interest` (car/truck/bus) is the one threat class COCO already
knows, so a stock, COCO-pretrained YOLO model is a fair, zero-fine-tuning
comparison against this project's own vehicle detector -- see
`COCO_TO_VEHICLE_SUBTYPE` and docs/METHODOLOGY_AND_LIMITATIONS.md. The other
three classes (`drone`, `dismount`, `launch_flash`) have no COCO equivalent;
comparing YOLO against them requires fine-tuning it first, which is itself a
measured, stated finding, not a shortcut.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Detection
from . import DetectorAdapter

# COCO class id -> UAVDT category id, for the one class (vehicles) a stock
# COCO-pretrained model can be compared on without fine-tuning.
COCO_TO_VEHICLE_SUBTYPE = {2: 1, 7: 2, 5: 3}  # car, truck, bus


class YOLOAdapter(DetectorAdapter):
    """Loads `ultralytics` lazily so the rest of the project has no hard
    dependency on it -- most of this pipeline runs without ever importing
    YOLO."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.25,
        device: str | None = None,
        class_map: dict | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Install ultralytics to use YOLOAdapter: pip install ultralytics") from e
        # Typed as Any: ultralytics' own return types (predict() can yield Results
        # or a raw Tensor depending on call mode) are broader than this adapter's
        # single, fixed usage pattern -- narrowing happens by how we call it below,
        # not by static typing against ultralytics' stubs.
        self.model: Any = YOLO(model_path)
        self.conf = float(conf)
        self.device = device
        self.class_map = class_map or {}

    def detect(self, frame: np.ndarray, frame_id: str) -> list[Detection]:
        result = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        names = result.names
        out: list[Detection] = []
        if result.boxes is None:
            return out
        for b in result.boxes:
            xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
            cls = int(b.cls[0].item())
            label = self.class_map.get(cls, names.get(cls, str(cls)))
            conf = float(b.conf[0].item())
            out.append(
                Detection(
                    frame_id=frame_id,
                    class_id=cls,
                    label=label,
                    confidence=conf,
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                    model="ultralytics_yolo",
                )
            )
        return out
