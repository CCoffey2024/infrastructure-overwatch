"""A teaching/testing adapter that replays known annotations instead of
running a model. Used to debug tracking and event logic independently from
detector error, and to build deterministic pipeline tests."""

from __future__ import annotations

import pandas as pd

from ..types import Detection
from . import DetectorAdapter


class GroundTruthAdapter(DetectorAdapter):
    def __init__(self, gt_df: pd.DataFrame):
        self.gt_df = gt_df.copy()
        labels = sorted(self.gt_df.label.unique().tolist())
        self.label_to_id = {v: i for i, v in enumerate(labels)}

    def detect_by_frame_number(self, frame_id: str, frame_number: int) -> list[Detection]:
        rows = self.gt_df[self.gt_df.frame_number == frame_number]
        out = []
        for _, r in rows.iterrows():
            out.append(
                Detection(
                    frame_id=frame_id,
                    class_id=self.label_to_id[str(r.label)],
                    label=str(r.label),
                    confidence=1.0,
                    x1=float(r.x1),
                    y1=float(r.y1),
                    x2=float(r.x2),
                    y2=float(r.y2),
                    model="ground_truth",
                    track_id=int(r.object_id) if int(r.object_id) >= 0 else None,
                )
            )
        return out

    def detect(self, frame, frame_id: str) -> list[Detection]:
        raise RuntimeError("GroundTruthAdapter requires detect_by_frame_number(frame_id, frame_number)")
