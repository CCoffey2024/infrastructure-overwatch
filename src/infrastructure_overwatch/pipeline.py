"""Orchestrates one run of the full pipeline: detect -> track -> events ->
calibrated triage -> an alert table. This is the module that replaces what
the source R&D notebooks did inline, cell by cell, in a monolithic notebook
-- the same steps, as reusable, testable code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import DEFAULT_TRIAGE_THRESHOLDS, ConfidenceCalibrator, TriageThresholds, route_detections
from .detectors import DetectorAdapter
from .events import PipelineEventEngine
from .tracking import MultiTracker
from .types import Detection, Event


@dataclass
class PipelineResult:
    detections: list[Detection]
    events: list[Event]

    def alerts_frame(self) -> pd.DataFrame:
        """One row per tracked detection, ready to write to `alerts.csv` or
        hand to an analyst dashboard."""
        return pd.DataFrame([d.to_dict() for d in self.detections])

    def events_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.events])


def run_frame_sequence(
    frames: list[np.ndarray],
    detector: DetectorAdapter,
    protected_zone: tuple[float, float, float, float],
    frame_ids: list[str] | None = None,
    timestamps_s: list[float] | None = None,
    fps: float = 10.0,
    tracker: MultiTracker | None = None,
    event_engine: PipelineEventEngine | None = None,
    calibrator: ConfidenceCalibrator | None = None,
    triage_thresholds: TriageThresholds = DEFAULT_TRIAGE_THRESHOLDS,
) -> PipelineResult:
    """Runs `detector` over each frame, tracks detections across frames,
    raises zone events, and (if `calibrator` is fitted) attaches calibrated
    triage bands. `calibrator` is optional: without one, `calibrated_confidence`
    and `triage_band` stay unset on every detection -- calibration needs a
    labeled validation run first (see `calibration.ConfidenceCalibrator.fit`),
    it isn't something this function can bootstrap on its own.
    """
    n = len(frames)
    frame_ids = frame_ids or [f"F{i:06d}" for i in range(n)]
    timestamps_s = timestamps_s or [i / fps for i in range(n)]
    tracker = tracker or MultiTracker()
    event_engine = event_engine or PipelineEventEngine(protected_zone=protected_zone)

    all_detections: list[Detection] = []
    all_events: list[Event] = []
    for frame, frame_id, ts in zip(frames, frame_ids, timestamps_s, strict=True):
        raw = detector.detect(frame, frame_id)
        tracked = tracker.step(raw)
        if calibrator is not None:
            tracked = route_detections(tracked, calibrator, triage_thresholds)
        all_detections.extend(tracked)
        # Every tracked detection still lands in alerts.csv above; only tracks the tracker
        # considers mature (see `MultiTracker.min_hits`) can raise a zone/loiter event, so a
        # dense scene's inevitable one-hit tracks don't each fire their own event.
        confirmed = [d for d in tracked if d.track_id is not None and tracker.is_confirmed(d.track_id)]
        all_events.extend(event_engine.update(confirmed, ts))

    return PipelineResult(detections=all_detections, events=all_events)
