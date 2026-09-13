"""Shared data contracts used across ingest, detection, tracking, and alerting.

Every module in this package passes `Detection`/`Event`/`FrameRecord` objects
across its boundaries instead of raw tuples or dicts, so a new detector
backend or a new alert consumer only has to agree on this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# The four threat classes this system is scoped to detect, and why each is
# modeled the way it is (see docs/ARCHITECTURE.md "Threat taxonomy" for the
# full reasoning):
#   - drone:              small UAS / loitering munition. Hard to see at
#                          range; the primary asymmetric threat.
#   - dismount:            an unauthorized person near the corridor or fence
#                          line. Narrow, upright silhouette.
#   - launch_flash:         a launch signature (flash + rising smoke), standing
#                          in for rocket/indirect-fire/stand-off threats. A
#                          fixed corridor camera cannot track a missile in
#                          flight; it can catch the launch signature. That is
#                          a deliberate scope-down, not a capability claim.
#   - vehicle_of_interest:  an unauthorized or "technical"-type vehicle
#                          approaching the corridor. The one class with a
#                          real-data-backed track (UAVDT vehicle footage)
#                          instead of a synthetic stand-in.
THREAT_CLASSES: tuple[str, ...] = ("drone", "dismount", "launch_flash", "vehicle_of_interest")

# Alert-triage bands a calibrated detection confidence is routed into. See
# calibration.py. "auto_confirm" still means "logged as a confirmed detection
# on the analyst's dashboard" -- there is no engagement action anywhere in
# this pipeline, at any band.
TRIAGE_BANDS: tuple[str, ...] = ("auto_confirm", "analyst_review", "auto_discard")


@dataclass(frozen=True)
class FrameRecord:
    """One sampled frame from an ingested video/sensor feed."""

    frame_id: str
    sensor_id: str
    timestamp_s: float
    modality: str
    source_video: str
    frame_number: int
    width: int
    height: int
    image_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Detection:
    """One detector output, before or after tracking/calibration.

    `class_id` is the detector's raw integer class index (backend-specific,
    e.g. a UAVDT category id for vehicle subtypes); `label` is the
    human-readable class name, normally one of THREAT_CLASSES. `track_id`,
    `calibrated_confidence`, and `triage_band` start unset and are filled in
    by the tracking and calibration stages respectively -- a Detection
    gains fields as it moves through the pipeline, it never loses them.
    """

    frame_id: str
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    model: str
    track_id: int | None = None
    calibrated_confidence: float | None = None
    triage_band: str | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Event:
    """A human-facing alert emitted by the event engine for one track.

    `severity` and `score` are triage hints for the analyst dashboard, not an
    automated decision -- every Event still ends at a person's judgment call.
    """

    frame_id: str
    timestamp_s: float
    event_type: str
    severity: str
    track_id: int | None
    description: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
