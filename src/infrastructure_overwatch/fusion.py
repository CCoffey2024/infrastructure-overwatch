"""Late fusion across sensors.

Two levels, kept separate because they answer different questions:

- `late_fuse_detections` fuses raw EO/IR *boxes* from two already
  pixel-registered sensors into one higher-confidence detection. A
  production system would calibrate each sensor into a common coordinate
  system and propagate geometric uncertainty; this assumes that's already
  done -- a secondary capability kept for completeness, not the primary
  detection path.
- `fuse_events` fuses *semantic events* (`ZONE_ENTRY`, `STOPPED_IN_ZONE`,
  ...) across any number of independently-run sensor jobs, each already
  through its own full detect/track/event pipeline. Corroboration is by
  time window and event type (optionally also label and/or pixel-space
  IoU), never raw pixels or detector features -- this is what "N sensors
  each already produced an Event, do 2+ of them agree" means. See
  docs/sensor-fusion.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import iou_xyxy
from .types import Detection


def late_fuse_detections(
    eo_dets: list[Detection], ir_dets: list[Detection], iou_threshold: float = 0.25
) -> list[Detection]:
    """Merges overlapping EO/IR detections into one higher-confidence
    detection; keeps unmatched detections from either sensor as-is."""
    fused = []
    used_ir: set[int] = set()
    for e in eo_dets:
        best_j, best_iou = None, 0.0
        for j, r in enumerate(ir_dets):
            if j in used_ir:
                continue
            score = iou_xyxy(e.xyxy, r.xyxy)
            if score > best_iou:
                best_j, best_iou = j, score
        if best_j is not None and best_iou >= iou_threshold:
            r = ir_dets[best_j]
            used_ir.add(best_j)
            w1, w2 = e.confidence, r.confidence
            s = max(1e-6, w1 + w2)
            fused.append(
                Detection(
                    frame_id=e.frame_id,
                    class_id=e.class_id if e.label != "motion" else r.class_id,
                    label=e.label if e.label != "motion" else r.label,
                    confidence=min(0.995, 1 - (1 - e.confidence) * (1 - r.confidence)),
                    x1=(e.x1 * w1 + r.x1 * w2) / s,
                    y1=(e.y1 * w1 + r.y1 * w2) / s,
                    x2=(e.x2 * w1 + r.x2 * w2) / s,
                    y2=(e.y2 * w1 + r.y2 * w2) / s,
                    model="late_fusion_eo_ir",
                )
            )
        else:
            fused.append(e)
    for j, r in enumerate(ir_dets):
        if j not in used_ir:
            fused.append(r)
    return fused


# --- Semantic event fusion across N independently-run sensor jobs -----------

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class SensorEventEvidence:
    """One sensor job's own `Event`, tagged with which job/sensor produced
    it. `label`/`box` are recovered by joining against that job's
    `alerts.csv` on `(frame_id, track_id)` -- `events.csv` itself carries
    neither, but label/spatial matching need them."""

    job_id: str
    sensor_id: str
    event_type: str
    timestamp_s: float
    track_id: int | None
    label: str | None
    severity: str
    score: float
    frame_id: str
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FusionConfig:
    max_time_delta_s: float = 0.5
    min_sensors: int = 2
    require_matching_label: bool = False
    # Opt-in, and only a meaningful signal when the contributing sensors are known to
    # already be pixel-registered into one common geometry -- see the module docstring.
    spatial_iou_threshold: float | None = None


# FusionConfig is frozen/immutable, so one module-level default instance is safe to share
# across every call site (mirrors calibration.DEFAULT_TRIAGE_THRESHOLDS).
DEFAULT_FUSION_CONFIG = FusionConfig()


def load_sensor_event_evidence(job_id: str, sensor_id: str, output_dir) -> list[SensorEventEvidence]:
    """Reads one completed run job's `events.csv` (+ `alerts.csv`, to recover each
    event's label/box) into evidence rows `fuse_events` can match across sensors."""
    output_dir = Path(output_dir)
    events_df = pd.read_csv(output_dir / "events.csv")
    alerts_path = output_dir / "alerts.csv"
    alerts_df = (
        pd.read_csv(alerts_path)
        if alerts_path.exists()
        else pd.DataFrame(columns=["frame_id", "track_id", "label", "x1", "y1", "x2", "y2"])
    )

    evidence = []
    for row in events_df.to_dict("records"):
        match = alerts_df[(alerts_df.frame_id == row["frame_id"]) & (alerts_df.track_id == row["track_id"])]
        label = str(match.iloc[0]["label"]) if len(match) else None
        box = tuple(match.iloc[0][["x1", "y1", "x2", "y2"]]) if len(match) else None
        evidence.append(
            SensorEventEvidence(
                job_id=job_id,
                sensor_id=sensor_id,
                event_type=str(row["event_type"]),
                timestamp_s=float(row["timestamp_s"]),
                track_id=(None if pd.isna(row["track_id"]) else int(row["track_id"])),
                label=label,
                severity=str(row["severity"]),
                score=float(row["score"]),
                frame_id=str(row["frame_id"]),
                box=box,
            )
        )
    return evidence


def _compatible(a: SensorEventEvidence, b: SensorEventEvidence, config: FusionConfig) -> bool:
    if a.sensor_id == b.sensor_id or a.event_type != b.event_type:
        return False
    if abs(a.timestamp_s - b.timestamp_s) > config.max_time_delta_s:
        return False
    if config.require_matching_label and a.label != b.label:
        return False
    if config.spatial_iou_threshold is not None:
        if a.box is None or b.box is None or iou_xyxy(a.box, b.box) < config.spatial_iou_threshold:
            return False
    return True


def fuse_events(
    evidence: list[SensorEventEvidence], config: FusionConfig = DEFAULT_FUSION_CONFIG
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily groups evidence from `evidence` (spanning any number of sensor jobs) by
    event type, time window, and optionally label/pixel-IoU, into fused events backed by
    at least `config.min_sensors` distinct sensors -- this is late fusion: it consumes
    each sensor's own persisted `Event` evidence, never raw pixels or detector features.
    Returns `(fused_events_df, contributors_df)`; the second is full provenance, one row
    per raw evidence item folded into each fused event, so a fused event's promotion can
    be audited back to exactly which sensors/tracks produced it.
    """
    order = sorted(range(len(evidence)), key=lambda i: evidence[i].timestamp_s)
    used: set[int] = set()
    fused_rows: list[dict] = []
    contributor_rows: list[dict] = []

    for idx in order:
        if idx in used:
            continue
        anchor = evidence[idx]
        group = [idx]
        for jdx in order:
            if jdx in used or jdx in group:
                continue
            candidate = evidence[jdx]
            if all(_compatible(evidence[g], candidate, config) for g in group):
                group.append(jdx)

        members = [evidence[g] for g in group]
        sensors_in_group = {m.sensor_id for m in members}
        if len(sensors_in_group) < config.min_sensors:
            continue
        used.update(group)

        fusion_event_id = f"fusion-{len(fused_rows) + 1:04d}"
        fused_rows.append(
            {
                "fusion_event_id": fusion_event_id,
                "event_type": anchor.event_type,
                "timestamp_s": float(np.mean([m.timestamp_s for m in members])),
                "n_sensors": len(sensors_in_group),
                "sensor_ids": ",".join(sorted(sensors_in_group)),
                "label": anchor.label,
                "max_severity": max(members, key=lambda m: _SEVERITY_RANK.get(m.severity, 0)).severity,
            }
        )
        for m in members:
            contributor_rows.append(
                {
                    "fusion_event_id": fusion_event_id,
                    "job_id": m.job_id,
                    "sensor_id": m.sensor_id,
                    "event_type": m.event_type,
                    "timestamp_s": m.timestamp_s,
                    "track_id": m.track_id,
                    "label": m.label,
                    "severity": m.severity,
                    "score": m.score,
                    "frame_id": m.frame_id,
                }
            )

    fused_columns = ["fusion_event_id", "event_type", "timestamp_s", "n_sensors", "sensor_ids", "label", "max_severity"]
    contributor_columns = [
        "fusion_event_id",
        "job_id",
        "sensor_id",
        "event_type",
        "timestamp_s",
        "track_id",
        "label",
        "severity",
        "score",
        "frame_id",
    ]
    return pd.DataFrame(fused_rows, columns=fused_columns), pd.DataFrame(contributor_rows, columns=contributor_columns)
