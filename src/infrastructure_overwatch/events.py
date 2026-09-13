"""Turns tracked detections into human-facing alerts about a protected zone.

Every `Event` this engine emits is a sensor-to-analyst alert-routing decision,
never an engagement decision -- see docs/USER_MANUAL.md for what each event
type means operationally and what it explicitly does not authorize.
"""

from __future__ import annotations

from collections import defaultdict, deque

from .geometry import euclidean, point_in_rect
from .types import Detection, Event


class PipelineEventEngine:
    def __init__(
        self,
        protected_zone: tuple[float, float, float, float],
        stop_px: float = 8.0,
        stop_frames: int = 15,
        loiter_frames: int = 25,
    ):
        self.protected_zone = protected_zone
        self.stop_px = float(stop_px)
        self.stop_frames = int(stop_frames)
        self.loiter_frames = int(loiter_frames)
        self.history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max(self.stop_frames, self.loiter_frames) + 5)
        )
        self.was_inside: dict[int, bool] = defaultdict(bool)
        self.emitted: set[tuple[int, str]] = set()

    def update(self, detections: list[Detection], timestamp_s: float) -> list[Event]:
        events = []
        for d in detections:
            if d.track_id is None:
                continue
            tid = d.track_id
            center = d.center
            inside = point_in_rect(center, self.protected_zone)
            self.history[tid].append((timestamp_s, center, d.label, inside, d.frame_id))

            if inside and not self.was_inside[tid]:
                key = (tid, "zone_entry")
                if key not in self.emitted:
                    events.append(
                        Event(
                            d.frame_id,
                            timestamp_s,
                            "ZONE_ENTRY",
                            "medium",
                            tid,
                            f"Track {tid} ({d.label}) entered the protected corridor.",
                            0.55,
                        )
                    )
                    self.emitted.add(key)
            self.was_inside[tid] = inside

            hist = list(self.history[tid])
            if len(hist) >= self.stop_frames:
                pts = [h[1] for h in hist[-self.stop_frames :]]
                movement = max(euclidean(pts[0], p) for p in pts[1:]) if len(pts) > 1 else 0
                if inside and movement <= self.stop_px:
                    key = (tid, "stopped")
                    if key not in self.emitted:
                        events.append(
                            Event(
                                d.frame_id,
                                timestamp_s,
                                "STOPPED_IN_ZONE",
                                "high",
                                tid,
                                f"Track {tid} remained nearly stationary inside the corridor.",
                                0.80,
                            )
                        )
                        self.emitted.add(key)

            if d.label == "dismount" and len(hist) >= self.loiter_frames:
                recent = hist[-self.loiter_frames :]
                if sum(1 for h in recent if h[3]) >= int(self.loiter_frames * 0.8):
                    key = (tid, "loiter")
                    if key not in self.emitted:
                        events.append(
                            Event(
                                d.frame_id,
                                timestamp_s,
                                "PERSON_LOITER",
                                "high",
                                tid,
                                f"Person track {tid} persisted within the protected corridor.",
                                0.85,
                            )
                        )
                        self.emitted.add(key)
        return events
