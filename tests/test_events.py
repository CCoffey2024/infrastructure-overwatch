from infrastructure_overwatch.events import PipelineEventEngine
from infrastructure_overwatch.types import Detection

ZONE = (0.0, 0.0, 100.0, 100.0)


def _det(track_id, cx, cy, label="vehicle_of_interest", size=10.0):
    return Detection(
        frame_id="F",
        class_id=0,
        label=label,
        confidence=0.9,
        x1=cx - size / 2,
        y1=cy - size / 2,
        x2=cx + size / 2,
        y2=cy + size / 2,
        model="test",
        track_id=track_id,
    )


def test_zone_entry_fires_once():
    engine = PipelineEventEngine(protected_zone=ZONE, stop_frames=1000, loiter_frames=1000)
    events_1 = engine.update([_det(1, 50, 50)], timestamp_s=0.0)
    events_2 = engine.update([_det(1, 51, 51)], timestamp_s=0.1)
    assert len(events_1) == 1
    assert events_1[0].event_type == "ZONE_ENTRY"
    assert events_2 == []  # already inside, no duplicate entry event


def test_zone_entry_ignores_detections_outside_zone():
    engine = PipelineEventEngine(protected_zone=ZONE)
    events = engine.update([_det(1, -50, -50)], timestamp_s=0.0)
    assert events == []


def test_zone_entry_ignores_untracked_detections():
    engine = PipelineEventEngine(protected_zone=ZONE)
    untracked = Detection(
        frame_id="F",
        class_id=0,
        label="vehicle_of_interest",
        confidence=0.9,
        x1=45,
        y1=45,
        x2=55,
        y2=55,
        model="test",
        track_id=None,
    )
    assert engine.update([untracked], timestamp_s=0.0) == []


def test_stopped_in_zone_fires_after_stop_frames_of_no_movement():
    engine = PipelineEventEngine(protected_zone=ZONE, stop_px=2.0, stop_frames=5, loiter_frames=1000)
    events = []
    for i in range(6):
        events += engine.update([_det(1, 50, 50)], timestamp_s=float(i))
    types = [e.event_type for e in events]
    assert "ZONE_ENTRY" in types
    assert "STOPPED_IN_ZONE" in types


def test_stopped_in_zone_does_not_fire_while_moving():
    engine = PipelineEventEngine(protected_zone=ZONE, stop_px=1.0, stop_frames=5, loiter_frames=1000)
    events = []
    for i in range(6):
        events += engine.update([_det(1, 50 + i * 5, 50)], timestamp_s=float(i))
    types = [e.event_type for e in events]
    assert "STOPPED_IN_ZONE" not in types


def test_person_loiter_requires_dismount_label():
    engine = PipelineEventEngine(protected_zone=ZONE, stop_frames=1000, loiter_frames=5)
    events = []
    for i in range(6):
        events += engine.update([_det(1, 50, 50, label="dismount")], timestamp_s=float(i))
    assert any(e.event_type == "PERSON_LOITER" for e in events)


def test_person_loiter_does_not_fire_for_non_dismount_label():
    engine = PipelineEventEngine(protected_zone=ZONE, stop_frames=1000, loiter_frames=5)
    events = []
    for i in range(6):
        events += engine.update([_det(1, 50, 50, label="vehicle_of_interest")], timestamp_s=float(i))
    assert not any(e.event_type == "PERSON_LOITER" for e in events)
