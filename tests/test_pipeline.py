from infrastructure_overwatch.detectors import DetectorAdapter
from infrastructure_overwatch.events import PipelineEventEngine
from infrastructure_overwatch.pipeline import run_frame_sequence
from infrastructure_overwatch.tracking import MultiTracker
from infrastructure_overwatch.types import Detection

ZONE = (0.0, 0.0, 100.0, 100.0)
# Deliberately huge -- big enough that every _ChurnyDetector detection below, however far
# apart, always counts as "inside" it. The flood this guards against is about track-id
# churn (a detection nothing gets to re-associate with), not about zone geometry, so the
# zone itself doesn't need to be realistic.
HUGE_ZONE = (0.0, 0.0, 1.0e7, 1.0e7)


class _ChurnyDetector(DetectorAdapter):
    """A stand-in for what a real dense scene actually does to a greedy IoU
    tracker: every frame it returns `n_per_frame` detections, each placed at
    its own globally unique slot 1000px apart from every other detection
    this detector will ever produce (box size 10 << 1000, so IoU between any
    two of them is always exactly 0) -- guaranteeing every single one spawns
    a brand-new track that can never be re-associated, exactly the track-id
    churn that produced ~2,000 ZONE_ENTRY events from a real, single
    VisDrone intersection sequence."""

    def __init__(self, n_per_frame: int = 8, size: float = 10.0, slot_spacing: float = 1000.0):
        self.n_per_frame = n_per_frame
        self.size = size
        self.slot_spacing = slot_spacing

    def detect(self, frame, frame_id: str) -> list[Detection]:
        i = int(frame_id[1:])
        dets = []
        for j in range(self.n_per_frame):
            slot = i * self.n_per_frame + j
            cx = cy = slot * self.slot_spacing
            dets.append(
                Detection(
                    frame_id=frame_id,
                    class_id=0,
                    label="vehicle_of_interest",
                    confidence=0.9,
                    x1=cx - self.size / 2,
                    y1=cy - self.size / 2,
                    x2=cx + self.size / 2,
                    y2=cy + self.size / 2,
                    model="test",
                )
            )
        return dets


def _run(min_hits: int, n_frames: int = 20, n_per_frame: int = 8):
    frame_ids = [f"F{i:06d}" for i in range(n_frames)]
    return run_frame_sequence(
        frames=[None] * n_frames,  # the churny detector ignores the frame array entirely
        detector=_ChurnyDetector(n_per_frame),
        protected_zone=HUGE_ZONE,
        frame_ids=frame_ids,
        tracker=MultiTracker(iou_thresh=0.25, min_hits=min_hits),
        event_engine=PipelineEventEngine(protected_zone=HUGE_ZONE, stop_frames=1000, loiter_frames=1000),
    )


def test_min_hits_1_reproduces_the_dense_scene_event_flood():
    # today's default behavior, unchanged: every churny, single-hit track counts
    result = _run(min_hits=1, n_frames=20, n_per_frame=8)
    zone_entries = [e for e in result.events if e.event_type == "ZONE_ENTRY"]
    assert len(zone_entries) == 20 * 8


def test_min_hits_3_suppresses_the_flood_from_track_churn():
    # the confirm-gate this test exists for: a track needs 3 real hits before it can
    # raise an event, and this detector never gives the same track a second hit
    result = _run(min_hits=3, n_frames=20, n_per_frame=8)
    zone_entries = [e for e in result.events if e.event_type == "ZONE_ENTRY"]
    assert len(zone_entries) == 0
    # detections/alerts themselves are unaffected -- only which ones can raise events
    assert len(result.detections) == 20 * 8


def test_min_hits_confirm_gate_still_allows_a_real_persistent_track_to_raise_an_event():
    class _SteadyDetector(DetectorAdapter):
        def detect(self, frame, frame_id: str) -> list[Detection]:
            return [
                Detection(
                    frame_id=frame_id,
                    class_id=0,
                    label="vehicle_of_interest",
                    confidence=0.9,
                    x1=45.0,
                    y1=45.0,
                    x2=55.0,
                    y2=55.0,
                    model="test",
                )
            ]

    n_frames = 5
    frame_ids = [f"F{i:06d}" for i in range(n_frames)]
    result = run_frame_sequence(
        frames=[None] * n_frames,
        detector=_SteadyDetector(),
        protected_zone=ZONE,
        frame_ids=frame_ids,
        tracker=MultiTracker(iou_thresh=0.25, min_hits=3),
        event_engine=PipelineEventEngine(protected_zone=ZONE, stop_frames=1000, loiter_frames=1000),
    )
    zone_entries = [e for e in result.events if e.event_type == "ZONE_ENTRY"]
    assert len(zone_entries) == 1  # a real, consistently-tracked object still raises exactly one event
