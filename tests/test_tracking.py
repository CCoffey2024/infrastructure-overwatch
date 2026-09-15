from infrastructure_overwatch.tracking import MultiTracker, count_id_switches, coverage, match_to_gt
from infrastructure_overwatch.types import Detection


def _det(x0, y0, size=10.0, label="vehicle_of_interest"):
    return Detection(
        frame_id="F",
        class_id=0,
        label=label,
        confidence=0.9,
        x1=x0,
        y1=y0,
        x2=x0 + size,
        y2=y0 + size,
        model="test",
    )


def test_new_detection_spawns_a_track():
    tracker = MultiTracker()
    out = tracker.step([_det(0, 0)])
    assert len(out) == 1
    assert out[0].track_id == 1


def test_overlapping_detection_next_frame_keeps_same_track_id():
    tracker = MultiTracker(iou_thresh=0.25)
    first = tracker.step([_det(0, 0)])
    second = tracker.step([_det(1, 1)])  # small shift, still heavily overlapping
    assert first[0].track_id == second[0].track_id


def test_far_away_detection_gets_a_new_track_id():
    tracker = MultiTracker(iou_thresh=0.25)
    first = tracker.step([_det(0, 0)])
    second = tracker.step([_det(500, 500)])
    assert first[0].track_id != second[0].track_id


def test_track_coasts_through_a_brief_miss_then_is_retired():
    tracker = MultiTracker(iou_thresh=0.25, max_coast=2)
    first = tracker.step([_det(0, 0)])
    tid = first[0].track_id
    tracker.step([])  # missed frame 1
    tracker.step([])  # missed frame 2
    assert tid in tracker.tracks  # still coasting (coast=2 <= max_coast)
    tracker.step([])  # missed frame 3 -- exceeds max_coast
    assert tid not in tracker.tracks


def test_min_hits_defaults_to_confirming_a_track_on_its_first_hit():
    tracker = MultiTracker()  # min_hits=1, today's existing behavior
    out = tracker.step([_det(0, 0)])
    assert tracker.is_confirmed(out[0].track_id)


def test_min_hits_gates_confirmation_until_enough_real_hits():
    tracker = MultiTracker(iou_thresh=0.25, min_hits=3)
    tid = tracker.step([_det(0, 0)])[0].track_id
    assert not tracker.is_confirmed(tid)  # hit 1
    tracker.step([_det(1, 1)])
    assert not tracker.is_confirmed(tid)  # hit 2, still a small overlapping shift -> same track
    tracker.step([_det(2, 2)])
    assert tracker.is_confirmed(tid)  # hit 3 -- now confirmed


def test_is_confirmed_is_false_for_an_unknown_or_retired_track():
    tracker = MultiTracker(min_hits=1)
    assert not tracker.is_confirmed(999)  # never existed
    tid = tracker.step([_det(0, 0)])[0].track_id
    for _ in range(tracker.max_coast + 1):
        tracker.step([])  # coast past max_coast -> retired
    assert not tracker.is_confirmed(tid)


def test_coasting_does_not_advance_hit_count():
    tracker = MultiTracker(iou_thresh=0.25, min_hits=2)
    tid = tracker.step([_det(0, 0)])[0].track_id
    tracker.step([])  # a missed/coasted frame must not count as a hit
    assert not tracker.is_confirmed(tid)
    tracker.step([_det(1, 1)])  # second real hit
    assert tracker.is_confirmed(tid)


def test_match_to_gt_finds_best_overlap():
    gt_boxes = [(0.0, 0.0, 10.0, 10.0, 101), (50.0, 50.0, 60.0, 60.0, 202)]
    assert match_to_gt((1.0, 1.0, 11.0, 11.0), gt_boxes) == 101
    assert match_to_gt((51.0, 51.0, 61.0, 61.0), gt_boxes) == 202
    assert match_to_gt((500.0, 500.0, 510.0, 510.0), gt_boxes) is None


def test_coverage_counts_matched_gt_object_frames():
    dets_per_frame = [[_det(0, 0)], [_det(0, 0)]]
    gt_per_frame = [[(0.0, 0.0, 10.0, 10.0, 1)], [(0.0, 0.0, 10.0, 10.0, 1)]]
    assert coverage(dets_per_frame, gt_per_frame) == 1.0


def test_coverage_is_nan_with_no_ground_truth():
    import math

    assert math.isnan(coverage([[]], [[]]))


def test_count_id_switches_detects_a_switch():
    d1 = _det(0, 0)
    d1_tracked = [type(d1)(**{**d1.to_dict(), "track_id": 1})]
    d2_tracked = [type(d1)(**{**d1.to_dict(), "track_id": 1})]
    gt_frame_1 = [(0.0, 0.0, 10.0, 10.0, 101)]
    gt_frame_2 = [(0.0, 0.0, 10.0, 10.0, 202)]  # same track id now matches a *different* real object

    switches, matched = count_id_switches([d1_tracked, d2_tracked], [gt_frame_1, gt_frame_2])
    assert switches == 1
    assert matched == 2
