from infrastructure_overwatch.geometry import euclidean, iou_xyxy, nms_xyxy, point_in_rect, rect_center


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou_xyxy(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou_xyxy((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 6.0, 6.0)) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection 5x10=50, union 100+100-50=150
    assert abs(iou_xyxy(a, b) - 50 / 150) < 1e-9


def test_point_in_rect():
    rect = (0.0, 0.0, 10.0, 10.0)
    assert point_in_rect((5.0, 5.0), rect)
    assert not point_in_rect((15.0, 5.0), rect)


def test_rect_center():
    assert rect_center((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


def test_euclidean():
    assert euclidean((0.0, 0.0), (3.0, 4.0)) == 5.0


def test_nms_drops_lower_scored_overlapping_box():
    boxes = [
        (0.0, 0.0, 10.0, 10.0, 0.9, 0),
        (1.0, 1.0, 11.0, 11.0, 0.5, 0),  # heavily overlaps the first, lower score
        (50.0, 50.0, 60.0, 60.0, 0.4, 1),  # disjoint, should survive
    ]
    kept = nms_xyxy(boxes, iou_thresh=0.4)
    assert len(kept) == 2
    kept_scores = sorted(b[4] for b in kept)
    assert kept_scores == [0.4, 0.9]
