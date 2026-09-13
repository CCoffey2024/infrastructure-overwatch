from infrastructure_overwatch.grid_codec import decode_grid_predictions, encode_grid_label


def test_encode_decode_roundtrip_single_box():
    img_w = img_h = 96
    grid_w = grid_h = 6
    n_classes = 3
    boxes = [((20.0, 20.0, 40.0, 50.0), 1)]

    label = encode_grid_label(boxes, img_w, img_h, grid_w, grid_h, n_classes)
    decoded = decode_grid_predictions(label, img_w, img_h, grid_w, grid_h, thresh=0.5, nms_iou=1.1)

    assert len(decoded) == 1
    x0, y0, x1, y1, conf, cls = decoded[0]
    assert cls == 1
    assert conf == 1.0
    # decoded box should be close to the original (grid quantization introduces some error)
    assert abs(x0 - 20.0) < 20
    assert abs(y0 - 20.0) < 20
    assert abs(x1 - 40.0) < 20
    assert abs(y1 - 50.0) < 20


def test_encode_empty_boxes_yields_no_objectness():
    label = encode_grid_label([], 96, 96, 6, 6, 3)
    assert label[..., 0].sum() == 0


def test_decode_below_threshold_yields_nothing():
    img_w = img_h = 96
    grid_w = grid_h = 6
    boxes = [((20.0, 20.0, 40.0, 50.0), 0)]
    label = encode_grid_label(boxes, img_w, img_h, grid_w, grid_h, 3)
    decoded = decode_grid_predictions(label, img_w, img_h, grid_w, grid_h, thresh=1.5)
    assert decoded == []


def test_encode_decode_multiple_boxes_different_cells():
    img_w = img_h = 96
    grid_w = grid_h = 6
    n_classes = 3
    boxes = [
        ((5.0, 5.0, 15.0, 15.0), 0),
        ((70.0, 70.0, 90.0, 90.0), 2),
    ]
    label = encode_grid_label(boxes, img_w, img_h, grid_w, grid_h, n_classes)
    decoded = decode_grid_predictions(label, img_w, img_h, grid_w, grid_h, thresh=0.5, nms_iou=1.1)
    assert len(decoded) == 2
    classes = sorted(d[5] for d in decoded)
    assert classes == [0, 2]
