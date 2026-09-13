import pandas as pd

from infrastructure_overwatch.evaluation import evaluate_framewise, precision_recall_f1, slice_metrics


def test_precision_recall_f1_perfect():
    assert precision_recall_f1(10, 0, 0) == (1.0, 1.0, 1.0)


def test_precision_recall_f1_no_predictions():
    assert precision_recall_f1(0, 0, 5) == (0.0, 0.0, 0.0)


def _row(frame_number, x1, y1, x2, y2, label="drone", **extra):
    return {"frame_number": frame_number, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label, **extra}


def test_evaluate_framewise_perfect_match():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10)])
    pred = pd.DataFrame([_row(0, 0, 0, 10, 10, confidence=0.9)])
    result = evaluate_framewise(gt, pred)
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["f1"] == 1.0


def test_evaluate_framewise_missed_detection():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10)])
    pred = pd.DataFrame(columns=["frame_number", "x1", "y1", "x2", "y2", "label", "confidence"])
    result = evaluate_framewise(gt, pred)
    assert result["fn"] == 1
    assert result["tp"] == 0


def test_slice_metrics_groups_by_column():
    gt = pd.DataFrame(
        [
            _row(0, 0, 0, 10, 10, occlusion="none"),
            _row(1, 0, 0, 10, 10, occlusion="heavy"),
        ]
    )
    pred = pd.DataFrame(
        [
            _row(0, 0, 0, 10, 10, confidence=0.9),
            # frame 1 (heavy occlusion) gets no matching prediction
        ]
    )
    result = slice_metrics(gt, pred, slice_col="occlusion")
    assert set(result["occlusion"]) == {"none", "heavy"}
    none_row = result[result["occlusion"] == "none"].iloc[0]
    heavy_row = result[result["occlusion"] == "heavy"].iloc[0]
    assert none_row["f1"] == 1.0
    assert heavy_row["fn"] == 1
