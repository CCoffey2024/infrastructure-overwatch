import numpy as np
import pandas as pd
import pytest

from infrastructure_overwatch.evaluation import (
    average_precision,
    calibration_reliability,
    evaluate_framewise,
    mean_average_precision,
    precision_recall_curve,
    precision_recall_f1,
    slice_metrics,
)


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


# --- precision_recall_curve / average_precision / mean_average_precision ----


def test_precision_recall_curve_is_empty_with_no_predictions():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10)])
    pred = pd.DataFrame(columns=["frame_number", "x1", "y1", "x2", "y2", "label", "confidence"])
    curve = precision_recall_curve(gt, pred)
    assert curve.empty


def test_precision_recall_curve_orders_by_descending_confidence():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10), _row(0, 50, 50, 60, 60)])
    pred = pd.DataFrame(
        [
            _row(0, 50, 50, 60, 60, confidence=0.4),  # lower confidence, listed first in the input
            _row(0, 0, 0, 10, 10, confidence=0.9),  # higher confidence, listed second
        ]
    )
    curve = precision_recall_curve(gt, pred)
    assert list(curve["confidence"]) == [0.9, 0.4]  # re-sorted descending regardless of input order
    assert list(curve["precision"]) == [1.0, 1.0]
    assert list(curve["recall"]) == [0.5, 1.0]


def test_average_precision_is_zero_with_no_ground_truth():
    gt = pd.DataFrame(columns=["frame_number", "x1", "y1", "x2", "y2", "label"])
    pred = pd.DataFrame([_row(0, 0, 0, 10, 10, confidence=0.9)])
    assert average_precision(gt, pred) == 0.0


def test_average_precision_is_zero_with_no_predictions():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10)])
    pred = pd.DataFrame(columns=["frame_number", "x1", "y1", "x2", "y2", "label", "confidence"])
    assert average_precision(gt, pred) == 0.0


def test_average_precision_is_one_for_a_perfect_detector():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10), _row(0, 50, 50, 60, 60)])
    pred = pd.DataFrame([_row(0, 0, 0, 10, 10, confidence=0.9), _row(0, 50, 50, 60, 60, confidence=0.8)])
    assert average_precision(gt, pred) == pytest.approx(1.0)


def test_average_precision_matches_hand_computed_value_for_partial_recall():
    # 2 ground-truth boxes, only 1 ever matched -> precision=1.0 plateaus for recall in
    # [0, 0.5] (51 of the 101 interpolation points), then drops to 0.0 beyond -- AP = 51/101.
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10), _row(0, 50, 50, 60, 60)])
    pred = pd.DataFrame([_row(0, 0, 0, 10, 10, confidence=0.9)])
    assert average_precision(gt, pred) == pytest.approx(51 / 101)


def test_mean_average_precision_averages_per_class_ap():
    # "drone" is perfectly detected (AP=1.0), "dismount" is entirely missed (AP=0.0)
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10, label="drone"), _row(0, 50, 50, 60, 60, label="dismount")])
    pred = pd.DataFrame([_row(0, 0, 0, 10, 10, label="drone", confidence=0.9)])

    result = mean_average_precision(gt, pred, iou_thresholds=(0.5,))

    assert result["per_class"]["drone"][0.5] == pytest.approx(1.0)
    assert result["per_class"]["dismount"][0.5] == pytest.approx(0.0)
    assert result["map_50"] == pytest.approx(0.5)
    assert result["map_50_95"] == pytest.approx(0.5)


def test_mean_average_precision_is_nan_with_no_ground_truth():
    gt = pd.DataFrame(columns=["frame_number", "x1", "y1", "x2", "y2", "label"])
    pred = pd.DataFrame([_row(0, 0, 0, 10, 10, confidence=0.9)])
    result = mean_average_precision(gt, pred)
    assert np.isnan(result["map_50"])
    assert np.isnan(result["map_50_95"])


# --- calibration_reliability --------------------------------------------


def test_calibration_reliability_buckets_confidence_and_observed_accuracy():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10), _row(1, 0, 0, 10, 10)])
    pred = pd.DataFrame(
        [
            # frame 0: a correct, high-confidence detection
            _row(0, 0, 0, 10, 10, confidence=0.9, calibrated_confidence=0.95),
            # frame 1: an incorrect (no matching gt box) detection at a similarly high
            # calibrated confidence -- same bin as the correct one above, lower accuracy
            _row(1, 500, 500, 510, 510, confidence=0.9, calibrated_confidence=0.92),
        ]
    )
    result = calibration_reliability(gt, pred, n_bins=10)

    assert len(result) == 1  # both predictions land in the same [0.9, 1.0) bucket
    bucket = result.iloc[0]
    assert bucket["n"] == 2
    assert bucket["observed_accuracy"] == pytest.approx(0.5)  # 1 of 2 was actually correct
    assert bucket["mean_confidence"] == pytest.approx((0.95 + 0.92) / 2)


def test_calibration_reliability_drops_rows_with_null_confidence():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10)])
    pred = pd.DataFrame(
        [
            _row(0, 0, 0, 10, 10, confidence=0.9, calibrated_confidence=0.8),
            _row(0, 0, 0, 10, 10, confidence=0.5, calibrated_confidence=None),  # never calibrated
        ]
    )
    result = calibration_reliability(gt, pred, n_bins=10)
    assert result["n"].sum() == 1


def test_calibration_reliability_is_empty_with_no_predictions():
    gt = pd.DataFrame([_row(0, 0, 0, 10, 10)])
    pred = pd.DataFrame(
        columns=["frame_number", "x1", "y1", "x2", "y2", "label", "confidence", "calibrated_confidence"]
    )
    result = calibration_reliability(gt, pred)
    assert result.empty
    assert list(result.columns) == ["bin_low", "bin_high", "mean_confidence", "observed_accuracy", "n"]
