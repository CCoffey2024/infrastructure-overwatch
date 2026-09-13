import numpy as np
import pytest

from infrastructure_overwatch.calibration import (
    ConfidenceCalibrator,
    TriageThresholds,
    route_detections,
    triage_band,
)
from infrastructure_overwatch.types import Detection


def _det(confidence: float) -> Detection:
    return Detection(
        frame_id="F0",
        class_id=0,
        label="drone",
        confidence=confidence,
        x1=0,
        y1=0,
        x2=1,
        y2=1,
        model="test",
    )


def test_triage_band_boundaries():
    thresholds = TriageThresholds(auto_confirm=0.9, discard=0.05)
    assert triage_band(0.95, thresholds) == "auto_confirm"
    assert triage_band(0.9, thresholds) == "auto_confirm"  # boundary is inclusive
    assert triage_band(0.5, thresholds) == "analyst_review"
    assert triage_band(0.05, thresholds) == "analyst_review"  # boundary is inclusive on the review side
    assert triage_band(0.04, thresholds) == "auto_discard"


def test_calibrator_requires_fit_before_use():
    calibrator = ConfidenceCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.calibrate(np.array([0.5]))


def test_calibrator_fit_then_calibrate_monotonic():
    rng = np.random.default_rng(0)
    confs = rng.uniform(0, 1, 200)
    is_tp = (confs > 0.5).astype(int)
    calibrator = ConfidenceCalibrator().fit(confs, is_tp)
    low = calibrator.calibrate(np.array([0.1]))[0]
    high = calibrator.calibrate(np.array([0.9]))[0]
    assert high >= low


def test_route_detections_attaches_band_and_calibrated_confidence():
    rng = np.random.default_rng(0)
    confs = rng.uniform(0, 1, 100)
    is_tp = (confs > 0.5).astype(int)
    calibrator = ConfidenceCalibrator().fit(confs, is_tp)

    dets = [_det(0.95), _det(0.5), _det(0.01)]
    routed = route_detections(dets, calibrator)

    assert len(routed) == 3
    for d in routed:
        assert d.calibrated_confidence is not None
        assert d.triage_band in ("auto_confirm", "analyst_review", "auto_discard")


def test_route_detections_empty_list():
    calibrator = ConfidenceCalibrator().fit(np.array([0.1, 0.9]), np.array([0, 1]))
    assert route_detections([], calibrator) == []
