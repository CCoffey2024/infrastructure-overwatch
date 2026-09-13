import pytest

from infrastructure_overwatch.types import THREAT_CLASSES, TRIAGE_BANDS, Detection


def _det():
    return Detection(
        frame_id="F0",
        class_id=0,
        label="drone",
        confidence=0.8,
        x1=0.0,
        y1=0.0,
        x2=10.0,
        y2=20.0,
        model="test",
    )


def test_threat_classes_are_the_four_documented_classes():
    assert THREAT_CLASSES == ("drone", "dismount", "launch_flash", "vehicle_of_interest")


def test_triage_bands():
    assert TRIAGE_BANDS == ("auto_confirm", "analyst_review", "auto_discard")


def test_detection_xyxy():
    assert _det().xyxy == (0.0, 0.0, 10.0, 20.0)


def test_detection_area():
    assert _det().area == 200.0


def test_detection_center():
    assert _det().center == (5.0, 10.0)


def test_detection_is_immutable():
    det = _det()
    with pytest.raises(AttributeError):
        det.confidence = 0.5  # type: ignore[misc]


def test_detection_to_dict_round_trips_fields():
    det = _det()
    d = det.to_dict()
    assert d["label"] == "drone"
    assert d["confidence"] == 0.8
