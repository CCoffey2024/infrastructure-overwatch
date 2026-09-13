from pathlib import Path

import pandas as pd

from infrastructure_overwatch.reporting import build_dashboard, build_report_card


def _alerts_df():
    return pd.DataFrame(
        [
            {"label": "drone", "confidence": 0.9, "triage_band": "auto_confirm"},
            {"label": "dismount", "confidence": 0.4, "triage_band": "analyst_review"},
            {"label": "drone", "confidence": 0.02, "triage_band": "auto_discard"},
        ]
    )


def _events_df():
    return pd.DataFrame(
        [
            {"event_type": "ZONE_ENTRY", "severity": "medium"},
            {"event_type": "STOPPED_IN_ZONE", "severity": "high"},
        ]
    )


def test_report_card_builds_with_data():
    fig = build_report_card(_alerts_df(), _events_df())
    assert fig is not None


def test_report_card_builds_with_empty_frames():
    empty_alerts = pd.DataFrame(columns=["label", "confidence", "triage_band"])
    empty_events = pd.DataFrame(columns=["severity"])
    fig = build_report_card(empty_alerts, empty_events)
    assert fig is not None


def test_report_card_writes_file(tmp_path: Path):
    out_path = tmp_path / "report_card.png"
    build_report_card(_alerts_df(), _events_df(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_dashboard_builds_with_data():
    fig = build_dashboard(_alerts_df(), _events_df())
    assert fig is not None


def test_dashboard_builds_with_empty_frames():
    empty_alerts = pd.DataFrame(columns=["label", "confidence", "triage_band"])
    empty_events = pd.DataFrame(columns=["severity"])
    fig = build_dashboard(empty_alerts, empty_events)
    assert fig is not None


def test_dashboard_writes_html_file(tmp_path: Path):
    out_path = tmp_path / "dashboard.html"
    build_dashboard(_alerts_df(), _events_df(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_dashboard_handles_missing_triage_band_column():
    alerts_no_band = pd.DataFrame([{"label": "drone", "confidence": 0.5}])
    fig = build_dashboard(alerts_no_band, _events_df())
    assert fig is not None
