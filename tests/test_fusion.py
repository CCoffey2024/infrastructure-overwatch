import pandas as pd

from infrastructure_overwatch.fusion import (
    FusionConfig,
    SensorEventEvidence,
    fuse_events,
    late_fuse_detections,
    load_sensor_event_evidence,
)
from infrastructure_overwatch.types import Detection


def _det(x0, y0, size=10.0, label="vehicle_of_interest", confidence=0.8, class_id=0, model="test"):
    return Detection(
        frame_id="F",
        class_id=class_id,
        label=label,
        confidence=confidence,
        x1=x0,
        y1=y0,
        x2=x0 + size,
        y2=y0 + size,
        model=model,
    )


# --- late_fuse_detections (docs/VALIDATION.md claimed this was already tested; it wasn't) --


def test_late_fuse_detections_merges_overlapping_eo_ir_boxes():
    eo = [_det(0, 0, confidence=0.6)]
    ir = [_det(1, 1, confidence=0.7)]  # small shift, still heavily overlapping

    fused = late_fuse_detections(eo, ir, iou_threshold=0.25)

    assert len(fused) == 1
    assert fused[0].model == "late_fusion_eo_ir"
    assert fused[0].confidence > max(eo[0].confidence, ir[0].confidence)  # combined evidence, more confident


def test_late_fuse_detections_keeps_unmatched_detections_from_both_sensors():
    eo = [_det(0, 0)]
    ir = [_det(500, 500)]  # nowhere near the EO detection

    fused = late_fuse_detections(eo, ir, iou_threshold=0.25)

    assert len(fused) == 2
    assert {d.model for d in fused} == {"test"}  # neither was fused, both pass through unchanged


def test_late_fuse_detections_classical_motion_label_is_replaced_by_the_named_ir_label():
    eo = [_det(0, 0, label="motion", class_id=-1, model="classical")]  # class-agnostic motion detector
    ir = [_det(1, 1, label="vehicle_of_interest", class_id=0)]

    fused = late_fuse_detections(eo, ir, iou_threshold=0.25)

    assert fused[0].label == "vehicle_of_interest"
    assert fused[0].class_id == 0


# --- load_sensor_event_evidence / fuse_events --------------------------------


def _write_run(tmp_path, name, events, alerts):
    out_dir = tmp_path / name
    out_dir.mkdir()
    pd.DataFrame(events).to_csv(out_dir / "events.csv", index=False)
    pd.DataFrame(alerts).to_csv(out_dir / "alerts.csv", index=False)
    return out_dir


def test_load_sensor_event_evidence_joins_label_and_box_from_alerts_csv(tmp_path):
    out_dir = _write_run(
        tmp_path,
        "eo_run",
        events=[
            {
                "frame_id": "F000001",
                "timestamp_s": 1.0,
                "event_type": "ZONE_ENTRY",
                "severity": "medium",
                "track_id": 7,
                "description": "...",
                "score": 0.55,
            }
        ],
        alerts=[{"frame_id": "F000001", "track_id": 7, "label": "car", "x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 12.0}],
    )
    evidence = load_sensor_event_evidence("job-eo", "EO_1", out_dir)

    assert len(evidence) == 1
    assert evidence[0].label == "car"
    assert evidence[0].box == (1.0, 2.0, 11.0, 12.0)
    assert evidence[0].sensor_id == "EO_1"
    assert evidence[0].job_id == "job-eo"


def test_load_sensor_event_evidence_handles_no_alerts_csv(tmp_path):
    out_dir = tmp_path / "no_alerts_run"
    out_dir.mkdir()
    pd.DataFrame(
        [
            {
                "frame_id": "F1",
                "timestamp_s": 1.0,
                "event_type": "ZONE_ENTRY",
                "severity": "medium",
                "track_id": 1,
                "description": "...",
                "score": 0.5,
            }
        ]
    ).to_csv(out_dir / "events.csv", index=False)

    evidence = load_sensor_event_evidence("job-1", "EO_1", out_dir)
    assert evidence[0].label is None
    assert evidence[0].box is None


def test_fuse_events_corroborates_two_sensors_within_the_time_window():
    evidence = [
        SensorEventEvidence("job-eo", "EO_1", "ZONE_ENTRY", 10.0, 1, "car", "medium", 0.55, "F1"),
        SensorEventEvidence("job-ir", "IR_1", "ZONE_ENTRY", 10.2, 3, "car", "medium", 0.55, "F1"),
    ]
    fused_df, contributors_df = fuse_events(evidence, FusionConfig(max_time_delta_s=0.5, min_sensors=2))

    assert len(fused_df) == 1
    assert fused_df.iloc[0]["n_sensors"] == 2
    assert set(fused_df.iloc[0]["sensor_ids"].split(",")) == {"EO_1", "IR_1"}
    assert len(contributors_df) == 2
    assert set(contributors_df["fusion_event_id"]) == {fused_df.iloc[0]["fusion_event_id"]}


def test_fuse_events_drops_a_single_sensor_observation_below_min_sensors():
    evidence = [SensorEventEvidence("job-eo", "EO_1", "ZONE_ENTRY", 10.0, 1, "car", "medium", 0.55, "F1")]
    fused_df, contributors_df = fuse_events(evidence, FusionConfig(min_sensors=2))
    assert fused_df.empty
    assert contributors_df.empty


def test_fuse_events_requires_events_within_the_time_window():
    evidence = [
        SensorEventEvidence("job-eo", "EO_1", "ZONE_ENTRY", 0.0, 1, "car", "medium", 0.55, "F1"),
        SensorEventEvidence("job-ir", "IR_1", "ZONE_ENTRY", 5.0, 3, "car", "medium", 0.55, "F1"),  # too far apart
    ]
    fused_df, _contributors_df = fuse_events(evidence, FusionConfig(max_time_delta_s=0.5, min_sensors=2))
    assert fused_df.empty


def test_fuse_events_requires_the_same_event_type():
    evidence = [
        SensorEventEvidence("job-eo", "EO_1", "ZONE_ENTRY", 10.0, 1, "car", "medium", 0.55, "F1"),
        SensorEventEvidence("job-ir", "IR_1", "STOPPED_IN_ZONE", 10.0, 3, "car", "high", 0.8, "F1"),
    ]
    fused_df, _contributors_df = fuse_events(evidence, FusionConfig(min_sensors=2))
    assert fused_df.empty


def test_fuse_events_never_matches_two_observations_from_the_same_sensor():
    evidence = [
        SensorEventEvidence("job-eo-1", "EO_1", "ZONE_ENTRY", 10.0, 1, "car", "medium", 0.55, "F1"),
        SensorEventEvidence("job-eo-2", "EO_1", "ZONE_ENTRY", 10.1, 2, "car", "medium", 0.55, "F1"),
    ]
    fused_df, _contributors_df = fuse_events(evidence, FusionConfig(min_sensors=2))
    assert fused_df.empty  # same sensor_id twice never counts as 2 distinct sensors


def test_fuse_events_require_matching_label_rejects_a_label_mismatch():
    evidence = [
        SensorEventEvidence("job-eo", "EO_1", "ZONE_ENTRY", 10.0, 1, "car", "medium", 0.55, "F1"),
        SensorEventEvidence("job-ir", "IR_1", "ZONE_ENTRY", 10.0, 3, "truck", "medium", 0.55, "F1"),
    ]
    fused_df, _c = fuse_events(evidence, FusionConfig(min_sensors=2, require_matching_label=True))
    assert fused_df.empty

    fused_df_unlabeled, _c = fuse_events(evidence, FusionConfig(min_sensors=2, require_matching_label=False))
    assert len(fused_df_unlabeled) == 1  # without the label requirement, time+type alone is enough


def test_fuse_events_spatial_iou_threshold_requires_overlapping_boxes():
    near = SensorEventEvidence("job-eo", "EO_1", "ZONE_ENTRY", 10.0, 1, "car", "medium", 0.55, "F1", box=(0, 0, 10, 10))
    far = SensorEventEvidence(
        "job-ir", "IR_1", "ZONE_ENTRY", 10.0, 3, "car", "medium", 0.55, "F1", box=(500, 500, 510, 510)
    )
    fused_df, _c = fuse_events(evidence := [near, far], FusionConfig(min_sensors=2, spatial_iou_threshold=0.3))
    assert fused_df.empty

    overlapping = SensorEventEvidence(
        "job-ir", "IR_1", "ZONE_ENTRY", 10.0, 3, "car", "medium", 0.55, "F1", box=(1, 1, 11, 11)
    )
    fused_df2, _c = fuse_events([near, overlapping], FusionConfig(min_sensors=2, spatial_iou_threshold=0.3))
    assert len(fused_df2) == 1


def test_fuse_events_contributor_rows_preserve_provenance_for_audit():
    evidence = [
        SensorEventEvidence("job-eo", "EO_1", "ZONE_ENTRY", 10.0, 1, "car", "medium", 0.55, "F1"),
        SensorEventEvidence("job-ir", "IR_1", "ZONE_ENTRY", 10.2, 3, "car", "medium", 0.55, "F1"),
    ]
    fused_df, contributors_df = fuse_events(evidence, FusionConfig(min_sensors=2))
    fid = fused_df.iloc[0]["fusion_event_id"]
    rows = contributors_df[contributors_df.fusion_event_id == fid]
    assert set(rows["job_id"]) == {"job-eo", "job-ir"}
    assert set(rows["track_id"]) == {1, 3}


def test_fuse_events_returns_empty_frames_with_the_right_columns_for_no_evidence():
    fused_df, contributors_df = fuse_events([])
    assert fused_df.empty
    assert list(fused_df.columns) == [
        "fusion_event_id",
        "event_type",
        "timestamp_s",
        "n_sensors",
        "sensor_ids",
        "label",
        "max_severity",
    ]
    assert contributors_df.empty
