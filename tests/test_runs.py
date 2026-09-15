import cv2
import numpy as np
import pytest
import torch

from infrastructure_overwatch import runs
from infrastructure_overwatch.detectors.grid_cnn import GridDetector
from infrastructure_overwatch.ingest import VEHICLE_GRID_H, VEHICLE_GRID_W


def _write_synthetic_video(path, n_frames: int = 12, size=(640, 352), fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), fill_value=i * 10 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_untrained_weights(path, n_classes: int = 3) -> None:
    """A randomly-initialized (never trained) grid detector, just to exercise the
    load-source -> detect -> track -> event wiring without needing a real trained
    checkpoint or real benchmark data. Must match the stage_channels
    `runs.run_real_pipeline` builds its GridCNNAdapter with."""
    model = GridDetector(
        n_classes=n_classes, grid_h=VEHICLE_GRID_H, grid_w=VEHICLE_GRID_W, stage_channels=(24, 48, 96, 96)
    )
    torch.save(model.state_dict(), path)


def test_load_real_source_rejects_a_source_without_a_colon():
    with pytest.raises(ValueError, match="source must be"):
        runs.load_real_source("not-a-valid-source", stride=1, cap=10, class_scheme="vehicle_only", split="val")


def test_load_real_source_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown source kind"):
        runs.load_real_source("carrier-pigeon:foo", stride=1, cap=10, class_scheme="vehicle_only", split="val")


def test_load_real_source_video_has_no_ground_truth(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, n_frames=5)

    frame_ids, frames, labels, rgb_frames, dataset_kind = runs.load_real_source(
        f"video:{video_path}", stride=1, cap=10, class_scheme="vehicle_only", split="val"
    )

    assert dataset_kind == "video"
    assert labels is None
    assert len(frame_ids) == len(frames) == len(rgb_frames) == 5


def test_run_real_pipeline_against_a_video_source_runs_uncalibrated(tmp_path):
    weights_path = tmp_path / "weights.pt"
    _write_untrained_weights(weights_path)
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, n_frames=6)

    config = runs.RealRunConfig(
        weights_path=str(weights_path),
        source=f"video:{video_path}",
        stride=1,
        calibrate=True,  # requested, but a video source has no ground truth to calibrate against
        conf_thresh=0.01,  # low enough that an untrained model still emits some detections
    )
    output = runs.run_real_pipeline(config)

    assert output.has_ground_truth is False
    assert output.calibration_warning is not None and "video" in output.calibration_warning
    assert len(output.frame_ids) == 6
    assert len(output.rgb_frames) == 6
    # every detection stays uncalibrated (no calibrator was ever built)
    assert all(d.calibrated_confidence is None for d in output.result.detections)


def test_run_real_pipeline_respects_track_min_hits(tmp_path):
    weights_path = tmp_path / "weights.pt"
    _write_untrained_weights(weights_path)
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, n_frames=6)

    config = runs.RealRunConfig(
        weights_path=str(weights_path),
        source=f"video:{video_path}",
        conf_thresh=0.01,
        track_min_hits=1000,  # impossibly high -- no track can ever be confirmed
    )
    output = runs.run_real_pipeline(config)

    assert output.result.events == []  # nothing could reach the confirm-gate
