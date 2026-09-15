"""Runs the real-data pipeline (detect -> track -> events -> calibrate -> optional anomaly
scoring) against one `--source` spec, independent of any particular entry point.

`cli.py`'s `demo-real` command and `service.py`'s job executor both need to turn a source
spec into a `pipeline.PipelineResult` the exact same way -- this module is the one place
that knows how, so the CLI and the operator console can never drift apart on what a "run"
actually does. Raises plain `ValueError`/`FileNotFoundError` on a bad source rather than
printing to stderr, since each caller has its own way of surfacing an error (an exit code
for the CLI, a failed job for the service).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import DEFAULT_TRIAGE_THRESHOLDS, ConfidenceCalibrator, self_calibrate
from .detectors.grid_cnn import GridCNNAdapter
from .events import PipelineEventEngine
from .pipeline import PipelineResult, run_frame_sequence
from .tracking import MultiTracker

# An illustrative "protected zone" for real-data sources -- UAVDT/VisDrone/an arbitrary
# video or folder are general drone/traffic footage, not a real perimeter, so there is no
# real protected-zone geometry to read off the data. Mirrors cli.REAL_DEMO_PROTECTED_ZONE;
# duplicated as a plain constant rather than imported, since cli.py is the argparse-facing
# entry point and shouldn't be a dependency of this module or of service.py.
DEFAULT_PROTECTED_ZONE = (96.0, 88.0, 544.0, 334.0)


@dataclass
class RealRunConfig:
    weights_path: str
    source: str  # "<uavdt|visdrone-vid|video|folder>:<sequence-or-path>"
    split: str = "val"  # VisDrone2019-VID split; ignored for uavdt/video/folder sources
    class_scheme: str = "vehicle_only"
    stride: int = 2
    cap: int = 150
    fps: float = 10.0
    conf_thresh: float = 0.4
    calibrate: bool = False
    track_min_hits: int = 1
    anomaly_ref: str | None = None
    zone: tuple[float, float, float, float] | None = None


@dataclass
class RealRunOutput:
    result: PipelineResult
    frame_ids: list[str]
    rgb_frames: list[np.ndarray] = field(repr=False)
    has_ground_truth: bool
    calibration_warning: str | None = None


def load_real_source(
    source: str, stride: int, cap: int, class_scheme: str, split: str
) -> tuple[list[str], list[np.ndarray], list | None, list[np.ndarray], str]:
    """Parses `source` and returns `(frame_ids, frames, labels, rgb_frames, dataset_kind)`.
    `frames` are letterboxed grayscale canvases ready for the grid-CNN detector; `labels`
    is one `grid_codec.encode_grid_label` tensor per frame for a named-benchmark source
    (ground truth to calibrate/score against), or `None` for a `video:`/`folder:` source
    (no ground truth exists for arbitrary footage)."""
    from .ingest import (
        WORK_H,
        WORK_W,
        UAVDTIndex,
        UAVDTVehicleDataset,
        VisDroneVIDIndex,
        VisDroneVIDVehicleDataset,
        letterbox,
        load_image_folder_frames,
        load_video_frames,
    )

    if ":" not in source:
        raise ValueError(f"source must be '<uavdt|visdrone-vid|video|folder>:<sequence-or-path>', got {source!r}")
    dataset_kind, seq = source.split(":", 1)

    ds: UAVDTVehicleDataset | VisDroneVIDVehicleDataset
    if dataset_kind == "uavdt":
        uavdt_index = UAVDTIndex()
        ds = UAVDTVehicleDataset(uavdt_index, [seq], stride=stride, cap=cap, class_scheme=class_scheme)
        if len(ds) == 0:
            raise FileNotFoundError(f"No frames found for source {source!r}; check the sequence name.")
        frame_ids = [f"{seq}_{fidx:06d}" for _seq, fidx in ds.items]
        frames, labels, rgb_frames = [], [], []
        for i, (_seq, fidx) in enumerate(ds.items):
            gray, label = ds[i]
            frames.append(gray)
            labels.append(label)
            canvas, _scale, _pad_x, _pad_y = letterbox(uavdt_index.load_frame(seq, fidx), WORK_W, WORK_H)
            rgb_frames.append(np.asarray(canvas))
        return frame_ids, frames, labels, rgb_frames, dataset_kind

    if dataset_kind == "visdrone-vid":
        visdrone_index = VisDroneVIDIndex()
        ds = VisDroneVIDVehicleDataset(
            visdrone_index, split=split, seq_list=[seq], stride=stride, cap=cap, class_scheme=class_scheme
        )
        if len(ds) == 0:
            raise FileNotFoundError(f"No frames found for source {source!r}; check the sequence name.")
        frame_ids = [f"{seq}_{fidx:06d}" for _seq, fidx in ds.items]
        frames, labels, rgb_frames = [], [], []
        for i, (_seq, fidx) in enumerate(ds.items):
            gray, label = ds[i]
            frames.append(gray)
            labels.append(label)
            canvas, _scale, _pad_x, _pad_y = letterbox(visdrone_index.load_frame(split, seq, fidx), WORK_W, WORK_H)
            rgb_frames.append(np.asarray(canvas))
        return frame_ids, frames, labels, rgb_frames, dataset_kind

    if dataset_kind in ("video", "folder"):
        loader = load_video_frames if dataset_kind == "video" else load_image_folder_frames
        frame_ids, frames, rgb_frames = loader(seq, stride=stride, cap=cap, work_w=WORK_W, work_h=WORK_H)
        return frame_ids, frames, None, rgb_frames, dataset_kind

    raise ValueError(f"Unknown source kind {dataset_kind!r}; expected 'uavdt', 'visdrone-vid', 'video', or 'folder'.")


def run_real_pipeline(config: RealRunConfig) -> RealRunOutput:
    from .ingest import VEHICLE_GRID_H, VEHICLE_GRID_W, WORK_H, WORK_W, class_names_for_scheme

    frame_ids, frames, labels, rgb_frames, dataset_kind = load_real_source(
        config.source, config.stride, config.cap, config.class_scheme, config.split
    )

    class_names = class_names_for_scheme(config.class_scheme)
    adapter = GridCNNAdapter.load(
        config.weights_path,
        n_classes=len(class_names),
        grid_h=VEHICLE_GRID_H,
        grid_w=VEHICLE_GRID_W,
        class_names=class_names,
        img_w=WORK_W,
        img_h=WORK_H,
        thresh=config.conf_thresh,
        stage_channels=(24, 48, 96, 96),
    )

    zone = config.zone or DEFAULT_PROTECTED_ZONE

    calibrator: ConfidenceCalibrator | None = None
    calibration_warning: str | None = None
    if config.calibrate:
        if labels is None:
            calibration_warning = (
                f"calibrate has no effect for a {dataset_kind!r} source: there is no ground truth to "
                "self-calibrate against. Detections/tracking/events/alerts still run, uncalibrated."
            )
        else:
            calibrator = self_calibrate(adapter, frames, labels, WORK_W, WORK_H, VEHICLE_GRID_W, VEHICLE_GRID_H)

    result = run_frame_sequence(
        frames=frames,
        detector=adapter,
        protected_zone=zone,
        frame_ids=frame_ids,
        fps=config.fps,
        tracker=MultiTracker(min_hits=config.track_min_hits),
        event_engine=PipelineEventEngine(protected_zone=zone),
        calibrator=calibrator,
        triage_thresholds=DEFAULT_TRIAGE_THRESHOLDS,
    )

    if config.anomaly_ref:
        from .anomaly import EmbeddingAnomalyScorer, score_track_anomalies

        scorer = EmbeddingAnomalyScorer.load(config.anomaly_ref)
        result.events.extend(score_track_anomalies(rgb_frames, frame_ids, result.detections, scorer, fps=config.fps))

    return RealRunOutput(
        result=result,
        frame_ids=frame_ids,
        rgb_frames=rgb_frames,
        has_ground_truth=labels is not None,
        calibration_warning=calibration_warning,
    )
