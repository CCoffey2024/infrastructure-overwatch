"""Command-line entry points. Two parallel tracks: `train-synthetic` / `demo` (the
synthetic corridor scene -- all four threat classes, no external data needed) and
`train-real` / `demo-real` (the real vehicle_of_interest(+dismount) track, backed by
UAVDT/VisDrone -- see `docs/DEVELOPMENT.md`). Run `python -m infrastructure_overwatch --help`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .calibration import ConfidenceCalibrator


def _cmd_train_synthetic(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader

    from .detectors.grid_cnn import GridDetector, evaluate_grid_detector, train_grid_detector
    from .synthetic import GRID, IMG_SIZE, N_SYNTHETIC_CLASSES, SyntheticCorridorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    day_train = SyntheticCorridorDataset(args.n_train, domain="day", seed=1)
    day_val = SyntheticCorridorDataset(200, domain="day", seed=2)
    night_val = SyntheticCorridorDataset(200, domain="night", seed=3)
    loader = DataLoader(day_train, batch_size=32, shuffle=True)

    model = GridDetector(n_classes=N_SYNTHETIC_CLASSES, grid_h=GRID, grid_w=GRID)
    t0 = time.time()
    train_grid_detector(model, loader, epochs=args.epochs, device=device)
    print(f"Trained in {time.time() - t0:.1f}s")

    day_score = evaluate_grid_detector(model, day_val, IMG_SIZE, IMG_SIZE, device=device)
    night_score = evaluate_grid_detector(model, night_val, IMG_SIZE, IMG_SIZE, device=device)
    print(f"Day   val -> F1 {day_score.f1:.2f}  (precision {day_score.precision:.2f}, recall {day_score.recall:.2f})")
    print(
        f"Night val -> F1 {night_score.f1:.2f}  (precision {night_score.precision:.2f}, recall {night_score.recall:.2f})"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}")
    return 0


# UAVDT sequences curated for the real vehicle_of_interest(+dismount) track -- the same
# split `notebooks/02_real_data_validation.ipynb` and `notebooks/08_visdrone_augmentation.ipynb`
# use, so a number measured here is directly comparable to what's already written up in
# docs/METHODOLOGY_AND_LIMITATIONS.md.
UAVDT_DAY_TRAIN_SEQS = ["M0101", "M0402", "M1201", "M1306"]
UAVDT_DAY_VAL_SEQS = ["M0403", "M1301"]
UAVDT_NIGHT_VAL_SEQS = ["M0601", "M0701", "M1009", "M1101"]

# An illustrative "protected zone" for the real-data demo -- UAVDT/VisDrone are general
# drone-traffic benchmarks, not footage of an actual perimeter, so there is no real
# protected-zone geometry to read off the data. This is a placeholder covering roughly the
# lower-central two-thirds of the working canvas (WORK_W/WORK_H); override with `--zone` for
# anything that should mean something for a specific sequence.
REAL_DEMO_PROTECTED_ZONE = (96.0, 88.0, 544.0, 334.0)


def _cmd_train_real(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    from .detectors.grid_cnn import GridDetector, evaluate_grid_detector, train_grid_detector
    from .ingest import (
        VEHICLE_GRID_H,
        VEHICLE_GRID_W,
        WORK_H,
        WORK_W,
        UAVDTIndex,
        UAVDTVehicleDataset,
        VisDroneDETIndex,
        VisDroneDETVehicleDataset,
        VisDroneVIDIndex,
        VisDroneVIDVehicleDataset,
        class_names_for_scheme,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    class_names = class_names_for_scheme(args.class_scheme)
    uavdt_index = UAVDTIndex()
    uavdt_train_ds = UAVDTVehicleDataset(
        uavdt_index, UAVDT_DAY_TRAIN_SEQS, stride=8, cap=80, seed=1, class_scheme=args.class_scheme
    )
    day_val_ds = UAVDTVehicleDataset(
        uavdt_index, UAVDT_DAY_VAL_SEQS, stride=10, cap=60, seed=2, class_scheme=args.class_scheme
    )
    night_val_ds = UAVDTVehicleDataset(
        uavdt_index, UAVDT_NIGHT_VAL_SEQS, stride=10, cap=50, seed=3, class_scheme=args.class_scheme
    )
    frame_counts = [len(uavdt_train_ds)]
    train_datasets: list[Dataset] = [uavdt_train_ds]

    if args.visdrone:
        visdrone_det_index = VisDroneDETIndex()
        visdrone_det_train_ds = VisDroneDETVehicleDataset(
            visdrone_det_index, split="train", class_scheme=args.class_scheme, stride=20, cap=350
        )
        visdrone_vid_index = VisDroneVIDIndex()
        vid_train_seqs = visdrone_vid_index.list_sequences("train")[::6][:10]
        visdrone_vid_train_ds = VisDroneVIDVehicleDataset(
            visdrone_vid_index,
            split="train",
            seq_list=vid_train_seqs,
            class_scheme=args.class_scheme,
            stride=15,
            cap=30,
        )
        frame_counts += [len(visdrone_det_train_ds), len(visdrone_vid_train_ds)]
        train_datasets += [visdrone_det_train_ds, visdrone_vid_train_ds]

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = GridDetector(
        n_classes=len(class_names), grid_h=VEHICLE_GRID_H, grid_w=VEHICLE_GRID_W, stage_channels=(24, 48, 96, 96)
    )
    t0 = time.time()
    train_grid_detector(model, loader, epochs=args.epochs, device=device)
    print(
        f"Trained in {time.time() - t0:.1f}s on {sum(frame_counts)} frames ({', '.join(str(c) for c in frame_counts)})"
    )

    day_score = evaluate_grid_detector(model, day_val_ds, WORK_W, WORK_H, device=device)
    night_score = evaluate_grid_detector(model, night_val_ds, WORK_W, WORK_H, device=device)
    print(f"Day   val -> F1 {day_score.f1:.2f}  (precision {day_score.precision:.2f}, recall {day_score.recall:.2f})")
    print(
        f"Night val -> F1 {night_score.f1:.2f}  (precision {night_score.precision:.2f}, recall {night_score.recall:.2f})"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}  (class_scheme={args.class_scheme!r}, classes={class_names})")
    return 0


def _self_calibrate(
    adapter, frames, labels, img_w: int, img_h: int, grid_w: int, grid_h: int
) -> ConfidenceCalibrator | None:
    """A quick self-calibration pass: label this run's own detections against its own
    ground truth (`labels`, one `grid_codec.encode_grid_label` tensor per frame -- the same
    format both the synthetic renderer and the real UAVDT/VisDrone datasets produce). A
    real deployment should calibrate on a held-out validation set instead (see
    `calibration.ConfidenceCalibrator`) -- this exists so `--calibrate` has something to
    show without requiring a separate validation run.
    """
    from .calibration import ConfidenceCalibrator
    from .geometry import iou_xyxy
    from .grid_codec import decode_grid_predictions

    confs, is_tp = [], []
    for frame, label in zip(frames, labels, strict=True):
        preds = adapter.detect(frame, "cal")
        gt_boxes = decode_grid_predictions(label, img_w, img_h, grid_w, grid_h, thresh=0.5, nms_iou=1.1)

        matched: set[int] = set()
        for p in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt_boxes):
                if j in matched:
                    continue
                v = iou_xyxy(p.xyxy, g[:4])
                if v > best_iou:
                    best_iou, best_j = v, j
            confs.append(p.confidence)
            if best_iou >= 0.3:
                is_tp.append(1)
                matched.add(best_j)
            else:
                is_tp.append(0)
    if not confs:
        return None
    return ConfidenceCalibrator().fit(np.array(confs), np.array(is_tp))


def _cmd_demo(args: argparse.Namespace) -> int:
    from .calibration import DEFAULT_TRIAGE_THRESHOLDS
    from .detectors.grid_cnn import GridCNNAdapter
    from .events import PipelineEventEngine
    from .grid_codec import encode_grid_label
    from .pipeline import run_frame_sequence
    from .synthetic import (
        GRID,
        IMG_SIZE,
        N_SYNTHETIC_CLASSES,
        PROTECTED_ZONE,
        SYNTHETIC_CLASS_NAMES,
        render_corridor_sequence,
    )
    from .tracking import MultiTracker

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"No weights at {weights_path}; run `train-synthetic` first.", file=sys.stderr)
        return 1

    adapter = GridCNNAdapter.load(
        str(weights_path),
        n_classes=N_SYNTHETIC_CLASSES,
        grid_h=GRID,
        grid_w=GRID,
        class_names=SYNTHETIC_CLASS_NAMES,
        img_w=IMG_SIZE,
        img_h=IMG_SIZE,
        thresh=args.conf_thresh,
    )

    rng = np.random.default_rng(args.seed)
    sequence = render_corridor_sequence(args.n_frames, rng, domain=args.domain, n_threats=args.n_threats)
    rgb_frames = [img for img, _boxes, _ids in sequence]
    frames = [
        np.transpose(img, (2, 0, 1)).mean(axis=0, keepdims=True).astype("float32") for img, _boxes, _ids in sequence
    ]
    frame_ids = [f"F{i:06d}" for i in range(len(frames))]

    calibrator = None
    if args.calibrate:
        labels = [
            encode_grid_label(boxes, IMG_SIZE, IMG_SIZE, GRID, GRID, N_SYNTHETIC_CLASSES)
            for _img, boxes, _ids in sequence
        ]
        calibrator = _self_calibrate(adapter, frames, labels, IMG_SIZE, IMG_SIZE, GRID, GRID)

    result = run_frame_sequence(
        frames=frames,
        detector=adapter,
        protected_zone=PROTECTED_ZONE,
        frame_ids=frame_ids,
        fps=args.fps,
        tracker=MultiTracker(min_hits=args.track_min_hits),
        event_engine=PipelineEventEngine(protected_zone=PROTECTED_ZONE),
        calibrator=calibrator,
        triage_thresholds=DEFAULT_TRIAGE_THRESHOLDS,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = out_dir / "alerts.csv"
    events_path = out_dir / "events.csv"
    result.alerts_frame().to_csv(alerts_path, index=False)
    result.events_frame().to_csv(events_path, index=False)

    print(f"{len(result.detections)} detections across {args.n_frames} frames")
    print(f"{len(result.events)} events raised")
    for e in result.events:
        print(f"  [{e.severity:6s}] {e.event_type:16s} t={e.timestamp_s:6.2f}s  {e.description}")
    print(f"Wrote {alerts_path} and {events_path}")

    if args.video:
        from .viz import build_annotated_gif

        video_path = out_dir / "annotated_demo.gif"
        build_annotated_gif(
            rgb_frames, frame_ids, result.detections, video_path, fps=args.fps, protected_zone=PROTECTED_ZONE
        )
        print(f"Wrote {video_path}")

    return 0


def _cmd_demo_real(args: argparse.Namespace) -> int:
    """Same pipeline as `demo` (detect -> track -> events -> calibrate -> alerts/GIF), run
    against real imagery instead of the synthetic renderer -- a structurally different model
    (see `train-real`), so this is a separate command rather than a flag on `demo`. `--source`
    accepts a named benchmark sequence (`uavdt:`/`visdrone-vid:`, with ground truth, so
    `--calibrate` and accuracy scoring apply) or an arbitrary local `video:`/`folder:` source
    (no ground truth, but works on any footage, not just the three benchmarks)."""
    from collections.abc import Callable

    from PIL import Image

    from .calibration import DEFAULT_TRIAGE_THRESHOLDS
    from .detectors.grid_cnn import GridCNNAdapter
    from .events import PipelineEventEngine
    from .ingest import (
        VEHICLE_GRID_H,
        VEHICLE_GRID_W,
        WORK_H,
        WORK_W,
        UAVDTIndex,
        UAVDTVehicleDataset,
        VisDroneVIDIndex,
        VisDroneVIDVehicleDataset,
        class_names_for_scheme,
        letterbox,
        load_image_folder_frames,
        load_video_frames,
    )
    from .pipeline import run_frame_sequence
    from .tracking import MultiTracker

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"No weights at {weights_path}; run `train-real` first.", file=sys.stderr)
        return 1

    if ":" not in args.source:
        print(
            f"--source must be '<uavdt|visdrone-vid|video|folder>:<sequence-or-path>', got {args.source!r}",
            file=sys.stderr,
        )
        return 1
    dataset_kind, seq = args.source.split(":", 1)

    ds: UAVDTVehicleDataset | VisDroneVIDVehicleDataset
    load_frame: Callable[[int], Image.Image]
    # Generic `video:`/`folder:` sources carry no ground truth, so `labels` stays `None` for
    # them -- `--calibrate` has nothing to self-calibrate against (checked below), and
    # mAP/F1 evaluation likewise isn't available for these sources.
    labels: list | None = None

    if dataset_kind == "uavdt":
        uavdt_index = UAVDTIndex()
        ds = UAVDTVehicleDataset(uavdt_index, [seq], stride=args.stride, cap=args.cap, class_scheme=args.class_scheme)
        load_frame = lambda frame_idx: uavdt_index.load_frame(seq, frame_idx)  # noqa: E731
        if len(ds) == 0:
            print(f"No frames found for --source {args.source!r}; check the sequence name.", file=sys.stderr)
            return 1
        frame_ids = [f"{seq}_{fidx:06d}" for _seq, fidx in ds.items]
        frames, labels, rgb_frames = [], [], []
        for i, (_seq, fidx) in enumerate(ds.items):
            gray, label = ds[i]
            frames.append(gray)
            labels.append(label)
            canvas, _scale, _pad_x, _pad_y = letterbox(load_frame(fidx), WORK_W, WORK_H)
            rgb_frames.append(np.asarray(canvas))
    elif dataset_kind == "visdrone-vid":
        visdrone_vid_index = VisDroneVIDIndex()
        ds = VisDroneVIDVehicleDataset(
            visdrone_vid_index,
            split=args.split,
            seq_list=[seq],
            stride=args.stride,
            cap=args.cap,
            class_scheme=args.class_scheme,
        )
        load_frame = lambda frame_idx: visdrone_vid_index.load_frame(args.split, seq, frame_idx)  # noqa: E731
        if len(ds) == 0:
            print(f"No frames found for --source {args.source!r}; check the sequence name.", file=sys.stderr)
            return 1
        frame_ids = [f"{seq}_{fidx:06d}" for _seq, fidx in ds.items]
        frames, labels, rgb_frames = [], [], []
        for i, (_seq, fidx) in enumerate(ds.items):
            gray, label = ds[i]
            frames.append(gray)
            labels.append(label)
            canvas, _scale, _pad_x, _pad_y = letterbox(load_frame(fidx), WORK_W, WORK_H)
            rgb_frames.append(np.asarray(canvas))
    elif dataset_kind in ("video", "folder"):
        loader = load_video_frames if dataset_kind == "video" else load_image_folder_frames
        try:
            frame_ids, frames, rgb_frames = loader(seq, stride=args.stride, cap=args.cap, work_w=WORK_W, work_h=WORK_H)
        except FileNotFoundError as source_error:
            print(str(source_error), file=sys.stderr)
            return 1
    else:
        print(
            f"Unknown --source kind {dataset_kind!r}; expected 'uavdt', 'visdrone-vid', 'video', or 'folder'.",
            file=sys.stderr,
        )
        return 1

    class_names = class_names_for_scheme(args.class_scheme)
    adapter = GridCNNAdapter.load(
        str(weights_path),
        n_classes=len(class_names),
        grid_h=VEHICLE_GRID_H,
        grid_w=VEHICLE_GRID_W,
        class_names=class_names,
        img_w=WORK_W,
        img_h=WORK_H,
        thresh=args.conf_thresh,
        stage_channels=(24, 48, 96, 96),
    )

    zone = tuple(args.zone) if args.zone else REAL_DEMO_PROTECTED_ZONE

    calibrator = None
    if args.calibrate:
        if labels is None:
            print(
                f"--calibrate has no effect for a {dataset_kind!r} source: there is no ground truth to "
                "self-calibrate against. Detections/tracking/events/alerts still run, uncalibrated.",
                file=sys.stderr,
            )
        else:
            calibrator = _self_calibrate(adapter, frames, labels, WORK_W, WORK_H, VEHICLE_GRID_W, VEHICLE_GRID_H)

    result = run_frame_sequence(
        frames=frames,
        detector=adapter,
        protected_zone=zone,
        frame_ids=frame_ids,
        fps=args.fps,
        tracker=MultiTracker(min_hits=args.track_min_hits),
        event_engine=PipelineEventEngine(protected_zone=zone),
        calibrator=calibrator,
        triage_thresholds=DEFAULT_TRIAGE_THRESHOLDS,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = out_dir / "alerts.csv"
    events_path = out_dir / "events.csv"
    result.alerts_frame().to_csv(alerts_path, index=False)
    result.events_frame().to_csv(events_path, index=False)

    print(f"{args.source}: {len(result.detections)} detections across {len(frames)} frames")
    print(f"{len(result.events)} events raised")
    for e in result.events:
        print(f"  [{e.severity:6s}] {e.event_type:16s} t={e.timestamp_s:6.2f}s  {e.description}")
    print(f"Wrote {alerts_path} and {events_path}")

    if args.video:
        from .viz import build_annotated_video

        video_path = out_dir / "annotated_demo.webm"
        # A WebM/VP8 video, not a GIF: real photographic frames have far more distinct
        # colors than GIF's 256-color palette can represent faithfully -- see viz.py's
        # module docstring for the measured difference this made.
        build_annotated_video(rgb_frames, frame_ids, result.detections, video_path, fps=args.fps, protected_zone=zone)
        print(f"Wrote {video_path}")

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    import pandas as pd

    from .reporting import build_dashboard, build_report_card

    out_dir = Path(args.out_dir)
    alerts_path = Path(args.alerts)
    events_path = Path(args.events)
    if not alerts_path.exists():
        print(f"No alerts file at {alerts_path}; run `demo` first (or pass --alerts).", file=sys.stderr)
        return 1

    alerts_df = pd.read_csv(alerts_path)
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(columns=["severity"])

    media_path = Path(args.video) if args.video else None
    if media_path is not None and not media_path.exists():
        print(f"No annotated GIF/video at {media_path}; dashboard will skip that panel.", file=sys.stderr)
        media_path = None

    report_path = out_dir / "report_card.png"
    dashboard_path = out_dir / "dashboard.html"
    build_report_card(alerts_df, events_df, out_path=report_path)
    build_dashboard(alerts_df, events_df, out_path=dashboard_path, media_path=media_path)

    print(f"{len(alerts_df)} alerts, {len(events_df)} events summarized")
    print(f"Wrote {report_path} and {dashboard_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infrastructure-overwatch")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train-synthetic", help="Train the synthetic-corridor grid detector.")
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--n-train", type=int, default=800)
    train_p.add_argument("--out", default="outputs/weights/corridor_detector.pt")
    train_p.set_defaults(func=_cmd_train_synthetic)

    demo_p = sub.add_parser("demo", help="Run the end-to-end pipeline on a synthetic sequence.")
    demo_p.add_argument("--weights", default="outputs/weights/corridor_detector.pt")
    demo_p.add_argument("--domain", choices=["day", "night"], default="night")
    demo_p.add_argument("--n-frames", type=int, default=120)
    demo_p.add_argument("--n-threats", type=int, default=2)
    demo_p.add_argument("--fps", type=float, default=10.0)
    demo_p.add_argument("--conf-thresh", type=float, default=0.4)
    demo_p.add_argument(
        "--calibrate", action="store_true", help="Self-calibrate confidence on this sequence's own ground truth."
    )
    demo_p.add_argument(
        "--track-min-hits",
        type=int,
        default=1,
        help="A track only raises zone/loiter events once it's been seen this many times "
        "(default 1, today's behavior: every track counts immediately). Raise this (e.g. 3) "
        "for a dense real-world scene, where a single-hit track is often detector noise, not "
        "a real object -- see MultiTracker.min_hits.",
    )
    demo_p.add_argument("--seed", type=int, default=7)
    demo_p.add_argument("--out-dir", default="outputs")
    demo_p.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an annotated GIF of the sequence (boxes, labels, confidence) to "
        "<out-dir>/annotated_demo.gif. See viz.py for why this is a GIF, not a compressed video.",
    )
    demo_p.set_defaults(func=_cmd_demo)

    train_real_p = sub.add_parser(
        "train-real",
        help="Train the real vehicle_of_interest(+dismount) grid detector on UAVDT (+ VisDrone).",
    )
    train_real_p.add_argument(
        "--class-scheme",
        choices=["vehicle_only", "vehicle_dismount"],
        default="vehicle_only",
        help="'vehicle_only' (car/truck/bus) or 'vehicle_dismount' (adds dismount, from "
        "VisDrone's pedestrian/people -- UAVDT alone can't train this).",
    )
    train_real_p.add_argument("--epochs", type=int, default=25)
    train_real_p.add_argument(
        "--visdrone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Combine VisDrone2019-DET + VisDrone2019-VID with UAVDT for training (requires "
        "VISDRONE_ROOT; see docs/DEVELOPMENT.md). --no-visdrone trains on UAVDT alone.",
    )
    train_real_p.add_argument("--out", default="outputs/weights/real_vehicle_detector.pt")
    train_real_p.set_defaults(func=_cmd_train_real)

    demo_real_p = sub.add_parser(
        "demo-real",
        help="Run the end-to-end pipeline on real imagery: a UAVDT/VisDrone2019-VID benchmark "
        "sequence, or an arbitrary local video file or image folder.",
    )
    demo_real_p.add_argument("--weights", default="outputs/weights/real_vehicle_detector.pt")
    demo_real_p.add_argument(
        "--source",
        default="uavdt:M0601",
        help="'<uavdt|visdrone-vid|video|folder>:<sequence-or-path>', e.g. 'uavdt:M0601', "
        "'visdrone-vid:uav0000086_00000_v', 'video:C:\\clips\\corridor.mp4', or "
        "'folder:C:\\clips\\frames'. UAVDT_NIGHT_VAL_SEQS/list_sequences('val') list uavdt/"
        "visdrone-vid candidates not used in `train-real`'s training split. A 'video:'/"
        "'folder:' source has no ground truth, so --calibrate is a no-op for it (a warning "
        "is printed) and it can't be scored for mAP/F1 -- detection/tracking/events/alerts "
        "still run normally.",
    )
    demo_real_p.add_argument(
        "--split",
        choices=["train", "val", "test-dev"],
        default="val",
        help="VisDrone2019-VID split to read --source from (ignored for uavdt/video/folder sources).",
    )
    demo_real_p.add_argument(
        "--class-scheme",
        choices=["vehicle_only", "vehicle_dismount"],
        default="vehicle_only",
        help="Must match whatever --weights was trained with.",
    )
    demo_real_p.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Sub-sample every Nth frame (Nth ground-truth frame for uavdt/visdrone-vid).",
    )
    demo_real_p.add_argument("--cap", type=int, default=150, help="Maximum number of frames to pull.")
    demo_real_p.add_argument("--fps", type=float, default=10.0)
    demo_real_p.add_argument("--conf-thresh", type=float, default=0.4)
    demo_real_p.add_argument(
        "--calibrate", action="store_true", help="Self-calibrate confidence on this sequence's own ground truth."
    )
    demo_real_p.add_argument(
        "--track-min-hits",
        type=int,
        default=1,
        help="A track only raises zone/loiter events once it's been seen this many times "
        "(default 1, today's behavior). Raise this (e.g. 3) for a dense real-world scene, "
        "where a single-hit track is often detector noise, not a real object -- see "
        "MultiTracker.min_hits.",
    )
    demo_real_p.add_argument(
        "--zone",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Protected-zone rectangle in the 640x352 working canvas. Defaults to an "
        "illustrative placeholder (see REAL_DEMO_PROTECTED_ZONE) -- UAVDT/VisDrone aren't "
        "shot around an actual perimeter, so there's no real zone to read off the data.",
    )
    demo_real_p.add_argument("--out-dir", default="outputs")
    demo_real_p.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an annotated WebM video of the sequence to <out-dir>/annotated_demo.webm "
        "(a video, not a GIF -- real footage has far more colors than a GIF palette can "
        "hold; see viz.py).",
    )
    demo_real_p.set_defaults(func=_cmd_demo_real)

    report_p = sub.add_parser("report", help="Build a report card + dashboard from alerts/events CSVs.")
    report_p.add_argument("--alerts", default="outputs/alerts.csv")
    report_p.add_argument("--events", default="outputs/events.csv")
    report_p.add_argument("--out-dir", default="outputs")
    report_p.add_argument(
        "--video",
        default="outputs/annotated_demo.gif",
        help="Annotated GIF/video to embed in the dashboard, if it exists (produced by "
        "`demo` or `demo-real`). Pass --video outputs/annotated_demo.webm after `demo-real`; "
        "pass an empty string to skip embedding one.",
    )
    report_p.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
