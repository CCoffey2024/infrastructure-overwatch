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

import numpy as np


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


def _apply_anomaly_scoring(anomaly_ref: str | None, result, rgb_frames, frame_ids, fps: float) -> None:
    """If `--anomaly-ref` points at a reference gallery saved by
    `fit-anomaly-reference`, scores each sufficiently mature track's crop
    against it and folds any `VISUAL_ANOMALY` events into `result.events` in
    place. Shared by `demo` and `demo-real` so both the synthetic and real
    sources get the same optional pass; a no-op when `anomaly_ref` is unset.
    """
    if not anomaly_ref:
        return
    from .anomaly import EmbeddingAnomalyScorer, score_track_anomalies

    scorer = EmbeddingAnomalyScorer.load(anomaly_ref)
    anomaly_events = score_track_anomalies(rgb_frames, frame_ids, result.detections, scorer, fps=fps)
    result.events.extend(anomaly_events)


def _cmd_demo(args: argparse.Namespace) -> int:
    from .calibration import DEFAULT_TRIAGE_THRESHOLDS, self_calibrate
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
        calibrator = self_calibrate(adapter, frames, labels, IMG_SIZE, IMG_SIZE, GRID, GRID)

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
    _apply_anomaly_scoring(args.anomaly_ref, result, rgb_frames, frame_ids, args.fps)

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
    (no ground truth, but works on any footage, not just the three benchmarks). The actual
    detect/track/calibrate/anomaly-score work lives in `runs.run_real_pipeline`, shared with
    `service.py`'s job executor -- this function is just the argparse/stdout/CSV shell
    around it.
    """
    from . import runs

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"No weights at {weights_path}; run `train-real` first.", file=sys.stderr)
        return 1

    config = runs.RealRunConfig(
        weights_path=str(weights_path),
        source=args.source,
        split=args.split,
        class_scheme=args.class_scheme,
        stride=args.stride,
        cap=args.cap,
        fps=args.fps,
        conf_thresh=args.conf_thresh,
        calibrate=args.calibrate,
        track_min_hits=args.track_min_hits,
        anomaly_ref=args.anomaly_ref,
        zone=tuple(args.zone) if args.zone else None,
    )
    try:
        output = runs.run_real_pipeline(config)
    except (ValueError, FileNotFoundError) as source_error:
        print(str(source_error), file=sys.stderr)
        return 1

    if output.calibration_warning:
        print(output.calibration_warning, file=sys.stderr)

    result = output.result
    zone = config.zone or runs.DEFAULT_PROTECTED_ZONE
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = out_dir / "alerts.csv"
    events_path = out_dir / "events.csv"
    result.alerts_frame().to_csv(alerts_path, index=False)
    result.events_frame().to_csv(events_path, index=False)

    print(f"{args.source}: {len(result.detections)} detections across {len(output.frame_ids)} frames")
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
        build_annotated_video(
            output.rgb_frames, output.frame_ids, result.detections, video_path, fps=args.fps, protected_zone=zone
        )
        print(f"Wrote {video_path}")

    return 0


def _cmd_fit_anomaly_reference(args: argparse.Namespace) -> int:
    """Builds a reference gallery of "normal" imagery for `anomaly.EmbeddingAnomalyScorer`
    and saves it for `demo`/`demo-real --anomaly-ref` to load. Splits the gallery folder
    in half: the first half becomes the reference gallery itself, the second half is held
    out to calibrate the alert threshold -- scoring an image against a gallery that already
    contains it would find itself as its own nearest neighbor (distance ~0) and silently
    calibrate an unusably strict near-zero threshold, so the two must be disjoint.
    """
    import numpy as np
    from PIL import Image

    from .anomaly import DinoV2Embedder, EmbeddingAnomalyScorer, HOGEmbedder
    from .ingest import list_image_paths

    try:
        paths = list_image_paths(args.gallery_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if len(paths) < 2:
        print(
            f"Only {len(paths)} image(s) under {args.gallery_dir}; need at least 2 to build a reference "
            "gallery and hold at least one out for threshold calibration.",
            file=sys.stderr,
        )
        return 1

    images = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    split = max(1, len(images) // 2)
    gallery_images, calibration_images = images[:split], images[split:] or images[:split]

    embedder = DinoV2Embedder() if args.embedder == "dinov2" else HOGEmbedder()
    scorer = EmbeddingAnomalyScorer(embedder)
    scorer.fit(gallery_images)
    scorer.calibrate_threshold(calibration_images, percentile=args.percentile)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scorer.save(out_path, embedder_name=args.embedder)

    print(
        f"Fit a {args.embedder} reference gallery from {len(gallery_images)} images under "
        f"{args.gallery_dir} ({len(calibration_images)} held out for calibration)"
    )
    print(f"Calibrated threshold={scorer.threshold:.3f} at the {args.percentile:.1f}th percentile")
    print(f"Saved reference to {out_path}")
    return 0


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _cmd_serve(args: argparse.Namespace) -> int:
    """Starts the operator-console API (see `service.py`). Every job-submission route
    accepts a local filesystem path (`video:`/`folder:`) and reads it directly -- there is
    no upload boundary yet (Phase 3) -- so binding anywhere but loopback means any client
    that can reach the port can ask this process to read any local file path it names.
    `--host` therefore defaults to `127.0.0.1`; binding elsewhere requires the explicit
    `--allow-remote` flag, so that exposure is a choice, not an accident. See
    docs/OPERATOR_CONSOLE.md.
    """
    if args.host not in _LOOPBACK_HOSTS and not args.allow_remote:
        print(
            f"Refusing to bind {args.host!r}: it isn't loopback ({sorted(_LOOPBACK_HOSTS)}), and every job-"
            "submission route reads a local filesystem path directly -- binding it without --allow-remote "
            "would let any client that can reach this port ask the process to read arbitrary local files. "
            "Pass --allow-remote if you understand and accept that for your network.",
            file=sys.stderr,
        )
        return 1
    if args.host not in _LOOPBACK_HOSTS:
        print(
            f"WARNING: binding {args.host!r} (not loopback). Every job-submission route reads a local "
            "filesystem path directly, with no authentication -- anyone who can reach this port can ask "
            "this process to read arbitrary local files.",
            file=sys.stderr,
        )

    try:
        import uvicorn
    except ImportError:
        print("The `web` extra is required: `pip install -e '.[web]'`", file=sys.stderr)
        return 1

    from .service import OperatorService, create_app

    service = OperatorService(workspace_dir=args.workspace, default_weights_path=args.weights)
    app = create_app(service)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        service.shutdown()
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
    demo_p.add_argument(
        "--anomaly-ref",
        default=None,
        help="Path to a reference gallery saved by `fit-anomaly-reference`. When set, scores "
        "each mature track's crop against it and adds any VISUAL_ANOMALY events it raises.",
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
        "--anomaly-ref",
        default=None,
        help="Path to a reference gallery saved by `fit-anomaly-reference`. When set, scores "
        "each mature track's crop against it and adds any VISUAL_ANOMALY events it raises.",
    )
    demo_real_p.add_argument(
        "--zone",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Protected-zone rectangle in the 640x352 working canvas. Defaults to an "
        "illustrative placeholder (see runs.DEFAULT_PROTECTED_ZONE) -- UAVDT/VisDrone aren't "
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

    fit_anomaly_p = sub.add_parser(
        "fit-anomaly-reference",
        help="Build a reference gallery of 'normal' imagery for --anomaly-ref, from a folder of images.",
    )
    fit_anomaly_p.add_argument("--gallery-dir", required=True, help="Folder of known-normal reference images.")
    fit_anomaly_p.add_argument(
        "--embedder",
        choices=["hog", "dinov2"],
        default="hog",
        help="hog (default): offline, no download, what CI exercises. dinov2: richer "
        "self-supervised features, needs network access on first use.",
    )
    fit_anomaly_p.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="What fraction of the held-out calibration images should score below the alert "
        "threshold (99.0 accepts a 1%% false-alarm rate on that held-out set).",
    )
    fit_anomaly_p.add_argument("--out", default="outputs/weights/anomaly_reference.npz")
    fit_anomaly_p.set_defaults(func=_cmd_fit_anomaly_reference)

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

    serve_p = sub.add_parser("serve", help="Start the operator-console API (needs the `web` extra).")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8765)
    serve_p.add_argument("--workspace", default="outputs", help="Where job records and their evidence are stored.")
    serve_p.add_argument(
        "--weights", default="outputs/weights/real_vehicle_detector.pt", help="Default detector weights for a run."
    )
    serve_p.add_argument(
        "--allow-remote", action="store_true", help="Required to bind anywhere but loopback -- see _cmd_serve."
    )
    serve_p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
