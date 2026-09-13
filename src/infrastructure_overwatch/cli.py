"""Command-line entry points: `train-synthetic` produces the weights the
`demo` command needs. Run `python -m infrastructure_overwatch --help`.
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


def _self_calibrate(adapter, sequence, frames) -> ConfidenceCalibrator | None:
    """A quick self-calibration pass: label this demo sequence's own
    detections against its own synthetic ground truth. A real deployment
    should calibrate on a held-out validation set instead (see
    `calibration.ConfidenceCalibrator`) -- this exists so `demo --calibrate`
    has something to show without requiring a separate validation run.
    """
    from .calibration import ConfidenceCalibrator
    from .geometry import iou_xyxy
    from .grid_codec import decode_grid_predictions, encode_grid_label
    from .synthetic import GRID, IMG_SIZE, N_SYNTHETIC_CLASSES

    confs, is_tp = [], []
    for (_img, boxes, _ids), frame in zip(sequence, frames, strict=True):
        preds = adapter.detect(frame, "cal")
        label = encode_grid_label(boxes, IMG_SIZE, IMG_SIZE, GRID, GRID, N_SYNTHETIC_CLASSES)
        gt_boxes = decode_grid_predictions(label, IMG_SIZE, IMG_SIZE, GRID, GRID, thresh=0.5, nms_iou=1.1)

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
    frames = [
        np.transpose(img, (2, 0, 1)).mean(axis=0, keepdims=True).astype("float32") for img, _boxes, _ids in sequence
    ]

    calibrator = None
    if args.calibrate:
        calibrator = _self_calibrate(adapter, sequence, frames)

    result = run_frame_sequence(
        frames=frames,
        detector=adapter,
        protected_zone=PROTECTED_ZONE,
        fps=args.fps,
        tracker=MultiTracker(),
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

    report_path = out_dir / "report_card.png"
    dashboard_path = out_dir / "dashboard.html"
    build_report_card(alerts_df, events_df, out_path=report_path)
    build_dashboard(alerts_df, events_df, out_path=dashboard_path)

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
    demo_p.add_argument("--seed", type=int, default=7)
    demo_p.add_argument("--out-dir", default="outputs")
    demo_p.set_defaults(func=_cmd_demo)

    report_p = sub.add_parser("report", help="Build a report card + dashboard from alerts/events CSVs.")
    report_p.add_argument("--alerts", default="outputs/alerts.csv")
    report_p.add_argument("--events", default="outputs/events.csv")
    report_p.add_argument("--out-dir", default="outputs")
    report_p.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
