"""Turns raw sensor input into what the rest of the pipeline consumes:
`FrameRecord`s for a live/recorded video feed, or an indexed, letterboxed
real-vehicle dataset for the `vehicle_of_interest` track.

The `vehicle_of_interest` class is the one threat class with a real-data
track: the UAVDT benchmark (Du et al., ECCV 2018) already has real drone-shot
car/truck/bus ground truth. `UAVDTIndex` reads it from a local copy of the
dataset (not included in this repo -- see docs/DEVELOPMENT.md for how to
obtain it); everything else in this module works without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .grid_codec import encode_grid_label
from .types import FrameRecord

# --- Video ingestion -------------------------------------------------------


class OpenCVVideoIngestAdapter:
    def __init__(self, sample_every_n: int = 1, jpeg_quality: int = 92):
        self.sample_every_n = max(1, int(sample_every_n))
        self.jpeg_quality = int(jpeg_quality)

    def extract(self, video_path, output_dir, sensor_id: str = "CAM01", modality: str = "EO") -> pd.DataFrame:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        records = []
        frame_number = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_number % self.sample_every_n == 0:
                timestamp_s = frame_number / fps
                frame_id = f"{sensor_id}_{frame_number:06d}"
                out = output_dir / f"{frame_id}.jpg"
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                rec = FrameRecord(
                    frame_id=frame_id,
                    sensor_id=sensor_id,
                    timestamp_s=timestamp_s,
                    modality=modality,
                    source_video=video_path.name,
                    frame_number=frame_number,
                    width=width,
                    height=height,
                    image_path=str(out),
                )
                records.append(rec.to_dict())
            frame_number += 1

        cap.release()
        return pd.DataFrame(records)


# --- Real vehicle data: UAVDT ----------------------------------------------

WORK_W, WORK_H = 640, 352  # multiple of 32 (backbone stride), close to native 1024x540 aspect
VEHICLE_GRID_W, VEHICLE_GRID_H = 40, 22
VEHICLE_N_CLASSES = 3  # UAVDT categories: 1=car, 2=truck, 3=bus
VEHICLE_CATEGORY_NAMES = {1: "car", 2: "truck", 3: "bus"}
ATTR_COLS = [
    "daylight",
    "night",
    "fog",
    "low_alt",
    "med_alt",
    "high_alt",
    "front_view",
    "side_view",
    "bird_view",
    "long_term",
]

DEFAULT_UAVDT_ROOT = os.environ.get("UAVDT_ROOT", r"D:\FMV\UAVDT\raw")


def letterbox(img: Image.Image, target_w: int = WORK_W, target_h: int = WORK_H, pad_value: int = 114):
    """Resizes preserving aspect ratio onto a fixed padded canvas, so real
    frames of varying native resolution all land on one working size the
    detector was trained for."""
    ow, oh = img.size
    scale = min(target_w / ow, target_h / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    resized = img.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), (pad_value, pad_value, pad_value))
    pad_x, pad_y = (target_w - nw) // 2, (target_h - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def letterbox_boxes(boxes_xywh, scale: float, pad_x: int, pad_y: int):
    return [(x * scale + pad_x, y * scale + pad_y, w * scale, h * scale) for (x, y, w, h) in boxes_xywh]


@dataclass
class UAVDTIndex:
    """Reads sequence metadata and ground truth from a local UAVDT copy.
    Raises `FileNotFoundError` with a clear message if `root` doesn't look
    like a UAVDT distribution -- callers should treat this as an optional,
    machine-local capability, not something the default demo depends on."""

    root: str = DEFAULT_UAVDT_ROOT
    frames_dir: str = field(init=False)
    gt_dir: str = field(init=False)
    attr_dir: str = field(init=False)

    def __post_init__(self):
        self.frames_dir = os.path.join(self.root, "UAV-benchmark-M")
        self.gt_dir = os.path.join(self.root, "UAV-benchmark-MOTD_v1.0", "GT")
        self.attr_dir = os.path.join(self.root, "M_attr")
        if not os.path.isdir(self.attr_dir):
            raise FileNotFoundError(
                f"No UAVDT distribution found at {self.root!r}. Set UAVDT_ROOT or pass "
                "`root=...` to UAVDTIndex. See docs/DEVELOPMENT.md for how to obtain the dataset."
            )

    def _find_attr_path(self, seq: str, split: str) -> str | None:
        for candidate in (f"{seq}_attr.txt", f"{seq} _attr.txt"):  # raw dist has one stray-space filename
            path = os.path.join(self.attr_dir, split, candidate)
            if os.path.exists(path):
                return path
        return None

    def parse_attrs(self, seq: str, split: str) -> dict:
        attr_path = self._find_attr_path(seq, split)
        if attr_path is None:
            raise FileNotFoundError(f"No attribute file for sequence {seq!r} (split={split!r}) under {self.attr_dir}")
        with open(attr_path) as f:
            vals = [int(v) for v in f.read().strip().split(",")]
        return dict(zip(ATTR_COLS, vals, strict=True))

    def parse_gt_whole(self, seq: str) -> pd.DataFrame:
        cols = ["frame", "target_id", "x", "y", "w", "h", "out_of_view", "occlusion", "category"]
        return pd.read_csv(os.path.join(self.gt_dir, f"{seq}_gt_whole.txt"), header=None, names=cols)

    def build_sequence_index(self) -> pd.DataFrame:
        rows = []
        for split in ["train", "test"]:
            split_dir = os.path.join(self.attr_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for fname in sorted(os.listdir(split_dir)):
                seq = fname.replace("_attr.txt", "").replace(" ", "")
                seq_dir = os.path.join(self.frames_dir, seq)
                row = {
                    "seq": seq,
                    "split": split,
                    "n_frames": len(os.listdir(seq_dir)) if os.path.isdir(seq_dir) else 0,
                }
                row.update(self.parse_attrs(seq, split))
                rows.append(row)
        return pd.DataFrame(rows)

    def load_frame(self, seq: str, frame_idx: int) -> Image.Image:
        return Image.open(os.path.join(self.frames_dir, seq, f"img{frame_idx:06d}.jpg")).convert("RGB")

    def sample_frame_indices(self, seq: str, stride: int = 8, cap: int = 80, offset: int = 0) -> list[int]:
        all_frames = sorted(self.parse_gt_whole(seq)["frame"].unique())
        return all_frames[offset::stride][:cap]


@dataclass
class UAVDTVehicleDataset(Dataset):
    """Real UAVDT frames -> letterboxed canvas + grid label, for training or
    evaluating the `vehicle_of_interest` detector (`detectors.grid_cnn`).
    `augment=True` applies the same blur/noise jitter used for the
    domain-randomization remedy in docs/METHODOLOGY_AND_LIMITATIONS.md."""

    index: UAVDTIndex
    seq_list: list[str]
    stride: int = 8
    cap: int = 80
    augment: bool = False
    seed: int = 0

    def __post_init__(self):
        self.items: list[tuple[str, int]] = []
        self.gt_cache: dict[str, pd.DataFrame] = {}
        for seq in self.seq_list:
            gt = self.index.parse_gt_whole(seq)
            self.gt_cache[seq] = gt
            self.items += [
                (seq, int(f)) for f in self.index.sample_frame_indices(seq, stride=self.stride, cap=self.cap)
            ]
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, img: Image.Image) -> Image.Image:
        blur = self.rng.uniform(0, 1.5)
        if blur > 0.1:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        arr = np.asarray(img).astype(np.float32) + self.rng.normal(0, 6, np.asarray(img).shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def __getitem__(self, idx: int):
        seq, fidx = self.items[idx]
        img = self.index.load_frame(seq, fidx)
        if self.augment:
            img = self._augment(img)
        frame_boxes = self.gt_cache[seq]
        frame_boxes = frame_boxes[frame_boxes.frame == fidx]
        canvas, scale, pad_x, pad_y = letterbox(img)
        lb_boxes = letterbox_boxes(frame_boxes[["x", "y", "w", "h"]].values, scale, pad_x, pad_y)

        boxes_with_class = []
        for (x, y, w, h), cat in zip(lb_boxes, frame_boxes["category"].values, strict=True):
            if w <= 1 or h <= 1:
                continue
            boxes_with_class.append(((x, y, x + w, y + h), int(cat) - 1))
        label = encode_grid_label(boxes_with_class, WORK_W, WORK_H, VEHICLE_GRID_W, VEHICLE_GRID_H, VEHICLE_N_CLASSES)

        gray = np.asarray(canvas.convert("L")).astype(np.float32) / 255.0
        return gray[np.newaxis, :, :], label
