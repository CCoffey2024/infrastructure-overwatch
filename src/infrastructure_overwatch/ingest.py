"""Turns raw sensor input into what the rest of the pipeline consumes: letterboxed frames
ready for a detector, either from an arbitrary local video/image-folder source
(`load_video_frames`/`load_image_folder_frames` -- no ground truth, works on anything) or
from an indexed real-vehicle(+dismount) benchmark dataset for the `vehicle_of_interest`
(and, under the `vehicle_dismount` class scheme, `dismount`) track.

The named-benchmark readers below are for training and accuracy evaluation, where labeled
ground truth is exactly the point. This ingestion layer is deliberately built to combine an
arbitrary number of independent real-data sources for the same track, not just one -- a
pipeline wired to a single sensor/dataset is a much weaker validation of "this generalizes"
than several independently-collected sources agreeing. Three sources feed it today, all
optional and machine-local (not included in this repo -- see docs/DEVELOPMENT.md for how to
obtain any of them):

- `UAVDTIndex` reads the UAVDT benchmark (Du et al., ECCV 2018): drone-video sequences with
  real car/truck/bus ground truth, but no person/pedestrian class.
- `VisDroneDETIndex` reads the VisDrone2019-DET benchmark (single images, one annotation
  file per image): the same car/truck/bus vehicle classes *and* pedestrian/people, which
  UAVDT has none of.
- `VisDroneVIDIndex` reads the VisDrone2019-VID benchmark (object detection in videos,
  structured as sequences of frames plus one whole-sequence ground-truth file, the same
  shape as UAVDT): the same category coverage as VisDrone-DET, but from continuous drone
  flight rather than unrelated stills.

Combining DET and VID rather than picking one is deliberate: they were collected
differently (unrelated photos vs. continuous flight), so together they're a broader, more
independent real-data signal than either alone -- and adding a *fourth* source later means
writing one more `Index`/`VehicleDataset` pair to this same shape, not touching the other
three. All three share one class-index convention via `class_names_for_scheme`, and all
produce the same letterboxed-canvas-plus-grid-label shape, so any combination of their
`*VehicleDataset` classes can be combined with `torch.utils.data.ConcatDataset` for
training. Everything else in this module works without any of them installed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .grid_codec import encode_grid_label

# --- Real vehicle data: UAVDT ----------------------------------------------

WORK_W, WORK_H = 640, 352  # multiple of 32 (backbone stride), close to native 1024x540 aspect
VEHICLE_GRID_W, VEHICLE_GRID_H = 40, 22
VEHICLE_N_CLASSES = 3  # UAVDT categories: 1=car, 2=truck, 3=bus
VEHICLE_CATEGORY_NAMES = {1: "car", 2: "truck", 3: "bus"}

# Two class schemes for the real vehicle(+dismount) track. "car"/"truck"/"bus" always occupy
# indices 0/1/2 in both, so UAVDT's `int(category) - 1` mapping is valid under either scheme
# -- UAVDT frames just never populate the "dismount" channel, which is correct (UAVDT has no
# person ground truth), not a bug.
VEHICLE_ONLY_CLASS_NAMES: tuple[str, ...] = ("car", "truck", "bus")
VEHICLE_DISMOUNT_CLASS_NAMES: tuple[str, ...] = ("car", "truck", "bus", "dismount")


def class_names_for_scheme(class_scheme: str) -> tuple[str, ...]:
    """`"vehicle_only"` (default, matches UAVDT's native 3 categories) or
    `"vehicle_dismount"` (adds the `dismount` class, populated only by VisDrone's
    pedestrian/people categories -- see `VISDRONE_CATEGORY_TO_CLASS_NAME`)."""
    if class_scheme == "vehicle_only":
        return VEHICLE_ONLY_CLASS_NAMES
    if class_scheme == "vehicle_dismount":
        return VEHICLE_DISMOUNT_CLASS_NAMES
    raise ValueError(f"Unknown class_scheme {class_scheme!r}; expected 'vehicle_only' or 'vehicle_dismount'")


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


# --- Generic ingestion: an arbitrary local video file or folder of still images ---------
#
# Unlike the named-benchmark readers below, these carry no ground truth -- there is no
# annotation file to read for "some mp4 I have" or "a folder of jpegs off a drone". They
# return the same (frame_ids, frames, rgb_frames) shape `cli.py`'s `demo-real` command
# already builds inline for UAVDT/VisDrone, so either source drops into the same
# detect/track/event pipeline; `--calibrate` and mAP/F1 evaluation just aren't available
# for them (nothing to self-calibrate or score against), which callers must check for
# rather than assume.

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_NATURAL_SORT_SPLIT = re.compile(r"(\d+)")


def _natural_sort_key(name: str) -> list:
    """Splits a filename into text/number chunks so `frame2.jpg` sorts before
    `frame10.jpg` -- a plain string sort gets this wrong for any source that
    isn't zero-padded (VisDrone/UAVDT always are; an arbitrary folder often
    isn't)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_SORT_SPLIT.split(name)]


def _letterboxed_frame(img: Image.Image, work_w: int, work_h: int) -> tuple[np.ndarray, np.ndarray]:
    """One image -> (rgb canvas array, normalized grayscale canvas array),
    the same letterbox-then-split-channel shape every named-dataset loader
    below produces per frame."""
    canvas, _scale, _pad_x, _pad_y = letterbox(img, work_w, work_h)
    rgb = np.asarray(canvas)
    gray = np.asarray(canvas.convert("L")).astype(np.float32) / 255.0
    return rgb, gray[np.newaxis, :, :]


def load_video_frames(
    video_path, stride: int = 2, cap: int = 150, work_w: int = WORK_W, work_h: int = WORK_H
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Reads an arbitrary local video file with OpenCV, letterboxes every
    `stride`-th frame (up to `cap` sampled frames) onto a fixed working
    canvas, and returns `(frame_ids, frames, rgb_frames)` -- `frames` ready
    for a grid-CNN detector, `rgb_frames` ready for annotated-video
    rendering. Raises `FileNotFoundError` if the file can't be opened or
    yields no frames."""
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    stem = video_path.stem
    frame_ids: list[str] = []
    frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    frame_number = 0
    try:
        while len(frame_ids) < cap:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_number % stride == 0:
                img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                rgb, gray = _letterboxed_frame(img, work_w, work_h)
                rgb_frames.append(rgb)
                frames.append(gray)
                frame_ids.append(f"{stem}_{frame_number:06d}")
            frame_number += 1
    finally:
        capture.release()

    if not frame_ids:
        raise FileNotFoundError(f"No frames could be read from {video_path}")
    return frame_ids, frames, rgb_frames


def list_image_paths(folder_path) -> list[Path]:
    """Every recognized image file directly under `folder_path`, ordered
    naturally by filename. Shared by `load_image_folder_frames` (which
    additionally letterboxes each one for the detector) and anything that
    just wants the raw files -- e.g. building an anomaly-reference gallery,
    which shouldn't be resized onto the detector's working canvas at all.
    Raises `FileNotFoundError` if the folder has no recognized image files."""
    folder_path = Path(folder_path)
    paths = sorted(
        (p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: _natural_sort_key(p.name),
    )
    if not paths:
        raise FileNotFoundError(f"No image files found under {folder_path}")
    return paths


def load_image_folder_frames(
    folder_path, stride: int = 1, cap: int = 150, work_w: int = WORK_W, work_h: int = WORK_H
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Reads an arbitrary local folder of still images (jpg/jpeg/png/bmp/tif),
    ordered naturally by filename, letterboxes every `stride`-th image (up to
    `cap` sampled frames), and returns the same `(frame_ids, frames,
    rgb_frames)` shape `load_video_frames` does. Raises `FileNotFoundError`
    if the folder has no recognized image files."""
    paths = list_image_paths(folder_path)[::stride][:cap]

    frame_ids: list[str] = []
    frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        rgb, gray = _letterboxed_frame(img, work_w, work_h)
        rgb_frames.append(rgb)
        frames.append(gray)
        frame_ids.append(p.stem)
    return frame_ids, frames, rgb_frames


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
    domain-randomization remedy in docs/METHODOLOGY_AND_LIMITATIONS.md.
    `class_scheme` defaults to `"vehicle_only"` (this dataset's native 3
    categories); pass `"vehicle_dismount"` only to match the class-index
    space of a `VisDroneDETVehicleDataset`/`VisDroneVIDVehicleDataset` it's
    being concatenated with -- UAVDT itself never has a dismount box to
    contribute."""

    index: UAVDTIndex
    seq_list: list[str]
    stride: int = 8
    cap: int = 80
    augment: bool = False
    seed: int = 0
    class_scheme: str = "vehicle_only"

    def __post_init__(self):
        self.class_names = class_names_for_scheme(self.class_scheme)
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
        label = encode_grid_label(
            boxes_with_class, WORK_W, WORK_H, VEHICLE_GRID_W, VEHICLE_GRID_H, len(self.class_names)
        )

        gray = np.asarray(canvas.convert("L")).astype(np.float32) / 255.0
        return gray[np.newaxis, :, :], label


# --- Real vehicle + dismount data: VisDrone (both the VisDrone2019-DET single-image task
# and the VisDrone2019-VID sequence-of-frames task -- see the module docstring for why this
# ingestion layer is built to combine an arbitrary number of real-data sources, not just one)

# VisDrone category id -> this project's class name. Shared by both the DET and VID tasks
# (they use the same 0-11 category taxonomy, just in different annotation-file layouts).
# Categories not listed here (0=ignored-region, 3=bicycle, 5=van, 7=tricycle,
# 8=awning-tricycle, 10=motor, 11=others) are outside this project's threat taxonomy and
# dropped, the same deliberate scoping as docs/ARCHITECTURE.md#why-four-threat-classes --
# not an oversight.
VISDRONE_CATEGORY_TO_CLASS_NAME: dict[int, str] = {
    1: "dismount",  # pedestrian
    2: "dismount",  # people
    4: "car",
    6: "truck",
    9: "bus",
}

DEFAULT_VISDRONE_ROOT = os.environ.get("VISDRONE_ROOT", r"D:\FMV\VisDrone")


def visdrone_category_to_class_index(category: int, class_names: tuple[str, ...]) -> int | None:
    """Maps a raw VisDrone category id to an index into `class_names`, or `None` if the
    category is outside this project's taxonomy, or excluded by the active `class_scheme`
    (e.g. a pedestrian/people detection when `class_names` is `VEHICLE_ONLY_CLASS_NAMES`,
    which has no `"dismount"` entry to map onto)."""
    name = VISDRONE_CATEGORY_TO_CLASS_NAME.get(int(category))
    if name is None or name not in class_names:
        return None
    return class_names.index(name)


@dataclass
class VisDroneDETIndex:
    """Reads image paths and ground-truth annotations from a local VisDrone2019-DET
    (single-image object detection) distribution: `root` containing
    `VisDrone2019-DET-{train,val,test-dev}/`, each with `images/` and `annotations/`
    subfolders (the layout the official archives unpack to). Raises `FileNotFoundError`
    with a clear message if `root` doesn't look like a VisDrone2019-DET distribution --
    like `UAVDTIndex`, callers should treat this as an optional, machine-local capability.
    See docs/DEVELOPMENT.md for how to obtain it, and `VisDroneVIDIndex` for the sequence
    (video) counterpart."""

    root: str = DEFAULT_VISDRONE_ROOT

    def __post_init__(self):
        if not any(os.path.isdir(self._split_dir(split)) for split in ("train", "val", "test-dev")):
            raise FileNotFoundError(
                f"No VisDrone2019-DET distribution found at {self.root!r}. Set VISDRONE_ROOT or pass "
                "`root=...` to VisDroneDETIndex. See docs/DEVELOPMENT.md for how to obtain the dataset."
            )

    def _split_dir(self, split: str) -> str:
        return os.path.join(self.root, f"VisDrone2019-DET-{split}")

    def list_image_stems(self, split: str) -> list[str]:
        images_dir = os.path.join(self._split_dir(split), "images")
        return sorted(Path(f).stem for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    def load_image(self, split: str, stem: str) -> Image.Image:
        images_dir = os.path.join(self._split_dir(split), "images")
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(images_dir, stem + ext)
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"No image file found for stem {stem!r} under {images_dir}")

    def parse_annotations(self, split: str, stem: str) -> pd.DataFrame:
        cols = ["x", "y", "w", "h", "score", "category", "truncation", "occlusion"]
        ann_path = os.path.join(self._split_dir(split), "annotations", f"{stem}.txt")
        if os.path.getsize(ann_path) == 0:
            return pd.DataFrame(columns=cols)
        return pd.read_csv(ann_path, header=None, names=cols)


@dataclass
class VisDroneDETVehicleDataset(Dataset):
    """Real VisDrone2019-DET frames -> letterboxed canvas + grid label, on the same
    working canvas, grid, and class-index convention as `UAVDTVehicleDataset` and
    `VisDroneVIDVehicleDataset` (see `class_names_for_scheme`), so any combination of the
    three can be combined with `torch.utils.data.ConcatDataset` to augment UAVDT's
    comparatively small real vehicle training set, and -- under
    `class_scheme="vehicle_dismount"` -- give the `dismount` class a real-data track UAVDT
    alone can't provide (UAVDT has no person/pedestrian ground truth)."""

    index: VisDroneDETIndex
    split: str = "train"
    class_scheme: str = "vehicle_only"
    stride: int = 1
    cap: int | None = None

    def __post_init__(self):
        self.class_names = class_names_for_scheme(self.class_scheme)
        stems = self.index.list_image_stems(self.split)[:: self.stride]
        self.stems = stems[: self.cap] if self.cap is not None else stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = self.index.load_image(self.split, stem)
        ann = self.index.parse_annotations(self.split, stem)
        canvas, scale, pad_x, pad_y = letterbox(img)
        lb_boxes = letterbox_boxes(ann[["x", "y", "w", "h"]].values, scale, pad_x, pad_y)

        boxes_with_class = []
        for (x, y, w, h), category, score in zip(lb_boxes, ann["category"].values, ann["score"].values, strict=True):
            if score == 0 or w <= 1 or h <= 1:
                continue
            class_idx = visdrone_category_to_class_index(category, self.class_names)
            if class_idx is None:
                continue
            boxes_with_class.append(((x, y, x + w, y + h), class_idx))

        label = encode_grid_label(
            boxes_with_class, WORK_W, WORK_H, VEHICLE_GRID_W, VEHICLE_GRID_H, len(self.class_names)
        )
        gray = np.asarray(canvas.convert("L")).astype(np.float32) / 255.0
        return gray[np.newaxis, :, :], label


@dataclass
class VisDroneVIDIndex:
    """Reads sequence frames and ground truth from a local VisDrone2019-VID (object
    detection in videos) distribution: `root` containing
    `VisDrone2019-VID-{train,val}/VisDrone2019-VID-{train,val}/{sequences,annotations}/`
    (the doubled directory name is how the official archives unpack). Structurally this is
    UAVDT's own shape -- one frames-folder and one whole-sequence ground-truth file per
    sequence -- just with VisDrone's 10-column annotation row
    (`frame,target_id,x,y,w,h,score,category,truncation,occlusion`) instead of UAVDT's
    9-column one. Raises `FileNotFoundError` with a clear message if `root` doesn't look
    like a VisDrone2019-VID distribution. See docs/DEVELOPMENT.md for how to obtain it, and
    `VisDroneDETIndex` for the single-image counterpart."""

    root: str = DEFAULT_VISDRONE_ROOT

    def __post_init__(self):
        if not any(os.path.isdir(self._split_root(split)) for split in ("train", "val", "test-dev")):
            raise FileNotFoundError(
                f"No VisDrone2019-VID distribution found at {self.root!r}. Set VISDRONE_ROOT or pass "
                "`root=...` to VisDroneVIDIndex. See docs/DEVELOPMENT.md for how to obtain the dataset."
            )

    def _split_root(self, split: str) -> str:
        # The official archive for split `s` unpacks to VisDrone2019-VID-{s}/VisDrone2019-VID-{s}/
        # (the zip's top-level folder shares its name with the archive itself); tolerate a
        # non-doubled layout too, in case a user's unzip flattened it.
        outer = os.path.join(self.root, f"VisDrone2019-VID-{split}")
        inner = os.path.join(outer, f"VisDrone2019-VID-{split}")
        return inner if os.path.isdir(os.path.join(inner, "sequences")) else outer

    def list_sequences(self, split: str) -> list[str]:
        ann_dir = os.path.join(self._split_root(split), "annotations")
        return sorted(Path(f).stem for f in os.listdir(ann_dir) if f.endswith(".txt"))

    def parse_gt(self, split: str, seq: str) -> pd.DataFrame:
        cols = ["frame", "target_id", "x", "y", "w", "h", "score", "category", "truncation", "occlusion"]
        return pd.read_csv(os.path.join(self._split_root(split), "annotations", f"{seq}.txt"), header=None, names=cols)

    def load_frame(self, split: str, seq: str, frame_idx: int) -> Image.Image:
        path = os.path.join(self._split_root(split), "sequences", seq, f"{frame_idx:07d}.jpg")
        return Image.open(path).convert("RGB")

    def sample_frame_indices(self, split: str, seq: str, stride: int = 8, cap: int = 80, offset: int = 0) -> list[int]:
        all_frames = sorted(self.parse_gt(split, seq)["frame"].unique())
        return all_frames[offset::stride][:cap]


@dataclass
class VisDroneVIDVehicleDataset(Dataset):
    """Real VisDrone2019-VID frames -> letterboxed canvas + grid label. Mirrors
    `UAVDTVehicleDataset`'s shape exactly (a chosen list of sequences, sub-sampled by
    `stride`/`cap`), on the same working canvas, grid, and class-index convention as
    `UAVDTVehicleDataset` and `VisDroneDETVehicleDataset`, so any combination of the three
    real sources can be combined with `torch.utils.data.ConcatDataset` for training."""

    index: VisDroneVIDIndex
    split: str
    seq_list: list[str]
    class_scheme: str = "vehicle_only"
    stride: int = 8
    cap: int = 80

    def __post_init__(self):
        self.class_names = class_names_for_scheme(self.class_scheme)
        self.items: list[tuple[str, int]] = []
        self.gt_cache: dict[str, pd.DataFrame] = {}
        for seq in self.seq_list:
            gt = self.index.parse_gt(self.split, seq)
            self.gt_cache[seq] = gt
            self.items += [
                (seq, int(f))
                for f in self.index.sample_frame_indices(self.split, seq, stride=self.stride, cap=self.cap)
            ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        seq, fidx = self.items[idx]
        img = self.index.load_frame(self.split, seq, fidx)
        frame_boxes = self.gt_cache[seq]
        frame_boxes = frame_boxes[frame_boxes.frame == fidx]
        canvas, scale, pad_x, pad_y = letterbox(img)
        lb_boxes = letterbox_boxes(frame_boxes[["x", "y", "w", "h"]].values, scale, pad_x, pad_y)

        boxes_with_class = []
        for (x, y, w, h), category, score in zip(
            lb_boxes, frame_boxes["category"].values, frame_boxes["score"].values, strict=True
        ):
            if score == 0 or w <= 1 or h <= 1:
                continue
            class_idx = visdrone_category_to_class_index(category, self.class_names)
            if class_idx is None:
                continue
            boxes_with_class.append(((x, y, x + w, y + h), class_idx))

        label = encode_grid_label(
            boxes_with_class, WORK_W, WORK_H, VEHICLE_GRID_W, VEHICLE_GRID_H, len(self.class_names)
        )
        gray = np.asarray(canvas.convert("L")).astype(np.float32) / 255.0
        return gray[np.newaxis, :, :], label
