"""A synthetic pipeline-corridor scene generator for the three classes that
have no real-data track (`drone`, `dismount`, `launch_flash` -- see
`types.THREAT_CLASSES`). `vehicle_of_interest` uses real UAVDT footage
instead; see `ingest.py`.

Two rendering modes, `render_day` / `render_night`, draw the *same*
underlying scene, so the day/night domain gap this project's detector has to
handle is visible directly rather than asserted. This is deliberately a cheap
procedural renderer, not a photorealistic one: the point is to isolate and
measure a specific degradation (low light, blur, sensor noise, JPEG
artifacting, oblique angle) rather than to simulate a real sensor end to end.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from .grid_codec import BoxClass, encode_grid_label

IMG_SIZE = 96  # chip size in pixels
GRID = 6  # 6x6 detection grid over the 96x96 chip

# Indices into types.THREAT_CLASSES for the three synthetically-rendered classes.
DRONE, DISMOUNT, LAUNCH_FLASH = 0, 1, 2
SYNTHETIC_CLASS_NAMES = ("drone", "dismount", "launch_flash")
N_SYNTHETIC_CLASSES = len(SYNTHETIC_CLASS_NAMES)

# An illustrative protected zone (the cleared right-of-way / pump-station pad)
# in chip coordinates, for demos that exercise events.PipelineEventEngine.
PROTECTED_ZONE = (18.0, 30.0, 84.0, 78.0)


def make_background(size: int, rng: np.random.Generator, roughness: int = 6, base: int = 130) -> Image.Image:
    """Cheap terrain-like texture: low-res noise, upsampled and blurred."""
    small = rng.normal(loc=base, scale=18, size=(roughness, roughness))
    bg = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(2))
    return bg.convert("L").convert("RGB")


def make_corridor_background(size: int, rng: np.random.Generator, base: int = 130) -> Image.Image:
    """Terrain texture + a fixed cleared right-of-way band + a pump-station pad,
    so every rendered chip is recognizably "the corridor", not an arbitrary
    patch of ground."""
    bg = make_background(size, rng, base=base)
    draw = ImageDraw.Draw(bg)
    band_w = size * 0.22
    row_shade = int(np.clip(base * 0.82, 0, 255))
    draw.polygon(
        [(0, size * 0.55), (size, size * 0.28), (size, size * 0.28 + band_w), (0, size * 0.55 + band_w)],
        fill=(row_shade, row_shade, row_shade),
    )
    pad_shade = int(np.clip(base * 0.55, 0, 255))
    px0, py0 = size * 0.68, size * 0.06
    draw.rectangle([px0, py0, px0 + size * 0.22, py0 + size * 0.16], fill=(pad_shade, pad_shade, pad_shade))
    return bg


def random_threats(rng: np.random.Generator, n_min: int = 1, n_max: int = 3, size: int = IMG_SIZE) -> list[dict]:
    """A small scene spec: 1-3 threat objects, each one of the three synthetic
    classes, at random positions."""
    n = rng.integers(n_min, n_max + 1)
    threats = []
    for _ in range(n):
        cls = int(rng.integers(0, N_SYNTHETIC_CLASSES))
        if cls == DRONE:  # small, roughly square, compact airborne signature
            w = h = rng.integers(6, 12)
        elif cls == DISMOUNT:  # narrow vertical silhouette
            w, h = rng.integers(4, 7), rng.integers(12, 20)
        else:  # launch_flash: broad bright burst
            w = h = rng.integers(10, 18)
        cx = rng.integers(w, size - w)
        cy = rng.integers(h, size - h)
        threats.append({"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h), "cls": cls})
    return threats


def _rot_shear_corners(cx, cy, w, h, angle_deg, shear):
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    shear_m = np.array([[1, shear], [0, 1]])
    theta = np.radians(angle_deg)
    rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ (rot_m @ shear_m).T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def render_day(threats: list[dict], rng: np.random.Generator, size: int = IMG_SIZE):
    """Clean daylight, close standoff -- the idealized case."""
    img = make_corridor_background(size, rng, base=int(rng.integers(120, 145)))
    draw = ImageDraw.Draw(img)
    boxes: list[BoxClass] = []
    for t in threats:
        cx, cy, w, h, cls = t["cx"], t["cy"], t["w"], t["h"], t["cls"]
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if cls == DRONE:  # small dark compact body + rotor cross
            shade = int(rng.integers(35, 65))
            draw.ellipse([x0, y0, x1, y1], fill=(shade, shade, shade))
            draw.line([x0 - 2, cy, x1 + 2, cy], fill=(shade, shade, shade), width=1)
            draw.line([cx, y0 - 2, cx, y1 + 2], fill=(shade, shade, shade), width=1)
        elif cls == DISMOUNT:  # body + head
            shade = int(rng.integers(50, 90))
            draw.rectangle([x0, y0 + h * 0.3, x1, y1], fill=(shade, shade, shade))
            draw.ellipse([cx - w * 0.45, y0, cx + w * 0.45, y0 + h * 0.35], fill=(shade, shade, shade))
        else:  # launch_flash: bright burst + faint rising smoke
            shade = int(rng.integers(225, 255))
            draw.ellipse([x0, y0, x1, y1], fill=(shade, shade, int(shade * 0.85)))
            smoke = int(rng.integers(150, 195))
            draw.ellipse([x0 - 2, y0 - h * 0.9, x1 + 2, y0 + h * 0.1], fill=(smoke, smoke, smoke))
        boxes.append(((x0, y0, x1, y1), cls))
    return np.asarray(img).astype(np.float32) / 255.0, boxes


def render_night(
    threats: list[dict],
    rng: np.random.Generator,
    size: int = IMG_SIZE,
    blur_len: float | None = None,
    jpeg_q: int | None = None,
    scale_jitter: float | None = None,
):
    """Night / long-standoff / oblique-look degradation. `launch_flash` is
    deliberately kept bright even here -- a flash/thermal signature doesn't
    dim the way a small cool airframe or a person does, and that asymmetry
    (one class gets *easier* at night, two get harder) is worth modeling
    directly rather than assuming every class degrades the same way."""
    base = int(rng.integers(30, 65))
    img = make_corridor_background(size, rng, base=base)
    draw = ImageDraw.Draw(img)
    boxes: list[BoxClass] = []
    for t in threats:
        cx, cy, w, h, cls = t["cx"], t["cy"], t["w"], t["h"], t["cls"]
        s = scale_jitter if scale_jitter is not None else rng.uniform(0.6, 1.4)
        w2, h2 = w * s, h * s
        if cls == LAUNCH_FLASH:
            shade = int(rng.integers(235, 255))
            x0, y0, x1, y1 = cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2
            draw.ellipse([x0, y0, x1, y1], fill=(shade, shade, int(shade * 0.8)))
            draw.ellipse([x0 - 3, y0 - h2 * 1.6, x1 + 3, y0 + h2 * 0.1], fill=(int(shade * 0.5),) * 3)
        else:
            angle = rng.uniform(-20, 20)
            shear = rng.uniform(-0.4, 0.4)
            pts = _rot_shear_corners(cx, cy, w2, h2, angle, shear)
            shade = int(rng.integers(35, 75)) if cls == DRONE else int(rng.integers(45, 90))
            draw.polygon([tuple(p) for p in pts], fill=(shade, shade, shade))
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        boxes.append(((max(x0, 0), max(y0, 0), min(x1, size), min(y1, size)), cls))

    blur_len = blur_len if blur_len is not None else float(rng.integers(1, 4))
    img = img.filter(ImageFilter.GaussianBlur(blur_len))
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, 9, np.asarray(img).shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    q = jpeg_q if jpeg_q is not None else int(rng.integers(18, 42))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0, boxes


def _to_gray_chw(frame: np.ndarray) -> np.ndarray:
    """`(H, W, 3)` float32 in [0, 1] -> `(1, H, W)` grayscale, matching what
    the grid detector was trained on."""
    return frame.mean(axis=2, keepdims=True).transpose(2, 0, 1).astype(np.float32)


@dataclass
class SyntheticCorridorDataset(Dataset):
    """Renders on the fly, so no data ever needs to be downloaded or stored
    to train on it. `fresh=True` reseeds every call (true online
    augmentation, used for domain-randomization training); `fresh=False` is
    deterministic per index, so repeated epochs/evaluations see a fixed
    set."""

    n_samples: int
    domain: str = "day"  # "day" or "night"
    seed: int = 0
    fresh: bool = False
    render_kwargs: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng() if self.fresh else np.random.default_rng(self.seed * 100_000 + idx)
        threats = random_threats(rng)
        if self.domain == "day":
            img, boxes = render_day(threats, rng)
        else:
            img, boxes = render_night(threats, rng, **self.render_kwargs)
        label = encode_grid_label(boxes, IMG_SIZE, IMG_SIZE, GRID, GRID, N_SYNTHETIC_CLASSES)
        return _to_gray_chw(img), label


def render_corridor_sequence(
    n_frames: int,
    rng: np.random.Generator,
    domain: str = "night",
    n_threats: int = 2,
    size: int = IMG_SIZE,
) -> list[tuple[np.ndarray, list[BoxClass], list[int]]]:
    """A short moving-threat sequence for demos that need continuity across
    frames (tracking, zone-entry/loiter events) rather than independent
    single-frame chips. Each of `n_threats` objects gets a persistent id, a
    fixed class, and a gentle random-walk path so it plausibly enters and
    lingers inside `PROTECTED_ZONE` at some point in the sequence.

    Returns a list of `(frame, boxes_with_class, object_ids)` per frame,
    where `boxes_with_class` aligns positionally with `object_ids`.
    """
    tracks = []
    for _ in range(n_threats):
        cls = int(rng.integers(0, N_SYNTHETIC_CLASSES))
        if cls == DISMOUNT:
            w, h = float(rng.integers(4, 7)), float(rng.integers(12, 20))
        else:
            w = h = float(rng.integers(8, 14))
        cx = float(rng.uniform(size * 0.1, size * 0.9))
        cy = float(rng.uniform(size * 0.1, size * 0.9))
        vx, vy = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2)
        tracks.append({"cls": cls, "w": w, "h": h, "cx": cx, "cy": cy, "vx": vx, "vy": vy})

    frames = []
    render_fn = render_day if domain == "day" else render_night
    for _t in range(n_frames):
        threats = []
        for tr in tracks:
            tr["cx"] = float(np.clip(tr["cx"] + tr["vx"] + rng.normal(0, 0.3), tr["w"], size - tr["w"]))
            tr["cy"] = float(np.clip(tr["cy"] + tr["vy"] + rng.normal(0, 0.3), tr["h"], size - tr["h"]))
            threats.append({"cx": tr["cx"], "cy": tr["cy"], "w": tr["w"], "h": tr["h"], "cls": tr["cls"]})
        img, boxes = render_fn(threats, rng)
        object_ids = list(range(len(tracks)))
        frames.append((img, boxes, object_ids))
    return frames
