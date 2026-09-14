"""Draws detections onto frames and renders an annotated GIF of a pipeline run --
what an operator actually looks at, as opposed to `reporting.py`'s summary
charts/tables. Uses OpenCV and Pillow only (both already hard dependencies
across this project), so this module needs no optional extra.

GIF, not a compressed video: an actual video codec turned out to be a bad fit for a
demo/audit tool. The PyPI `opencv-python` wheel doesn't bundle an H.264 encoder (patent
licensing), so `cv2.VideoWriter` with `mp4v`/`avc1` silently produces a file most browsers
can't decode -- `writer.isOpened()` even reports success; the failure only shows up later
as a browser decode error. Falling back to VP8/WebM (which every major browser does play)
traded that problem for a worse one: measured directly against this project's own sparse
detections, VP8's lossy inter-frame compression visibly washed out a detection box that
was only drawn on a single frame, down from its real (160, 160, 160) gray to near-black --
exactly the kind of thing an operator needs to see correctly. A GIF has no inter-frame
prediction to lose a one-frame box to, and every major browser renders it natively via a
plain `<img>` tag, no codec install required on any machine this runs on. The tradeoffs:
no scrub bar/play-pause controls, and a larger file for a long, high-resolution sequence.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .types import Detection

# BGR (OpenCV's native channel order), not RGB.
THREAT_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "drone": (255, 191, 0),
    "dismount": (0, 215, 255),
    "launch_flash": (0, 69, 255),
    "vehicle_of_interest": (76, 175, 80),
}
TRIAGE_BAND_COLORS: dict[str, tuple[int, int, int]] = {
    "auto_confirm": (76, 175, 80),
    "analyst_review": (0, 152, 255),
    "auto_discard": (160, 160, 160),
}
DEFAULT_COLOR = (255, 255, 255)


def _color_for(d: Detection) -> tuple[int, int, int]:
    """Colored by triage band when the run was calibrated (matching the
    band colors `docs/USER_MANUAL.md` and `reporting.py` already use
    elsewhere), by threat class otherwise -- either way conveys something,
    unlike one flat color for every box."""
    if d.triage_band is not None:
        return TRIAGE_BAND_COLORS.get(d.triage_band, DEFAULT_COLOR)
    return THREAT_CLASS_COLORS.get(d.label, DEFAULT_COLOR)


def draw_detections(
    frame_rgb: np.ndarray,
    detections: list[Detection],
    protected_zone: tuple[float, float, float, float] | None = None,
    scale: int = 1,
) -> np.ndarray:
    """`frame_rgb`: `(H, W, 3)`, either float in `[0, 1]` (this project's own
    renderer output) or already `uint8`. Returns a `(H*scale, W*scale, 3)`
    `uint8` BGR image (OpenCV's native order, ready for `cv2.VideoWriter` or
    `cv2.imwrite`) with the protected zone, each detection's box, and its
    label + confidence drawn on it. `scale` upscales before drawing so boxes
    and text stay legible on this project's small (96x96) synthetic chips."""
    img = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8) if frame_rgb.dtype != np.uint8 else frame_rgb.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if scale != 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    if protected_zone is not None:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in protected_zone)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    for d in detections:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in d.xyxy)
        color = _color_for(d)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)

        conf = d.calibrated_confidence if d.calibrated_confidence is not None else d.confidence
        text = f"{d.label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty0 = max(0, y0 - th - baseline - 2)
        cv2.rectangle(img, (x0, ty0), (x0 + tw + 4, ty0 + th + baseline + 2), color, -1)
        cv2.putText(img, text, (x0 + 2, ty0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def build_annotated_gif(
    frames_rgb: list[np.ndarray],
    frame_ids: list[str],
    detections: list[Detection],
    out_path: str | Path,
    fps: float = 10.0,
    protected_zone: tuple[float, float, float, float] | None = None,
    scale: int = 5,
) -> Path:
    """Writes an animated GIF with every frame's detections drawn on it -- groups
    `detections` by `frame_id` internally, so callers just pass the flat list
    `PipelineResult.detections` already produces, alongside the same
    `frames_rgb`/`frame_ids` given to `pipeline.run_frame_sequence`. See the module
    docstring for why this is a GIF and not a compressed video."""
    if not frames_rgb:
        raise ValueError("build_annotated_gif requires at least one frame")
    if len(frames_rgb) != len(frame_ids):
        raise ValueError(f"frames_rgb ({len(frames_rgb)}) and frame_ids ({len(frame_ids)}) must be the same length")

    by_frame: dict[str, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_id, []).append(d)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rendered = [
        draw_detections(f, by_frame.get(fid, []), protected_zone, scale)
        for f, fid in zip(frames_rgb, frame_ids, strict=True)
    ]
    pil_frames = [Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)) for r in rendered]
    pil_frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"{out_path} was written but is empty -- the GIF encoder failed silently.")
    return out_path
