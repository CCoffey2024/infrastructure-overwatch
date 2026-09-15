import cv2
import numpy as np
import pytest
from PIL import Image

from infrastructure_overwatch.types import Detection
from infrastructure_overwatch.viz import (
    THREAT_CLASS_COLORS,
    TRIAGE_BAND_COLORS,
    build_annotated_gif,
    build_annotated_video,
    draw_detections,
)


def _detection(**overrides):
    base = dict(
        frame_id="F0",
        class_id=0,
        label="drone",
        confidence=0.8,
        x1=10.0,
        y1=10.0,
        x2=40.0,
        y2=40.0,
        model="test",
    )
    base.update(overrides)
    return Detection(**base)


def _frame():
    return np.random.default_rng(0).uniform(0, 1, (96, 96, 3)).astype("float32")


def test_draw_detections_upscales_and_returns_uint8_bgr():
    out = draw_detections(_frame(), [_detection()], scale=5)
    assert out.shape == (480, 480, 3)
    assert out.dtype == np.uint8


def test_draw_detections_with_no_detections_still_returns_a_frame():
    out = draw_detections(_frame(), [], scale=1)
    assert out.shape == (96, 96, 3)


def test_draw_detections_accepts_uint8_input():
    frame_u8 = (np.random.default_rng(0).uniform(0, 1, (96, 96, 3)) * 255).astype(np.uint8)
    out = draw_detections(frame_u8, [_detection()])
    assert out.dtype == np.uint8


def test_draw_detections_colors_by_triage_band_when_calibrated():
    frame = np.zeros((20, 20, 3), dtype=np.float32)
    d = _detection(x1=2, y1=2, x2=10, y2=10, triage_band="auto_confirm")
    out = draw_detections(frame, [d], scale=1)
    # the box's top edge should now contain the auto_confirm color (BGR)
    assert tuple(out[2, 5].tolist()) == TRIAGE_BAND_COLORS["auto_confirm"]


def test_draw_detections_colors_by_class_when_uncalibrated():
    frame = np.zeros((20, 20, 3), dtype=np.float32)
    d = _detection(x1=2, y1=2, x2=10, y2=10, label="vehicle_of_interest", triage_band=None)
    out = draw_detections(frame, [d], scale=1)
    assert tuple(out[2, 5].tolist()) == THREAT_CLASS_COLORS["vehicle_of_interest"]


def test_build_annotated_gif_writes_a_nonempty_file(tmp_path):
    frames = [_frame(), _frame(), _frame()]
    ids = ["F0", "F1", "F2"]
    dets = [_detection(frame_id="F0"), _detection(frame_id="F2", label="dismount")]
    out_path = tmp_path / "demo.gif"

    result_path = build_annotated_gif(frames, ids, dets, out_path, fps=5.0, protected_zone=(18.0, 30.0, 84.0, 78.0))

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_build_annotated_gif_preserves_a_detection_present_on_only_one_frame(tmp_path):
    """Regression test: an earlier video-based implementation used lossy inter-frame
    compression (VP8/WebM) that visibly washed out a box drawn on only a single frame out
    of many -- measured directly against this project's own sparse detections, the box's
    real (160, 160, 160) gray came back near-black after round-tripping through the video
    encoder. A GIF has no inter-frame prediction to lose that box to.

    Frames use per-frame-varying pixel noise, not flat zeros: real sequence frames always
    differ slightly frame to frame (rendering noise), and pixel-identical consecutive
    frames make PIL's GIF writer silently collapse them into one displayed-longer frame --
    a real behavior worth knowing about, but not what this test is checking.
    """
    rng = np.random.default_rng(0)
    frames = [rng.uniform(0, 0.2, (20, 20, 3)).astype(np.float32) for _ in range(6)]
    ids = [f"F{i}" for i in range(6)]
    # the detection only appears on frame F3, surrounded by frames with nothing drawn
    dets = [_detection(frame_id="F3", x1=2, y1=2, x2=10, y2=10, triage_band="auto_discard")]
    out_path = tmp_path / "sparse.gif"

    build_annotated_gif(frames, ids, dets, out_path, fps=5.0, scale=1)

    gif = Image.open(out_path)
    assert gif.n_frames == 6
    gif.seek(3)
    frame3 = np.array(gif.convert("RGB"))
    assert tuple(frame3[2, 5].tolist()) == TRIAGE_BAND_COLORS["auto_discard"][::-1]  # BGR -> RGB

    gif.seek(0)
    frame0 = np.array(gif.convert("RGB"))
    assert tuple(frame0[2, 5].tolist()) != tuple(frame3[2, 5].tolist())


def test_build_annotated_gif_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        build_annotated_gif([_frame()], ["F0", "F1"], [], tmp_path / "demo.gif")


def test_build_annotated_gif_rejects_empty_frames(tmp_path):
    with pytest.raises(ValueError):
        build_annotated_gif([], [], [], tmp_path / "demo.gif")


def _high_color_frame(seed: int) -> np.ndarray:
    """A frame with far more than 256 distinct colors, like a real photographic UAVDT/
    VisDrone frame -- uniform noise already trivially exceeds a GIF palette's budget from
    a single frame (96*96 pixels, each essentially independently colored)."""
    return np.random.default_rng(seed).uniform(0, 1, (96, 96, 3)).astype(np.float32)


def test_build_annotated_video_writes_a_nonempty_file(tmp_path):
    frames = [_high_color_frame(i) for i in range(3)]
    ids = ["F0", "F1", "F2"]
    dets = [_detection(frame_id="F0"), _detection(frame_id="F2", label="dismount")]
    out_path = tmp_path / "demo.webm"

    result_path = build_annotated_video(frames, ids, dets, out_path, fps=5.0, protected_zone=(18.0, 30.0, 84.0, 78.0))

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_build_annotated_video_preserves_color_on_photographic_content(tmp_path):
    """Regression test for the opposite failure mode from GIF's: a real UAVDT frame with
    boxes drawn on it measured out to 105,825 distinct colors in a single 640x352 frame,
    nowhere near GIF's 256-color palette -- a detection box's exact (160, 160, 160) gray
    came back a muddy, indistinguishable (111, 129, 112) after a GIF round-trip on content
    like this. VP8/WebM has no such fixed palette; verify it keeps the color recognizable
    (allow a few units of lossy-compression drift, not GIF-style quantization collapse).
    """
    frames = [_high_color_frame(i) for i in range(4)] * 3  # denser: repeats, not sparse
    ids = [f"F{i}" for i in range(12)]
    dets = [_detection(frame_id=fid, x1=20, y1=20, x2=60, y2=60, triage_band="auto_discard") for fid in ids]
    out_path = tmp_path / "photographic.webm"

    build_annotated_video(frames, ids, dets, out_path, fps=5.0, scale=1)

    cap = cv2.VideoCapture(str(out_path))
    ok, frame0 = cap.read()
    assert ok
    top_edge_bgr = frame0[20, 40].astype(int)
    expected = np.array(TRIAGE_BAND_COLORS["auto_discard"])
    assert np.abs(top_edge_bgr - expected).max() <= 15


def test_build_annotated_video_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        build_annotated_video([_frame()], ["F0", "F1"], [], tmp_path / "demo.webm")


def test_build_annotated_video_rejects_empty_frames(tmp_path):
    with pytest.raises(ValueError):
        build_annotated_video([], [], [], tmp_path / "demo.webm")
