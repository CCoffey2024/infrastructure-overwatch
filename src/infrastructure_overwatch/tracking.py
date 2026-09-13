"""Multi-target tracking: greedy IoU data association with a constant-
velocity motion model, plus the evaluation helpers (ID-switch count, ground-
truth coverage) used to judge whether it's actually working.

Not Hungarian-optimal, and not ByteTrack -- deliberately simple so the
association mechanics stay visible, while still handling many simultaneous
targets, which is what a real corridor scene actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .geometry import iou_xyxy
from .types import Detection


@dataclass
class _TrackState:
    box: tuple[float, float, float, float]
    label: str
    vx: float = 0.0
    vy: float = 0.0
    coast: int = 0


class MultiTracker:
    """Predict each existing track forward with its last-known velocity,
    then greedily assign this frame's detections to tracks by descending
    IoU. Unmatched tracks coast (keep their predicted box) for `max_coast`
    frames before being retired; unmatched detections spawn new tracks."""

    def __init__(self, iou_thresh: float = 0.25, max_coast: int = 4):
        self.tracks: dict[int, _TrackState] = {}
        self.next_id = 1
        self.iou_thresh = float(iou_thresh)
        self.max_coast = int(max_coast)

    def _predict(self, tr: _TrackState) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = tr.box
        return (x0 + tr.vx, y0 + tr.vy, x1 + tr.vx, y1 + tr.vy)

    def step(self, detections: list[Detection]) -> list[Detection]:
        """Returns `detections` with `track_id` filled in (as new `Detection`
        instances -- `Detection` is frozen)."""
        preds = {tid: self._predict(tr) for tid, tr in self.tracks.items()}
        pairs = sorted(
            ((iou_xyxy(preds[tid], det.xyxy), tid, di) for tid in preds for di, det in enumerate(detections)),
            key=lambda p: -p[0],
        )
        matched_tracks: set[int] = set()
        matched_dets: dict[int, int] = {}  # detection index -> track id
        for score, tid, di in pairs:
            if score < self.iou_thresh or tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets[di] = tid
            det = detections[di]
            old_box = self.tracks[tid].box
            self.tracks[tid] = _TrackState(
                box=det.xyxy, label=det.label, vx=det.x1 - old_box[0], vy=det.y1 - old_box[1], coast=0
            )

        for tid in list(self.tracks):
            if tid not in matched_tracks:
                tr = self.tracks[tid]
                tr.coast += 1
                if tr.coast > self.max_coast:
                    del self.tracks[tid]
                else:
                    tr.box = preds[tid]

        out = []
        for di, det in enumerate(detections):
            if di in matched_dets:
                out.append(replace(det, track_id=matched_dets[di]))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = _TrackState(box=det.xyxy, label=det.label)
                out.append(replace(det, track_id=tid))
        return out


GtBox = tuple[float, float, float, float, int]  # x0, y0, x1, y1, ground_truth_object_id


def match_to_gt(
    pred_box: tuple[float, float, float, float], gt_boxes: list[GtBox], iou_thresh: float = 0.3
) -> int | None:
    """The best-matching ground-truth object id for one predicted box, or
    `None` if nothing overlaps enough."""
    best_iou, best_gid = 0.0, None
    for x0, y0, x1, y1, gid in gt_boxes:
        v = iou_xyxy(pred_box, (x0, y0, x1, y1))
        if v > best_iou:
            best_iou, best_gid = v, gid
    return best_gid if best_iou >= iou_thresh else None


def coverage(frame_detections: list[list[Detection]], gt_per_frame: list[list[GtBox]]) -> float:
    """Fraction of real ground-truth object-frames that have some predicted
    box overlapping them (IoU >= 0.3). Compare the raw per-frame detector
    output against the tracked estimate to see what tracking (coasting
    through brief misses) actually buys."""
    covered = total = 0
    for dets, gt_boxes in zip(frame_detections, gt_per_frame, strict=True):
        pred_xyxy = [d.xyxy for d in dets]
        for x0, y0, x1, y1, _gid in gt_boxes:
            total += 1
            covered += int(any(iou_xyxy((x0, y0, x1, y1), pb) >= 0.3 for pb in pred_xyxy))
    return covered / total if total else float("nan")


def count_id_switches(
    tracked_frames: list[list[Detection]], gt_per_frame: list[list[GtBox]], iou_thresh: float = 0.3
) -> tuple[int, int]:
    """How often a track id's best-matching real object id changes from one
    frame to the next. Returns `(switches, matched_track_frames)`."""
    last_gid_for_track: dict[int, int] = {}
    switches = matched_track_frames = 0
    for dets, gt_boxes in zip(tracked_frames, gt_per_frame, strict=True):
        for d in dets:
            if d.track_id is None:
                continue
            gid = match_to_gt(d.xyxy, gt_boxes, iou_thresh=iou_thresh)
            if gid is None:
                continue
            matched_track_frames += 1
            if d.track_id in last_gid_for_track and last_gid_for_track[d.track_id] != gid:
                switches += 1
            last_gid_for_track[d.track_id] = gid
    return switches, matched_track_frames
