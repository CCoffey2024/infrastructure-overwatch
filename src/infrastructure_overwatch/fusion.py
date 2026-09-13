"""Late fusion across sensor modalities (e.g. EO + IR).

A production system would calibrate each sensor into a common coordinate
system and propagate geometric uncertainty. This assumes already-aligned
imagery -- a secondary capability kept for completeness, not the primary
detection path.
"""

from __future__ import annotations

from .geometry import iou_xyxy
from .types import Detection


def late_fuse_detections(
    eo_dets: list[Detection], ir_dets: list[Detection], iou_threshold: float = 0.25
) -> list[Detection]:
    """Merges overlapping EO/IR detections into one higher-confidence
    detection; keeps unmatched detections from either sensor as-is."""
    fused = []
    used_ir: set[int] = set()
    for e in eo_dets:
        best_j, best_iou = None, 0.0
        for j, r in enumerate(ir_dets):
            if j in used_ir:
                continue
            score = iou_xyxy(e.xyxy, r.xyxy)
            if score > best_iou:
                best_j, best_iou = j, score
        if best_j is not None and best_iou >= iou_threshold:
            r = ir_dets[best_j]
            used_ir.add(best_j)
            w1, w2 = e.confidence, r.confidence
            s = max(1e-6, w1 + w2)
            fused.append(
                Detection(
                    frame_id=e.frame_id,
                    class_id=e.class_id if e.label != "motion" else r.class_id,
                    label=e.label if e.label != "motion" else r.label,
                    confidence=min(0.995, 1 - (1 - e.confidence) * (1 - r.confidence)),
                    x1=(e.x1 * w1 + r.x1 * w2) / s,
                    y1=(e.y1 * w1 + r.y1 * w2) / s,
                    x2=(e.x2 * w1 + r.x2 * w2) / s,
                    y2=(e.y2 * w1 + r.y2 * w2) / s,
                    model="late_fusion_eo_ir",
                )
            )
        else:
            fused.append(e)
    for j, r in enumerate(ir_dets):
        if j not in used_ir:
            fused.append(r)
    return fused
