"""Framewise detection scoring, plus attribute-sliced error analysis.

A single aggregate F1 hides where a model actually fails. `slice_metrics`
groups the same precision/recall/f1 computation by any column you have
(occlusion level, sensor domain, threat class, time of day) so "the model is
weakest on small, occluded, high-altitude frames" is a measured slice, not a
guess.
"""

from __future__ import annotations

import pandas as pd

from .geometry import iou_xyxy


def match_detections(gt_rows, pred_rows, iou_threshold: float = 0.5, class_aware: bool = False) -> tuple[int, int, int]:
    gt = list(gt_rows)
    pred = sorted(list(pred_rows), key=lambda x: float(x.get("confidence", 1.0)), reverse=True)
    used_gt: set[int] = set()
    tp = fp = 0
    for p in pred:
        best_i, best_iou = None, 0.0
        for i, g in enumerate(gt):
            if i in used_gt:
                continue
            if class_aware and str(p.get("label")) != str(g.get("label")):
                continue
            score = iou_xyxy(
                (float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])),
                (float(g["x1"]), float(g["y1"]), float(g["x2"]), float(g["y2"])),
            )
            if score > best_iou:
                best_i, best_iou = i, score
        if best_i is not None and best_iou >= iou_threshold:
            tp += 1
            used_gt.add(best_i)
        else:
            fp += 1
    fn = len(gt) - len(used_gt)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_framewise(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float = 0.3, class_aware: bool = False
) -> dict:
    tps = fps = fns = 0
    frames = sorted(set(gt_df.frame_number.unique()).union(set(pred_df.frame_number.unique())))
    for f in frames:
        gt_rows = gt_df[gt_df.frame_number == f].to_dict("records")
        pred_rows = pred_df[pred_df.frame_number == f].to_dict("records")
        tp, fp, fn = match_detections(gt_rows, pred_rows, iou_threshold, class_aware)
        tps += tp
        fps += fp
        fns += fn
    p, r, f1 = precision_recall_f1(tps, fps, fns)
    return {"tp": tps, "fp": fps, "fn": fns, "precision": p, "recall": r, "f1": f1}


def slice_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    slice_col: str,
    iou_threshold: float = 0.3,
    class_aware: bool = False,
) -> pd.DataFrame:
    """`evaluate_framewise`, grouped by `slice_col` (present on `gt_df`) --
    one row per slice value, so a class- or attribute-dependent failure mode
    shows up as a table instead of getting averaged away."""
    rows = []
    for value, gt_slice in gt_df.groupby(slice_col):
        frame_ids = set(gt_slice.frame_number.unique())
        pred_slice = pred_df[pred_df.frame_number.isin(frame_ids)]
        result = evaluate_framewise(gt_slice, pred_slice, iou_threshold, class_aware)
        rows.append({slice_col: value, "n_frames": len(frame_ids), **result})
    return pd.DataFrame(rows).sort_values(slice_col).reset_index(drop=True)
