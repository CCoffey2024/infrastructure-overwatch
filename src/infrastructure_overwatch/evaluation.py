"""Framewise detection scoring, plus attribute-sliced error analysis.

A single aggregate F1 hides where a model actually fails. `slice_metrics`
groups the same precision/recall/f1 computation by any column you have
(occlusion level, sensor domain, threat class, time of day) so "the model is
weakest on small, occluded, high-altitude frames" is a measured slice, not a
guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import iou_xyxy


def _match_predictions(gt_rows, pred_rows, iou_threshold: float = 0.5, class_aware: bool = False) -> list[bool]:
    """Greedy IoU matching, walked in descending-confidence order (ties keep
    their original relative order) -- the same association `match_detections`
    aggregates into tp/fp/fn counts, but returned as one True/False per
    *original* `pred_rows` entry (True = matched a not-yet-claimed
    ground-truth box at >= `iou_threshold`) so a caller can pair each
    prediction's own confidence with whether it was actually correct, e.g.
    to build a PR curve or a calibration reliability diagram."""
    gt = list(gt_rows)
    pred = list(pred_rows)
    order = sorted(range(len(pred)), key=lambda i: -float(pred[i].get("confidence", 1.0)))
    is_tp = [False] * len(pred)
    used_gt: set[int] = set()
    for i in order:
        p = pred[i]
        best_j, best_iou = None, 0.0
        for j, g in enumerate(gt):
            if j in used_gt:
                continue
            if class_aware and str(p.get("label")) != str(g.get("label")):
                continue
            score = iou_xyxy(
                (float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])),
                (float(g["x1"]), float(g["y1"]), float(g["x2"]), float(g["y2"])),
            )
            if score > best_iou:
                best_j, best_iou = j, score
        if best_j is not None and best_iou >= iou_threshold:
            is_tp[i] = True
            used_gt.add(best_j)
    return is_tp


def match_detections(gt_rows, pred_rows, iou_threshold: float = 0.5, class_aware: bool = False) -> tuple[int, int, int]:
    is_tp = _match_predictions(gt_rows, pred_rows, iou_threshold, class_aware)
    tp = sum(is_tp)
    fp = len(is_tp) - tp
    fn = len(list(gt_rows)) - tp  # each match claims exactly one distinct gt box
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


# --- mAP / PR curves / calibration reliability ------------------------------
#
# `evaluate_framewise`/`slice_metrics` above answer "how good is the detector
# at one fixed confidence threshold". These answer the question a proper
# accuracy dashboard actually needs: how does precision/recall trade off
# across every threshold (`precision_recall_curve`), what's the standard
# single-number accuracy score at that trade-off (`average_precision`/
# `mean_average_precision`, COCO's own definitions), and separately -- once a
# detector's confidence has been through `calibration.ConfidenceCalibrator`
# -- does its *calibrated* confidence actually track how often it's right
# (`calibration_reliability`)? All three build on `_match_predictions` above,
# so "correct" means exactly what `match_detections`/`evaluate_framewise`
# already mean: a class-aware-if-requested IoU match against `gt_df`.


def precision_recall_curve(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float = 0.5, class_aware: bool = False
) -> pd.DataFrame:
    """Every prediction across every frame, matched against its own frame's
    ground truth, then sorted globally by descending confidence so
    cumulative precision/recall can be read at any operating point -- the
    data an interactive PR-curve chart needs, and what `average_precision`
    integrates over. Empty (zero rows) if `pred_df` has no rows."""
    frames = sorted(set(gt_df.frame_number.unique()) | set(pred_df.frame_number.unique()))
    total_gt = len(gt_df)
    confidences: list[float] = []
    is_tp: list[bool] = []
    for f in frames:
        gt_rows = gt_df[gt_df.frame_number == f].to_dict("records")
        pred_rows = pred_df[pred_df.frame_number == f].to_dict("records")
        for p, tp in zip(pred_rows, _match_predictions(gt_rows, pred_rows, iou_threshold, class_aware), strict=True):
            confidences.append(float(p.get("confidence", 1.0)))
            is_tp.append(tp)

    if not confidences:
        return pd.DataFrame(columns=["confidence", "precision", "recall"])

    order = np.argsort(-np.asarray(confidences), kind="stable")
    conf_sorted = np.asarray(confidences)[order]
    tp_sorted = np.asarray(is_tp, dtype=float)[order]
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1.0 - tp_sorted)
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.zeros_like(cum_tp), where=denom > 0)
    recall = cum_tp / total_gt if total_gt > 0 else np.zeros_like(cum_tp)
    return pd.DataFrame({"confidence": conf_sorted, "precision": precision, "recall": recall})


def average_precision(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float = 0.5, class_aware: bool = False
) -> float:
    """COCO-style 101-point interpolated average precision at one IoU
    threshold: the precision/recall curve's precision is first made
    monotonically non-increasing as recall falls (the standard "precision
    envelope" -- a detector's precision at recall r is credited as the best
    precision achieved at any recall >= r), then averaged over 101 evenly
    spaced recall points (0.00, 0.01, ..., 1.00). 0.0 if there's no ground
    truth, or no predictions ever reach a given recall point."""
    if len(gt_df) == 0:
        return 0.0
    curve = precision_recall_curve(gt_df, pred_df, iou_threshold, class_aware)
    if curve.empty:
        return 0.0
    recall = curve["recall"].to_numpy()  # non-decreasing: rows are in descending-confidence order
    precision_envelope = np.maximum.accumulate(curve["precision"].to_numpy()[::-1])[::-1]
    recall_points = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, recall_points, side="left")
    safe_idx = np.clip(idx, 0, len(precision_envelope) - 1)
    precision_at_points = np.where(idx < len(precision_envelope), precision_envelope[safe_idx], 0.0)
    return float(precision_at_points.mean())


DEFAULT_COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50:0.05:0.95


def mean_average_precision(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_thresholds: tuple[float, ...] = DEFAULT_COCO_IOU_THRESHOLDS,
    class_aware: bool = True,
) -> dict:
    """COCO-style mAP: `average_precision` at each of `iou_thresholds`
    (default 0.50:0.05:0.95, COCO's own sweep), per class when
    `class_aware` and `gt_df` has a `label` column (else pooled across all
    classes), then averaged. Returns `{"map_50": AP@0.5 averaged over
    classes, "map_50_95": the full sweep averaged over classes and
    thresholds, "per_class": {label: {iou_threshold: ap, ...}, ...}}` --
    `map_50`/`map_50_95` are NaN if there's no ground truth to score."""
    classes: list[str | None] = sorted(gt_df["label"].unique()) if class_aware and "label" in gt_df.columns else [None]
    per_class: dict[str, dict[float, float]] = {}
    for cls in classes:
        gt_cls = gt_df[gt_df["label"] == cls] if cls is not None else gt_df
        pred_cls = pred_df[pred_df["label"] == cls] if cls is not None and "label" in pred_df.columns else pred_df
        key = cls if cls is not None else "all"
        per_class[key] = {thr: average_precision(gt_cls, pred_cls, iou_threshold=thr) for thr in iou_thresholds}

    all_aps = [ap for thr_to_ap in per_class.values() for ap in thr_to_ap.values()]
    ap_at_50 = [thr_to_ap[0.5] for thr_to_ap in per_class.values() if 0.5 in thr_to_ap]
    return {
        "map_50": float(np.mean(ap_at_50)) if ap_at_50 else float("nan"),
        "map_50_95": float(np.mean(all_aps)) if all_aps else float("nan"),
        "per_class": per_class,
    }


def calibration_reliability(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = 0.3,
    class_aware: bool = False,
    confidence_col: str = "calibrated_confidence",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability-diagram data: bins predictions by `confidence_col` into
    `n_bins` equal-width buckets over [0, 1] and reports, per non-empty
    bucket, the mean predicted confidence vs. the observed fraction that
    were actually true positives (matched against `gt_df` at
    `iou_threshold`) -- a well-calibrated detector's two columns should
    track each other closely. Rows with a null `confidence_col` (not yet
    run through `calibration.ConfidenceCalibrator`) are dropped rather than
    treated as zero confidence. Columns: `bin_low`, `bin_high`,
    `mean_confidence`, `observed_accuracy`, `n`.
    """
    frames = sorted(set(gt_df.frame_number.unique()) | set(pred_df.frame_number.unique()))
    confidences: list[float] = []
    is_tp: list[bool] = []
    for f in frames:
        gt_rows = gt_df[gt_df.frame_number == f].to_dict("records")
        pred_rows = pred_df[pred_df.frame_number == f].to_dict("records")
        labels = _match_predictions(gt_rows, pred_rows, iou_threshold, class_aware)
        for p, tp in zip(pred_rows, labels, strict=True):
            conf = p.get(confidence_col)
            if conf is None or (isinstance(conf, float) and np.isnan(conf)):
                continue
            confidences.append(float(conf))
            is_tp.append(tp)

    columns = ["bin_low", "bin_high", "mean_confidence", "observed_accuracy", "n"]
    if not confidences:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame({"confidence": confidences, "is_tp": is_tp})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = np.clip(np.digitize(df["confidence"], edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        bucket = df[df["bin"] == b]
        if bucket.empty:
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "mean_confidence": float(bucket["confidence"].mean()),
                "observed_accuracy": float(bucket["is_tp"].mean()),
                "n": int(len(bucket)),
            }
        )
    return pd.DataFrame(rows, columns=columns)
