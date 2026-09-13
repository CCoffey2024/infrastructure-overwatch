"""Confidence calibration and alert-triage routing.

Every detection this pipeline produces routes to a human analyst, who
decides whether to escalate or dismiss -- this system never decides to
engage anything. That framing sets the two costs that matter: a **false
negative** (missed threat) is the highest-consequence error but isn't
recoverable by simply lowering every threshold, because that trades it
directly against analyst alert fatigue; a **false positive** costs analyst
attention, the actual scarce resource in a small fusion cell watching
multiple feeds. That asymmetry is why this module calibrates confidence and
routes detections into triage bands instead of reporting one accuracy
number: the operationally relevant question is which detections a human even
needs to look at.

Default thresholds (`auto_confirm=0.90`, `discard=0.05`) are the *measured*
operating point from this project's own validation runs, not defaults picked
blind -- see docs/METHODOLOGY_AND_LIMITATIONS.md for the before/after
comparison that justified moving `discard` down from an untuned 0.20.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .types import TRIAGE_BANDS, Detection


class ConfidenceCalibrator:
    """Wraps `sklearn.isotonic.IsotonicRegression` to map a detector's raw
    confidence to an empirically-grounded probability of being a true
    positive. Fit once on a labeled validation set, then reused at inference
    time -- a raw model confidence is not a calibrated probability, and
    treating it as one silently miscounts how much attention an alert queue
    actually needs.
    """

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, confidences: np.ndarray, is_true_positive: np.ndarray) -> ConfidenceCalibrator:
        self._iso.fit(confidences, is_true_positive)
        self._fitted = True
        return self

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ConfidenceCalibrator.fit(...) must be called before calibrate(...)")
        return self._iso.predict(confidences)


def reliability_curve(
    confidences: np.ndarray, is_true_positive: np.ndarray, n_bins: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Bins predicted confidence and compares it against empirical precision
    in that bin -- the standard way to see whether "0.8 confidence" actually
    means "80% of the time this is real" for a given model."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bins) - 1, 0, n_bins - 1)
    mean_conf, empirical_precision = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_conf.append(confidences[mask].mean())
        empirical_precision.append(is_true_positive[mask].mean())
    return np.array(mean_conf), np.array(empirical_precision)


@dataclass(frozen=True)
class TriageThresholds:
    auto_confirm: float = 0.90
    discard: float = 0.05


# TriageThresholds is frozen/immutable, so one module-level default instance
# is safe to share across every call site (avoids constructing a new one on
# every function call, which ruff flags as a mutable-default-style hazard).
DEFAULT_TRIAGE_THRESHOLDS = TriageThresholds()


def triage_band(calibrated_confidence: float, thresholds: TriageThresholds = DEFAULT_TRIAGE_THRESHOLDS) -> str:
    if calibrated_confidence >= thresholds.auto_confirm:
        return "auto_confirm"
    if calibrated_confidence < thresholds.discard:
        return "auto_discard"
    return "analyst_review"


def route_detections(
    detections: list[Detection],
    calibrator: ConfidenceCalibrator,
    thresholds: TriageThresholds = DEFAULT_TRIAGE_THRESHOLDS,
) -> list[Detection]:
    """Attaches `calibrated_confidence` and `triage_band` to each detection.
    Detections in the `auto_discard` band are kept (not dropped) so a
    reviewer can audit what got filtered and why -- see
    `docs/USER_MANUAL.md` on reading the discard band."""
    if not detections:
        return []
    raw = np.array([d.confidence for d in detections])
    calibrated = calibrator.calibrate(raw)
    out = []
    for d, c in zip(detections, calibrated, strict=True):
        band = triage_band(float(c), thresholds)
        assert band in TRIAGE_BANDS
        out.append(replace(d, calibrated_confidence=float(c), triage_band=band))
    return out
