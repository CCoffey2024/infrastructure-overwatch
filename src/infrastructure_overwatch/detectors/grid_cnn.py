"""This project's own lightweight detector: a small CNN backbone with a
per-cell objectness/box/class head, trained from scratch on the synthetic
corridor scene (and, for `vehicle_of_interest`, on real UAVDT frames -- see
ingest.py). ~110K params at the 96x96/6x6 synthetic scale.

Why a lightweight model at all, when a production detector (YOLO) is
available: it is worth measuring whether a small, fast, from-scratch model
covers this project's specific, narrow threat taxonomy before reaching for a
heavier general-purpose one. See docs/METHODOLOGY_AND_LIMITATIONS.md for the
measured comparison -- the answer turns out to depend on the class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..geometry import iou_xyxy
from ..grid_codec import decode_grid_predictions
from ..types import Detection
from . import DetectorAdapter


class GridDetector(nn.Module):
    """`Conv-BN-ReLU-MaxPool` stages -> adaptive pool to the target grid ->
    a 1x1 conv head emitting `(objectness, box, class-logits)` per cell.

    One parametrized architecture serves both the small synthetic chip
    detector and a larger real-scene detector; only `stage_channels` and the
    target grid differ. The source R&D notebooks defined this network twice
    with those as the only real differences -- this replaces both.
    """

    def __init__(self, n_classes: int, grid_h: int, grid_w: int, stage_channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in stage_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU()]
        layers += [nn.AdaptiveAvgPool2d((grid_h, grid_w))]
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_ch, 5 + n_classes, 1)
        self.n_classes = n_classes
        self.grid_h, self.grid_w = grid_h, grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features(x)).permute(0, 2, 3, 1)
        obj = torch.sigmoid(out[..., 0:1])
        box = torch.sigmoid(out[..., 1:5])
        return torch.cat([obj, box, out[..., 5:]], dim=-1)  # class logits stay raw


def grid_detection_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
    lambda_cls: float = 1.0,
) -> torch.Tensor:
    obj_mask = target[..., 0]
    raw = F.binary_cross_entropy(pred[..., 0], obj_mask, reduction="none")
    obj_loss = torch.where(obj_mask > 0, raw, lambda_noobj * raw).mean()

    box_mask = obj_mask.unsqueeze(-1)
    box_loss = F.mse_loss(pred[..., 1:5] * box_mask, target[..., 1:5] * box_mask, reduction="sum") / (
        box_mask.sum() + 1e-6
    )

    cls_logits = pred[..., 5:].reshape(-1, n_classes)
    cls_target = target[..., 5:].argmax(dim=-1).reshape(-1)
    cls_raw = F.cross_entropy(cls_logits, cls_target, reduction="none").reshape(obj_mask.shape)
    cls_loss = (cls_raw * obj_mask).sum() / (obj_mask.sum() + 1e-6)

    return obj_loss + lambda_coord * box_loss + lambda_cls * cls_loss


@dataclass
class DetectionScore:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def evaluate_grid_detector(
    model: GridDetector,
    dataset: Dataset,
    img_w: int,
    img_h: int,
    n: int | None = None,
    thresh: float = 0.5,
    iou_thresh: float = 0.3,
    device: torch.device | None = None,
) -> DetectionScore:
    """Precision/recall/F1 by matching decoded predictions to decoded ground
    truth via greedy IoU -- the same evaluation logic serves the synthetic
    and real-scene detectors, since both emit the same grid-tensor format."""
    device = device or torch.device("cpu")
    model.eval()
    n = len(dataset) if n is None else min(n, len(dataset))  # type: ignore[arg-type]
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(n):
            img, label = dataset[i]
            img_t = torch.as_tensor(img).unsqueeze(0).to(device)
            pred = model(img_t)[0].cpu()
            pred_boxes = decode_grid_predictions(pred, img_w, img_h, model.grid_w, model.grid_h, thresh=thresh)
            gt_boxes = decode_grid_predictions(
                torch.as_tensor(label), img_w, img_h, model.grid_w, model.grid_h, thresh=0.5, nms_iou=1.1
            )
            matched: set[int] = set()
            for pb in pred_boxes:
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    if j in matched:
                        continue
                    v = iou_xyxy(pb[:4], gb[:4])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_thresh:
                    tp += 1
                    matched.add(best_j)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionScore(precision, recall, f1, tp, fp, fn)


def train_grid_detector(
    model: GridDetector,
    loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    quiet: bool = False,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        model.train()
        total, count = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = grid_detection_loss(model(imgs), labels, model.n_classes)
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)
            count += imgs.size(0)
        history.append(total / count)
        if not quiet and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1:2d}/{epochs}  loss {history[-1]:.4f}")
    return history


class GridCNNAdapter(DetectorAdapter):
    """Wraps a trained `GridDetector` behind the shared `DetectorAdapter`
    interface, so it can be swapped for `ClassicalMotionAdapter` or
    `YOLOAdapter` anywhere in the pipeline."""

    def __init__(
        self,
        model: GridDetector,
        class_names: tuple[str, ...],
        img_w: int,
        img_h: int,
        thresh: float = 0.4,
        device: torch.device | None = None,
    ):
        self.model = model
        self.class_names = class_names
        self.img_w, self.img_h = img_w, img_h
        self.thresh = thresh
        self.device = device or torch.device("cpu")
        self.model.to(self.device).eval()

    @classmethod
    def load(
        cls,
        weights_path: str,
        n_classes: int,
        grid_h: int,
        grid_w: int,
        class_names,
        img_w,
        img_h,
        stage_channels=(16, 32, 64),
        **kwargs,
    ) -> GridCNNAdapter:
        model = GridDetector(n_classes=n_classes, grid_h=grid_h, grid_w=grid_w, stage_channels=stage_channels)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return cls(model, class_names=class_names, img_w=img_w, img_h=img_h, **kwargs)

    def detect(self, frame: np.ndarray, frame_id: str) -> list[Detection]:
        """`frame` is a grayscale `(1, H, W)` float32 array in [0, 1], the
        same format `synthetic.SyntheticCorridorDataset` yields."""
        img_t = torch.as_tensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img_t)[0].cpu()
        boxes = decode_grid_predictions(
            pred, self.img_w, self.img_h, self.model.grid_w, self.model.grid_h, thresh=self.thresh
        )
        out = []
        for x0, y0, x1, y1, conf, cls in boxes:
            out.append(
                Detection(
                    frame_id=frame_id,
                    class_id=cls,
                    label=self.class_names[cls],
                    confidence=float(conf),
                    x1=float(x0),
                    y1=float(y0),
                    x2=float(x1),
                    y2=float(y1),
                    model="grid_cnn",
                )
            )
        return out
