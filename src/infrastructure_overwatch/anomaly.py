"""Embedding-based anomaly scoring: is this crop or frame unlike anything in
a reference gallery of "normal" corridor imagery?

This is a complementary signal to the class-based detectors in `detectors/`,
not a replacement for them. A fixed four-class taxonomy (`types.THREAT_CLASSES`)
can only ever flag what it was built to recognize; a nearest-neighbor
embedding distance can flag "this looks unlike anything the gallery has seen"
without needing a name for what it is -- useful for genuinely novel visual
patterns (new equipment, an unfamiliar vehicle type, unusual activity) that
the taxonomy wouldn't otherwise catch. Like every other signal in this
project, an anomaly score is a routing hint for a human analyst, never an
automated decision.

Two embedder backends, both lazy-imported so the rest of the project has no
hard dependency on either:

- `HOGEmbedder` -- offline, no model download, what tests and CI exercise.
  Requires the `anomaly` extra (`pip install -e ".[anomaly]"`).
- `DinoV2Embedder` -- self-supervised DINOv2 features via `torch.hub`.
  Requires network access on first use to download the model (cached
  afterward), so it is not exercised by automated tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .types import Detection, Event


class Embedder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...


class HOGEmbedder:
    """Histogram-of-oriented-gradients embedding. A deliberately simple,
    dependency-light fallback: no model weights to download, deterministic,
    fast enough to run in CI. Not as semantically rich as a learned
    embedding, but useful as a default and as a way to validate the anomaly-
    scoring logic itself independently of any particular embedding model."""

    def __init__(self, size: tuple[int, int] = (128, 128)):
        self.size = size

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        import cv2
        from skimage.feature import hog

        feats = []
        for img in images:
            resized = cv2.resize(img, self.size)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
            v = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            ).astype(np.float32, copy=False)
            v = v / (np.linalg.norm(v) + 1e-8)
            feats.append(v)
        if not feats:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(feats)


class DinoV2Embedder:
    """DINOv2 self-supervised features (`facebookresearch/dinov2` via
    `torch.hub`). Requires internet access the first time a given model
    variant is loaded; cached by `torch.hub` afterward."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        import cv2

        torch = self.torch
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        batch = torch.stack([self.transform(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images]).to(self.device)
        with torch.inference_mode():
            out = self.model(batch)
        x = out.detach().float().cpu().numpy()
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x


@dataclass
class AnomalyScore:
    index: int
    score: float
    is_anomalous: bool


class EmbeddingAnomalyScorer:
    """Fits a reference gallery of embeddings from known-normal imagery, then
    scores new images by cosine distance to the *nearest* gallery embedding
    (on L2-normalized vectors). A higher score means "less like anything
    already in the gallery."""

    def __init__(self, embedder: Embedder, threshold: float | None = None):
        self.embedder = embedder
        self.threshold = threshold
        self._gallery: np.ndarray | None = None

    def fit(self, reference_images: list[np.ndarray]) -> EmbeddingAnomalyScorer:
        gallery = self.embedder.encode(reference_images)
        if gallery.size == 0:
            raise ValueError("reference_images must be non-empty")
        self._gallery = gallery
        return self

    def score(self, images: list[np.ndarray]) -> list[AnomalyScore]:
        if self._gallery is None:
            raise RuntimeError("EmbeddingAnomalyScorer.fit(...) must be called with a reference gallery first")
        embeddings = self.embedder.encode(images)
        threshold = self.threshold if self.threshold is not None else _DEFAULT_THRESHOLD
        if embeddings.size == 0:
            return []
        sims = embeddings @ self._gallery.T  # (n_images, gallery_size)
        nearest_sim = sims.max(axis=1)
        distances = 1.0 - nearest_sim
        return [
            AnomalyScore(index=i, score=float(d), is_anomalous=bool(d >= threshold)) for i, d in enumerate(distances)
        ]

    def calibrate_threshold(self, normal_images: list[np.ndarray], percentile: float = 99.0) -> float:
        """Sets `self.threshold` to the given percentile of anomaly scores
        measured on a held-out set of *known-normal* images -- the standard
        way to pick an operating point for a one-class scorer without
        needing labeled anomalies. Read `percentile` as "what fraction of
        genuinely normal imagery should score below the alert threshold";
        99.0 accepts a 1% false-alarm rate on the calibration set itself.
        `normal_images` must be held out from `fit`'s reference gallery --
        scoring an image against a gallery that already contains it finds
        itself as its own nearest neighbor (distance ~0), which would
        silently calibrate an unusably strict near-zero threshold."""
        scores = [s.score for s in self.score(normal_images)]
        self.threshold = float(np.percentile(scores, percentile))
        return self.threshold

    def save(self, path, embedder_name: str) -> None:
        """Persists the fitted gallery, threshold, and which embedder class
        built it, so `load(...)` can reconstruct a working scorer without
        the caller having to remember which embedder was used."""
        if self._gallery is None:
            raise RuntimeError("EmbeddingAnomalyScorer.fit(...) must be called before save(...)")
        threshold = self.threshold if self.threshold is not None else _DEFAULT_THRESHOLD
        with open(path, "wb") as f:
            np.savez(f, gallery=self._gallery, threshold=np.float64(threshold), embedder_name=np.str_(embedder_name))

    @classmethod
    def load(cls, path, embedder: Embedder | None = None) -> EmbeddingAnomalyScorer:
        """Loads a reference gallery written by `save(...)`. Pass `embedder`
        to reuse a specific already-constructed instance; omit it to
        auto-instantiate the same embedder class (`HOGEmbedder` or
        `DinoV2Embedder`) the gallery was fit with."""
        with open(path, "rb") as f:
            data = np.load(f)
            gallery = data["gallery"]
            threshold = float(data["threshold"])
            embedder_name = str(data["embedder_name"])
        if embedder is None:
            embedder = DinoV2Embedder() if embedder_name == "dinov2" else HOGEmbedder()
        scorer = cls(embedder, threshold=threshold)
        scorer._gallery = gallery
        return scorer


_DEFAULT_THRESHOLD = 0.5


def score_track_anomalies(
    rgb_frames: list[np.ndarray],
    frame_ids: list[str],
    detections: list[Detection],
    scorer: EmbeddingAnomalyScorer,
    min_hits: int = 3,
    sample_every_n_hits: int = 5,
    fps: float = 10.0,
) -> list[Event]:
    """Crops each sufficiently-mature track's box out of its RGB frame,
    scores the crop against `scorer`'s persisted reference gallery, and
    emits a `VISUAL_ANOMALY` `Event` the first time that track's score
    crosses the scorer's threshold -- reusing the existing `Event` type and
    the same fire-once-per-track discipline `PipelineEventEngine` already
    uses for its own event types, so this is a routing hint for a human
    analyst like every other event, not a new kind of decision.

    `min_hits` skips a track's crops until it's been seen at least this many
    times (the same idea as `MultiTracker.min_hits`, applied independently
    here since a caller may want a different bar for "worth embedding" than
    for "worth raising a zone event"). `sample_every_n_hits` then scores
    only every Nth hit of an already-mature track -- an appearance embedding
    changes slowly frame to frame, so embedding every single one is wasted
    compute for a signal this stable.
    """
    from .types import Event

    frame_index = {fid: i for i, fid in enumerate(frame_ids)}
    hit_count: dict[int, int] = {}
    already_flagged: set[int] = set()
    events: list[Event] = []

    for det in detections:
        if det.track_id is None or det.track_id in already_flagged:
            continue
        hit_count[det.track_id] = hit_count.get(det.track_id, 0) + 1
        hits = hit_count[det.track_id]
        if hits < min_hits or (hits - min_hits) % sample_every_n_hits != 0:
            continue

        frame = rgb_frames[frame_index[det.frame_id]]
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        [result] = scorer.score([crop])
        if result.is_anomalous:
            threshold = scorer.threshold if scorer.threshold is not None else _DEFAULT_THRESHOLD
            events.append(
                Event(
                    frame_id=det.frame_id,
                    timestamp_s=frame_index[det.frame_id] / fps,
                    event_type="VISUAL_ANOMALY",
                    severity="medium",
                    track_id=det.track_id,
                    description=(
                        f"Track {det.track_id} ({det.label}) looks unlike anything in the reference "
                        f"gallery (distance {result.score:.2f} >= threshold {threshold:.2f})."
                    ),
                    score=result.score,
                )
            )
            already_flagged.add(det.track_id)
    return events
