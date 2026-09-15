import numpy as np
import pytest

from infrastructure_overwatch.anomaly import EmbeddingAnomalyScorer, HOGEmbedder, score_track_anomalies
from infrastructure_overwatch.types import Detection


def _stripes(seed: int, size: int = 64) -> np.ndarray:
    """A deterministic vertical-stripe pattern whose spacing depends on
    `seed`, so different seeds are HOG-distinguishable but each seed is
    reproducible."""
    rng = np.random.default_rng(seed)
    spacing = int(rng.integers(4, 12))
    x = np.arange(size)
    col = (x // spacing) % 2 * 255
    img = np.tile(col, (size, 1)).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _solid(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_hog_embedder_output_is_normalized():
    embedder = HOGEmbedder(size=(64, 64))
    feats = embedder.encode([_stripes(1), _stripes(2)])
    assert feats.shape[0] == 2
    norms = np.linalg.norm(feats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_hog_embedder_empty_input():
    embedder = HOGEmbedder()
    feats = embedder.encode([])
    assert feats.shape == (0, 0)


def test_scorer_requires_fit_before_score():
    scorer = EmbeddingAnomalyScorer(HOGEmbedder(size=(64, 64)))
    with pytest.raises(RuntimeError):
        scorer.score([_stripes(1)])


def test_fit_rejects_empty_gallery():
    scorer = EmbeddingAnomalyScorer(HOGEmbedder(size=(64, 64)))
    with pytest.raises(ValueError):
        scorer.fit([])


def test_familiar_image_scores_lower_than_novel_pattern():
    embedder = HOGEmbedder(size=(64, 64))
    gallery = [_stripes(seed) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder).fit(gallery)

    # Same distribution as the gallery (another stripe pattern).
    familiar_score = scorer.score([_stripes(1)])[0].score
    # A solid block is a very different texture from any striped gallery image.
    novel_score = scorer.score([_solid(128)])[0].score

    assert familiar_score < novel_score


def test_calibrate_threshold_sets_and_returns_float():
    embedder = HOGEmbedder(size=(64, 64))
    gallery = [_stripes(seed) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder).fit(gallery)

    held_out_normal = [_stripes(seed) for seed in range(10, 15)]
    threshold = scorer.calibrate_threshold(held_out_normal, percentile=90.0)

    assert isinstance(threshold, float)
    assert scorer.threshold == threshold


def test_is_anomalous_flag_respects_threshold():
    embedder = HOGEmbedder(size=(64, 64))
    gallery = [_stripes(seed) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder, threshold=0.0).fit(gallery)
    # threshold=0.0 means every non-negative distance counts as anomalous
    results = scorer.score([_stripes(1)])
    assert results[0].is_anomalous is True


# --- Persistence --------------------------------------------------------


def test_save_requires_a_fitted_scorer(tmp_path):
    scorer = EmbeddingAnomalyScorer(HOGEmbedder(size=(64, 64)))
    with pytest.raises(RuntimeError):
        scorer.save(tmp_path / "ref.npz", embedder_name="hog")


def test_save_then_load_round_trips_gallery_and_threshold(tmp_path):
    embedder = HOGEmbedder(size=(64, 64))
    gallery = [_stripes(seed) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder, threshold=0.42).fit(gallery)

    path = tmp_path / "ref.npz"
    scorer.save(path, embedder_name="hog")
    loaded = EmbeddingAnomalyScorer.load(path, embedder=HOGEmbedder(size=(64, 64)))

    assert loaded.threshold == pytest.approx(0.42)
    # a familiar (gallery-distribution) image should score identically either way
    original_score = scorer.score([_stripes(1)])[0].score
    loaded_score = loaded.score([_stripes(1)])[0].score
    assert loaded_score == pytest.approx(original_score)


def test_load_without_an_embedder_auto_instantiates_the_saved_kind(tmp_path):
    embedder = HOGEmbedder(size=(64, 64))
    gallery = [_stripes(seed) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder).fit(gallery)
    scorer.calibrate_threshold([_stripes(seed) for seed in range(10, 13)], percentile=90.0)

    path = tmp_path / "ref.npz"
    scorer.save(path, embedder_name="hog")
    loaded = EmbeddingAnomalyScorer.load(path)  # no embedder passed -- must build its own HOGEmbedder

    assert isinstance(loaded.embedder, HOGEmbedder)
    assert loaded.threshold == pytest.approx(scorer.threshold)


# --- score_track_anomalies -----------------------------------------------


def _det(track_id, frame_id, label="vehicle_of_interest", size=10.0):
    return Detection(
        frame_id=frame_id,
        class_id=0,
        label=label,
        confidence=0.9,
        x1=0.0,
        y1=0.0,
        x2=size,
        y2=size,
        model="test",
        track_id=track_id,
    )


def test_score_track_anomalies_flags_a_track_whose_crop_is_novel():
    embedder = HOGEmbedder(size=(32, 32))
    gallery = [_stripes(seed, size=32) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder, threshold=0.05).fit(gallery)

    n_frames = 4
    frame_ids = [f"F{i:06d}" for i in range(n_frames)]
    rgb_frames = [_solid(128, size=32) for _ in range(n_frames)]  # a novel texture every frame
    detections = [_det(track_id=1, frame_id=fid) for fid in frame_ids]

    events = score_track_anomalies(rgb_frames, frame_ids, detections, scorer, min_hits=2, sample_every_n_hits=1)

    assert len(events) == 1  # fires once, not once per remaining hit
    assert events[0].event_type == "VISUAL_ANOMALY"
    assert events[0].track_id == 1


def test_score_track_anomalies_does_not_flag_a_familiar_track():
    embedder = HOGEmbedder(size=(32, 32))
    gallery = [_stripes(seed, size=32) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder, threshold=0.5).fit(gallery)

    n_frames = 4
    frame_ids = [f"F{i:06d}" for i in range(n_frames)]
    rgb_frames = [_stripes(1, size=32) for _ in range(n_frames)]  # same distribution as the gallery
    detections = [_det(track_id=1, frame_id=fid) for fid in frame_ids]

    events = score_track_anomalies(rgb_frames, frame_ids, detections, scorer, min_hits=2, sample_every_n_hits=1)

    assert events == []


def test_score_track_anomalies_respects_min_hits():
    embedder = HOGEmbedder(size=(32, 32))
    gallery = [_stripes(seed, size=32) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder, threshold=0.05).fit(gallery)

    frame_ids = ["F000000"]
    rgb_frames = [_solid(128, size=32)]
    detections = [_det(track_id=1, frame_id="F000000")]

    # a single hit, but min_hits=2 -- the track hasn't proven itself mature yet
    events = score_track_anomalies(rgb_frames, frame_ids, detections, scorer, min_hits=2, sample_every_n_hits=1)
    assert events == []


def test_score_track_anomalies_ignores_untracked_detections():
    embedder = HOGEmbedder(size=(32, 32))
    gallery = [_stripes(seed, size=32) for seed in range(1, 6)]
    scorer = EmbeddingAnomalyScorer(embedder, threshold=0.05).fit(gallery)

    untracked = _det(track_id=None, frame_id="F000000")
    events = score_track_anomalies([_solid(128, size=32)], ["F000000"], [untracked], scorer, min_hits=1)
    assert events == []
