import numpy as np
import pytest

from infrastructure_overwatch.anomaly import EmbeddingAnomalyScorer, HOGEmbedder


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
