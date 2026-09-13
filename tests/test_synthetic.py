import numpy as np

from infrastructure_overwatch.synthetic import (
    GRID,
    IMG_SIZE,
    N_SYNTHETIC_CLASSES,
    SyntheticCorridorDataset,
    random_threats,
    render_corridor_sequence,
    render_day,
    render_night,
)


def test_random_threats_within_bounds():
    rng = np.random.default_rng(0)
    threats = random_threats(rng, n_min=2, n_max=2, size=IMG_SIZE)
    assert len(threats) == 2
    for t in threats:
        assert 0 <= t["cx"] <= IMG_SIZE
        assert 0 <= t["cy"] <= IMG_SIZE
        assert 0 <= t["cls"] < N_SYNTHETIC_CLASSES


def test_render_day_shape_and_range():
    rng = np.random.default_rng(1)
    threats = random_threats(rng)
    img, boxes = render_day(threats, rng)
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert len(boxes) == len(threats)


def test_render_night_shape_and_range():
    rng = np.random.default_rng(2)
    threats = random_threats(rng)
    img, boxes = render_night(threats, rng)
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_synthetic_corridor_dataset_item_shapes():
    ds = SyntheticCorridorDataset(5, domain="day", seed=0)
    assert len(ds) == 5
    img, label = ds[0]
    assert img.shape == (1, IMG_SIZE, IMG_SIZE)
    assert label.shape == (GRID, GRID, 5 + N_SYNTHETIC_CLASSES)


def test_synthetic_corridor_dataset_deterministic_when_not_fresh():
    ds = SyntheticCorridorDataset(5, domain="day", seed=0, fresh=False)
    img_a, _ = ds[2]
    img_b, _ = ds[2]
    assert np.array_equal(img_a, img_b)


def test_render_corridor_sequence_has_persistent_object_ids():
    rng = np.random.default_rng(3)
    frames = render_corridor_sequence(10, rng, domain="night", n_threats=2)
    assert len(frames) == 10
    for _img, _boxes, object_ids in frames:
        assert len(object_ids) == 2
        assert object_ids == [0, 1]
