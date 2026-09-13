import pytest
from PIL import Image

from infrastructure_overwatch.ingest import (
    VEHICLE_DISMOUNT_CLASS_NAMES,
    VEHICLE_ONLY_CLASS_NAMES,
    VisDroneDETIndex,
    VisDroneDETVehicleDataset,
    VisDroneVIDIndex,
    VisDroneVIDVehicleDataset,
    class_names_for_scheme,
    visdrone_category_to_class_index,
)


def test_class_names_for_scheme_vehicle_only():
    assert class_names_for_scheme("vehicle_only") == VEHICLE_ONLY_CLASS_NAMES == ("car", "truck", "bus")


def test_class_names_for_scheme_vehicle_dismount():
    assert class_names_for_scheme("vehicle_dismount") == VEHICLE_DISMOUNT_CLASS_NAMES
    assert class_names_for_scheme("vehicle_dismount")[:3] == class_names_for_scheme("vehicle_only")


def test_class_names_for_scheme_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        class_names_for_scheme("something_else")


def test_visdrone_category_to_class_index_vehicle_only():
    names = class_names_for_scheme("vehicle_only")
    assert visdrone_category_to_class_index(4, names) == 0  # car
    assert visdrone_category_to_class_index(6, names) == 1  # truck
    assert visdrone_category_to_class_index(9, names) == 2  # bus
    assert visdrone_category_to_class_index(1, names) is None  # pedestrian, no dismount slot
    assert visdrone_category_to_class_index(2, names) is None  # people, no dismount slot


def test_visdrone_category_to_class_index_vehicle_dismount():
    names = class_names_for_scheme("vehicle_dismount")
    assert visdrone_category_to_class_index(1, names) == 3  # pedestrian
    assert visdrone_category_to_class_index(2, names) == 3  # people
    assert visdrone_category_to_class_index(4, names) == 0  # car


def test_visdrone_category_to_class_index_drops_out_of_scope_categories():
    names = class_names_for_scheme("vehicle_dismount")
    for out_of_scope_cat in (0, 3, 5, 7, 8, 10, 11):
        assert visdrone_category_to_class_index(out_of_scope_cat, names) is None


# --- VisDrone2019-DET (single-image) ----------------------------------------


def test_visdrone_det_index_raises_when_root_is_not_a_visdrone_distribution(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisDroneDETIndex(root=str(tmp_path))


def _make_visdrone_det_fixture(tmp_path):
    split_dir = tmp_path / "VisDrone2019-DET-train"
    images_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    images_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(images_dir / "0000001_img.jpg")
    # car (cat 4, score 1), pedestrian (cat 1, score 1), bicycle (cat 3, score 1, out of
    # taxonomy), and a van (cat 5, score 0) that should be dropped for its zero score too.
    (ann_dir / "0000001_img.txt").write_text(
        "10,10,20,30,1,4,0,0\n50,40,15,15,1,1,0,0\n5,5,10,10,1,3,0,0\n70,60,10,10,0,5,0,0\n"
    )
    # an image with no annotated objects at all
    Image.new("RGB", (100, 80), color=(10, 10, 10)).save(images_dir / "0000002_img.jpg")
    (ann_dir / "0000002_img.txt").write_text("")
    return tmp_path


def test_visdrone_det_index_lists_images_and_parses_annotations(tmp_path):
    root = _make_visdrone_det_fixture(tmp_path)
    index = VisDroneDETIndex(root=str(root))
    stems = index.list_image_stems("train")
    assert stems == ["0000001_img", "0000002_img"]

    ann = index.parse_annotations("train", "0000001_img")
    assert len(ann) == 4
    assert list(ann["category"]) == [4, 1, 3, 5]

    empty_ann = index.parse_annotations("train", "0000002_img")
    assert len(empty_ann) == 0

    img = index.load_image("train", "0000001_img")
    assert img.size == (100, 80)


def test_visdrone_det_vehicle_dataset_vehicle_only_drops_pedestrian_and_bicycle(tmp_path):
    root = _make_visdrone_det_fixture(tmp_path)
    index = VisDroneDETIndex(root=str(root))
    ds = VisDroneDETVehicleDataset(index, split="train", class_scheme="vehicle_only")
    assert len(ds) == 2

    _img, label = ds[0]
    n_classes = len(VEHICLE_ONLY_CLASS_NAMES)
    assert label.shape[-1] == 5 + n_classes
    # exactly one object encoded (the car) -- pedestrian, bicycle, and the zero-score van
    # are all dropped under this scheme
    assert label[..., 0].sum() == 1
    assert label[..., 5 + 0].sum() == 1  # car channel (class index 0)
    assert label[..., 5 + 1].sum() == 0  # truck
    assert label[..., 5 + 2].sum() == 0  # bus


def test_visdrone_det_vehicle_dataset_vehicle_dismount_keeps_pedestrian(tmp_path):
    root = _make_visdrone_det_fixture(tmp_path)
    index = VisDroneDETIndex(root=str(root))
    ds = VisDroneDETVehicleDataset(index, split="train", class_scheme="vehicle_dismount")

    _img, label = ds[0]
    n_classes = len(VEHICLE_DISMOUNT_CLASS_NAMES)
    assert label.shape[-1] == 5 + n_classes
    # car + pedestrian(->dismount) encoded; bicycle and the zero-score van still dropped
    assert label[..., 0].sum() == 2
    assert label[..., 5 + 0].sum() == 1  # car
    assert label[..., 5 + 3].sum() == 1  # dismount


def test_visdrone_det_vehicle_dataset_handles_image_with_no_objects(tmp_path):
    root = _make_visdrone_det_fixture(tmp_path)
    index = VisDroneDETIndex(root=str(root))
    ds = VisDroneDETVehicleDataset(index, split="train", class_scheme="vehicle_only")

    _img, label = ds[1]
    assert label[..., 0].sum() == 0


# --- VisDrone2019-VID (sequence-of-frames) ----------------------------------


def test_visdrone_vid_index_raises_when_root_is_not_a_visdrone_distribution(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisDroneVIDIndex(root=str(tmp_path))


def _make_visdrone_vid_fixture(tmp_path):
    # official archives unpack with a doubled directory name -- exercise that layout
    split_dir = tmp_path / "VisDrone2019-VID-train" / "VisDrone2019-VID-train"
    seq_dir = split_dir / "sequences" / "uav0000001_v"
    ann_dir = split_dir / "annotations"
    seq_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for i in range(1, 4):
        Image.new("RGB", (100, 80), color=(120, 120, 120)).save(seq_dir / f"{i:07d}.jpg")

    # frame 1: car (cat 4, score 1) and pedestrian (cat 1, score 1)
    # frame 2: bicycle (cat 3, score 1, out of taxonomy) and a van (cat 5, score 0, dropped)
    # frame 3: no annotated objects
    (ann_dir / "uav0000001_v.txt").write_text(
        "1,0,10,10,20,30,1,4,0,0\n1,1,50,40,15,15,1,1,0,0\n2,2,5,5,10,10,1,3,0,0\n2,3,70,60,10,10,0,5,0,0\n"
    )
    return tmp_path


def test_visdrone_vid_index_lists_sequences_and_parses_gt(tmp_path):
    root = _make_visdrone_vid_fixture(tmp_path)
    index = VisDroneVIDIndex(root=str(root))
    assert index.list_sequences("train") == ["uav0000001_v"]

    gt = index.parse_gt("train", "uav0000001_v")
    assert len(gt) == 4
    assert sorted(gt["frame"].unique().tolist()) == [1, 2]

    img = index.load_frame("train", "uav0000001_v", 1)
    assert img.size == (100, 80)

    frame_indices = index.sample_frame_indices("train", "uav0000001_v", stride=1, cap=10)
    assert frame_indices == [1, 2]  # frame 3 has no gt rows, so it's never a distinct "frame" value


def test_visdrone_vid_vehicle_dataset_vehicle_only_drops_pedestrian_and_bicycle(tmp_path):
    root = _make_visdrone_vid_fixture(tmp_path)
    index = VisDroneVIDIndex(root=str(root))
    ds = VisDroneVIDVehicleDataset(
        index, split="train", seq_list=["uav0000001_v"], class_scheme="vehicle_only", stride=1, cap=10
    )
    assert len(ds) == 2  # frames 1 and 2

    _img, label_f1 = ds[0]
    n_classes = len(VEHICLE_ONLY_CLASS_NAMES)
    assert label_f1.shape[-1] == 5 + n_classes
    assert label_f1[..., 0].sum() == 1  # only the car -- pedestrian dropped
    assert label_f1[..., 5 + 0].sum() == 1  # car

    _img, label_f2 = ds[1]
    assert label_f2[..., 0].sum() == 0  # bicycle out of taxonomy, van dropped for zero score


def test_visdrone_vid_vehicle_dataset_vehicle_dismount_keeps_pedestrian(tmp_path):
    root = _make_visdrone_vid_fixture(tmp_path)
    index = VisDroneVIDIndex(root=str(root))
    ds = VisDroneVIDVehicleDataset(
        index, split="train", seq_list=["uav0000001_v"], class_scheme="vehicle_dismount", stride=1, cap=10
    )

    _img, label_f1 = ds[0]
    n_classes = len(VEHICLE_DISMOUNT_CLASS_NAMES)
    assert label_f1.shape[-1] == 5 + n_classes
    assert label_f1[..., 0].sum() == 2  # car + pedestrian(->dismount)
    assert label_f1[..., 5 + 0].sum() == 1  # car
    assert label_f1[..., 5 + 3].sum() == 1  # dismount
