"""Builds the pruned evidence notebooks in notebooks/ from code, so their
narrative markdown and code cells are versioned as this script rather than as
fragile notebook JSON. Run: python scripts/build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


SETUP_CELL = code(
    """
from pathlib import Path
import sys
HERE = Path.cwd().resolve()
ROOT = HERE.parent if HERE.name == "notebooks" else HERE
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
print("Project root:", ROOT)
"""
)


def save(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    path = NB_DIR / name
    nbf.write(nb, path)
    return path


# --- 01: domain gap evidence -----------------------------------------------

nb01 = [
    md(
        """
# 01 — Domain-gap evidence: Day vs. Night on the synthetic corridor scene

This notebook reproduces this project's central measured finding: a detector trained only
on **Day** (clean daylight, close standoff) imagery loses most or all of its accuracy on
**Night** (low light, long standoff, oblique angle) imagery of the *same* underlying scene,
and shows what recovers it. See `docs/METHODOLOGY_AND_LIMITATIONS.md#domain-gap-day-vs-night`
for the full write-up this notebook backs.

It imports the actual application code (`infrastructure_overwatch.synthetic`,
`infrastructure_overwatch.detectors.grid_cnn`) rather than redefining anything — if you
change the detector or renderer in `src/`, re-running this notebook re-measures the claim.
"""
    ),
    SETUP_CELL,
    code(
        """
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from infrastructure_overwatch.synthetic import GRID, IMG_SIZE, N_SYNTHETIC_CLASSES, SyntheticCorridorDataset
from infrastructure_overwatch.detectors.grid_cnn import GridDetector, evaluate_grid_detector, train_grid_detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""
    ),
    md("## Train on Day only, then measure both domains"),
    code(
        """
day_train_ds = SyntheticCorridorDataset(800, domain="day", seed=1)
day_val_ds = SyntheticCorridorDataset(200, domain="day", seed=2)
night_val_ds = SyntheticCorridorDataset(200, domain="night", seed=3)
day_train_loader = DataLoader(day_train_ds, batch_size=32, shuffle=True)

baseline_model = GridDetector(n_classes=N_SYNTHETIC_CLASSES, grid_h=GRID, grid_w=GRID)
t0 = time.time()
train_grid_detector(baseline_model, day_train_loader, epochs=20, device=device)
print(f"done in {time.time()-t0:.1f}s")
"""
    ),
    code(
        """
day_score = evaluate_grid_detector(baseline_model, day_val_ds, IMG_SIZE, IMG_SIZE, device=device)
night_score = evaluate_grid_detector(baseline_model, night_val_ds, IMG_SIZE, IMG_SIZE, device=device)
print(f"Day   val -> precision {day_score.precision:.2f}  recall {day_score.recall:.2f}  F1 {day_score.f1:.2f}")
print(f"Night val -> precision {night_score.precision:.2f}  recall {night_score.recall:.2f}  F1 {night_score.f1:.2f}")

fig, ax = plt.subplots(figsize=(5, 3.2))
bars = ax.bar(["Day val", "Night val"], [day_score.f1, night_score.f1], color=["#4C78A8", "#E45756"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1")
ax.set_title("Same model, two domains (synthetic corridor scene)")
for b, v in zip(bars, [day_score.f1, night_score.f1], strict=True):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Two remedies, and what each costs

- **Domain randomization** — train directly on Night-style rendering, reseeding every
  epoch (`fresh=True`), using no real Night-domain labels.
- **Fine-tuning** — start from the Day-trained weights, fine-tune on a small, fixed batch
  of Night-style samples, simulating "a field team sent back a small batch of labeled
  night frames."
"""
    ),
    code(
        """
import copy

# Remedy A: domain randomization
augmented_train_ds = SyntheticCorridorDataset(500, domain="night", fresh=True)
augmented_loader = DataLoader(augmented_train_ds, batch_size=32, shuffle=True)
augmented_model = GridDetector(n_classes=N_SYNTHETIC_CLASSES, grid_h=GRID, grid_w=GRID)
train_grid_detector(augmented_model, augmented_loader, epochs=20, device=device, quiet=True)
augmented_score = evaluate_grid_detector(augmented_model, night_val_ds, IMG_SIZE, IMG_SIZE, device=device)

# Remedy B: limited fine-tuning on a small Night-style batch
night_small_train_ds = SyntheticCorridorDataset(100, domain="night", seed=99)
night_small_loader = DataLoader(night_small_train_ds, batch_size=16, shuffle=True)
finetuned_model = copy.deepcopy(baseline_model)
train_grid_detector(finetuned_model, night_small_loader, epochs=20, lr=2e-4, device=device, quiet=True)
finetuned_score = evaluate_grid_detector(finetuned_model, night_val_ds, IMG_SIZE, IMG_SIZE, device=device)

print(f"Night val, Day-only baseline:     F1 {night_score.f1:.2f}")
print(f"Night val, domain randomization:  F1 {augmented_score.f1:.2f}")
print(f"Night val, fine-tuned (100 imgs): F1 {finetuned_score.f1:.2f}")

fig, ax = plt.subplots(figsize=(6.5, 3.5))
names = ["Day-only\\n(baseline)", "Domain\\nrandomization", "Fine-tuned on\\n100 images"]
f1s = [night_score.f1, augmented_score.f1, finetuned_score.f1]
bars = ax.bar(names, f1s, color=["#E45756", "#F58518", "#54A24B"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1 (Night validation)")
ax.set_title("Two remedies for the same domain gap")
for b, v in zip(bars, f1s, strict=True):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Takeaway

The collapse from Day-only training to Night evaluation, and the partial recovery from
either remedy, is the mechanism `docs/METHODOLOGY_AND_LIMITATIONS.md` and
`docs/USER_MANUAL.md` both build on: this system is measurably worse in degraded
conditions, and a missing alert at night is not evidence nothing is there. See
`02_real_data_validation.ipynb` for the same mechanism measured independently on real
drone footage.
"""
    ),
]

# --- 02: real data validation (UAVDT) ---------------------------------------

nb02 = [
    md(
        """
# 02 — Real-data validation: the `vehicle_of_interest` track on UAVDT

`drone`, `dismount`, and `launch_flash` are trained on a synthetic renderer, deliberately,
so each degradation factor could be isolated and controlled. `vehicle_of_interest` is
different: real drone-video ground truth already exists for it (the UAVDT benchmark, Du et
al., ECCV 2018), so this notebook re-runs the same collapse-and-remedy measurement as
`01_domain_gap_evidence.ipynb` on independently-sourced real footage instead of synthetic
renders — a second, real-data check of the same mechanism.

**Requires a local UAVDT copy.** Set `UAVDT_ROOT` or edit the path below; see
`docs/DEVELOPMENT.md#obtaining-uavdt-for-the-real-vehicle_of_interest-track`. If the dataset
isn't available, this notebook will raise `FileNotFoundError` on the indexing cell rather
than silently skipping — that failure is expected and documented, not a bug.
"""
    ),
    SETUP_CELL,
    code(
        """
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from infrastructure_overwatch.ingest import UAVDTIndex, UAVDTVehicleDataset, WORK_W, WORK_H, VEHICLE_GRID_W, VEHICLE_GRID_H, VEHICLE_N_CLASSES
from infrastructure_overwatch.detectors.grid_cnn import GridDetector, evaluate_grid_detector, train_grid_detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
index = UAVDTIndex()  # reads UAVDT_ROOT, or pass root=r"D:\\FMV\\UAVDT\\raw" explicitly
print(f"Using device: {device}, UAVDT root: {index.root}")
"""
    ),
    md("## Curated sequence split (same day/night framing as the synthetic track)"),
    code(
        """
DAY_TRAIN_SEQS = ["M0101", "M0402", "M1201", "M1306"]
DAY_VAL_SEQS = ["M0403", "M1301"]
NIGHT_FINETUNE_SEQS = ["M0206", "M1008"]
NIGHT_VAL_SEQS = ["M0601", "M0701", "M1009", "M1101"]

veh_day_train_ds = UAVDTVehicleDataset(index, DAY_TRAIN_SEQS, stride=8, cap=80, seed=1)
veh_day_val_ds = UAVDTVehicleDataset(index, DAY_VAL_SEQS, stride=10, cap=60, seed=2)
veh_night_val_ds = UAVDTVehicleDataset(index, NIGHT_VAL_SEQS, stride=10, cap=50, seed=3)
print(f"day-train: {len(veh_day_train_ds)} frames   day-val: {len(veh_day_val_ds)} frames   "
      f"night-val: {len(veh_night_val_ds)} frames")
"""
    ),
    code(
        """
veh_day_train_loader = DataLoader(veh_day_train_ds, batch_size=8, shuffle=True)
veh_day_model = GridDetector(n_classes=VEHICLE_N_CLASSES, grid_h=VEHICLE_GRID_H, grid_w=VEHICLE_GRID_W, stage_channels=(24, 48, 96, 96))

t0 = time.time()
train_grid_detector(veh_day_model, veh_day_train_loader, epochs=25, device=device)
print(f"done in {time.time()-t0:.1f}s")
"""
    ),
    code(
        """
veh_day_score = evaluate_grid_detector(veh_day_model, veh_day_val_ds, WORK_W, WORK_H, device=device)
veh_night_score = evaluate_grid_detector(veh_day_model, veh_night_val_ds, WORK_W, WORK_H, device=device)
print(f"Day   val (real) -> F1 {veh_day_score.f1:.2f}")
print(f"Night val (real, day-only model) -> F1 {veh_night_score.f1:.2f}")

import copy
veh_night_finetune_ds = UAVDTVehicleDataset(index, NIGHT_FINETUNE_SEQS, stride=6, cap=60, seed=21)
veh_night_finetune_loader = DataLoader(veh_night_finetune_ds, batch_size=8, shuffle=True)
veh_finetuned_model = copy.deepcopy(veh_day_model)
train_grid_detector(veh_finetuned_model, veh_night_finetune_loader, epochs=20, lr=2e-4, device=device, quiet=True)
veh_finetuned_score = evaluate_grid_detector(veh_finetuned_model, veh_night_val_ds, WORK_W, WORK_H, device=device)
print(f"Night val (real, fine-tuned) -> F1 {veh_finetuned_score.f1:.2f}")

fig, ax = plt.subplots(figsize=(6.5, 3.5))
names = ["Day val\\n(real)", "Night val\\n(real, day-only)", "Night val\\n(real, fine-tuned)"]
f1s = [veh_day_score.f1, veh_night_score.f1, veh_finetuned_score.f1]
bars = ax.bar(names, f1s, color=["#4C78A8", "#E45756", "#54A24B"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1")
ax.set_title("vehicle_of_interest: same collapse-and-remedy mechanism, on real footage")
for b, v in zip(bars, f1s, strict=True):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Takeaway

Read this against `01_domain_gap_evidence.ipynb` rather than in isolation: the same
collapse-then-partial-recovery shape showing up on real, independently-sourced drone
footage is evidence the synthetic mechanism generalizes, not just an artifact of a renderer
built to make a point. This trained `veh_finetuned_model` is also what
`03_model_comparison_bakeoff.ipynb` compares against stock YOLO11n.
"""
    ),
]

# --- 03: model comparison bakeoff -------------------------------------------

nb03 = [
    md(
        """
# 03 — Model bakeoff: this project's lightweight detector vs. YOLO11n

This notebook compares this project's own `GridDetector` against a fine-tuned YOLO11n on
`drone`/`dismount`/`launch_flash` (the synthetic corridor scene) — the harder of the two
comparisons discussed in `docs/METHODOLOGY_AND_LIMITATIONS.md#model-comparison-lightweight-vs-production`,
since COCO has no classes matching this scenario's taxonomy, so a fair comparison requires
fine-tuning YOLO first rather than using it zero-shot.

The other comparison that document discusses — `vehicle_of_interest` on real UAVDT frames,
where stock COCO-pretrained YOLO already knows `car`/`truck`/`bus` and wins for free — isn't
re-run here to keep this notebook's runtime reasonable; it uses the same
`detectors.yolo_adapter.YOLOAdapter` / `COCO_TO_VEHICLE_SUBTYPE` machinery demonstrated in
`02_real_data_validation.ipynb`.

Requires `ultralytics` (`pip install -e ".[yolo]"`) and a `yolo11n.pt` weights file in the
project root.
"""
    ),
    SETUP_CELL,
    code(
        """
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from infrastructure_overwatch.synthetic import GRID, IMG_SIZE, N_SYNTHETIC_CLASSES, SYNTHETIC_CLASS_NAMES, SyntheticCorridorDataset, random_threats, render_day, render_night
from infrastructure_overwatch.detectors.grid_cnn import GridDetector, GridCNNAdapter, evaluate_grid_detector, train_grid_detector
from infrastructure_overwatch.detectors.yolo_adapter import YOLOAdapter
from infrastructure_overwatch.geometry import iou_xyxy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
    ),
    md("## 6b. Fine-tune our own detector, and a fine-tuned YOLO11n, on the synthetic classes"),
    code(
        """
day_train_ds = SyntheticCorridorDataset(800, domain="day", seed=1)
night_val_ds = SyntheticCorridorDataset(200, domain="night", seed=3)
day_train_loader = DataLoader(day_train_ds, batch_size=32, shuffle=True)

our_model = GridDetector(n_classes=N_SYNTHETIC_CLASSES, grid_h=GRID, grid_w=GRID)
train_grid_detector(our_model, day_train_loader, epochs=20, device=device, quiet=True)

night_small_train_ds = SyntheticCorridorDataset(100, domain="night", seed=99)
night_small_loader = DataLoader(night_small_train_ds, batch_size=16, shuffle=True)
train_grid_detector(our_model, night_small_loader, epochs=15, lr=2e-4, device=device, quiet=True)

our_adapter = GridCNNAdapter(our_model, class_names=SYNTHETIC_CLASS_NAMES, img_w=IMG_SIZE, img_h=IMG_SIZE, thresh=0.4, device=device)
our_score = evaluate_grid_detector(our_model, night_val_ds, IMG_SIZE, IMG_SIZE, device=device)
print(f"Our model, Night val -> F1 {our_score.f1:.2f}")
"""
    ),
    code(
        """
import shutil
from pathlib import Path
from PIL import Image

YOLO_DATA_DIR = Path("../yolo_threat_data")
if YOLO_DATA_DIR.exists():
    shutil.rmtree(YOLO_DATA_DIR)
for split in ["train", "val"]:
    (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yolo_split(split, n, domain, seed):
    for i in range(n):
        rng = np.random.default_rng(seed * 100_000 + i)
        threats = random_threats(rng)
        img, boxes = (render_day(threats, rng) if domain == "day" else render_night(threats, rng))
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(YOLO_DATA_DIR / "images" / split / f"{domain}_{i:04d}.jpg", quality=90)
        lines = []
        for (x0, y0, x1, y1), cls in boxes:
            cx, cy = (x0 + x1) / 2 / IMG_SIZE, (y0 + y1) / 2 / IMG_SIZE
            w, h = (x1 - x0) / IMG_SIZE, (y1 - y0) / IMG_SIZE
            lines.append(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
        (YOLO_DATA_DIR / "labels" / split / f"{domain}_{i:04d}.txt").write_text("\\n".join(lines))


write_yolo_split("train", 400, "day", seed=1001)
write_yolo_split("val", 100, "night", seed=1002)
data_yaml = YOLO_DATA_DIR / "data.yaml"
data_yaml.write_text(
    f"path: {YOLO_DATA_DIR.resolve()}\\ntrain: images/train\\nval: images/val\\nnames: {list(SYNTHETIC_CLASS_NAMES)}\\n"
)
print(f"Wrote synthetic YOLO-format dataset to {YOLO_DATA_DIR.resolve()}")
"""
    ),
    code(
        """
import contextlib, io as _io

from ultralytics import YOLO

yolo_threat_model = YOLO("../yolo11n.pt")
t0 = time.time()
with contextlib.redirect_stdout(_io.StringIO()):
    yolo_threat_model.train(
        data=str(data_yaml.resolve()), epochs=15, imgsz=96, batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False, plots=False, val=True,
        project="../yolo_threat_runs", name="finetune", exist_ok=True,
    )
print(f"YOLO fine-tune done in {time.time()-t0:.1f}s")
yolo_threat_model.save("../yolo_threat_finetuned.pt")
yolo_adapter = YOLOAdapter(model_path="../yolo_threat_finetuned.pt", conf=0.15)
"""
    ),
    code(
        """
def compare_on_val(dataset, n=100):
    ours_tp = ours_fp = ours_fn = 0
    yolo_tp = yolo_fp = yolo_fn = 0
    n = min(n, len(dataset))
    for i in range(n):
        img_t, label = dataset[i]
        gray_img = np.repeat((img_t[0] * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        from infrastructure_overwatch.grid_codec import decode_grid_predictions
        gt_boxes = [(b[0], b[1], b[2], b[3]) for b in decode_grid_predictions(label, IMG_SIZE, IMG_SIZE, GRID, GRID, thresh=0.5, nms_iou=1.1)]

        ours_preds = our_adapter.detect(img_t, f"F{i}")
        yolo_preds = yolo_adapter.detect(gray_img, f"F{i}")

        for preds, which in [(ours_preds, "ours"), (yolo_preds, "yolo")]:
            matched, tp, fp = set(), 0, 0
            for p in preds:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gt_boxes):
                    if j not in matched:
                        v = iou_xyxy(p.xyxy, g)
                        if v > best_iou:
                            best_iou, best_j = v, j
                if best_iou >= 0.3:
                    tp += 1
                    matched.add(best_j)
                else:
                    fp += 1
            fn = len(gt_boxes) - len(matched)
            if which == "ours":
                ours_tp += tp; ours_fp += fp; ours_fn += fn
            else:
                yolo_tp += tp; yolo_fp += fp; yolo_fn += fn

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)

    return {"ours": f1(ours_tp, ours_fp, ours_fn), "yolo": f1(yolo_tp, yolo_fp, yolo_fn)}


threat_compare = compare_on_val(night_val_ds, n=100)
for name, (p, r, f) in threat_compare.items():
    print(f"  {name:5s} -> precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}")

fig, ax = plt.subplots(figsize=(5, 3.5))
names = ["Our model", "YOLO11n\\n(fine-tuned)"]
f1s = [threat_compare["ours"][2], threat_compare["yolo"][2]]
bars = ax.bar(names, f1s, color=["#4C78A8", "#F58518"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1 (Night validation)")
ax.set_title("drone/dismount/launch_flash: our model vs. fine-tuned YOLO11n")
for b, v in zip(bars, f1s, strict=True):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Takeaway

Whether YOLO11n or this project's own small model wins depends on the class, not on "our
model" vs. "YOLO" in the abstract — see `docs/METHODOLOGY_AND_LIMITATIONS.md#model-comparison-lightweight-vs-production`
for the full reasoning, including the vehicle-track comparison where YOLO wins for free.
"""
    ),
]

# --- 04: edge deployment benchmarks ------------------------------------------

nb04 = [
    md(
        """
# 04 — Edge deployment: ONNX export and latency benchmarking

A corridor overwatch sensor is field hardware, not a data-center GPU. This notebook
exports this project's detector to ONNX (fp32 and dynamic int8 quantization) and measures
inference latency for PyTorch eager and ONNX Runtime, on CPU and (if available) GPU. See
`docs/METHODOLOGY_AND_LIMITATIONS.md#edge-deployment`.

Requires the `onnx` extra: `pip install -e ".[onnx]"`.
"""
    ),
    SETUP_CELL,
    code(
        """
import torch
from torch.utils.data import DataLoader

from infrastructure_overwatch.synthetic import GRID, IMG_SIZE, N_SYNTHETIC_CLASSES, SyntheticCorridorDataset
from infrastructure_overwatch.detectors.grid_cnn import GridDetector, train_grid_detector
from infrastructure_overwatch.export import benchmark_onnx, benchmark_torch, export_onnx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
day_train_ds = SyntheticCorridorDataset(400, domain="day", seed=1)
loader = DataLoader(day_train_ds, batch_size=32, shuffle=True)
model = GridDetector(n_classes=N_SYNTHETIC_CLASSES, grid_h=GRID, grid_w=GRID)
train_grid_detector(model, loader, epochs=10, device=device, quiet=True)
print("Trained a quick model for benchmarking (accuracy isn't the point of this notebook).")
"""
    ),
    code(
        """
input_shape = (1, IMG_SIZE, IMG_SIZE)
exported = export_onnx(model, input_shape, out_dir="../outputs/onnx_artifacts", name="corridor_detector")
print(f"fp32: {exported.fp32_size_kb:.1f} KB   int8: {exported.int8_size_kb:.1f} KB "
      f"({100*(1 - exported.int8_size_kb/exported.fp32_size_kb):.0f}% smaller)")
"""
    ),
    code(
        """
torch_cpu_latency = benchmark_torch(model, input_shape, device=torch.device("cpu"))
onnx_fp32_latency = benchmark_onnx(exported.fp32_path, input_shape)
onnx_int8_latency = benchmark_onnx(exported.int8_path, input_shape)

print(f"PyTorch eager (CPU):     {torch_cpu_latency:.3f} ms/inference")
print(f"ONNX Runtime fp32 (CPU): {onnx_fp32_latency:.3f} ms/inference")
print(f"ONNX Runtime int8 (CPU): {onnx_int8_latency:.3f} ms/inference")

if torch.cuda.is_available():
    torch_gpu_latency = benchmark_torch(model, input_shape, device=torch.device("cuda"))
    print(f"PyTorch eager (GPU, {torch.cuda.get_device_name(0)}): {torch_gpu_latency:.3f} ms/inference")
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        onnx_gpu_latency = benchmark_onnx(exported.fp32_path, input_shape, provider="CUDAExecutionProvider")
        print(f"ONNX Runtime fp32 (GPU): {onnx_gpu_latency:.3f} ms/inference")
    else:
        print("onnxruntime has no CUDAExecutionProvider installed -- GPU ONNX latency not measured.")
else:
    print("No CUDA GPU available on this machine -- GPU latency not measured.")
"""
    ),
    md(
        """
## Takeaway

Read the int8-vs-fp32 numbers above the way `docs/METHODOLOGY_AND_LIMITATIONS.md` insists
on: quantization's storage win is close to guaranteed, its *latency* win depends on model
scale and this runtime's kernel support on this hardware -- report what was actually
measured here, not the generic "quantization is faster" assumption.
"""
    ),
]

# --- 05: calibration and triage ---------------------------------------------

nb05 = [
    md(
        """
# 05 — Confidence calibration and the analyst review queue

Every detection this pipeline produces routes to a human analyst. This notebook fits an
isotonic calibration from raw detector confidence to empirical true-positive rate, then
routes detections into `auto_confirm` / `analyst_review` / `auto_discard` bands. See
`docs/METHODOLOGY_AND_LIMITATIONS.md#calibration-and-analyst-review-queue` and
`docs/USER_MANUAL.md` for what each band means operationally.
"""
    ),
    SETUP_CELL,
    code(
        """
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from infrastructure_overwatch.synthetic import GRID, IMG_SIZE, N_SYNTHETIC_CLASSES, SyntheticCorridorDataset
from infrastructure_overwatch.detectors.grid_cnn import GridDetector, train_grid_detector
from infrastructure_overwatch.grid_codec import decode_grid_predictions
from infrastructure_overwatch.geometry import iou_xyxy
from infrastructure_overwatch.calibration import ConfidenceCalibrator, TriageThresholds, reliability_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
day_train_ds = SyntheticCorridorDataset(800, domain="day", seed=1)
night_val_ds = SyntheticCorridorDataset(300, domain="night", seed=3)
loader = DataLoader(day_train_ds, batch_size=32, shuffle=True)
model = GridDetector(n_classes=N_SYNTHETIC_CLASSES, grid_h=GRID, grid_w=GRID)
train_grid_detector(model, loader, epochs=20, device=device, quiet=True)
model.eval().to(device)
print("Trained a detector to calibrate.")
"""
    ),
    code(
        """
def collect_confidence_labels(model, dataset, n=300, thresh=0.1, iou_thresh=0.3):
    confs, is_tp = [], []
    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            img, label = dataset[i]
            pred = model(torch.as_tensor(img).unsqueeze(0).to(device))[0].cpu()
            pred_boxes = decode_grid_predictions(pred, IMG_SIZE, IMG_SIZE, GRID, GRID, thresh=thresh)
            gt_boxes = decode_grid_predictions(label, IMG_SIZE, IMG_SIZE, GRID, GRID, thresh=0.5, nms_iou=1.1)
            matched = set()
            for pb in sorted(pred_boxes, key=lambda b: -b[4]):
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    if j in matched:
                        continue
                    v = iou_xyxy(pb[:4], gb[:4])
                    if v > best_iou:
                        best_iou, best_j = v, j
                confs.append(pb[4])
                if best_iou >= iou_thresh:
                    is_tp.append(1)
                    matched.add(best_j)
                else:
                    is_tp.append(0)
    return np.array(confs), np.array(is_tp)


confs, is_tp = collect_confidence_labels(model, night_val_ds)
print(f"Collected {len(confs)} predicted boxes ({is_tp.sum()} true positives, {len(is_tp)-is_tp.sum()} false positives)")
"""
    ),
    code(
        """
calibrator = ConfidenceCalibrator().fit(confs, is_tp)
calibrated = calibrator.calibrate(confs)

raw_x, raw_y = reliability_curve(confs, is_tp)
cal_x, cal_y = reliability_curve(calibrated, is_tp)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, x, y, title in [(axes[0], raw_x, raw_y, "Raw model confidence"), (axes[1], cal_x, cal_y, "Isotonic-calibrated confidence")]:
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.plot(x, y, "o-", color="#4C78A8")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical precision")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
"""
    ),
    md("## Triage routing, and revisiting the discard threshold"),
    code(
        """
for thresholds_label, thresholds in [("untuned default (0.20)", TriageThresholds(discard=0.20)),
                                      ("recommended (0.05)", TriageThresholds(discard=0.05))]:
    n_total = len(calibrated)
    confirm_mask = calibrated >= thresholds.auto_confirm
    discard_mask = calibrated < thresholds.discard
    review_mask = ~confirm_mask & ~discard_mask
    print(f"discard_thresh={thresholds_label}:")
    print(f"  auto_confirm:   {int(confirm_mask.sum()):4d}  ({int(is_tp[confirm_mask].sum())} true positives)")
    print(f"  analyst_review: {int(review_mask.sum()):4d}  ({int(is_tp[review_mask].sum())} true positives)")
    print(f"  auto_discard:   {int(discard_mask.sum()):4d}  ({int(is_tp[discard_mask].sum())} true positives, i.e. missed threats)")
"""
    ),
    md(
        """
## Takeaway

The discard-threshold comparison above is exactly the trade `docs/USER_MANUAL.md` explains:
lowering `discard` recovers real threats that were previously falling into the auto-discard
band, at the cost of a larger analyst review volume. `calibration.TriageThresholds` is
where a real deployment tunes this trade for its own operating environment.
"""
    ),
]


# --- 06: analyst reporting and embedding-based anomaly detection -----------

nb06 = [
    md(
        """
# 06 — Analyst reporting, and embedding-based anomaly detection

Two lighter-weight Phase 2 capabilities, demonstrated together since neither needs a long
training run:

1. **`reporting.py`** — turns `pipeline.PipelineResult`'s alert/event tables into a
   Seaborn report-card figure and a Plotly interactive dashboard, for a shift-handoff
   briefing or a live monitoring view.
2. **`anomaly.py`** — a nearest-neighbor embedding distance (via a HOG embedder here, no
   model download required) that flags imagery unlike anything in a reference gallery of
   "normal" corridor scenes, independent of the four-class detector taxonomy.

Requires the `dataviz` and `anomaly` extras: `pip install -e ".[dataviz,anomaly]"`.
"""
    ),
    SETUP_CELL,
    md("## Part 1 — Reporting, from a real (if quick) pipeline run"),
    code(
        """
import numpy as np
import torch
from torch.utils.data import DataLoader

from infrastructure_overwatch.synthetic import GRID, IMG_SIZE, N_SYNTHETIC_CLASSES, PROTECTED_ZONE, SYNTHETIC_CLASS_NAMES, SyntheticCorridorDataset, render_corridor_sequence
from infrastructure_overwatch.detectors.grid_cnn import GridDetector, GridCNNAdapter, train_grid_detector
from infrastructure_overwatch.calibration import ConfidenceCalibrator, DEFAULT_TRIAGE_THRESHOLDS
from infrastructure_overwatch.tracking import MultiTracker
from infrastructure_overwatch.events import PipelineEventEngine
from infrastructure_overwatch.pipeline import run_frame_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A quick detector -- this notebook is about reporting, not detector accuracy.
day_train_ds = SyntheticCorridorDataset(400, domain="day", seed=1)
loader = DataLoader(day_train_ds, batch_size=32, shuffle=True)
model = GridDetector(n_classes=N_SYNTHETIC_CLASSES, grid_h=GRID, grid_w=GRID)
train_grid_detector(model, loader, epochs=15, device=device, quiet=True)
adapter = GridCNNAdapter(model, class_names=SYNTHETIC_CLASS_NAMES, img_w=IMG_SIZE, img_h=IMG_SIZE, thresh=0.4, device=device)
print("Trained a quick detector for this notebook's demo run.")
"""
    ),
    code(
        """
rng = np.random.default_rng(11)
sequence = render_corridor_sequence(100, rng, domain="night", n_threats=2)
frames = [np.transpose(img, (2, 0, 1)).mean(axis=0, keepdims=True).astype("float32") for img, _boxes, _ids in sequence]

result = run_frame_sequence(
    frames=frames, detector=adapter, protected_zone=PROTECTED_ZONE, fps=10.0,
    tracker=MultiTracker(), event_engine=PipelineEventEngine(protected_zone=PROTECTED_ZONE),
    calibrator=None, triage_thresholds=DEFAULT_TRIAGE_THRESHOLDS,
)
alerts_df = result.alerts_frame()
events_df = result.events_frame()
print(f"{len(alerts_df)} detections, {len(events_df)} events")
"""
    ),
    code(
        """
from infrastructure_overwatch.reporting import build_report_card, build_dashboard

fig = build_report_card(alerts_df, events_df, out_path="../outputs/notebook06_report_card.png")
import matplotlib.pyplot as plt
plt.show()

dash = build_dashboard(alerts_df, events_df, out_path="../outputs/notebook06_dashboard.html")
print("Wrote outputs/notebook06_report_card.png and outputs/notebook06_dashboard.html")
"""
    ),
    md(
        """
## Part 2 — Embedding-based anomaly detection

Fit a reference gallery from Day-domain corridor renders (the "normal" case), then score
Day imagery (should look familiar) against Night imagery (a genuinely different visual
domain) to see the anomaly-distance mechanism actually separate them.
"""
    ),
    code(
        """
from infrastructure_overwatch.anomaly import EmbeddingAnomalyScorer, HOGEmbedder
from infrastructure_overwatch.synthetic import random_threats, render_day, render_night

def render_uint8(render_fn, seed):
    rng = np.random.default_rng(seed)
    threats = random_threats(rng)
    img, _boxes = render_fn(threats, rng)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

gallery_images = [render_uint8(render_day, seed) for seed in range(1, 31)]
day_val_images = [render_uint8(render_day, seed) for seed in range(100, 110)]
night_val_images = [render_uint8(render_night, seed) for seed in range(100, 110)]

scorer = EmbeddingAnomalyScorer(HOGEmbedder(size=(96, 96))).fit(gallery_images)
threshold = scorer.calibrate_threshold([render_uint8(render_day, seed) for seed in range(200, 220)], percentile=95.0)
print(f"Calibrated threshold (95th percentile of held-out Day scores): {threshold:.4f}")

day_scores = [s.score for s in scorer.score(day_val_images)]
night_scores = [s.score for s in scorer.score(night_val_images)]
print(f"Day   val anomaly score: mean {np.mean(day_scores):.4f}")
print(f"Night val anomaly score: mean {np.mean(night_scores):.4f}")
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(day_scores, bins=10, alpha=0.6, label="Day (familiar domain)", color="#4C78A8")
ax.hist(night_scores, bins=10, alpha=0.6, label="Night (novel domain)", color="#E45756")
ax.axvline(threshold, color="black", linestyle="--", label="calibrated threshold")
ax.set_xlabel("anomaly score (cosine distance to nearest gallery embedding)")
ax.set_ylabel("count")
ax.legend()
ax.set_title("A HOG-embedding gallery of Day imagery separates Day from Night")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Takeaway

The anomaly scorer is deliberately *not* a replacement for the class-based detectors --
it has no notion of "drone" or "vehicle," only "unlike the reference gallery." Read a high
anomaly score as "worth a human glance because it's visually unusual," not as a threat
classification; see `docs/USER_MANUAL.md` and `docs/ARCHITECTURE.md` for how this
complements, rather than replaces, the four-class taxonomy.
"""
    ),
]


# --- 07: free-text alert-note triage (LoRA fine-tuning) ---------------------

nb07 = [
    md(
        """
# 07 — Free-text alert-note triage: LoRA fine-tuning end to end

`triage_nlp.NoteTriageClassifier` triages an analyst's free-text note into `NOTE_CATEGORIES`
(`confirmed_threat`/`false_alarm`/`sensor_or_equipment_issue`/`needs_more_information`) using
a small transformer (`distilbert-base-uncased`) with a LoRA adapter (parameter-efficient
fine-tuning, not a full fine-tune). Until this notebook, this module had never been
exercised end to end -- see `docs/VALIDATION.md` and
`docs/METHODOLOGY_AND_LIMITATIONS.md#free-text-alert-note-triage-optional`. This notebook
downloads the real pretrained base model, fine-tunes a LoRA adapter on a small hand-written,
synthetic-but-plausible labeled dataset of analyst notes, saves the adapter to
`outputs/weights/note_triage_adapter`, and measures held-out accuracy -- closing that gap
with a real (if small) measured number rather than an unexercised mechanism.

Requires the `nlp` extra: `pip install -e ".[nlp]"` (downloads `distilbert-base-uncased`
from the Hugging Face Hub on first run).
"""
    ),
    SETUP_CELL,
    md(
        """
## A small, hand-written labeled dataset

40 short analyst notes, 10 per category, written to be representative of what an analyst
might actually type against an alert -- not scraped or auto-generated. Split 8 train / 2
validation per category (stratified) so the accuracy measured below is on notes the model
never trained on, not a training-set score.
"""
    ),
    code(
        """
NOTES_BY_CATEGORY = {
    "confirmed_threat": [
        "Visual confirmed on second camera, subject carrying a weapon, escalating to response team.",
        "Thermal signature matches known threat profile, tracked crossing the perimeter fence, notifying QRF.",
        "Multiple witnesses report an armed intruder near the north gate, detection corroborated by patrol.",
        "Drone visually identified, flight path directly toward the restricted zone, confirmed hostile intent.",
        "Ground team confirms an unauthorized vehicle breached the checkpoint, occupants uncooperative, treat as hostile.",
        "Secondary sensor cross-check confirms the detection, subject matches the wanted bulletin, escalate immediately.",
        "Launch flash detection corroborated by the acoustic sensor, confirmed real event, not a glare artifact.",
        "Patrol visually confirmed the dismount is carrying a rifle, this is a real threat.",
        "Two independent cameras track the same subject climbing the fence, confirmed intrusion in progress.",
        "Analyst and patrol both confirm a hostile actor in the corridor, requesting immediate response.",
    ],
    "false_alarm": [
        "Reviewed the footage, this was wind-blown debris, not a real target, closing the alert.",
        "Confirmed false positive, a shadow from cloud movement triggered the detector.",
        "A bird flock crossed the frame, no actual threat present, stand down.",
        "Analyst review shows this was a stray dog, not a dismount, false alarm.",
        "Glare off a vehicle windshield caused the trigger, nothing there, closing out.",
        "Reviewed the clip, it's a plastic bag caught in the wind, no threat.",
        "Second look confirms this was foliage movement, not a person, disregard.",
        "This was our own maintenance vehicle on a scheduled patrol, not an intruder.",
        "Rain streaking across the lens triggered the alert, no object actually present.",
        "Confirmed this is livestock wandering near the fence line, not a person, false alarm.",
    ],
    "sensor_or_equipment_issue": [
        "Camera 3 feed is frozen again, needs a reboot, can't confirm anything from this alert.",
        "Detector keeps flagging the same coordinates every night, looks like a stuck pixel.",
        "Thermal camera lens is fogged over, image quality too poor to assess.",
        "Sensor calibration drifted after the storm, false triggers spiking since last night.",
        "Gimbal appears stuck at one azimuth, the feed is not actually panning, ticket filed.",
        "Network link to the north tower camera is dropping frames intermittently.",
        "IR illuminator failed, the night feed is unusable until it's replaced.",
        "A firmware update seems to have broken the bounding box overlay, visually garbled.",
        "Camera 5 has a hardware fault, image is corrupted with horizontal banding.",
        "Motion detector is miscalibrated after the lens swap, generating alerts on empty frames.",
    ],
    "needs_more_information": [
        "Can't tell from this angle if it's a person or equipment, need a second camera.",
        "Image is too dark to make a call, requesting a daylight recheck.",
        "Partial occlusion behind the treeline, unclear what triggered the alert.",
        "Need the track history before deciding, this could be the same false alarm as earlier.",
        "Not enough context in a single frame, want the full clip before triaging.",
        "Ambiguous shape at long range, requesting a closer pass from patrol to confirm.",
        "Ops center audio was garbled, unclear if this matches a reported event, need more input.",
        "Ground truth is unclear, waiting on the analyst supervisor's confirmation before categorizing.",
        "Object is at the edge of sensor range, resolution too low to classify confidently.",
        "Conflicting reports from two patrols, need clarification before this can be triaged.",
    ],
}

train_texts, train_labels, val_texts, val_labels = [], [], [], []
for category, notes in NOTES_BY_CATEGORY.items():
    train, val = notes[:8], notes[8:]
    train_texts += train
    train_labels += [category] * len(train)
    val_texts += val
    val_labels += [category] * len(val)

print(f"{len(train_texts)} train notes, {len(val_texts)} validation notes, "
      f"{len(NOTES_BY_CATEGORY)} categories")
"""
    ),
    md("## Load the base model and wrap it with a LoRA adapter"),
    code(
        """
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from infrastructure_overwatch.triage_nlp import NOTE_CATEGORIES, apply_lora, train_note_triage

torch.manual_seed(0)
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(NOTE_CATEGORIES))
lora_model = apply_lora(base_model)
lora_model.print_trainable_parameters()
"""
    ),
    md("## Fine-tune the LoRA adapter on the training split"),
    code(
        """
t0 = time.time()
loss_history = train_note_triage(lora_model, tokenizer, train_texts, train_labels, epochs=30, lr=2e-4, batch_size=8)
print(f"Trained in {time.time()-t0:.1f}s. Final epoch loss: {loss_history[-1]:.4f}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(range(1, len(loss_history) + 1), loss_history, "o-", color="#4C78A8")
ax.set_xlabel("epoch")
ax.set_ylabel("mean training loss")
ax.set_title("LoRA fine-tune on 32 synthetic analyst notes")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Held-out accuracy, then save the adapter

Evaluate on the 8 validation notes the model never trained on, then save the LoRA adapter to
`outputs/weights/note_triage_adapter` and reload it through `NoteTriageClassifier` -- the same
adapter-loading path a real caller would use -- as a round-trip check that the saved adapter
actually reproduces these predictions.
"""
    ),
    code(
        """
lora_model.eval().to("cpu")
encoded = tokenizer(val_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    val_logits = lora_model(**encoded).logits
val_preds = [NOTE_CATEGORIES[i] for i in val_logits.argmax(dim=-1).tolist()]
val_correct = sum(p == t for p, t in zip(val_preds, val_labels, strict=True))
val_accuracy = val_correct / len(val_labels)
print(f"Validation accuracy: {val_correct}/{len(val_labels)} = {val_accuracy:.3f}")

for text, pred, gold in zip(val_texts, val_preds, val_labels, strict=True):
    mark = "OK " if pred == gold else "ERR"
    print(f"  [{mark}] pred={pred:28s} gold={gold:28s} {text[:60]}")
"""
    ),
    code(
        """
ADAPTER_DIR = ROOT / "outputs" / "weights" / "note_triage_adapter"
ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)
lora_model.save_pretrained(str(ADAPTER_DIR))
print(f"Saved LoRA adapter to {ADAPTER_DIR}")

from infrastructure_overwatch.triage_nlp import NoteTriageClassifier

reloaded = NoteTriageClassifier(adapter_path=str(ADAPTER_DIR))
reloaded_results = reloaded.predict(val_texts)
reloaded_preds = [r.category for r in reloaded_results]
assert reloaded_preds == val_preds, "reloaded adapter should reproduce the in-memory model's predictions"
print("Reloaded adapter from disk reproduces the same validation predictions.")
"""
    ),
    md(
        """
## Takeaway

Fine-tuning a LoRA adapter (~1% of the base model's parameters) on just 32 hand-written
training notes drives training loss from ~1.4 to well under 0.01, and the adapter generalizes
to held-out notes far better than chance (4 categories, so 25% is the no-signal floor) --
see the accuracy printed above for this run's actual number, and
`docs/METHODOLOGY_AND_LIMITATIONS.md#free-text-alert-note-triage-optional` for how to read
it. This is a small hand-written dataset, not a production-scale labeled corpus: read the
measured accuracy as "the mechanism works end to end," not as a validated production
accuracy rate -- the honest next step for real deployment is real analyst-labeled notes at
far greater volume.
"""
    ),
]


ALL_NOTEBOOKS = {
    "01": ("01_domain_gap_evidence.ipynb", nb01),
    "02": ("02_real_data_validation.ipynb", nb02),
    "03": ("03_model_comparison_bakeoff.ipynb", nb03),
    "04": ("04_edge_deployment_benchmarks.ipynb", nb04),
    "05": ("05_calibration_and_triage.ipynb", nb05),
    "06": ("06_reporting_and_anomaly_detection.ipynb", nb06),
    "07": ("07_note_triage.ipynb", nb07),
}

if __name__ == "__main__":
    import sys

    NB_DIR.mkdir(parents=True, exist_ok=True)
    selected = sys.argv[1:] or list(ALL_NOTEBOOKS)
    for key in selected:
        name, cells = ALL_NOTEBOOKS[key]
        path = save(name, cells)
        print(f"Wrote {path}")
