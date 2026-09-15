# Development

## Setup

```bash
uv venv
uv pip install -e ".[dev,onnx,anomaly,dataviz,web]"   # what CI installs
# add `yolo` for the Ultralytics backend, `nlp` for the LoRA note-triage classifier
```

Requires Python 3.12+. Torch/torchvision install from PyPI by default; on a CUDA-capable
machine, install a CUDA build separately if you want GPU acceleration (`pip install torch
--index-url https://download.pytorch.org/whl/cu126`, matching your driver's CUDA version).

## Running the checks CI runs

```bash
ruff check .
ruff format --check .
mypy src
pytest -v
```

`pytest`, `ruff`, and `mypy` all run without GPU and without the optional YOLO/UAVDT/NLP
dependencies, in well under a minute — this covers geometry, grid encode/decode,
calibration, tracking (including the `min_hits` confirm-gate), event logic, evaluation
(framewise P/R/F1 and COCO-style mAP/PR-curve/calibration-reliability), HOG-based anomaly
scoring and its pipeline wiring, reporting, generic video/image-folder ingestion, N-sensor
event fusion, the job store, and the FastAPI operator-console service end to end via
`TestClient` (`web`/`httpx`) — none of which need a GPU or a model download. `triage_nlp.py`'s
tests only cover its pure-Python parts (`NOTE_CATEGORIES`, `TriageResult`) for the same
reason ONNX/YOLO are kept out of the required dependency set — downloading real
transformer weights isn't something CI should depend on.

## Training and running the demo

```bash
python -m infrastructure_overwatch train-synthetic --epochs 20 --n-train 800
python -m infrastructure_overwatch demo --domain night --calibrate
```

`train-synthetic` trains the grid-CNN detector on the synthetic corridor renderer (no
external data needed) and reports Day/Night validation F1. `demo` generates a synthetic
moving-threat sequence, runs the full detect → track → event → calibrate pipeline, and
writes `outputs/alerts.csv` / `outputs/events.csv` / `outputs/annotated_demo.gif`
(`--no-video` to skip the last one). `report` (requires the `dataviz` extra) turns the two
CSVs (plus the GIF, if present) into `outputs/report_card.png` and `outputs/dashboard.html`:

```bash
python -m infrastructure_overwatch report
```

## Training and running the demo on real footage

A second, parallel pair of commands runs the identical detect → track → event → calibrate
pipeline against real UAVDT/VisDrone footage instead of the synthetic renderer, for the
`vehicle_of_interest` (and, with `--class-scheme vehicle_dismount`, `dismount`) track --
see "Obtaining UAVDT" / "Obtaining VisDrone" below for the local data these need:

```bash
python -m infrastructure_overwatch train-real --epochs 25
python -m infrastructure_overwatch demo-real --source "uavdt:M0601" --calibrate
python -m infrastructure_overwatch report --video outputs/annotated_demo.webm
```

`train-real` trains on UAVDT alone by default, and reports Day/Night validation F1 on the
same `UAVDT_DAY_VAL_SEQS`/`UAVDT_NIGHT_VAL_SEQS` split `notebooks/02_real_data_validation.ipynb`
and `notebooks/08_visdrone_augmentation.ipynb` use, so a number measured here is directly
comparable to what's already written up in `docs/METHODOLOGY_AND_LIMITATIONS.md`. Pass
`--visdrone` (on by default when VisDrone is available; `--no-visdrone` to opt out) to
combine VisDrone2019-DET + VisDrone2019-VID with UAVDT for training, matching
`notebooks/08_visdrone_augmentation.ipynb`'s measured improvement.

`demo-real --source` takes `<uavdt|visdrone-vid|video|folder>:<sequence-or-path>` --
`uavdt:M0601` or `visdrone-vid:uav0000086_00000_v` for a named benchmark sequence (with
ground truth, so `--calibrate` and accuracy scoring apply), or `video:C:\clips\
corridor.mp4` / `folder:C:\clips\frames` for an arbitrary local video file or image
folder (no ground truth -- `--calibrate` prints a warning and is a no-op, but detection/
tracking/events/alerts run the same as any other source; see `ingest.load_video_frames`/
`load_image_folder_frames`). `UAVDT_NIGHT_VAL_SEQS` (in `cli.py`) and
`VisDroneVIDIndex.list_sequences("val")` list named-sequence candidates not used in
`train-real`'s own training split. Its `--zone` defaults to an illustrative placeholder
(`runs.DEFAULT_PROTECTED_ZONE`) -- none of the four source kinds are shot around an actual
perimeter, so there is no real protected-zone geometry to read off the data; pass `--zone
X0 Y0 X1 Y1` (in the 640x352 working canvas) for anything that should mean something for a
specific sequence. `--track-min-hits N` (default 1, unchanged behavior) requires a track to
be seen `N` times before it can raise a zone/loiter event -- raise it (e.g. 3) for a dense
real-world scene, where a single-hit track is usually detector noise, not a real object;
see `docs/METHODOLOGY_AND_LIMITATIONS.md#tracking`. `--anomaly-ref <path>` (a gallery saved
by `fit-anomaly-reference`) scores mature tracks against it and adds any `VISUAL_ANOMALY`
events it raises.

`demo-real` writes an annotated **WebM video**, not a GIF: a real photographic frame has
far more distinct colors than a GIF's 256-color palette can hold (measured directly against
this project's own UAVDT frames -- see `viz.py`'s module docstring and `docs/VALIDATION.md`
for the numbers), so `report` needs the `--video outputs/annotated_demo.webm` flag to embed
it (its default still looks for `outputs/annotated_demo.gif`, matching the synthetic path).

The identical `detect -> track -> event -> calibrate -> anomaly` pipeline this section runs
from the CLI is also reachable from a browser: `python -m infrastructure_overwatch serve`
(needs the `web` extra) starts a local operator console over the same `runs.
run_real_pipeline` this section's commands call -- see
[docs/OPERATOR_CONSOLE.md](OPERATOR_CONSOLE.md).

## Obtaining UAVDT (for the real `vehicle_of_interest` track)

`ingest.UAVDTIndex` reads a local copy of the UAVDT benchmark (Du et al., ECCV 2018):

1. Download the UAVDT-M detection benchmark (frames + `M_attr` attribute files +
   `UAV-benchmark-MOTD_v1.0` ground truth) from the dataset's official distribution.
2. Point the project at it either by setting `UAVDT_ROOT` in your environment, or passing
   `root=...` explicitly: `UAVDTIndex(root=r"D:\FMV\UAVDT\raw")`.
3. `UAVDTIndex.build_sequence_index()` lists available sequences with their
   daylight/night/fog/altitude attributes; `ingest.UAVDTVehicleDataset` turns a chosen
   sequence list into a training/eval dataset for `detectors.grid_cnn`.

This is an optional, machine-local capability — nothing in the default `train-synthetic` /
`demo` path or in CI depends on it.

## Obtaining VisDrone (for augmenting UAVDT, and a real-data track for `dismount`)

This project's real-data ingestion layer is deliberately built to combine an arbitrary
number of independent real-data sources for the same track, not just one — a pipeline
wired to a single sensor/dataset is a much weaker validation of "this generalizes" than
several independently-collected sources agreeing. UAVDT (drone-shot, MOT-style sequences)
was the first source; VisDrone adds two more, in two different formats, both optional and
machine-local like UAVDT:

- **VisDrone2019-DET** (`ingest.VisDroneDETIndex` / `VisDroneDETVehicleDataset`) — single
  images, one annotation file per image. The simpler of the two to work with.
- **VisDrone2019-VID** (`ingest.VisDroneVIDIndex` / `VisDroneVIDVehicleDataset`) — drone
  video, structured as UAVDT already is (a folder of numbered frames plus one whole-sequence
  ground-truth file per sequence), so it's the closer match to UAVDT's own real-video
  character rather than DET's single unrelated photos.

Both tasks share the same underlying VisDrone category taxonomy and are worth having *both*
of, not one instead of the other — DET and VID were collected differently (unrelated stills
vs. continuous flight sequences), so together they're a broader, more independent real-data
signal than either alone. All three real sources (UAVDT, VisDrone-DET, VisDrone-VID) can be
mixed in any combination via `torch.utils.data.ConcatDataset`, because they all produce the
same letterboxed-canvas-plus-grid-label shape (`WORK_W`/`WORK_H`,
`VEHICLE_GRID_W`/`VEHICLE_GRID_H`) under the same `class_scheme` convention (below). Adding a
*fourth* real source later (a different sensor, a different country's drone-traffic
benchmark, whatever it turns out to be) means writing one more `Index`/`VehicleDataset` pair
that honors this same shape — not touching the other three.

1. Download VisDrone2019-DET and/or VisDrone2019-VID train/val splits from the dataset's
   official distribution (GoogleDrive/BaiduYun links — there's no scriptable direct
   download). Both live under one shared root:
   - DET unpacks to `VisDrone2019-DET-{train,val,test-dev}/`, each with `images/` and
     `annotations/`.
   - VID unpacks to `VisDrone2019-VID-{train,val}/VisDrone2019-VID-{train,val}/` (the
     doubled directory name is the official archive's own layout — `VisDroneVIDIndex`
     tolerates a flattened layout too), each containing `sequences/<seq>/<7-digit-frame>.jpg`
     and `annotations/<seq>.txt`.
2. Point the project at the shared parent directory either by setting `VISDRONE_ROOT` in
   your environment, or passing `root=...` explicitly to either index class, e.g.
   `VisDroneVIDIndex(root=r"D:\FMV\VisDrone")`.
3. `VisDroneVIDIndex.list_sequences(split)` lists available sequences (there is no
   day/night/fog attribute file like UAVDT's `M_attr`, since VisDrone doesn't publish one);
   pass a chosen list to `VisDroneVIDVehicleDataset(index, split, seq_list, ...)`, mirroring
   `UAVDTVehicleDataset`'s own shape exactly. `VisDroneDETVehicleDataset(index, split, ...)`
   takes no sequence list — DET has no sequence structure to choose from.

**Two class schemes**, both defined by `ingest.class_names_for_scheme`:

- `"vehicle_only"` (default) — `("car", "truck", "bus")`, matching UAVDT's native 3
  categories. Use this to augment the existing vehicle-only real-data track with more
  vehicle examples; VisDrone's pedestrian/people boxes are dropped.
- `"vehicle_dismount"` — adds a 4th `"dismount"` class, populated from VisDrone's
  `pedestrian`/`people` categories (UAVDT frames simply never populate this channel, which
  is correct — UAVDT has no person ground truth to contribute). This trains a 4-class
  real-vehicle-track detector rather than the synthetic corridor scene's own `dismount`
  class — a separate model on a separate working canvas, not a change to
  `detectors.grid_cnn`'s synthetic-scene training path.

`UAVDTVehicleDataset`, `VisDroneDETVehicleDataset`, and `VisDroneVIDVehicleDataset` all take
a `class_scheme` argument that must match across whichever of them you combine, so their
label tensors share one class-index space. VisDrone categories outside this project's
taxonomy (`van`, `bicycle`, `tricycle`, `awning-tricycle`, `motor`, `others`, and ignored
regions) are dropped by `ingest.visdrone_category_to_class_index` regardless of scheme or
task — the same deliberate scoping as the four synthetic threat classes (see
`docs/ARCHITECTURE.md#why-four-threat-classes`).

Like UAVDT, all of this is an optional, machine-local capability — nothing in the default
`train-synthetic` / `demo` path or in CI depends on it.

## Adding a new detector backend

Every detector implements `detectors.DetectorAdapter`:

```python
class DetectorAdapter(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: str) -> list[Detection]: ...
```

Look at `detectors/classical.py` (no model) or `detectors/yolo_adapter.py` (lazy-imported
external model) as templates. Once your adapter returns `types.Detection` objects, it
plugs into `pipeline.run_frame_sequence` without changing tracking, events, or calibration
— that's the point of the shared interface (see `docs/ARCHITECTURE.md`).

## How `notebooks/` relates to `src/`

The notebooks in `notebooks/` are evidence/reproduction notebooks, not the application.
Each one imports from `infrastructure_overwatch` and runs one specific experiment (the
domain-gap collapse, the real-UAVDT validation, the model bakeoff, edge-deployment
benchmarks, calibration, and — in `06_reporting_and_anomaly_detection.ipynb` — the
reporting and anomaly-scoring modules) to reproduce the findings written up in
`docs/METHODOLOGY_AND_LIMITATIONS.md`. If you change detector, calibration, or tracking
logic in `src/`, re-run the relevant notebook and update that doc's numbers rather than
letting the two drift apart. They're generated from `scripts/build_notebooks.py` rather
than hand-edited in Jupyter — edit that script and re-run
`python scripts/build_notebooks.py <NN>` (a two-digit key, e.g. `06`) to regenerate one
notebook without touching the others' already-executed outputs.

## Project origin note

This package was built by extracting and restructuring code and findings from two research
notebooks (`pipeline_defense_scenario.ipynb`, `bah_cv_field_deployment.ipynb`) that are not
part of this repository. If you're comparing behavior against those notebooks: the grid
detector architecture here is a single parametrized `GridDetector` (see
`detectors/grid_cnn.py`) replacing what the notebooks implemented as two separate,
near-duplicate classes — expect matching *behavior*, not an identical class name or file
layout.
