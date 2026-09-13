# Development

## Setup

```bash
uv venv
uv pip install -e ".[dev,onnx,anomaly,dataviz]"   # what CI installs
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
calibration, tracking, event logic, evaluation, HOG-based anomaly scoring, and reporting,
none of which need a GPU or a model download. `triage_nlp.py`'s tests only cover its
pure-Python parts (`NOTE_CATEGORIES`, `TriageResult`) for the same reason ONNX/YOLO are
kept out of the required dependency set — downloading real transformer weights isn't
something CI should depend on.

## Training and running the demo

```bash
python -m infrastructure_overwatch train-synthetic --epochs 20 --n-train 800
python -m infrastructure_overwatch demo --domain night --calibrate
```

`train-synthetic` trains the grid-CNN detector on the synthetic corridor renderer (no
external data needed) and reports Day/Night validation F1. `demo` generates a synthetic
moving-threat sequence, runs the full detect → track → event → calibrate pipeline, and
writes `outputs/alerts.csv` / `outputs/events.csv`. `report` (requires the `dataviz`
extra) turns those two CSVs into `outputs/report_card.png` and `outputs/dashboard.html`:

```bash
python -m infrastructure_overwatch report
```

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
