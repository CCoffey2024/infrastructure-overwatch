# Architecture

## Module map

```
src/infrastructure_overwatch/
  types.py           Detection / Event / FrameRecord data contracts; THREAT_CLASSES, TRIAGE_BANDS
  geometry.py         IoU, point-in-rect, NMS -- dependency-free primitives
  grid_codec.py        encode/decode between (box, class) ground truth and the grid-tensor
                       label format the project's own detector predicts
  synthetic.py          the day/night corridor scene renderer (drone/dismount/launch_flash)
  ingest.py              video -> FrameRecord ingestion; real UAVDT vehicle-data reader
  detectors/
    classical.py          background-subtraction motion detector (no model, no classes)
    grid_cnn.py             this project's own lightweight detector: model, training,
                           evaluation, and a DetectorAdapter wrapper
    yolo_adapter.py          Ultralytics YOLO backend, same adapter interface
    ground_truth.py           replays known annotations, for testing tracking/events
  tracking.py             MultiTracker: greedy IoU + constant-velocity data association
  events.py                PipelineEventEngine: protected-zone entry/stopped/loiter alerts
  calibration.py            isotonic confidence calibration + auto_confirm/review/discard
                           triage routing
  evaluation.py              precision/recall/F1, incl. attribute-sliced error analysis
  export.py                   ONNX export + fp32/int8, CPU/GPU latency benchmarking
  fusion.py                    optional EO/IR late fusion
  anomaly.py                    embedding-based anomaly scoring (HOG / DINOv2), a
                                complementary signal to the class-based detectors
  reporting.py                   Seaborn report-card + Plotly dashboard (charts and
                                alerts/events tables), built from PipelineResult's tables
  viz.py                          draws detections on frames and renders an annotated
                                GIF (synthetic) or WebM video (real footage) of a run --
                                what the dashboard's media panel shows; see viz.py's own
                                docstring for why the format depends on the frame source
  triage_nlp.py                   optional LoRA-fine-tuned free-text alert-note triage
  pipeline.py                       orchestration: wires detector -> tracker -> events ->
                                    calibration into one run
  cli.py                             `train-synthetic` / `demo` / `train-real` /
                                    `demo-real` / `report` entry points
```

## Data flow

```
frame (ndarray)
   │
   ▼
DetectorAdapter.detect(frame, frame_id) -> list[Detection]     (classical | grid_cnn | yolo)
   │
   ▼
MultiTracker.step(detections) -> list[Detection]                (track_id filled in)
   │
   ├──────────────────────────────┐
   ▼                              ▼
PipelineEventEngine.update()   calibration.route_detections()
   -> list[Event]                 -> Detection.calibrated_confidence, .triage_band
   │                              │
   └──────────────┬───────────────┘
                   ▼
         PipelineResult.alerts_frame() / .events_frame()
```

Every stage passes `types.Detection`/`Event` objects, not backend-specific structures --
`pipeline.run_frame_sequence` doesn't know or care whether the detector behind it is a
20-line OpenCV background subtractor or a trained CNN.

## Why four threat classes

A fixed or patrol EO sensor watching a pipeline corridor realistically catches a specific
set of *visual signatures*, not every threat a planner might list:

- **`drone`** — a small UAS/loitering munition. Modeled as a small, compact airborne
  shape: hard to see at range, which is the point of most of the domain-gap work
  in `docs/METHODOLOGY_AND_LIMITATIONS.md`.
- **`dismount`** — an unauthorized person near the fence line or manifold. A narrow,
  upright silhouette; the one class `events.PipelineEventEngine` treats specially for
  loitering (`PERSON_LOITER`), since a stationary vehicle and a loitering person warrant
  different analyst framing.
- **`launch_flash`** — a launch signature (flash + rising smoke), standing in for
  rocket/indirect-fire/stand-off threats. A pipeline-perimeter camera cannot realistically
  track a rocket or missile in flight — it's too fast, too far, and off-axis for most of its
  flight. What a fixed camera *can* catch is the launch signature itself. Modeling "detect
  the missile" would model a sensor capability that doesn't exist at this range; modeling
  "detect the launch flash" models what the sensor actually sees.
- **`vehicle_of_interest`** — an unauthorized or "technical"-type vehicle approaching the
  corridor. The one class with a real-data-backed track (`ingest.UAVDTIndex` /
  `UAVDTVehicleDataset`, from the UAVDT drone-video benchmark) instead of a synthetic
  stand-in, because real labeled vehicle footage already exists.

## Why one adapter interface for every detector

`ClassicalMotionAdapter`, `GridCNNAdapter`, and `YOLOAdapter` all implement
`detect(frame, frame_id) -> list[Detection]` and nothing else. That lets `pipeline.py`,
`tracking.py`, and `events.py` stay entirely detector-agnostic, and lets a reviewer swap or
A/B compare detectors without touching anything downstream — see the model bakeoff in
`docs/METHODOLOGY_AND_LIMITATIONS.md`, which runs the identical tracking/event/calibration
stack against two different detector backends.

## Why one parametrized grid-detector architecture, not two

The source research notebooks this project was built from defined two nearly-identical CNN
architectures: a 96x96/6x6-grid model for the synthetic classes, and a 640x352/40x22-grid
model for real UAVDT frames. `detectors.grid_cnn.GridDetector` takes `(n_classes, grid_h,
grid_w, stage_channels)` as constructor arguments instead, so one implementation serves
both scales. `grid_codec.py`'s encode/decode functions are similarly parametrized over
image and grid size. This is a direct example of what changed converting the original
notebooks into this package: duplicated, scale-specific code became one tested,
parametrized module.

## Why calibration is a separate stage from detection

A raw detector confidence is not a calibrated probability. `calibration.py` fits an
isotonic regression from raw confidence to empirical true-positive rate on a labeled
validation set, then buckets calibrated confidence into `auto_confirm` / `analyst_review` /
`auto_discard`. The threshold choice is a measured operating point, not a default picked
blind — see `docs/METHODOLOGY_AND_LIMITATIONS.md` for the before/after comparison that
justified the current `discard` threshold. This stage is optional in `pipeline.py`
(`calibrator=None` skips it) because calibration requires its own labeled validation run;
it cannot bootstrap itself from a single inference pass.

## Secondary capabilities (Phase 2)

Three modules extend the core detect/track/event/calibrate pipeline without being part of
its critical path — each behind its own optional dependency extra, and each lazy-importing
its heavy dependencies so the core package has no hard dependency on any of them:

- **`anomaly.py`** (`pip install -e ".[anomaly]"`) — `EmbeddingAnomalyScorer` fits a
  reference gallery of "normal" imagery embeddings, then flags new imagery by cosine
  distance to the nearest gallery embedding. Two embedder backends: `HOGEmbedder`
  (offline, no model download, what CI exercises) and `DinoV2Embedder` (self-supervised
  DINOv2 features via `torch.hub`, requires network access on first use). This is a
  complementary signal to the four-class detectors, not a replacement — it has no notion
  of "drone" or "vehicle," only "unlike anything the gallery has seen," which is useful
  for novel visual patterns the fixed taxonomy wouldn't otherwise catch.
- **`reporting.py`** (`pip install -e ".[dataviz]"`) — `build_report_card` (a static
  Seaborn figure) and `build_dashboard` (an interactive Plotly dashboard: the same four
  summary charts, plus an alerts table and an events table, plus — if `demo`/`demo-real
  --video` wrote one — an embedded annotated GIF or video of the run), both built from
  `pipeline.PipelineResult.alerts_frame()` / `.events_frame()`. Wired into the CLI as
  `infrastructure-overwatch report`. The GIF/video itself comes from `viz.py`, which only
  needs the base dependencies (OpenCV, Pillow) — no `dataviz` extra — since drawing boxes
  on frames doesn't touch matplotlib/seaborn/plotly at all.
- **`triage_nlp.py`** (`pip install -e ".[nlp]"`) — `NoteTriageClassifier` triages a
  free-text analyst note (e.g. "confirmed on second camera, escalating") into a small
  category taxonomy (`NOTE_CATEGORIES`), using a small transformer with a LoRA adapter
  (parameter-efficient fine-tuning) rather than full fine-tuning. This classifies the
  analyst's own written assessment, not the visual content of a detection — a separate
  signal from `types.THREAT_CLASSES`. Like calibration's triage bands, its output is a
  routing suggestion for a human, never an automated action. Requires downloading real
  pretrained weights on first use, so it is not exercised in CI (see
  `docs/VALIDATION.md`).

## What this system explicitly does not do

There is no code path in this repository that takes a `Detection` or `Event` and produces
an engagement, targeting, or fires action. `PipelineEventEngine` and
`calibration.route_detections` both terminate at human-facing outputs (`Event`, a triage
band on a `Detection`) meant for an analyst dashboard. See
[docs/USER_MANUAL.md](USER_MANUAL.md) for what an analyst should and should not infer from
an alert.
