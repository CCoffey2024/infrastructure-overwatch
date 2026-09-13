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
  pipeline.py                   orchestration: wires detector -> tracker -> events ->
                                calibration into one run
  cli.py                         `train-synthetic` / `demo` entry points
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
  shape: genuinely hard to see at range, which is the point of most of the domain-gap work
  in `docs/METHODOLOGY_AND_LIMITATIONS.md`.
- **`dismount`** — an unauthorized person near the fence line or manifold. A narrow,
  upright silhouette; the one class `events.PipelineEventEngine` treats specially for
  loitering (`PERSON_LOITER`), since a stationary vehicle and a loitering person warrant
  different analyst framing.
- **`launch_flash`** — a launch signature (flash + rising smoke), standing in for
  rocket/indirect-fire/stand-off threats. **This is a deliberate scope-down**: a
  pipeline-perimeter camera cannot realistically track a rocket or missile in flight — it's
  too fast, too far, and off-axis for most of its flight. What a fixed camera *can* catch is
  the launch signature itself. Modeling "detect the missile" would model a sensor capability
  that doesn't exist at this range; modeling "detect the launch flash" models what the
  sensor actually sees.
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

## What this system explicitly does not do

There is no code path in this repository that takes a `Detection` or `Event` and produces
an engagement, targeting, or fires action. `PipelineEventEngine` and
`calibration.route_detections` both terminate at human-facing outputs (`Event`, a triage
band on a `Detection`) meant for an analyst dashboard. See
[docs/USER_MANUAL.md](USER_MANUAL.md) for what an analyst should and should not infer from
an alert.
