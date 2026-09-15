# Architecture

## Module map

```
src/infrastructure_overwatch/
  types.py           Detection / Event / FrameRecord data contracts; THREAT_CLASSES, TRIAGE_BANDS
  geometry.py         IoU, point-in-rect, NMS -- dependency-free primitives
  grid_codec.py        encode/decode between (box, class) ground truth and the grid-tensor
                       label format the project's own detector predicts
  synthetic.py          the day/night corridor scene renderer (drone/dismount/launch_flash)
  ingest.py              real-data ingestion: named-benchmark readers (UAVDT, VisDrone-DET,
                         VisDrone-VID, with ground truth) plus load_video_frames/
                         load_image_folder_frames (an arbitrary local video file or image
                         folder, no ground truth, works on anything)
  detectors/
    classical.py          background-subtraction motion detector (no model, no classes)
    grid_cnn.py             this project's own lightweight detector: model, training,
                           evaluation, and a DetectorAdapter wrapper
    yolo_adapter.py          Ultralytics YOLO backend, same adapter interface
    ground_truth.py           replays known annotations, for testing tracking/events
  tracking.py             MultiTracker: greedy IoU + constant-velocity data association,
                         plus a min_hits confirm-gate before a track can raise events
  events.py                PipelineEventEngine: protected-zone entry/stopped/loiter alerts
  calibration.py            isotonic confidence calibration + auto_confirm/review/discard
                           triage routing; self_calibrate() (shared by cli.py and runs.py)
  evaluation.py              precision/recall/F1 + attribute-sliced error analysis, plus
                             COCO-style average_precision/mean_average_precision,
                            precision_recall_curve, and calibration_reliability
  export.py                   ONNX export + fp32/int8, CPU/GPU latency benchmarking
  fusion.py                    late_fuse_detections (2-sensor, box-level, pre-registered
                               imagery) and fuse_events (N-sensor, semantic-event-level,
                              across independently-run sensor jobs -- see below)
  anomaly.py                    embedding-based anomaly scoring (HOG / DINOv2) plus
                                score_track_anomalies, which turns scored track crops
                               into VISUAL_ANOMALY pipeline events
  reporting.py                   Seaborn report-card + Plotly dashboard (charts and
                                alerts/events tables), built from PipelineResult's tables
  viz.py                          draws detections on frames and renders an annotated
                                GIF (synthetic) or WebM video (real footage) of a run --
                                what the dashboard's media panel shows; see viz.py's own
                                docstring for why the format depends on the frame source
  triage_nlp.py                   optional LoRA-fine-tuned free-text alert-note triage
  pipeline.py                       orchestration: wires detector -> tracker -> events ->
                                    calibration into one run (run_frame_sequence)
  runs.py                            RealRunConfig/run_real_pipeline: turns a --source
                                     spec into a PipelineResult, shared verbatim by cli.py
                                    and service.py so neither can drift from the other
  jobs.py                              JobStore: one JSON file per job, no database --
                                      the operator console's run/fusion job records
  service.py                            OperatorService + the FastAPI app: submit a run
                                       or fusion job, poll status, read back evidence
                                      (alerts/events/metrics/frames) -- see
                                     docs/OPERATOR_CONSOLE.md
  web/                                   the operator-console frontend (index.html/
                                        app.js/style.css, vanilla JS, no build step)
  cli.py                             `train-synthetic` / `demo` / `train-real` /
                                    `demo-real` / `fit-anomaly-reference` / `report` /
                                   `serve` entry points
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
  for novel visual patterns the fixed taxonomy wouldn't otherwise catch. `EmbeddingAnomaly
  Scorer.save`/`.load` persist a fitted gallery (`fit-anomaly-reference` builds one from a
  folder of normal imagery, holding half out for threshold calibration — scoring an image
  against a gallery that already contains it would find itself as a perfect match and
  silently calibrate an unusably strict near-zero threshold). `score_track_anomalies`
  crops each sufficiently mature track, scores it against a loaded gallery, and emits a
  `VISUAL_ANOMALY` `Event` the first time a track crosses the threshold — reusing the
  existing `Event` type and the same fire-once-per-track discipline
  `PipelineEventEngine` already uses, wired into `demo`/`demo-real` via `--anomaly-ref`.
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

## The operator console (Phase 3)

`cli.py`'s `demo`/`demo-real` commands and the web operator console (`service.py` + `web/`)
are two front ends over the same engine, not two implementations of it — the same
discipline as "one adapter interface for every detector" above, extended to the ingest/
serving boundary:

```
--source spec
     │
     ▼
runs.load_real_source() -> frames, labels (or None), rgb_frames
     │
     ▼
runs.run_real_pipeline() -> PipelineResult   (same detect/track/event/calibrate/anomaly
     │                                        stack cli.py's demo-real always ran)
     ├── cli.py: writes alerts.csv/events.csv, prints a summary, optionally an annotated
     │           video (boxes burned into pixels -- a static, shareable export)
     │
     └── service.py: writes alerts.csv/events.csv/frame_ids.json under a JobStore-tracked
                      job directory; the web console's player draws boxes *client-side*,
                      live, from the raw frame (GET .../frame/{n}, re-derived from the
                      original source, never burned in) + the alerts table -- which is
                      what makes a confidence-threshold slider, per-class visibility
                      toggles, and click-a-box-to-highlight-its-track possible at all. A
                      pre-rendered video can't support any of that no matter which codec
                      survives the browser (see docs/VALIDATION.md's own GIF/WebM color-
                      fidelity findings for why boxes get burned into pixels for the
                      *static* export in the first place, and why the interactive path
                      deliberately avoids doing that at all).
```

`jobs.JobStore` persists one JSON file per job (no database) under the workspace's `jobs/`
directory, with crash recovery (a job left `queued`/`running` from a killed process is
reclassified `failed` on the next load — that process is gone). `Job.config` records the
exact `RealRunConfig` a run was submitted with, so a later `/metrics` or `/frame/{n}` read
can reconstruct an identical config instead of guessing at `stride`/`cap`/`class_scheme`.

`fusion.fuse_events` operates one level above `late_fuse_detections`: it fuses *events*
already produced by independently completed sensor jobs (time window + event type,
optionally label and/or pixel-space IoU), not raw boxes from two live, pre-registered
streams — see the module's own docstring and `docs/OPERATOR_CONSOLE.md` for the full
contract. `events.csv` carries neither label nor box, so `load_sensor_event_evidence`
joins each event back to its own run's `alerts.csv` on `(frame_id, track_id)` to recover
them before fusion can use them for label/spatial matching.

The console has no authentication and binds to loopback (`127.0.0.1`) by default —
`docs/OPERATOR_CONSOLE.md` covers the full security posture and why every job-submission
route reading a local filesystem path directly makes that the correct default, not just a
cautious one.

## What this system explicitly does not do

There is no code path in this repository that takes a `Detection` or `Event` and produces
an engagement, targeting, or fires action. `PipelineEventEngine` and
`calibration.route_detections` both terminate at human-facing outputs (`Event`, a triage
band on a `Detection`) meant for an analyst dashboard. See
[docs/USER_MANUAL.md](USER_MANUAL.md) for what an analyst should and should not infer from
an alert.
