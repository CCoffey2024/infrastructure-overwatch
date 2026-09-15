# Infrastructure Overwatch

A computer-vision pipeline for critical-infrastructure corridor monitoring (e.g. a pipeline
pumping station and its right-of-way). It detects, tracks, and routes sensor alerts to a
human analyst.

> This system contains no targeting, engagement, or fires logic, and nothing here should be
> extended into one. Every alert this pipeline produces ends at a person's judgment call,
> never at an automated action. It is trained on synthetic renderings of a schematic
> corridor scene plus a curated slice of one public real-drone-video benchmark (UAVDT). It
> is not validated against any real sensor, real threat imagery, or real facility, and
> should be treated as an engineering prototype, not an operational system. Read
> [docs/METHODOLOGY_AND_LIMITATIONS.md](docs/METHODOLOGY_AND_LIMITATIONS.md) before trusting
> any output, and [docs/USER_MANUAL.md](docs/USER_MANUAL.md) for what an alert does and does
> not mean.

## What it does

Given a video feed (or a generated synthetic one), the pipeline:

1. **Detects** one of four threat classes — `drone`, `dismount`, `launch_flash`
   (a rocket/indirect-fire launch signature), and `vehicle_of_interest` — using either its
   own lightweight grid-CNN, an Ultralytics YOLO model, or a classical motion baseline,
   all behind one `DetectorAdapter` interface.
2. **Tracks** detections across frames (`MultiTracker`: greedy IoU + constant-velocity
   prediction), so a flickering per-frame detector still yields stable identities.
3. **Raises events** when a track enters, stops inside, or loiters in a protected zone
   (`PipelineEventEngine`).
4. **Calibrates** detector confidence against measured true-positive rates and routes each
   detection into an `auto_confirm` / `analyst_review` / `auto_discard` triage band, so the
   analyst spends time on the detections that actually warrant it.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full module map and the reasoning
behind the threat taxonomy and the protected-zone event model.

Three secondary capabilities extend that core pipeline, each behind its own optional
dependency extra:

- **Analyst reporting** (`reporting.py`, `pip install -e ".[dataviz]"`) — a Seaborn
  report-card figure and a Plotly interactive dashboard (summary charts, plus alerts and
  events tables, plus an embedded annotated GIF of the run — see `viz.py`), built from the
  same alert/event tables the pipeline produces (`infrastructure-overwatch report`).
- **Embedding-based anomaly detection** (`anomaly.py`, `pip install -e ".[anomaly]"`) — a
  nearest-neighbor embedding distance (HOG by default, or DINOv2) that flags imagery
  unlike anything in a reference gallery, independent of the four-class taxonomy.
- **Free-text alert-note triage** (`triage_nlp.py`, `pip install -e ".[nlp]"`) — a small
  transformer with a LoRA adapter that triages an analyst's free-text note into a
  category taxonomy (`confirmed_threat` / `false_alarm` / `sensor_or_equipment_issue` /
  `needs_more_information`).

## Quickstart

```bash
uv venv
uv pip install -e ".[dev,onnx]"

# Train the synthetic-corridor detector (a few seconds to a couple minutes on CPU)
python -m infrastructure_overwatch train-synthetic

# Run the end-to-end demo: generates a synthetic night-domain sequence, detects, tracks,
# raises events, and writes outputs/alerts.csv + outputs/events.csv + an annotated GIF
# (outputs/annotated_demo.gif — boxes, labels, confidence; pass --no-video to skip it)
python -m infrastructure_overwatch demo --calibrate

# Optional: build a report card + dashboard from that run (needs the `dataviz` extra) --
# the dashboard embeds the annotated GIF above its charts and alerts/events tables
uv pip install -e ".[dataviz]"
python -m infrastructure_overwatch report
```

There is a second, parallel `train-real` / `demo-real` pair of commands that run the same
pipeline against real UAVDT/VisDrone footage instead of the synthetic renderer, for the
`vehicle_of_interest` (and, with `--class-scheme vehicle_dismount`, `dismount`) track —
requires a local copy of one or both datasets, see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). `demo-real` writes a WebM video rather than a
GIF (real footage has far more colors than a GIF palette can hold faithfully — see
`viz.py`); pass `report --video outputs/annotated_demo.webm` to embed it.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map, data flow, and the design
  decisions behind the threat taxonomy and event model
- [docs/USER_MANUAL.md](docs/USER_MANUAL.md) — for an analyst or technical/tactical
  operator: what each alert type and triage band means, and what this system does not do
- [docs/METHODOLOGY_AND_LIMITATIONS.md](docs/METHODOLOGY_AND_LIMITATIONS.md) — measured
  findings (domain-gap collapse, remedies, model bakeoff, edge-deployment latency,
  calibration) and a model-card-style limitations statement
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — engineer setup, tests, extending the
  pipeline with a new detector backend
- [docs/VALIDATION.md](docs/VALIDATION.md) — an honest record of what has actually been
  run and measured on this codebase, and what hasn't

## Project origin

This repository turns a set of CV/ML engineering exercises, originally built as narrative
research notebooks (`pipeline_defense_scenario.ipynb` and `bah_cv_field_deployment.ipynb`),
into a tested, importable package. Those notebooks' measured findings and design reasoning
are preserved in `docs/`; their code is no longer kept as one-off notebook cells.

## License

[MIT](LICENSE)
