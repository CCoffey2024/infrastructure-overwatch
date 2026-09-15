# Operator console

A local web UI for submitting analysis runs, watching them complete, and reviewing their
evidence — an alternative front end to `demo`/`demo-real` for someone who'd rather click
through a form and an interactive video player than run CLI commands and open CSVs. It
runs the identical detect/track/event/calibrate/anomaly pipeline underneath (`runs.
run_real_pipeline` — see `docs/ARCHITECTURE.md#the-operator-console-phase-3`); this
document covers the console itself, not the pipeline.

## Starting it

```bash
pip install -e ".[web]"
python -m infrastructure_overwatch serve
```

Opens on `http://127.0.0.1:8765` by default (loopback only — see **Security posture**
below). `--host`/`--port` change the bind address/port; `--workspace` changes where job
records and their evidence are stored (default `outputs/`); `--weights` sets the default
detector checkpoint a submitted run uses if it doesn't specify its own.

## What it does

- **Start analysis run** — point at a local video file, an image folder, or a named
  benchmark sequence (`uavdt:`/`visdrone-vid:`), the same four source kinds
  `demo-real --source` accepts. Set sensor ID, class scheme, and any of `RealRunConfig`'s
  knobs (stride/cap/fps/confidence threshold/track-min-hits/calibrate/anomaly reference).
- **Job queue** — every submitted run or fusion, its status
  (`queued`/`running`/`completed`/`failed`), and a one-line result summary, polling every
  2 seconds. Backed by `jobs.JobStore` (one JSON file per job, no database).
- **Evidence panel**, once a run completes:
  - Summary tiles (frames/detections/tracks/events/alerts/anomalies).
  - An **interactive player**: play/pause/scrub/speed/loop, a confidence-threshold
    slider, per-class visibility toggles, and click-a-box-to-highlight-its-track. This
    works because the player draws boxes *client-side*, live, on top of a raw frame
    (`GET /api/jobs/{id}/frame/{n}`, re-derived from the original source on demand, never
    burned into pixels) using the same `alerts.csv` data the Detections tab shows — not
    because it's playing a pre-rendered annotated video. A baked video literally cannot
    support any of that interactivity, whichever codec survives the browser.
  - Tabs: Detections, Events, Alerts (events at `medium`/`high` severity), Anomalies
    (`VISUAL_ANOMALY` events, only present if the run used `--anomaly-ref`), Metrics
    (mAP@0.5, mAP@0.5:0.95, per-class AP, calibration reliability — only available when
    the source has ground truth, i.e. a named benchmark sequence, not a plain video/
    folder), Fusion evidence.
- **Fuse completed runs** — select 2+ completed runs with distinct sensor IDs, set a
  time-window (and optionally a label-match) tolerance, and see the fused events plus
  full contributor provenance (which raw event, from which job/sensor/track, fed which
  fused event).

## API

Every console feature is backed by a plain JSON REST API (`service.create_app`), usable
directly (`curl`, a script, another tool) without the web UI at all:

| Method & path | What it does |
|---|---|
| `POST /api/jobs` | Submit a run. Body: `source` (required), plus any `RealRunConfig` field. Returns `202` + the job. |
| `GET /api/jobs` | List all jobs, newest first. |
| `GET /api/jobs/{id}` | One job's current status/summary. |
| `GET /api/jobs/{id}/alerts` \| `/events` | Paginated (`offset`/`limit`) table rows as JSON. |
| `GET /api/jobs/{id}/frame_ids` | The ordered list of string `frame_id`s the run produced — maps `/frame/{n}`'s positional index onto the `frame_id` alerts/events rows carry. |
| `GET /api/jobs/{id}/frame/{n}` | Raw JPEG for frame `n`, re-derived from the original source. No boxes burned in. |
| `GET /api/jobs/{id}/metrics` | mAP/PR/calibration-reliability. `404` if the source has no ground truth. |
| `POST /api/jobs/fuse` | Submit a fusion. Body: `source_job_ids` (≥2), plus any `fusion.FusionConfig` field. |
| `GET /api/jobs/{id}/fusion_events` \| `/fusion_contributors` | Paginated fusion output tables. |
| `DELETE /api/jobs/{id}` | Delete a job and its evidence. Refuses an active (`queued`/`running`) job. |

## Security posture

**This console has no authentication.** It binds to `127.0.0.1` (loopback) by default,
and `serve` refuses to bind anywhere else unless you pass `--allow-remote` explicitly.

That default isn't just cautious — it's load-bearing. Every job-submission route accepts
a local filesystem path (`video:<path>`, `folder:<path>`) and reads it directly; there is
no upload boundary. Binding this process to a non-loopback address means **any client
that can reach the port can ask this process to read any local file path it names.** If
you pass `--allow-remote` anyway, understand that trade-off for your network before you
do — don't expose this beyond a host you trust, and don't put it behind a reverse proxy
you assume adds authentication unless you've actually configured that yourself.

## Limitations

- No file upload — every source is read in place from a local path. Large frame
  collections were always meant to be read this way (see `README.md`'s ingest note), but
  there is currently no way to hand the console a file that isn't already on the machine
  it runs on.
- `/metrics` only exists for a named-benchmark source (`uavdt:`/`visdrone-vid:`) — a
  plain `video:`/`folder:` source has no ground truth to score against, by design (see
  `runs.py`'s module docstring).
- `evaluation.slice_metrics` (attribute-sliced P/R/F1) has no route or UI yet.
- Job dispatch is a single-worker thread pool by default — two runs submitted at once
  queue rather than run in parallel. Fine for a solo analyst's workstation; raise
  `OperatorService(max_workers=...)` if you need more concurrency and have the CPU/GPU to
  back it.
