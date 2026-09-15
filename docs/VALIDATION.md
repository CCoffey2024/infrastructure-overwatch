# Validation

An honest record of what has actually been run on this codebase, on what machine, and
what it measured — not a claim that everything has been exercised everywhere.

## Automated checks (CI, and reproducible anywhere)

`ruff check .`, `ruff format --check .`, `mypy src`, and `pytest -v` (177 tests covering
geometry, grid encode/decode, calibration, tracking (including the `MultiTracker.min_hits`
confirm-gate), event logic, evaluation (framewise P/R/F1 and COCO-style mAP/PR-curve/
calibration-reliability), synthetic rendering, HOG-based anomaly scoring plus the
`score_track_anomalies` pipeline wiring, reporting, VisDrone-DET/VisDrone-VID
ingest/class-mapping on on-disk fixtures, generic video/image-folder ingestion,
detection-drawing/GIF/video rendering, N-sensor semantic event fusion, the JSON-file job
store, the FastAPI operator-console service end to end via `TestClient`, and the `serve`
CLI command's loopback-only refusal gate) all pass with no GPU. CI installs the `dev`,
`onnx`, `anomaly`, `dataviz`, and `web` extras (all lightweight, no model downloads); the
`yolo` and `nlp` extras are not installed in CI since they pull in either a GPU-oriented
package or real pretrained weights. These run in CI (`.github/workflows/ci.yml`) on every
push/PR.

## End-to-end pipeline

`python -m infrastructure_overwatch train-synthetic --epochs 20 --n-train 800`,
`python -m infrastructure_overwatch demo --domain night --calibrate`, and
`python -m infrastructure_overwatch report` were all run on this machine and completed
without error, producing `outputs/weights/corridor_detector.pt`, `outputs/alerts.csv`,
`outputs/events.csv`, `outputs/report_card.png`, and `outputs/dashboard.html`.

The dashboard's annotated-GIF panel (`viz.build_annotated_gif`) was checked past "the file
exists" — loaded in a real browser, played, and its pixels read back programmatically
against the exact detection coordinates in that run's `alerts.csv`. That check caught a
real bug before it shipped: the first implementation wrote an actual compressed video
(WebM/VP8 via `cv2.VideoWriter`), which played, but its lossy inter-frame compression
visibly destroyed a detection box that was only drawn on a single frame — real
`(160, 160, 160)` gray came back near-black. Switching to an animated GIF (no inter-frame
prediction to lose a one-frame box to) fixed it; `tests/test_viz.py` now has a regression
test for exactly this case.

`python -m infrastructure_overwatch train-real --epochs 2 --no-visdrone` and
`demo-real --source "uavdt:M0601" --calibrate` were run against a real local UAVDT copy
(`D:\FMV\UAVDT\raw`) and completed without error — but not on the first try, and the same
"read the pixels back, don't just check the file exists" discipline that validated the GIF
panel caught two more real bugs before they shipped:

- `cli._self_calibrate` was called with `VEHICLE_GRID_W`/`VEHICLE_GRID_H` swapped for the
  real-data path (an easy mistake: the synthetic path's grid is square, `GRID`/`GRID`, so
  the same bug there would have been invisible). Crashed immediately with an `IndexError`
  the first time it ran against real (non-square, 40x22) grid data — a case CI's synthetic
  tests can't reach, since they never exercise a non-square grid.
- The GIF panel's own fix doesn't transfer to real footage: a real UAVDT frame with boxes
  drawn on it measured out to **105,825 distinct colors in a single 640x352 frame**, nowhere
  close to a GIF's 256-color palette. A detection box's exact `(76, 175, 80)` green came
  back a muddy, indistinguishable `(111, 129, 112)` after a GIF round-trip on real footage
  — the *opposite* failure mode from the one that ruled video out for the synthetic path.
  `viz.py` now has two functions, `build_annotated_gif` (synthetic) and
  `build_annotated_video` (WebM/VP8, real footage) — measured to preserve that same green
  to `(74, 174, 79)`, since real detections are also denser (tracked objects usually span
  many consecutive frames), so VP8's sparse-content failure mode doesn't apply here. Both
  bugs have regression tests in `tests/test_viz.py`.

## CV engine hardening and the operator console

A later pass hardened the pipeline itself (generic ingestion, tracker robustness, real
accuracy metrics, anomaly wiring) and built a local web operator console on top of it —
job store, FastAPI service, N-sensor fusion, and an interactive analyst frontend. All of
it was run against real footage on this machine, not just unit-tested:

- **Generic ingestion**: `demo-real --source "video:...\test_subset_55frames_30fps.mp4"`
  and `--source "folder:...\uav0000140_01590_v"` (a real VisDrone2019-VID sequence's raw
  frame folder, read as a plain image folder rather than through the named `visdrone-vid:`
  reader) both ran end to end and produced detections/tracks/events identical in kind to
  the named-dataset path — confirms `load_video_frames`/`load_image_folder_frames` aren't
  just unit-tested in isolation.
- **Tracker confirm-gate**: `demo-real --source "visdrone-vid:uav0000140_01590_v"
  --split train --calibrate` against a dense real intersection sequence produced 1,707
  events from 128 frames with the (unchanged) default `--track-min-hits 1` — almost
  entirely `ZONE_ENTRY` events from single-hit, never-re-associated tracks, not real zone
  entries. Re-running the identical sequence with `--track-min-hits 3` produced 856
  events, roughly half. `MultiTracker.min_hits`/`is_confirmed()` measurably fixes what it
  was built to fix, on the sequence that originally surfaced the problem, not a
  synthetic reproduction of it (a synthetic reproduction also exists, deterministically,
  in `tests/test_pipeline.py`).
- **mAP/calibration-reliability against real detector output**: `evaluation.
  mean_average_precision`/`calibration_reliability` were run against a real trained
  detector's predictions on real UAVDT footage (`M0601`, not just hand-built fixtures)
  and produced plausible, low-but-nonzero numbers consistent with this project's
  already-documented weak real-data accuracy (see "Real-data augmentation" below) — mAP
  wasn't just correct on a toy example, it was sanity-checked against a model whose
  accuracy ceiling was already independently measured elsewhere in this document.
- **Anomaly wiring**: `fit-anomaly-reference` (HOG embedder, a 4-image gallery split
  2/2 for gallery/calibration) followed by `demo-real --anomaly-ref ...` produced
  `VISUAL_ANOMALY` events end to end, deduplicated once per track as designed. The
  gallery used for this smoke test was small and drawn from a different scene than the
  footage it scored, so the resulting anomaly rate was high — a calibration-data-quality
  artifact of the smoke test, not a defect in `score_track_anomalies` itself.
- **The operator console, live, in a real browser**: started `serve`, then — through the
  actual HTML form, not curl — submitted a real run against
  `visdrone-vid:uav0000140_01590_v`, watched it move through `queued`/`running`/
  `completed` in the polling job-queue table, opened the evidence panel, played the
  annotated sequence (canvas overlay drawn from `/frame/{n}` + the alerts table, boxes/
  labels/confidence/track-id rendered correctly), filtered by confidence threshold and by
  class, clicked a box and confirmed its track highlighted, read real mAP/calibration
  numbers off the Metrics tab, then ran a second job under a different `sensor_id` and
  fused the two through the fusion panel, confirming the fused-event and contributor
  tables. This caught one real bug before it shipped: `GET /api/jobs/{id}/metrics` was
  silently shadowed by a more general `GET /api/jobs/{id}/{table_name}` route registered
  before it (FastAPI/Starlette match routes in registration order), so every metrics
  request was being rejected as "no such table" — invisible to `pytest` because the only
  automated `/metrics` test used a source with no ground truth, which is a 404 either
  way. Fixed by reordering the routes; a regression test now stubs `read_metrics` and
  asserts the real handler runs, and the fix was re-verified live against the same real
  run afterward.
- A second, narrower bug surfaced by the *test suite* rather than manual testing: adding
  `test_service.py` made `tests/test_reporting.py::test_report_card_writes_file` fail
  intermittently with `_tkinter.TclError` when the full suite ran together (never in
  isolation). `reporting.build_report_card` never called `matplotlib.use(...)`, so it
  silently depended on whatever GUI backend matplotlib auto-selected in the calling
  process; something in the FastAPI/Starlette/anyio `TestClient` machinery left Tk in a
  state the next `matplotlib.pyplot` import couldn't recover from. Fixed by forcing the
  `Agg` backend explicitly in `build_report_card` — it only ever writes a PNG file, it
  never needed a GUI backend in the first place. Confirmed stable across three
  consecutive full-suite runs afterward.

All of the above is a single pass on one development machine, same caveat as everything
else in this document.

## Evidence notebooks

Executed top-to-bottom (`jupyter nbconvert --execute`) on this development machine (Windows,
Python 3.12, CPU-only — no CUDA GPU exercised in this validation pass):

| Notebook | Status | Headline measured result |
|---|---|---|
| `01_domain_gap_evidence.ipynb` | Executed successfully | Day F1 0.86 → Night F1 0.00 (Day-only baseline); domain randomization F1 0.50, fine-tune (100 imgs) F1 0.43 |
| `02_real_data_validation.ipynb` | Executed successfully (real UAVDT data, `D:\FMV\UAVDT\raw`) | Day F1 0.40 → Night F1 0.07 (Day-only baseline); fine-tuned F1 0.20 |
| `03_model_comparison_bakeoff.ipynb` | Executed successfully | Our detector F1 0.41 vs. fine-tuned YOLO11n F1 0.06 (synthetic threat classes, Night val) |
| `04_edge_deployment_benchmarks.ipynb` | Executed successfully | ONNX export + fp32/int8 CPU latency measured; no CUDA GPU on this validation run |
| `05_calibration_and_triage.ipynb` | Executed successfully | Isotonic calibration + triage-band routing measured on a freshly trained detector |
| `06_reporting_and_anomaly_detection.ipynb` | Executed successfully | Report card + dashboard generated from a live pipeline run; HOG-embedding anomaly scorer separated Day (mean 0.36) from Night (mean 0.47) imagery around a calibrated threshold of 0.43 |
| `07_note_triage.ipynb` | Executed successfully | LoRA fine-tune of `distilbert-base-uncased` on 32 hand-written synthetic analyst notes: training loss 1.41 → 0.004 over 30 epochs, held-out validation accuracy 6/8 (0.75) on 8 notes never seen in training; adapter saved to `outputs/weights/note_triage_adapter` and reloaded via `NoteTriageClassifier` to confirm the round trip reproduces the same predictions |
| `08_visdrone_augmentation.ipynb` | Executed successfully (real UAVDT + VisDrone2019-DET + VisDrone2019-VID data) | Combining all three real sources raised `vehicle_of_interest` Day F1 0.39→0.44 and Night F1 0.08→0.15 over UAVDT alone; a first real-data `dismount` detector (4-class, all three sources combined) scored dismount-only F1 0.05 on **both** VisDrone-DET val and VisDrone-VID val independently (precision/recall 0.06-0.07/0.04 on each) — consistent across two independently-collected held-out sources, but a weak first result, not yet a usable detector |

All numbers above are from a **single run** on one development machine, not an aggregate
over multiple seeds — see `docs/METHODOLOGY_AND_LIMITATIONS.md` for how to read them
(re-running will shift exact numbers; the qualitative findings have been consistent).

## Known quirk

`notebooks/03_model_comparison_bakeoff.ipynb`'s YOLO fine-tuning step wrote its training
run artifacts (`weights/`, `results.csv`, etc.) to a path under a *different* local
project's `runs/` directory rather than this repo's own tree — Ultralytics resolves its
`project=` argument against a machine-wide settings file rather than the working directory,
and that file happened to point elsewhere on this development machine. Nothing in this
repository was affected; this only produced a couple of extra untracked files in an
already-untracked scratch directory of an unrelated local project. Noted here for
transparency, and as a heads-up if you re-run that notebook and go looking for its
`runs/` output.

## Not yet validated

- No CUDA GPU was exercised during this validation pass (edge-deployment GPU latency
  numbers are machine-dependent; re-run `04_edge_deployment_benchmarks.ipynb` on a CUDA
  machine to get them).
- The classical motion detector (`detectors/classical.py`) is covered by unit tests but
  not exercised in an evidence notebook. `fusion.py`'s box-level `late_fuse_detections`
  and its N-sensor semantic `fuse_events` are both now covered by `tests/test_fusion.py`
  (added from scratch — this file previously claimed unit-test coverage for `fusion.py`
  that didn't actually exist; there were zero tests for it before), and `fuse_events` is
  additionally exercised end to end through the live operator-console walkthrough below.
  Neither has an evidence notebook.
- `anomaly.DinoV2Embedder` (the richer, self-supervised embedding backend) is implemented
  but not exercised — only the offline `HOGEmbedder` backend was run, since DINOv2 requires
  a network download on first use.
- `triage_nlp.py`'s mechanism is now exercised end to end (`notebooks/07_note_triage.ipynb`,
  see the table above), but only on a 40-example hand-written dataset — real analyst-labeled
  notes at production volume remain untried.
- `ingest.py`'s VisDrone-DET/VisDrone-VID support is now exercised end to end
  (`notebooks/08_visdrone_augmentation.ipynb`, see the table above), but the first
  real-data `dismount` detector it trained is a weak result (F1 0.05) — a capability gap
  worth investigating (small-object grid resolution for person-scale boxes,
  training budget, or loss-term balance tuned for vehicle-scale objects), not yet a claim
  that real-data `dismount` detection works. The vehicle augmentation result (Day/Night F1
  improved over UAVDT alone) is a single run on one machine, same caveat as everything else
  in this section.
- No real overwatch sensor, real facility, or real threat imagery has been used anywhere
  in this project — see `docs/METHODOLOGY_AND_LIMITATIONS.md#limitations`.
- The operator console has no file-upload path — every job-submission route reads a
  local filesystem path directly (see `docs/OPERATOR_CONSOLE.md`). `score_track_
  anomalies`'s live smoke test above only exercised the `HOGEmbedder` backend through the
  console, matching the existing `DinoV2Embedder` gap already noted above.
- `evaluation.slice_metrics` (attribute-sliced P/R/F1) has no consumer anywhere in
  `service.py`/the web console yet — the same gap this document already notes for
  `mean_average_precision`/`calibration_reliability` before they were wired into
  `/api/jobs/{id}/metrics`, just not yet closed for slicing.
- `fusion.fuse_events`'s live smoke test above fused two runs built from the *same*
  underlying footage under different `sensor_id`s, as a mechanism check — it has not been
  run against genuinely independent EO/IR sensor feeds of the same scene.
