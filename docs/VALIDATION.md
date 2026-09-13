# Validation

An honest record of what has actually been run on this codebase, on what machine, and
what it measured — not a claim that everything has been exercised everywhere.

## Automated checks (CI, and reproducible anywhere)

`ruff check .`, `ruff format --check .`, `mypy src`, and `pytest -v` (65 tests covering
geometry, grid encode/decode, calibration, tracking, event logic, evaluation, synthetic
rendering, HOG-based anomaly scoring, and reporting) all pass with no GPU. CI installs the
`dev`, `onnx`, `anomaly`, and `dataviz` extras (all lightweight, no model downloads); the
`yolo` and `nlp` extras are not installed in CI since they pull in either a GPU-oriented
package or real pretrained weights. These run in CI (`.github/workflows/ci.yml`) on every
push/PR.

## End-to-end pipeline

`python -m infrastructure_overwatch train-synthetic --epochs 20 --n-train 800`,
`python -m infrastructure_overwatch demo --domain night --calibrate`, and
`python -m infrastructure_overwatch report` were all run on this machine and completed
without error, producing `outputs/weights/corridor_detector.pt`, `outputs/alerts.csv`,
`outputs/events.csv`, `outputs/report_card.png`, and `outputs/dashboard.html`.

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
- The classical motion detector (`detectors/classical.py`) and EO/IR fusion
  (`fusion.py`) are covered by unit tests but not exercised in an evidence notebook.
- `anomaly.DinoV2Embedder` (the richer, self-supervised embedding backend) is implemented
  but not exercised — only the offline `HOGEmbedder` backend was run, since DINOv2 requires
  a network download on first use.
- `triage_nlp.py` has **not been trained or run at all** in this repository — it requires
  downloading real pretrained transformer weights and labeled note data, neither of which
  are part of this validation pass. Its pure-Python parts (`NOTE_CATEGORIES`,
  `TriageResult`) are unit-tested; `NoteTriageClassifier`, `train_note_triage`, and the
  LoRA config helpers are not.
- No real overwatch sensor, real facility, or real threat imagery has been used anywhere
  in this project — see `docs/METHODOLOGY_AND_LIMITATIONS.md#limitations`.
