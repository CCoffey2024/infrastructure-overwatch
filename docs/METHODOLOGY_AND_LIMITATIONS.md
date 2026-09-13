# Methodology and Limitations

A model-card-style summary: what was measured, how, and what it does and does not license
you to believe. Read the [Limitations](#limitations) section before the results — it was
written that way on purpose.

## Limitations

> Trained on synthetic renderings of a schematic corridor scene (`drone`/`dismount`/
> `launch_flash`) and a small curated slice of one real drone-video benchmark
> (`vehicle_of_interest`, UAVDT). **Not validated against any real overwatch sensor, real
> threat imagery, or real facility layout.** `launch_flash` models a launch signature only,
> not missile/rocket flight — a deliberate scope decision, not a capability claim. Weakest
> on small/far objects and anything past light occlusion. This is an engineering prototype
> for CV/ML practice, not an operational detection system, and it makes no targeting,
> engagement, or fires recommendation anywhere in its pipeline.

Specific, measured weaknesses:

- **Night/long-standoff performance is substantially worse than day/close-standoff** for
  every class except `launch_flash` (see [Domain gap](#domain-gap-day-vs-night) below) —
  this is the single largest source of missed detections in this system.
- **Small and partially-occluded objects are the hardest case**, consistent with what a
  40x22-cell grid over a 640x352 real frame can represent — an object spanning less than
  one grid cell is fundamentally hard for this architecture to localize regardless of
  training.
- **Tracking can lose or swap identity** across occlusion or in a crowded scene (an "ID
  switch") — see [Tracking](#tracking) below for the measured rate on a real sequence.
- **A stock production detector (YOLO11n) is not a strictly-better fallback** — see
  [Model comparison](#model-comparison-lightweight-vs-production) — it needs its own
  investment (resolution, fine-tuning) for classes it wasn't trained on, and even after
  fine-tuning underperformed this project's small custom model on the synthetic threat
  classes, likely due to a backbone/resolution mismatch at this scenario's small object
  scale.

## Threat taxonomy and scope decisions

See [ARCHITECTURE.md](ARCHITECTURE.md#why-four-threat-classes) for the full reasoning
behind the four classes. The one worth restating here: **`launch_flash` detects a launch
signature (flash + rising smoke), not a missile or rocket in flight.** A fixed
corridor-perimeter camera cannot realistically track a fast-moving, mostly off-axis
projectile — modeling "detect the missile" would model a sensor capability that doesn't
exist at this range. Modeling "detect the launch flash" models what the sensor actually
sees. Treat a missing `launch_flash` alert as "no launch signature was observed by this
sensor," never as "no stand-off fire occurred."

## Domain gap: day vs. night

The core methodology throughout this project: render (or select) the *same* underlying
scene under two conditions — clean daylight/close-standoff ("Day") vs. low-light/long-
standoff/oblique-angle ("Night") — and measure the same model's F1 on both, rather than
reporting a single blended accuracy number that would hide the gap.

Two independent measurements of the same underlying mechanism:

1. **Synthetic corridor scene** (`drone`/`dismount`/`launch_flash`, this project's own
   `GridDetector`): a model trained on Day-domain data only scored **F1 0.86 on Day** but
   **F1 0.00 on Night** in this repository's own validation run (`notebooks/01_domain_gap_evidence.ipynb`)
   — essentially a total collapse, not a gradual degradation. Reproduce with
   `train-synthetic` or by re-running that notebook; exact numbers will vary run to run
   (different random seeds, no fixed release model), but the collapse itself has been
   consistent across every run so far.
2. **Real UAVDT vehicle footage** (`vehicle_of_interest`, real drone-shot car/truck/bus
   footage): the same collapse shape appears on independently-sourced real data, not just a
   synthetic renderer built to make a point. In this repository's own run
   (`notebooks/02_real_data_validation.ipynb`): **F1 0.40 on real Day frames, F1 0.07 on
   real Night frames** with the Day-only-trained model — a smaller absolute gap than the
   synthetic track, but the same direction, on genuinely independent real footage. Fine-
   tuning on a small batch of real Night-domain frames recovered it to **F1 0.20** (requires
   a local UAVDT copy, see `DEVELOPMENT.md`).

Two remedies are implemented and compared for both tracks:

- **Domain randomization** — train directly on Night-style rendering/augmentation, no real
  Night-domain labels used.
- **Fine-tuning** — start from the Day-trained weights, fine-tune on a small batch of
  Night-domain samples, simulating "a field team sent back a small batch of labeled
  night/hard-condition frames."

Both remedies recover substantial accuracy from the Day-only baseline's total collapse; in
this repository's own run, domain randomization (F1 0.50) slightly outperformed the
100-image fine-tune (F1 0.43) on the synthetic track — the opposite ranking from what the
predecessor research notebooks this project was built from originally reported. That
discrepancy is itself worth taking at face value rather than smoothing over: which remedy
wins is evidently sensitive to exact architecture, seed, and batch composition, not a fixed
fact about "fine-tuning always beats augmentation" or vice versa. Don't treat either
ranking as settled — re-run `notebooks/01_domain_gap_evidence.ipynb` and
`02_real_data_validation.ipynb` and read whatever they currently measure.

## Model comparison: lightweight vs. production

The fair comparison depends on whether the production model already knows the classes
involved:

- **`drone`/`dismount`/`launch_flash` (synthetic corridor scene) — measured**
  (`notebooks/03_model_comparison_bakeoff.ipynb`): COCO has no matching classes, so YOLO11n
  was fine-tuned on this scenario's own synthetic training set first, for a fair
  comparison. In this repository's own run: **this project's own detector scored F1 0.41
  (precision 0.69, recall 0.29) vs. fine-tuned YOLO11n's F1 0.06 (precision 0.06, recall
  0.07)** on the Night validation set — a large gap in favor of the small custom model,
  plausibly because YOLO11n's backbone downsampling factor loses very small objects before
  its detection head ever sees them at this scenario's small (96x96) chip resolution, and
  15 epochs on 400 images is a small fine-tuning budget by production standards. Both are
  real, stated constraints on this specific comparison, not a general claim that YOLO is
  bad.
- **`vehicle_of_interest` (real UAVDT frames) — not yet re-measured in this repository.**
  Stock, COCO-pretrained `yolo11n.pt` already has `car`/`truck`/`bus` as native classes
  (`detectors.yolo_adapter.COCO_TO_VEHICLE_SUBTYPE`), so no fine-tuning would be needed for
  a like-for-like comparison against this project's own fine-tuned vehicle detector — the
  predecessor research notebooks this project was built from measured YOLO winning this
  comparison comfortably, which is plausible given YOLO already carries relevant pretrained
  knowledge here, but that specific number has not been reproduced against this
  repository's own `GridDetector` implementation. Treat it as an expected-but-unverified
  claim until someone runs it (the pieces — `YOLOAdapter`, `UAVDTVehicleDataset` — are both
  already in this codebase and demonstrated separately in `02_real_data_validation.ipynb`).

The actionable takeaway from what *is* measured: **a production model is not a
strictly-better default you fall back to
after testing something lighter** — it wins when the class and operating resolution match
what it was built for, and needs its own investment (resolution, fine-tune budget, or both)
when they don't.

## Edge deployment

`export.py` exports to ONNX (fp32) and dynamic int8 quantization, and benchmarks inference
latency for PyTorch eager and ONNX Runtime, on both CPU and (when available) GPU
(`notebooks/04_edge_deployment_benchmarks.ipynb`). The consistent finding worth stating
plainly: **quantization's storage win is close to guaranteed (an int8 model is
predictably smaller); its *latency* win is not** — it depends on model scale and the
runtime's hardware kernel support, and this project reports what was actually measured on
its own small model rather than assuming the generic "quantization is faster" story holds
at every scale.

## Tracking

`tracking.MultiTracker` is a deliberately simple greedy-IoU associator with a
constant-velocity motion model — not Hungarian-optimal, not ByteTrack. Two measurements
matter for trusting its output:

- **Coverage**: the fraction of real ground-truth object-frames with some predicted box
  overlapping them, comparing raw per-frame detections against the tracked estimate — this
  shows what coasting through brief misses actually buys.
- **ID switches** (`tracking.count_id_switches`): how often a track id's best-matching real
  object changes from one frame to the next. A nonzero rate is expected and should be read
  as "don't treat two alerts with different track IDs as guaranteed-different real objects"
  (see `docs/USER_MANUAL.md`), not as a bug to eliminate entirely with this associator.

## Calibration and analyst review queue

Every detection routes to a human analyst — this system's actual engineering question is
"how much of the analyst's attention does this consume," not just "what's the accuracy."
`calibration.py` fits an isotonic calibration from raw detector confidence to empirical
true-positive rate on a labeled validation set, then buckets calibrated confidence into
`auto_confirm` / `analyst_review` / `auto_discard` (see `docs/USER_MANUAL.md` for what each
band means operationally).

The current default thresholds (`auto_confirm=0.90`, `discard=0.05`) reflect a measured
operating-point revision from an untuned `discard=0.20` default: lowering the discard
threshold recovers most of the previously-missed real threats sitting just above the old
cutoff, at the cost of a larger analyst review volume. That is a deliberate trade in favor
of fewer missed threats, made explicit and measurable rather than left as an unexamined
default — see `notebooks/05_calibration_and_triage.ipynb` for the current before/after
comparison and `calibration.TriageThresholds` to adjust it for a different operating
environment.

## Embedding-based anomaly detection

`anomaly.EmbeddingAnomalyScorer` is a nearest-neighbor cosine-distance scorer over a
reference gallery of "normal" imagery embeddings — a complementary signal to the four-class
detectors, not a replacement (see `docs/ARCHITECTURE.md#secondary-capabilities-phase-2`).
In this repository's own run (`notebooks/06_reporting_and_anomaly_detection.ipynb`, using
the offline `HOGEmbedder` backend — no model download involved): a gallery built entirely
from **Day**-domain corridor renders scored held-out **Day** imagery at a mean anomaly
distance of **0.36**, and **Night**-domain imagery (a genuinely different visual domain,
never seen by the gallery) at **0.47** — a clear separation, with a 95th-percentile
Day-calibrated threshold of **0.43** landing between the two. That is a sanity check that
the distance metric behaves as intended (a known-different domain scores as more anomalous
than the reference domain), not a validated real-world anomaly-detection accuracy number;
no labeled real anomalies have been used to evaluate false-positive/false-negative rates.
The `DinoV2Embedder` backend (richer, self-supervised features) is implemented but not
exercised in this validation pass — see `docs/VALIDATION.md`.

## Free-text alert-note triage (optional)

`triage_nlp.NoteTriageClassifier` is a secondary, optional module (requires the `nlp`
extra) that triages an analyst's free-text note into `NOTE_CATEGORIES`
(`confirmed_threat`/`false_alarm`/`sensor_or_equipment_issue`/`needs_more_information`)
using a small transformer with a LoRA adapter. **This has not been trained or evaluated
in this repository** — it requires downloading real pretrained weights and labeled
training examples, neither of which are part of this repo's automated validation (see
`docs/VALIDATION.md`). Treat it as a demonstrated *mechanism* (the LoRA fine-tuning
approach, following the same parameter-efficient-fine-tuning pattern as a general
LoRA/PEFT reference exercise this project draws on) rather than a measured capability
until someone runs `triage_nlp.train_note_triage` on real labeled note data.
