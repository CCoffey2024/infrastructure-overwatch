# User Manual — for the analyst or technical/tactical operator

This document explains what this system's output means operationally: what to trust, what
not to trust, and what an alert does and does not authorize. It assumes no CV/ML
background. If you need the engineering "why," see
[ARCHITECTURE.md](ARCHITECTURE.md) and
[METHODOLOGY_AND_LIMITATIONS.md](METHODOLOGY_AND_LIMITATIONS.md).

## The one thing to understand before anything else

**This system never decides to engage, target, or act on anything.** Every output it
produces — a detection, a track, an event, a triage band — is a routing decision about
where a human analyst's attention should go next. Nothing here should be read as, or
extended into, a targeting or fires tool. If you are looking at this system's output to
make an engagement decision, stop: that is not what it is for, and it was never validated
for that use.

## What an alert (`Event`) means

The pipeline raises three kinds of event, all scoped to a "protected zone" (the corridor
or facility perimeter being watched):

| Event type | What triggered it | What it means | What it does *not* mean |
|---|---|---|---|
| `ZONE_ENTRY` | A tracked object's center crossed into the protected zone | Something the detector classified as one of the four threat classes is now inside the watched area | Not a confirmed threat — see "reading a detection's confidence" below |
| `STOPPED_IN_ZONE` | A tracked object stayed inside the zone with almost no movement for a sustained window | An object has stopped inside the protected area — historically the higher-consequence pattern (e.g. a vehicle stopping near a manifold) | Not evidence of intent; a legitimate vehicle can also stop |
| `PERSON_LOITER` | A `dismount`-classified track stayed inside the zone for a sustained window | A person-shaped track has persisted in the protected area | Not identification of a person; the detector does not recognize individuals |

Each event carries a `severity` (`medium`/`high`) and a `score` — both are triage hints to
help you prioritize a queue of alerts, not a confidence that something bad is happening.

## Reading a detection's confidence

A detector's raw confidence number is **not** a calibrated probability — a "0.8 confidence"
detection is not automatically "80% likely to be real." This system's calibration stage
(`calibration.py`) corrects for that using a validation-set-measured mapping, and then
routes each detection into one of three bands:

- **`auto_confirm`** — high enough calibrated confidence that it's logged as a confirmed
  detection on the dashboard without requiring a first look. **This still means "logged,"
  never "acted on."** A human can still review and dismiss it.
- **`analyst_review`** — the band where a human's judgment adds the most value over the
  model's own confidence. This is normally the largest band, and is expected to be — see
  the measured review-volume numbers in `METHODOLOGY_AND_LIMITATIONS.md`.
- **`auto_discard`** — low enough calibrated confidence that it's filtered from the primary
  queue. **Detections in this band are not deleted** — they're retained in the underlying
  data (`Detection.triage_band == "auto_discard"`) so they can be audited if a real event is
  later confirmed to have been missed. If you suspect the discard threshold is too
  aggressive for your operating environment, that is a configuration question for whoever
  deployed this system (`calibration.TriageThresholds`), not something to work around by
  distrusting every alert.

## What this system is confident about, and what it is not

Read this before trusting a specific alert in a specific condition:

- It was trained on **synthetic renderings** of a schematic corridor scene for `drone`,
  `dismount`, and `launch_flash`, and on a **curated slice of one real drone-video
  benchmark** (UAVDT) for `vehicle_of_interest`. It has not been validated against any real
  overwatch sensor, real threat imagery, or your specific facility's real layout, lighting,
  or terrain.
- It is measurably worse at night, at long standoff, and on small or partially-occluded
  objects than in daylight/close-standoff conditions — see the measured F1 numbers in
  `METHODOLOGY_AND_LIMITATIONS.md`. A missing alert in poor conditions is not proof nothing
  is there.
- `launch_flash` detects a **launch signature only** — a flash and rising smoke plume. It
  cannot and does not track a rocket, missile, or round in flight. Do not read a missing
  `launch_flash` alert as "no stand-off fire in progress"; it only means no launch signature
  was seen by this sensor's field of view.
- Detection tracking can lose or swap an identity across a brief occlusion or a crowded
  scene (an "ID switch") — see the measured switch rate in
  `METHODOLOGY_AND_LIMITATIONS.md`. Two consecutive alerts with different track IDs are not
  guaranteed to be two different real objects.

## Two additional signals you may see

If your deployment has these optional modules enabled, two more things can show up
alongside a detection:

- **An anomaly score.** A separate, unlabeled "this looks unlike anything already seen"
  signal (`anomaly.py`) — it has no notion of `drone`/`dismount`/`launch_flash`/
  `vehicle_of_interest`, only visual unfamiliarity relative to a reference gallery. Read a
  high anomaly score as "worth a glance because it's unusual," never as a threat
  classification in its own right.
- **A note-triage suggestion.** If you attach a free-text note to an alert, an optional
  module (`triage_nlp.py`) can suggest one of four categories
  (`confirmed_threat`/`false_alarm`/`sensor_or_equipment_issue`/
  `needs_more_information`) based on what you wrote. It is reading your own written
  assessment back to you as a suggested label, not independently verifying anything — it
  has not been trained or validated as of this writing (see
  `METHODOLOGY_AND_LIMITATIONS.md`), so treat any suggestion from it as provisional.

## If something looks wrong

If you believe this system is missing real threats, raising too many false alerts, or
behaving inconsistently with what this manual describes, treat that as an engineering
finding to report (see `DEVELOPMENT.md`), not something to silently route around. This
system's entire value proposition is that its behavior is measured and documented rather
than assumed — an operator's field observations are exactly the kind of measurement that
should feed back into it.
