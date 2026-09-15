"""The operator-console API: submit an ingest+detect+track+event run (or a fusion of
several completed runs) as a background job, poll its status, and read back its evidence.

Every route delegates to `runs.py`/`fusion.py`/`evaluation.py` -- this module is
orchestration only, no pipeline logic of its own, so the CLI and the console can never
silently diverge on what a run or a fusion actually does (see `runs.py`'s docstring).

Security posture: binds to `127.0.0.1` by default via `cli.py`'s `serve` command; the
console has no authentication, so a non-loopback bind is an explicit opt-in the operator
must choose, understanding what that means for their network. See docs/OPERATOR_CONSOLE.md.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .jobs import Job, JobStore

DEFAULT_WEIGHTS_PATH = "outputs/weights/real_vehicle_detector.pt"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class OperatorService:
    """Owns the job store and the thread pool that actually executes runs/fusions.
    Framework-agnostic on purpose -- `create_app()` below is a thin FastAPI shell
    around this, so the execution logic can be unit-tested without spinning up a
    server (see `tests/test_service.py`).
    """

    def __init__(self, workspace_dir, default_weights_path: str = DEFAULT_WEIGHTS_PATH, max_workers: int = 1):
        self.workspace_dir = Path(workspace_dir)
        self.jobs_dir = self.workspace_dir / "jobs"
        self.store = JobStore(self.jobs_dir)
        self.default_weights_path = default_weights_path
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    # --- run jobs ---------------------------------------------------------

    def submit_run(
        self,
        source: str,
        sensor_id: str = "CAM01",
        display_name: str = "",
        weights_path: str | None = None,
        **run_kwargs: Any,
    ) -> Job:
        weights_path = weights_path or self.default_weights_path
        # Persisted verbatim so a later /metrics or /frame/{n} read can rebuild the exact
        # same RealRunConfig this run actually used -- see Job.config's docstring.
        job_config = {"weights_path": weights_path, **run_kwargs}
        job = self.store.create(
            kind="run", source=source, sensor_id=sensor_id, display_name=display_name or source, config=job_config
        )
        out_dir = self.jobs_dir / job.job_id
        job = self.store.update(job.job_id, output_dir=str(out_dir))
        self.executor.submit(self._execute_run, job.job_id)
        return job

    def _execute_run(self, job_id: str) -> None:
        from . import runs

        job = self.store.update(job_id, status="running", started_at=_now_iso())
        try:
            config = runs.RealRunConfig(source=job.source, **job.config)
            output = runs.run_real_pipeline(config)

            out_dir = Path(job.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output.result.alerts_frame().to_csv(out_dir / "alerts.csv", index=False)
            output.result.events_frame().to_csv(out_dir / "events.csv", index=False)

            summary = {
                "frames": len(output.frame_ids),
                "detections": len(output.result.detections),
                "tracks": len({d.track_id for d in output.result.detections if d.track_id is not None}),
                "events": len(output.result.events),
                "alerts": sum(1 for e in output.result.events if e.severity in ("medium", "high")),
                "anomalies": sum(1 for e in output.result.events if e.event_type == "VISUAL_ANOMALY"),
                "has_ground_truth": output.has_ground_truth,
                "calibration_warning": output.calibration_warning,
            }
            self.store.update(job_id, status="completed", completed_at=_now_iso(), summary=summary)
        except Exception as exc:  # noqa: BLE001 -- a failed job records *why*, for the operator to read
            self.store.update(job_id, status="failed", completed_at=_now_iso(), error=str(exc))

    # --- fusion jobs --------------------------------------------------------

    def submit_fusion(self, source_job_ids: list[str], **fusion_kwargs: Any) -> Job:
        job = self.store.create(
            kind="fusion",
            display_name=f"fusion of {len(source_job_ids)} runs",
            source_job_ids=list(source_job_ids),
        )
        out_dir = self.jobs_dir / job.job_id
        job = self.store.update(job.job_id, output_dir=str(out_dir))
        self.executor.submit(self._execute_fusion, job.job_id, fusion_kwargs)
        return job

    def _execute_fusion(self, job_id: str, fusion_kwargs: dict) -> None:
        from .fusion import FusionConfig, fuse_events, load_sensor_event_evidence

        job = self.store.update(job_id, status="running", started_at=_now_iso())
        try:
            maybe_jobs = [self.store.get(jid) for jid in job.source_job_ids]
            missing = [jid for jid, j in zip(job.source_job_ids, maybe_jobs, strict=True) if j is None]
            if missing:
                raise ValueError(f"Unknown source job id(s): {missing}")
            source_jobs: list[Job] = [j for j in maybe_jobs if j is not None]
            not_completed = [j.job_id for j in source_jobs if j.status != "completed"]
            if not_completed:
                raise ValueError(f"Every fused job must be completed; not completed: {not_completed}")
            distinct_sensors = {j.sensor_id for j in source_jobs}
            if len(distinct_sensors) < 2:
                raise ValueError("Fusion needs at least 2 completed runs with distinct sensor ids.")

            evidence = []
            for j in source_jobs:
                evidence += load_sensor_event_evidence(j.job_id, j.sensor_id, j.output_dir)
            fused_df, contributors_df = fuse_events(evidence, FusionConfig(**fusion_kwargs))

            out_dir = Path(job.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fused_df.to_csv(out_dir / "fusion_events.csv", index=False)
            contributors_df.to_csv(out_dir / "fusion_contributors.csv", index=False)

            summary = {
                "source_runs": len(source_jobs),
                "distinct_sensors": len(distinct_sensors),
                "fused_events": len(fused_df),
                "contributors": len(contributors_df),
            }
            self.store.update(job_id, status="completed", completed_at=_now_iso(), summary=summary)
        except Exception as exc:  # noqa: BLE001
            self.store.update(job_id, status="failed", completed_at=_now_iso(), error=str(exc))

    # --- reads ---------------------------------------------------------

    def _require_completed(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "completed":
            raise ValueError(f"Job {job_id} is {job.status!r}, not completed.")
        return job

    def read_table(self, job_id: str, name: str, offset: int = 0, limit: int = 200) -> list[dict]:
        """`name` is one of `alerts`/`events` for a run job, or
        `fusion_events`/`fusion_contributors` for a fusion job."""
        job = self._require_completed(job_id)
        path = Path(job.output_dir) / f"{name}.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        df = df.where(pd.notna(df), None)  # NaN isn't valid JSON
        return df.iloc[offset : offset + limit].to_dict("records")

    def read_metrics(self, job_id: str) -> dict | None:
        """mAP/PR-curve/calibration-reliability for a run job, or `None` if its source
        has no ground truth to score against (a `video:`/`folder:` source, or a fusion
        job)."""
        from . import runs
        from .evaluation import average_precision, calibration_reliability, mean_average_precision

        job = self._require_completed(job_id)
        if job.kind != "run" or not job.summary.get("has_ground_truth"):
            return None

        alerts_path = Path(job.output_dir) / "alerts.csv"
        alerts_df = pd.read_csv(alerts_path)
        config = runs.RealRunConfig(source=job.source, **job.config)
        tables = runs.build_evaluation_tables(config, alerts_df)
        if tables is None:
            return None
        gt_df, pred_df = tables

        map_result = mean_average_precision(gt_df, pred_df)
        reliability_df = calibration_reliability(gt_df, pred_df)
        return {
            "map_50": map_result["map_50"],
            "map_50_95": map_result["map_50_95"],
            "per_class_ap_50": {label: thr_to_ap.get(0.5) for label, thr_to_ap in map_result["per_class"].items()},
            "f1_at_iou_30": average_precision(gt_df, pred_df, iou_threshold=0.3),
            "calibration_reliability": reliability_df.to_dict("records"),
        }

    def read_frame_jpeg(self, job_id: str, frame_index: int, quality: int = 90) -> bytes:
        """Re-derives one RAW frame (no boxes burned in -- an interactive viewer draws
        those itself from `read_table(..., "alerts")`) from the job's original source,
        JPEG-encoded server-side. Cheap and codec-safe: no annotated video ever has to
        survive a browser's own decoder (see docs/VALIDATION.md on why that's a real
        risk), because there is no annotated video in this path at all."""
        import cv2

        from . import runs

        job = self._require_completed(job_id)
        config = runs.RealRunConfig(source=job.source, **job.config)
        _frame_ids, _frames, _labels, rgb_frames, _kind = runs.load_real_source(
            job.source, config.stride, config.cap, config.class_scheme, config.split
        )
        if not (0 <= frame_index < len(rgb_frames)):
            raise IndexError(f"frame {frame_index} out of range (0..{len(rgb_frames) - 1})")
        bgr = np.asarray(rgb_frames[frame_index])[:, :, ::-1]  # rgb_frames are RGB; cv2 wants BGR
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"Failed to JPEG-encode frame {frame_index} for job {job_id}")
        return io.BytesIO(buf.tobytes()).getvalue()

    def delete(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        if job.status in ("queued", "running"):
            raise ValueError(f"Job {job_id} is {job.status!r}; cannot delete an active job.")
        out_dir = Path(job.output_dir) if job.output_dir else None
        self.store.delete(job_id)
        if out_dir and out_dir.exists() and out_dir.is_relative_to(self.jobs_dir):
            import shutil

            shutil.rmtree(out_dir, ignore_errors=True)


def create_app(service: OperatorService | None = None):
    """Builds the FastAPI app. Imports fastapi lazily -- the `web` extra is optional,
    so nothing outside `serve`/this module should have a hard dependency on it."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response

    service = service or OperatorService(workspace_dir="outputs")
    app = FastAPI(title="Infrastructure Overwatch Operator Console")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [j.to_dict() for j in service.store.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = service.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No job {job_id!r}")
        return job.to_dict()

    @app.post("/api/jobs", status_code=202)
    def submit_run(body: dict) -> dict:
        source = body.get("source")
        if not source:
            raise HTTPException(status_code=422, detail="'source' is required.")
        run_kwargs = {
            k: body[k]
            for k in (
                "split",
                "class_scheme",
                "stride",
                "cap",
                "fps",
                "conf_thresh",
                "calibrate",
                "track_min_hits",
                "anomaly_ref",
                "zone",
            )
            if k in body
        }
        job = service.submit_run(
            source=source,
            sensor_id=body.get("sensor_id", "CAM01"),
            display_name=body.get("display_name", ""),
            weights_path=body.get("weights_path"),
            **run_kwargs,
        )
        return job.to_dict()

    @app.post("/api/jobs/fuse", status_code=202)
    def submit_fusion(body: dict) -> dict:
        source_job_ids = body.get("source_job_ids") or []
        if len(source_job_ids) < 2:
            raise HTTPException(status_code=422, detail="'source_job_ids' needs at least 2 completed run jobs.")
        fusion_kwargs = {
            k: body[k]
            for k in ("max_time_delta_s", "min_sensors", "require_matching_label", "spatial_iou_threshold")
            if k in body
        }
        job = service.submit_fusion(source_job_ids, **fusion_kwargs)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/{table_name}")
    def read_table(job_id: str, table_name: str, offset: int = 0, limit: int = 200) -> list[dict]:
        if table_name not in ("alerts", "events", "fusion_events", "fusion_contributors"):
            raise HTTPException(status_code=404, detail=f"No such table {table_name!r}")
        try:
            return service.read_table(job_id, table_name, offset=offset, limit=min(limit, 1000))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/metrics")
    def read_metrics(job_id: str) -> dict:
        try:
            metrics = service.read_metrics(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if metrics is None:
            raise HTTPException(status_code=404, detail="No ground truth available for this job's source.")
        return metrics

    @app.get("/api/jobs/{job_id}/frame/{frame_index}")
    def read_frame(job_id: str, frame_index: int) -> Response:
        try:
            jpeg_bytes = service.read_frame_jpeg(job_id, frame_index)
        except (KeyError, ValueError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        try:
            service.delete(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app
