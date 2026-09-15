"""Lightweight job tracking for the operator console: one JSON file per job under a jobs
directory, no database. A single local analyst's tool doesn't need transactional
guarantees beyond "the file on disk is the source of truth" -- see
docs/OPERATOR_CONSOLE.md. `service.py` dispatches the actual pipeline work (via
`runs.run_real_pipeline`/`fusion.fuse_events`) on a thread pool and updates a `Job`'s
status/summary through this store as it progresses.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

JOB_STATUSES = ("queued", "running", "completed", "failed")
JOB_KINDS = ("run", "fusion")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class Job:
    """One operator-console job. `source_job_ids` is only meaningful for a
    `kind="fusion"` job (the completed run jobs it fuses); `summary` holds
    whatever result counts make sense for `kind` (frames/detections/tracks/
    events/alerts/anomalies for a run, fused-event/contributor counts for a
    fusion) -- deliberately a loose dict rather than a fixed schema, since
    Phase 3's UI is the only consumer and different job kinds legitimately
    report different things.
    """

    job_id: str
    kind: str  # "run" | "fusion"
    status: str = "queued"  # "queued" | "running" | "completed" | "failed"
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    display_name: str = ""
    source: str = ""  # e.g. "video:C:\\clips\\corridor.mp4" or "uavdt:M0601"
    sensor_id: str = "CAM01"
    output_dir: str = ""
    source_job_ids: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        return cls(**data)


class JobStore:
    """Persists `Job`s as one JSON file per job under `root/{job_id}.json`. Rebuilds its
    in-memory index from disk on construction; any job still `queued`/`running` from a
    previous process is marked `failed` on load -- that process is gone, so it can't
    still be running."""

    def __init__(self, root):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            job = Job.from_dict(data)
            if job.status in ("queued", "running"):
                job.status = "failed"
                job.error = "The operator service restarted while this job was in progress."
                job.completed_at = _now_iso()
                self._write(job)
            self._jobs[job.job_id] = job

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _write(self, job: Job) -> None:
        tmp = self._path(job.job_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2))
        os.replace(tmp, self._path(job.job_id))

    def create(self, kind: str, **fields) -> Job:
        if kind not in JOB_KINDS:
            raise ValueError(f"Unknown job kind {kind!r}; expected one of {JOB_KINDS}")
        job = Job(job_id=uuid.uuid4().hex, kind=kind, **fields)
        self._jobs[job.job_id] = job
        self._write(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update(self, job_id: str, **fields) -> Job:
        job = self._jobs[job_id]
        for key, value in fields.items():
            setattr(job, key, value)
        self._write(job)
        return job

    def delete(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        path = self._path(job_id)
        if path.exists():
            path.unlink()
