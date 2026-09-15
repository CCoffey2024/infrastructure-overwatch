import time

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from infrastructure_overwatch.detectors.grid_cnn import GridDetector
from infrastructure_overwatch.ingest import VEHICLE_GRID_H, VEHICLE_GRID_W
from infrastructure_overwatch.service import OperatorService, create_app


def _write_synthetic_video(path, n_frames: int = 8, size=(640, 352), fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), fill_value=i * 10 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_untrained_weights(path, n_classes: int = 3) -> None:
    model = GridDetector(
        n_classes=n_classes, grid_h=VEHICLE_GRID_H, grid_w=VEHICLE_GRID_W, stage_channels=(24, 48, 96, 96)
    )
    torch.save(model.state_dict(), path)


@pytest.fixture
def client(tmp_path):
    weights_path = tmp_path / "weights.pt"
    _write_untrained_weights(weights_path)
    service = OperatorService(workspace_dir=tmp_path / "workspace", default_weights_path=str(weights_path))
    app = create_app(service)
    with TestClient(app) as c:
        c.service = service  # stashed for direct access where a test needs it
        c.video_path = tmp_path / "clip.mp4"
        _write_synthetic_video(c.video_path, n_frames=8)
        yield c
    service.shutdown()


def _wait_for_status(client, job_id: str, status: str = "completed", timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not reach a terminal status within {timeout_s}s")


def test_health(tmp_path):
    # create_app()'s default OperatorService(workspace_dir="outputs") would otherwise
    # create a real outputs/jobs/ in the repo's own working directory as a side effect.
    service = OperatorService(workspace_dir=tmp_path)
    with TestClient(create_app(service)) as c:
        assert c.get("/api/health").json() == {"status": "ok"}
    service.shutdown()


def test_submit_run_returns_202_and_a_queued_or_running_job(client):
    resp = client.post("/api/jobs", json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] in ("queued", "running", "completed")
    assert body["source"] == f"video:{client.video_path}"


def test_submit_run_without_a_source_is_rejected(client):
    resp = client.post("/api/jobs", json={})
    assert resp.status_code == 422


def test_get_unknown_job_is_404(client):
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_full_run_lifecycle_completes_and_produces_evidence(client):
    resp = client.post(
        "/api/jobs",
        json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1, "sensor_id": "EO_1"},
    )
    job_id = resp.json()["job_id"]

    job = _wait_for_status(client, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["summary"]["frames"] == 8
    assert job["summary"]["has_ground_truth"] is False

    events = client.get(f"/api/jobs/{job_id}/events").json()
    assert isinstance(events, list)

    alerts = client.get(f"/api/jobs/{job_id}/alerts").json()
    assert isinstance(alerts, list)


def test_metrics_404_for_a_source_without_ground_truth(client):
    resp = client.post("/api/jobs", json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)

    resp = client.get(f"/api/jobs/{job_id}/metrics")
    assert resp.status_code == 404


def test_frame_endpoint_returns_a_jpeg(client):
    resp = client.post("/api/jobs", json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)

    resp = client.get(f"/api/jobs/{job_id}/frame/0")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_frame_endpoint_404_out_of_range(client):
    resp = client.post("/api/jobs", json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)

    resp = client.get(f"/api/jobs/{job_id}/frame/9999")
    assert resp.status_code == 404


def test_run_job_for_a_nonexistent_video_fails_with_a_readable_error(client):
    resp = client.post("/api/jobs", json={"source": "video:C:\\does\\not\\exist.mp4"})
    job_id = resp.json()["job_id"]
    job = _wait_for_status(client, job_id)
    assert job["status"] == "failed"
    assert job["error"]


def test_delete_removes_a_completed_job(client):
    resp = client.post("/api/jobs", json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)

    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_fusion_requires_at_least_two_source_jobs(client):
    resp = client.post("/api/jobs/fuse", json={"source_job_ids": ["only-one"]})
    assert resp.status_code == 422


def test_fusion_end_to_end_across_two_completed_runs(client):
    r1 = client.post(
        "/api/jobs",
        json={
            "source": f"video:{client.video_path}",
            "conf_thresh": 0.01,
            "stride": 1,
            "track_min_hits": 1,
            "sensor_id": "EO_1",
        },
    )
    r2 = client.post(
        "/api/jobs",
        json={
            "source": f"video:{client.video_path}",
            "conf_thresh": 0.01,
            "stride": 1,
            "track_min_hits": 1,
            "sensor_id": "IR_1",
        },
    )
    job1 = _wait_for_status(client, r1.json()["job_id"])
    job2 = _wait_for_status(client, r2.json()["job_id"])
    assert job1["status"] == job2["status"] == "completed"

    resp = client.post("/api/jobs/fuse", json={"source_job_ids": [job1["job_id"], job2["job_id"]]})
    assert resp.status_code == 202
    fusion_job = _wait_for_status(client, resp.json()["job_id"])
    assert fusion_job["status"] == "completed", fusion_job.get("error")
    assert fusion_job["kind"] == "fusion"
    assert fusion_job["summary"]["distinct_sensors"] == 2

    fused_events = client.get(f"/api/jobs/{fusion_job['job_id']}/fusion_events").json()
    assert isinstance(fused_events, list)


def test_fusion_fails_with_fewer_than_two_distinct_sensors(client):
    # both runs default to the same sensor_id -- fusion should refuse, not silently no-op
    r1 = client.post("/api/jobs", json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1})
    r2 = client.post("/api/jobs", json={"source": f"video:{client.video_path}", "conf_thresh": 0.01, "stride": 1})
    job1 = _wait_for_status(client, r1.json()["job_id"])
    job2 = _wait_for_status(client, r2.json()["job_id"])

    resp = client.post("/api/jobs/fuse", json={"source_job_ids": [job1["job_id"], job2["job_id"]]})
    fusion_job = _wait_for_status(client, resp.json()["job_id"])
    assert fusion_job["status"] == "failed"
    assert "distinct sensor" in fusion_job["error"]
