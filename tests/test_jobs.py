import json

import pytest

from infrastructure_overwatch.jobs import Job, JobStore


def test_create_persists_a_job_file(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(kind="run", source="video:clip.mp4", sensor_id="EO_1")

    assert job.status == "queued"
    assert job.job_id  # a real id was assigned
    path = tmp_path / f"{job.job_id}.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["source"] == "video:clip.mp4"
    assert on_disk["sensor_id"] == "EO_1"


def test_create_rejects_an_unknown_kind(tmp_path):
    store = JobStore(tmp_path)
    with pytest.raises(ValueError):
        store.create(kind="teleport")


def test_get_returns_none_for_an_unknown_id(tmp_path):
    store = JobStore(tmp_path)
    assert store.get("does-not-exist") is None


def test_update_mutates_and_persists(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(kind="run")
    # "completed", not "running" -- a "running" job left on disk is exactly what the
    # crash-recovery test below exercises, and would be reclassified "failed" on reload.
    store.update(job.job_id, status="completed", started_at="2026-01-01T00:00:00.000Z")

    reloaded = JobStore(tmp_path)  # force a fresh read from disk
    updated = reloaded.get(job.job_id)
    assert updated.status == "completed"
    assert updated.started_at == "2026-01-01T00:00:00.000Z"


def test_list_returns_newest_first(tmp_path):
    store = JobStore(tmp_path)
    first = store.create(kind="run", created_at="2026-01-01T00:00:00.000Z")
    second = store.create(kind="run", created_at="2026-01-02T00:00:00.000Z")

    ids_in_order = [j.job_id for j in store.list()]
    assert ids_in_order == [second.job_id, first.job_id]


def test_delete_removes_the_job_file(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(kind="run")
    path = tmp_path / f"{job.job_id}.json"
    assert path.exists()

    store.delete(job.job_id)
    assert not path.exists()
    assert store.get(job.job_id) is None


def test_delete_is_a_no_op_for_an_unknown_id(tmp_path):
    store = JobStore(tmp_path)
    store.delete("does-not-exist")  # must not raise


def test_a_job_left_running_is_marked_failed_on_the_next_load(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(kind="run")
    store.update(job.job_id, status="running", started_at="2026-01-01T00:00:00.000Z")

    # simulate the process restarting -- a fresh JobStore over the same directory
    restarted = JobStore(tmp_path)
    recovered = restarted.get(job.job_id)
    assert recovered.status == "failed"
    assert recovered.error is not None
    assert recovered.completed_at is not None


def test_a_completed_job_is_untouched_by_a_restart(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(kind="run")
    store.update(job.job_id, status="completed", summary={"detections": 42})

    restarted = JobStore(tmp_path)
    recovered = restarted.get(job.job_id)
    assert recovered.status == "completed"
    assert recovered.error is None
    assert recovered.summary == {"detections": 42}


def test_job_round_trips_through_to_dict_and_from_dict():
    job = Job(job_id="abc123", kind="run", source="uavdt:M0601", summary={"frames": 10})
    restored = Job.from_dict(job.to_dict())
    assert restored == job
