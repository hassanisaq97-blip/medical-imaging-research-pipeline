"""Tests for the Redis-backed job queue.

Requires a real Redis instance reachable at REDIS_URL (default
redis://localhost:6379/0) -- these tests are skipped automatically if
none is reachable, so the fast unit-test suite still runs without Redis
installed. CI starts a Redis service container for this reason (see
.github/workflows/ci.yml).
"""

from __future__ import annotations

import pytest

from medimg_pipeline.exceptions import JobNotFoundError, QueueUnavailableError
from medimg_pipeline.queue.jobs import (
    QUEUE_NAME,
    enqueue_inference_job,
    enqueue_training_job,
    get_job,
    get_job_logs,
    get_redis_connection,
    run_inference_job,
    run_training_job,
)


@pytest.fixture()
def redis_conn():
    try:
        conn = get_redis_connection()
    except QueueUnavailableError:
        pytest.skip("Redis is not reachable at REDIS_URL; skipping queue tests.")
    yield conn
    conn.flushdb()


def test_get_job_raises_for_unknown_id(redis_conn):
    with pytest.raises(JobNotFoundError):
        get_job(redis_conn, "does-not-exist")


def test_enqueue_inference_job_starts_queued(redis_conn):
    record = enqueue_inference_job(
        redis_conn, image_path="img.nii.gz", checkpoint_path="model.pt", output_dir="out/"
    )
    fetched = get_job(redis_conn, record.job_id)
    assert fetched.status == "QUEUED"
    assert fetched.job_type == "inference"

    logs = get_job_logs(redis_conn, record.job_id)
    assert any("queued" in line for line in logs)


def test_worker_completes_inference_job(redis_conn, monkeypatch):
    import medimg_pipeline.queue.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "_inference_work", lambda **kwargs: "/tmp/fake_output.nii.gz")

    record = enqueue_inference_job(
        redis_conn, image_path="img.nii.gz", checkpoint_path="model.pt", output_dir="out/"
    )
    # Simulate what an RQ worker process does when it picks up the job.
    run_inference_job(
        job_id=record.job_id, image_path="img.nii.gz", checkpoint_path="model.pt", output_dir="out/"
    )

    fetched = get_job(redis_conn, record.job_id)
    assert fetched.status == "COMPLETED"
    assert fetched.output_path == "/tmp/fake_output.nii.gz"
    assert fetched.started_at is not None
    assert fetched.ended_at is not None


def test_worker_marks_job_failed_on_exception(redis_conn, monkeypatch):
    import medimg_pipeline.queue.jobs as jobs_module

    def _boom(**kwargs):
        raise RuntimeError("synthetic failure for testing")

    monkeypatch.setattr(jobs_module, "_training_work", _boom)

    record = enqueue_training_job(redis_conn, config_path="configs/train_small.yaml")

    with pytest.raises(RuntimeError):
        run_training_job(job_id=record.job_id, config_path="configs/train_small.yaml")

    fetched = get_job(redis_conn, record.job_id)
    assert fetched.status == "FAILED"
    assert "synthetic failure" in fetched.error


def test_real_rq_worker_processes_queued_job(redis_conn, monkeypatch):
    """More faithful integration test: an actual RQ Worker (burst mode)
    pulls the job off the Redis queue and executes it, rather than the
    test calling the job function directly."""

    from rq import Queue, Worker

    import medimg_pipeline.queue.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "_inference_work", lambda **kwargs: "/tmp/fake_output.nii.gz")

    record = enqueue_inference_job(
        redis_conn, image_path="img.nii.gz", checkpoint_path="model.pt", output_dir="out/"
    )

    queue = Queue(QUEUE_NAME, connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    worker.work(burst=True)  # process everything currently queued, then return

    fetched = get_job(redis_conn, record.job_id)
    assert fetched.status == "COMPLETED"
