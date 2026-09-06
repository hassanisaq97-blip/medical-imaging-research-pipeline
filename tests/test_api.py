"""Tests for the FastAPI job-queue API. Requires a reachable Redis (see
tests/test_queue.py); skipped automatically if none is available.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medimg_pipeline.exceptions import QueueUnavailableError
from medimg_pipeline.queue.jobs import get_redis_connection


@pytest.fixture()
def client():
    try:
        conn = get_redis_connection()
        conn.flushdb()
    except QueueUnavailableError:
        pytest.skip("Redis is not reachable at REDIS_URL; skipping API tests.")

    from medimg_pipeline.api.main import app

    return TestClient(app)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_submit_inference_job_and_query_status(client):
    response = client.post(
        "/jobs/inference",
        json={"image_path": "img.nii.gz", "checkpoint_path": "model.pt", "output_dir": "out/"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "QUEUED"
    assert body["job_type"] == "inference"
    job_id = body["job_id"]

    status_response = client.get(f"/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["job_id"] == job_id

    logs_response = client.get(f"/jobs/{job_id}/logs")
    assert logs_response.status_code == 200
    assert logs_response.json()["job_id"] == job_id


def test_submit_training_job(client):
    response = client.post("/jobs/training", json={"config_path": "configs/train_small.yaml"})
    assert response.status_code == 200
    assert response.json()["job_type"] == "training"


def test_job_status_404_for_unknown_id(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_job_logs_404_for_unknown_id(client):
    response = client.get("/jobs/does-not-exist/logs")
    assert response.status_code == 404
