"""FastAPI job-queue API.

    POST /jobs/inference   -> enqueue an inference job, returns job_id
    POST /jobs/training    -> enqueue a training job, returns job_id
    GET  /jobs/{job_id}    -> job status/metadata
    GET  /jobs/{job_id}/logs -> log lines captured from the worker
    GET  /health           -> liveness + Redis connectivity

See README §12 for the full architecture. Errors from this project's own
exception hierarchy (medimg_pipeline.exceptions) are translated into
appropriate HTTP status codes with a sanitized message rather than a raw
500 traceback; unexpected errors still return 500 but are logged
server-side with a request-scoped logger.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from medimg_pipeline.api.schemas import (
    HealthResponse,
    InferenceJobRequest,
    JobLogsResponse,
    JobResponse,
    TrainingJobRequest,
)
from medimg_pipeline.exceptions import JobNotFoundError, MedImgError, QueueUnavailableError
from medimg_pipeline.queue.jobs import (
    enqueue_inference_job,
    enqueue_training_job,
    get_job,
    get_job_logs,
    get_redis_connection,
)
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("api.main")

app = FastAPI(
    title="medimg-pipeline API",
    description=(
        "Job-queue API for medical-imaging segmentation training/inference. "
        "Research software; not a certified medical device."
    ),
    version="0.1.0",
)


def _job_to_response(record) -> JobResponse:
    return JobResponse(**record.to_dict())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_redis_connection()
        return HealthResponse(status="ok", redis="ok")
    except QueueUnavailableError as exc:
        logger.warning("Health check: Redis unavailable: %s", exc)
        return HealthResponse(status="degraded", redis="unavailable")


@app.post("/jobs/inference", response_model=JobResponse)
def submit_inference_job(request: InferenceJobRequest) -> JobResponse:
    try:
        conn = get_redis_connection()
        record = enqueue_inference_job(
            conn,
            image_path=request.image_path,
            checkpoint_path=request.checkpoint_path,
            output_dir=request.output_dir,
            device=request.device,
        )
        return _job_to_response(record)
    except QueueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/jobs/training", response_model=JobResponse)
def submit_training_job(request: TrainingJobRequest) -> JobResponse:
    try:
        conn = get_redis_connection()
        record = enqueue_training_job(conn, config_path=request.config_path)
        return _job_to_response(record)
    except QueueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str) -> JobResponse:
    try:
        conn = get_redis_connection()
        record = get_job(conn, job_id)
        return _job_to_response(record)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QueueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
def job_logs(job_id: str) -> JobLogsResponse:
    try:
        conn = get_redis_connection()
        logs = get_job_logs(conn, job_id)
        return JobLogsResponse(job_id=job_id, logs=logs)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QueueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.exception_handler(MedImgError)
def _medimg_error_handler(request, exc: MedImgError) -> JSONResponse:
    logger.error("Unhandled MedImgError on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
