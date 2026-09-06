"""Job model and Redis-backed job store/queue.

Architecture (see README §12):

    FastAPI (medimg_pipeline.api) -> Redis-backed queue (RQ) -> worker
        -> segmentation/training/inference job -> result -> status/log

Job metadata (id, type, status, timestamps, sanitized error, output path)
is stored as a JSON blob in Redis under `medimg:job:{id}:meta`; log lines
are stored in a Redis list under `medimg:job:{id}:logs`. RQ itself is used
only to actually dispatch work to a worker process; this project's own
job metadata is the source of truth for status shown to API clients, so
that the API's job-state contract (QUEUED/RUNNING/COMPLETED/FAILED) does
not depend on RQ's internal job-state vocabulary.

No patient-identifying information is ever stored in job metadata or
logs: job parameters are limited to file paths and config, and error
messages are the sanitized messages produced by this project's own
`medimg_pipeline.exceptions` (or `str(exc)` for unexpected errors, which
in this codebase never interpolates DICOM tag values -- see
medimg_pipeline.anonymization.dicom for why).
"""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from medimg_pipeline.exceptions import JobNotFoundError, QueueUnavailableError
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("queue.jobs")

_KEY_PREFIX = "medimg:job"
QUEUE_NAME = "medimg_pipeline"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobType(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: str
    created_at: str
    params: dict[str, Any]
    started_at: str | None = None
    ended_at: str | None = None
    output_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def get_redis_connection(url: str | None = None):
    import redis

    from medimg_pipeline.utils.env import redis_url

    conn = redis.from_url(url or redis_url())
    try:
        conn.ping()
    except Exception as exc:  # noqa: BLE001
        raise QueueUnavailableError(exc) from exc
    return conn


def _meta_key(job_id: str) -> str:
    return f"{_KEY_PREFIX}:{job_id}:meta"


def _logs_key(job_id: str) -> str:
    return f"{_KEY_PREFIX}:{job_id}:logs"


def _save(conn, record: JobRecord) -> None:
    conn.set(_meta_key(record.job_id), json.dumps(record.to_dict()))


def get_job(conn, job_id: str) -> JobRecord:
    raw = conn.get(_meta_key(job_id))
    if raw is None:
        raise JobNotFoundError(job_id)
    return JobRecord(**json.loads(raw))


def get_job_logs(conn, job_id: str) -> list[str]:
    if conn.get(_meta_key(job_id)) is None:
        raise JobNotFoundError(job_id)
    raw_lines = conn.lrange(_logs_key(job_id), 0, -1)
    return [line.decode("utf-8") if isinstance(line, bytes) else line for line in raw_lines]


def _append_log(conn, job_id: str, message: str) -> None:
    conn.rpush(_logs_key(job_id), message)


@contextlib.contextmanager
def _capture_logs_to_redis(conn, job_id: str):
    """Temporarily mirror `medimg_pipeline`-logger output into Redis for this job."""

    class _RedisHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                _append_log(conn, job_id, self.format(record))
            except Exception:  # noqa: BLE001 - never let logging break the job
                pass

    handler = _RedisHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root = logging.getLogger("medimg_pipeline")
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)


def _create_job(conn, job_type: JobType, params: dict[str, Any]) -> JobRecord:
    job_id = str(uuid.uuid4())
    record = JobRecord(
        job_id=job_id,
        job_type=job_type.value,
        status=JobStatus.QUEUED.value,
        created_at=datetime.now(timezone.utc).isoformat(),
        params=params,
    )
    _save(conn, record)
    _append_log(conn, job_id, f"Job {job_id} ({job_type.value}) queued.")
    return record


def _run_job(job_id: str, job_type: JobType, work_fn, params: dict[str, Any]) -> None:
    """Executed by the RQ worker process. Updates job status/logs/result."""

    conn = get_redis_connection()
    record = get_job(conn, job_id)
    record.status = JobStatus.RUNNING.value
    record.started_at = datetime.now(timezone.utc).isoformat()
    _save(conn, record)

    with _capture_logs_to_redis(conn, job_id):
        try:
            output_path = work_fn(**params)
            record = get_job(conn, job_id)
            record.status = JobStatus.COMPLETED.value
            record.ended_at = datetime.now(timezone.utc).isoformat()
            record.output_path = str(output_path) if output_path is not None else None
            _save(conn, record)
            logger.info("Job %s completed successfully.", job_id)
        except Exception as exc:  # noqa: BLE001
            record = get_job(conn, job_id)
            record.status = JobStatus.FAILED.value
            record.ended_at = datetime.now(timezone.utc).isoformat()
            record.error = str(exc)
            _save(conn, record)
            logger.error("Job %s failed: %s", job_id, exc)
            raise


def _inference_work(
    image_path: str, checkpoint_path: str, output_dir: str, device: str = "auto"
) -> str:
    from medimg_pipeline.inference.infer import run_inference

    result = run_inference(image_path, checkpoint_path, output_dir, device=device)
    return result.output_mask_path


def _training_work(config_path: str) -> str:
    from medimg_pipeline.training.config import TrainConfig
    from medimg_pipeline.training.train import run_training

    config = TrainConfig.from_yaml(config_path)
    checkpoint_path = run_training(config)
    return str(checkpoint_path)


def run_inference_job(
    job_id: str, image_path: str, checkpoint_path: str, output_dir: str, device: str = "auto"
) -> None:
    """RQ entry point for an inference job (must be importable by workers)."""

    _run_job(
        job_id,
        JobType.INFERENCE,
        _inference_work,
        {
            "image_path": image_path,
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir,
            "device": device,
        },
    )


def run_training_job(job_id: str, config_path: str) -> None:
    """RQ entry point for a training job (must be importable by workers)."""

    _run_job(job_id, JobType.TRAINING, _training_work, {"config_path": config_path})


def enqueue_inference_job(
    conn, image_path: str, checkpoint_path: str, output_dir: str, *, device: str = "auto"
) -> JobRecord:
    from rq import Queue

    record = _create_job(
        conn,
        JobType.INFERENCE,
        {
            "image_path": image_path,
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir,
            "device": device,
        },
    )
    queue = Queue(QUEUE_NAME, connection=conn)
    queue.enqueue(
        run_inference_job,
        job_id=record.job_id,
        kwargs={
            "job_id": record.job_id,
            "image_path": image_path,
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir,
            "device": device,
        },
    )
    return record


def enqueue_training_job(conn, config_path: str) -> JobRecord:
    from rq import Queue

    record = _create_job(conn, JobType.TRAINING, {"config_path": config_path})
    queue = Queue(QUEUE_NAME, connection=conn)
    queue.enqueue(
        run_training_job,
        job_id=record.job_id,
        kwargs={"job_id": record.job_id, "config_path": config_path},
    )
    return record
