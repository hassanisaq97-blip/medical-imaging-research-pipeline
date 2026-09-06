"""RQ worker entry point: `python -m medimg_pipeline worker`.

Listens on the `medimg_pipeline` queue and executes inference/training
jobs enqueued by the API (see medimg_pipeline.queue.jobs). Run one or
more of these as separate processes/containers (see docker-compose.yml);
RQ's default worker executes one job at a time per process.
"""

from __future__ import annotations

from medimg_pipeline.queue.jobs import QUEUE_NAME, get_redis_connection
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("queue.worker")


def main() -> None:
    from rq import Queue, Worker

    conn = get_redis_connection()
    queue = Queue(QUEUE_NAME, connection=conn)
    logger.info("Starting RQ worker on queue '%s'", QUEUE_NAME)
    worker = Worker([queue], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
