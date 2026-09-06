"""Structured logging.

Every log record carries a timestamp, component name, severity, and an
optional job_id, and never carries patient identifiers (see README
Security/Privacy section and anonymization docs). Callers should pass
subject IDs only in their already-pseudonymous form (e.g. "sub-000"),
never a real name, MRN, or accession number.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


class JobContextFilter(logging.Filter):
    """Injects a `job_id` field (default "-") so the format string always works."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        return True


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("MEDIMG_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(JobContextFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | job=%(job_id)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root = logging.getLogger("medimg_pipeline")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(component: str) -> logging.Logger:
    """Return a logger named `medimg_pipeline.<component>`.

    Usage: ``logger = get_logger(__name__)`` or a short component name like
    ``get_logger("curation")``.
    """

    _configure_root()
    name = component if component.startswith("medimg_pipeline") else f"medimg_pipeline.{component}"
    return logging.getLogger(name)


def with_job_id(logger: logging.Logger, job_id: str) -> logging.LoggerAdapter:
    """Return an adapter that stamps every record with the given job_id."""

    return logging.LoggerAdapter(logger, {"job_id": job_id})
