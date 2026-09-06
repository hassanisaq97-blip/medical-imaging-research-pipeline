"""Centralized environment-variable configuration.

Loads a local `.env` file (if present) via python-dotenv, mirroring the
convention observed directly in Multimodal-HC's own `utils.py` (see
docs/DEPICT_TECHNICAL_RESEARCH.md §1.6). No path in this project is ever
hard-coded; everything goes through this module so `.env.example`
documents every configurable value in one place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def dataset_root() -> str | None:
    return os.environ.get("DATASET_ROOT")


def output_root() -> str:
    return os.environ.get("MEDIMG_OUTPUT_ROOT", "./outputs")


def device_preference() -> str:
    return os.environ.get("MEDIMG_DEVICE", "auto")


def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def api_host() -> str:
    return os.environ.get("MEDIMG_API_HOST", "0.0.0.0")


def api_port() -> int:
    return int(os.environ.get("MEDIMG_API_PORT", "8000"))


def anonymization_salt() -> str | None:
    """Secret salt for deterministic DICOM pseudonymization.

    Deliberately not given a hard-coded default: a shared default salt
    would make pseudonymization trivially reversible for anyone who reads
    this source code. If unset, callers should generate a random one and
    warn that it must be saved to reproduce the same pseudonym mapping.
    """

    return os.environ.get("MEDIMG_ANON_SALT")
