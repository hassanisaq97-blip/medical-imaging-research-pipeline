"""Pydantic request/response models for the job-queue API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceJobRequest(BaseModel):
    image_path: str = Field(..., description="Path to a NIfTI image, reachable from the worker.")
    checkpoint_path: str = Field(..., description="Path to a trained model checkpoint (.pt).")
    output_dir: str = Field(
        ..., description="Directory to write the predicted mask + metadata into."
    )
    device: str = Field("auto", description="auto | cpu | mps | cuda")


class TrainingJobRequest(BaseModel):
    config_path: str = Field(
        ..., description="Path to a training YAML config, reachable from the worker."
    )


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None
    output_path: str | None = None
    error: str | None = None
    params: dict


class JobLogsResponse(BaseModel):
    job_id: str
    logs: list[str]


class HealthResponse(BaseModel):
    status: str
    redis: str
