# CPU base image. Works for api/worker/tests without a GPU; see README
# §14 for the separate NVIDIA Docker notes needed for real GPU training.
FROM python:3.11-slim AS base

# libgomp1: required by torch/MONAI's OpenMP-based ops.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e .

COPY README.md ./

ENV PYTHONUNBUFFERED=1 \
    MEDIMG_LOG_LEVEL=INFO

EXPOSE 8000

# Default: run the API. docker-compose overrides the command for the
# worker service.
CMD ["python", "-m", "medimg_pipeline", "api"]
