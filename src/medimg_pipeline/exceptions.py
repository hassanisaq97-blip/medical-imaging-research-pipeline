"""Domain-specific exceptions.

Debugging is a first-class feature of this project (see README §14). Every
exception here carries a short, specific message and, where useful, a
`hint` attribute with a concrete recommended next action, so callers
(including the job-queue worker and the FastAPI layer) can surface
something actionable instead of a bare traceback. None of these exceptions
should ever be constructed with patient-identifying data in the message.
"""

from __future__ import annotations


class MedImgError(Exception):
    """Base class for all medimg_pipeline errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.hint = hint
        full = f"{message} (hint: {hint})" if hint else message
        super().__init__(full)


# --- Imaging I/O -----------------------------------------------------------


class ImageNotFoundError(MedImgError):
    """A referenced image file does not exist on disk."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Image not found: {path}",
            hint="Check the path and that the dataset volume is mounted.",
        )


class CorruptImageError(MedImgError):
    """A NIfTI/DICOM file exists but cannot be parsed or read."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Could not read image at {path}: {reason}",
            hint="Re-download or re-export the file; it may be truncated.",
        )


class UnsupportedModalityError(MedImgError):
    """An image modality is not one this pipeline knows how to handle."""

    def __init__(self, modality: str) -> None:
        super().__init__(
            f"Unsupported modality: {modality}",
            hint="Supported modalities: CT, PET, MRI. Extend "
            "medimg_pipeline.imaging if you need another.",
        )


# --- Shape / geometry validation -------------------------------------------


class ShapeMismatchError(MedImgError):
    """An image and its corresponding mask have incompatible shapes."""

    def __init__(self, image_shape: tuple, mask_shape: tuple) -> None:
        super().__init__(
            f"Image shape {image_shape} does not match mask shape {mask_shape}",
            hint="Resample the mask onto the image grid using nearest-neighbour "
            "interpolation before training or inference.",
        )


class AffineMismatchError(MedImgError):
    """An image and its mask have incompatible affine matrices."""

    def __init__(self, max_abs_diff: float, tolerance: float) -> None:
        super().__init__(
            f"Affine matrices differ by {max_abs_diff:.4g}, exceeding tolerance {tolerance:.4g}",
            hint="Image and mask are not co-registered; check that the mask "
            "was generated from this exact image, not a different session.",
        )


class EmptyMaskError(MedImgError):
    """A segmentation mask has zero foreground voxels."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Mask has no foreground voxels: {path}",
            hint="Exclude this subject from training, or check whether the "
            "correct label index was selected when binarizing the mask.",
        )


class MissingModalityError(MedImgError):
    """A subject is missing a modality required by the current task."""

    def __init__(self, subject_id: str, modality: str) -> None:
        super().__init__(
            f"Subject {subject_id} is missing required modality: {modality}",
            hint="This subject will be excluded from the manifest; see the "
            "QC report for the full exclusion list.",
        )


class MissingSegmentationError(MedImgError):
    """A subject is missing the segmentation mask required by the task."""

    def __init__(self, subject_id: str, label: str) -> None:
        super().__init__(
            f"Subject {subject_id} is missing segmentation label: {label}",
            hint="This subject will be excluded from the manifest.",
        )


# --- Model / checkpoint -----------------------------------------------------


class CheckpointNotFoundError(MedImgError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Model checkpoint not found: {path}",
            hint="Run `medimg-pipeline train` first, or point --checkpoint at "
            "an existing .pt file.",
        )


class DeviceUnavailableError(MedImgError):
    """Requested compute device (cuda/mps) is not available."""

    def __init__(self, requested: str) -> None:
        super().__init__(
            f"Requested device '{requested}' is not available on this machine",
            hint="Use --device auto to fall back automatically, or --device cpu.",
        )


class OutOfMemoryError(MedImgError):
    """GPU/MPS ran out of memory during training or inference."""

    def __init__(self, original: Exception) -> None:
        super().__init__(
            f"Out of device memory: {original}",
            hint="Reduce --batch-size and/or --patch-size in your config, or switch --device cpu.",
        )


# --- DICOM / ingestion -------------------------------------------------------


class DicomConversionError(MedImgError):
    def __init__(self, series_path: str, reason: str) -> None:
        super().__init__(
            f"Failed to convert DICOM series at {series_path} to NIfTI: {reason}",
            hint="Verify the series is a single, complete acquisition and that "
            "dicom2nifti/dcm2niix supports its transfer syntax.",
        )


class MissingExternalToolError(MedImgError):
    """A required external CLI tool (e.g. dcm2niix) is not installed."""

    def __init__(self, tool: str, install_hint: str) -> None:
        super().__init__(
            f"Required external tool not found on PATH: {tool}",
            hint=install_hint,
        )


# --- Job queue ---------------------------------------------------------------


class JobNotFoundError(MedImgError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job not found: {job_id}")


class QueueUnavailableError(MedImgError):
    """Redis / the job queue backend could not be reached."""

    def __init__(self, original: Exception) -> None:
        super().__init__(
            f"Could not reach the job queue backend: {original}",
            hint="Check that Redis is running (`make docker-up` or `redis-cli "
            "ping`) and REDIS_URL is set correctly.",
        )


class WorkerFailureError(MedImgError):
    """A background job raised an exception while running."""

    def __init__(self, job_id: str, reason: str) -> None:
        super().__init__(f"Job {job_id} failed: {reason}")
