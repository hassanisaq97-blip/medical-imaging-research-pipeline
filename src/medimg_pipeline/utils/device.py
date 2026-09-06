"""Automatic compute device selection: CUDA -> MPS -> CPU.

See README §8/§13: the project must run somewhere sensible on any
machine, and must fail with a clear, actionable error rather than a raw
stack trace when a specific device is requested but unavailable.
"""

from __future__ import annotations

from medimg_pipeline.exceptions import DeviceUnavailableError
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("utils.device")


def resolve_device(requested: str = "auto") -> torch.device:  # noqa: F821
    """Resolve a device string to a torch.device.

    Parameters
    ----------
    requested:
        One of "auto", "cpu", "mps", "cuda". "auto" tries CUDA, then MPS,
        then CPU, and never raises.
    """

    import torch

    requested = (requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            chosen = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            chosen = "mps"
        else:
            chosen = "cpu"
        logger.info("Auto-selected device: %s", chosen)
        return torch.device(chosen)

    if requested == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("cuda")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise DeviceUnavailableError("mps")
    if requested not in {"cpu", "cuda", "mps"}:
        raise DeviceUnavailableError(requested)

    return torch.device(requested)


def supports_amp(device: torch.device) -> bool:  # noqa: F821
    """Whether automatic mixed precision is supported on this device.

    CUDA: yes. MPS: PyTorch's autocast support for MPS is partial/version
    dependent, so we conservatively disable it and fall back to full
    precision rather than risk silently-wrong results. CPU: no.
    """

    return device.type == "cuda"
