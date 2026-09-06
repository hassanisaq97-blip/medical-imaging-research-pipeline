"""Inference pipeline: NIfTI in, segmentation NIfTI + metadata JSON out.

Runs the same deterministic preprocessing as validation
(`medimg_pipeline.preprocessing.transforms.get_inference_transforms`),
then resamples the predicted mask back onto the ORIGINAL image's grid
before saving, so the output `.nii.gz` overlays correctly on the input
image in any viewer -- it deliberately does not just save the prediction
at the (possibly resampled) model-space resolution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from medimg_pipeline.exceptions import CheckpointNotFoundError, OutOfMemoryError
from medimg_pipeline.imaging.nifti import load_nifti
from medimg_pipeline.utils.device import resolve_device
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("inference.infer")


@dataclass
class InferenceResult:
    input_path: str
    output_mask_path: str
    metadata_path: str
    foreground_voxels: int
    input_shape: tuple[int, ...]
    device: str
    runtime_seconds: float
    model_checkpoint: str


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def run_inference(
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
) -> InferenceResult:
    import torch
    from monai.inferers import sliding_window_inference

    from medimg_pipeline.models.unet import build_unet
    from medimg_pipeline.preprocessing.transforms import get_inference_transforms

    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(str(checkpoint_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = checkpoint.get("config", {})

    model = build_unet(
        channels=tuple(config.get("channels", (16, 32, 64, 128, 256))),
        strides=tuple(config.get("strides", (2, 2, 2, 2))),
        num_res_units=config.get("num_res_units", 2),
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, original_img, original_info = load_nifti(image_path)

    transforms = get_inference_transforms()
    data = transforms({"image": str(image_path)})
    image_tensor = data["image"].unsqueeze(0).to(resolved_device)  # add batch dim

    patch_size = tuple(config.get("patch_size", (96, 96, 96)))

    try:
        with torch.no_grad():
            logits = sliding_window_inference(
                image_tensor, roi_size=patch_size, sw_batch_size=1, predictor=model
            )
        pred = torch.argmax(logits, dim=1, keepdim=True)[0, 0].cpu().numpy().astype(np.uint8)
    except RuntimeError as exc:
        if _is_oom_error(exc):
            raise OutOfMemoryError(exc) from exc
        raise

    # Resample the prediction back onto the original image's grid using
    # nearest-neighbour interpolation (it is a label map), so the output
    # overlays correctly on the un-preprocessed input.
    resampled_pred = _resample_nearest_to_shape(pred, original_info.shape)

    import nibabel as nib

    output_mask_path = output_dir / f"{image_path.stem.replace('.nii', '')}_seg.nii.gz"
    nib.save(nib.Nifti1Image(resampled_pred, original_info.affine), str(output_mask_path))

    foreground_voxels = int(np.sum(resampled_pred > 0))
    runtime = time.time() - start

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(image_path),
        "output_mask_path": str(output_mask_path),
        "model_checkpoint": str(checkpoint_path),
        "device": str(resolved_device),
        "runtime_seconds": runtime,
        "input_shape": list(original_info.shape),
        "input_voxel_spacing": [round(v, 4) for v in original_info.voxel_spacing],
        "foreground_voxels": foreground_voxels,
        "qc_status": "pass" if foreground_voxels > 0 else "warning_empty_prediction",
        "disclaimer": "Research use only. Not a certified medical device. Not for clinical decision-making.",
    }
    metadata_path = output_dir / f"{image_path.stem.replace('.nii', '')}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    result = InferenceResult(
        input_path=str(image_path),
        output_mask_path=str(output_mask_path),
        metadata_path=str(metadata_path),
        foreground_voxels=foreground_voxels,
        input_shape=original_info.shape,
        device=str(resolved_device),
        runtime_seconds=runtime,
        model_checkpoint=str(checkpoint_path),
    )
    logger.info(
        "Inference complete: %s -> %s (%d foreground voxels, %.2fs)",
        image_path,
        output_mask_path,
        foreground_voxels,
        runtime,
    )
    return result


def _resample_nearest_to_shape(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Nearest-neighbour resample of a label map to `target_shape`.

    Used only to bring the model-space prediction back onto the original
    image's voxel grid. Implemented via `scipy.ndimage.zoom(order=0)`
    rather than any interpolation that could introduce fractional /
    invalid label values.
    """

    if tuple(mask.shape) == tuple(target_shape):
        return mask

    from scipy.ndimage import zoom

    factors = [t / s for t, s in zip(target_shape, mask.shape, strict=True)]
    resampled = zoom(mask, factors, order=0, mode="nearest")
    # Guard against off-by-one rounding in zoom's output shape.
    resampled = resampled[: target_shape[0], : target_shape[1], : target_shape[2]]
    if resampled.shape != tuple(target_shape):
        padded = np.zeros(target_shape, dtype=resampled.dtype)
        slices = tuple(slice(0, s) for s in resampled.shape)
        padded[slices] = resampled
        resampled = padded
    return resampled.astype(np.uint8)
