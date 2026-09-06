"""MONAI-based preprocessing and augmentation pipelines.

Every documented preprocessing choice below matters for correctness, not
just style:

- **Interpolation for masks is always nearest-neighbour** (`mode=("bilinear",
  "nearest")` in every `Spacingd`/`Resized`/`Rotate90d`/`RandAffined`-style
  transform that touches both image and label). Any other interpolation
  mode on a label map invents intermediate class values that never
  existed in the original segmentation (e.g. a voxel with value 0.5
  between background=0 and liver=1), which silently corrupts training
  labels. This mirrors the convention observed directly in
  Multimodal-HC's own derivative-resampling code (`order=0` throughout,
  see docs/DEPICT_TECHNICAL_RESEARCH.md §1.4).
- **Intensity clipping/normalization is CT-appropriate**: CT values are
  physically meaningful Hounsfield Units, so this project clips to a
  fixed HU window before z-scoring rather than doing dataset-relative
  percentile normalization (which would make the same tissue map to
  different values depending on what else was in the field of view).
- **Augmentation only happens in the training pipeline.** The eval/
  inference pipeline is deterministic: same input always produces the
  same preprocessed tensor, which matters for reproducible QC and for
  inference correctness.
- **Foreground cropping** during training focuses patches near the organ
  of interest (using `RandCropByPosNegLabeld`) so a small 3D patch model
  actually sees positive examples often enough to learn, given how small
  a fixed-size patch is relative to a whole-body volume.
"""

from __future__ import annotations

# CT intensity window for soft tissue / abdominal organs (liver, spleen,
# kidneys). Chosen as a standard abdominal soft-tissue window, not fit to
# any particular dataset. Values in Hounsfield Units.
CT_WINDOW_MIN = -200.0
CT_WINDOW_MAX = 250.0


def get_train_transforms(patch_size: tuple[int, int, int] = (96, 96, 96)):
    """Training-time transform pipeline: preprocessing + augmentation.

    Expects dict samples with keys "image" and "label", each a path to a
    NIfTI file (LoadImaged handles the actual file I/O).
    """

    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        RandAffined,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandShiftIntensityd,
        ScaleIntensityRanged,
        Spacingd,
    )

    keys = ["image", "label"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=CT_WINDOW_MIN,
                a_max=CT_WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandAffined(
                keys=keys,
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
    )


def get_eval_transforms(patch_size: tuple[int, int, int] | None = None):
    """Deterministic preprocessing pipeline used for validation and inference.

    No augmentation, no random cropping -- the full (resampled, normalized)
    volume is returned so validation metrics and inference outputs are
    computed on the whole image, at a fixed, reproducible resolution.
    """

    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    keys = ["image", "label"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
            EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
            Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=keys,
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=CT_WINDOW_MIN,
                a_max=CT_WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )


def get_inference_transforms():
    """Deterministic pipeline for a single image with no label available."""

    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    keys = ["image"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            ScaleIntensityRanged(
                keys=keys,
                a_min=CT_WINDOW_MIN,
                a_max=CT_WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )
