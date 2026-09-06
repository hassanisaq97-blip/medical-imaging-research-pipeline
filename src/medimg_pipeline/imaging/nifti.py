"""NIfTI loading and validation helpers.

This is the single place responsible for opening a `.nii`/`.nii.gz` file
and checking it is sane before it is used anywhere else in the pipeline
(curation, preprocessing, training, inference). Centralizing this means
every consumer gets the same corrupt-file / shape / affine handling and
the same custom exceptions instead of a raw nibabel traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medimg_pipeline.exceptions import (
    AffineMismatchError,
    CorruptImageError,
    ImageNotFoundError,
    ShapeMismatchError,
)


@dataclass
class ImageInfo:
    """Lightweight, non-identifying summary of a loaded NIfTI volume."""

    path: str
    shape: tuple[int, ...]
    voxel_spacing: tuple[float, ...]
    affine: np.ndarray
    dtype: str

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "voxel_spacing": [round(float(v), 4) for v in self.voxel_spacing],
            "dtype": self.dtype,
        }


def load_nifti(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image, ImageInfo]:  # noqa: F821
    """Load a NIfTI file, returning (array, nibabel image, ImageInfo).

    Raises ImageNotFoundError if the path does not exist, CorruptImageError
    if it exists but cannot be parsed as NIfTI.
    """

    import nibabel as nib

    path = Path(path)
    if not path.exists():
        raise ImageNotFoundError(str(path))

    try:
        img = nib.load(str(path))
        array = np.asanyarray(img.dataobj)
    except Exception as exc:  # noqa: BLE001 - re-raise as our own type
        raise CorruptImageError(str(path), str(exc)) from exc

    info = ImageInfo(
        path=str(path),
        shape=tuple(array.shape),
        voxel_spacing=tuple(float(z) for z in img.header.get_zooms()[:3]),
        affine=np.asarray(img.affine),
        dtype=str(array.dtype),
    )
    return array, img, info


def voxel_spacing(img: nib.Nifti1Image) -> tuple[float, ...]:  # noqa: F821
    return tuple(float(z) for z in img.header.get_zooms()[:3])


def validate_affine_match(
    affine_a: np.ndarray, affine_b: np.ndarray, *, tolerance: float = 1e-3
) -> None:
    """Raise AffineMismatchError if two affines differ beyond tolerance."""

    diff = np.abs(np.asarray(affine_a) - np.asarray(affine_b))
    max_diff = float(diff.max())
    if max_diff > tolerance:
        raise AffineMismatchError(max_diff, tolerance)


def validate_image_mask_pair(
    image_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    image_affine: np.ndarray,
    mask_affine: np.ndarray,
    *,
    affine_tolerance: float = 1e-3,
) -> None:
    """Raise ShapeMismatchError / AffineMismatchError if image and mask
    are not on the same grid. Segmentation masks must always be resampled
    onto the image grid with nearest-neighbour interpolation upstream of
    this check (see docs/DEPICT_TECHNICAL_RESEARCH.md §1.4 and
    medimg_pipeline.preprocessing.transforms).
    """

    if tuple(image_shape) != tuple(mask_shape):
        raise ShapeMismatchError(tuple(image_shape), tuple(mask_shape))
    validate_affine_match(image_affine, mask_affine, tolerance=affine_tolerance)


def foreground_voxel_count(mask: np.ndarray, label: int = 1) -> int:
    """Count voxels equal to `label` in a mask array."""

    return int(np.sum(mask == label))
