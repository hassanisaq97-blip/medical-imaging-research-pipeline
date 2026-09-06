from pathlib import Path

import numpy as np
import pytest
from tests.helpers import make_synthetic_volume_pair

from medimg_pipeline.exceptions import (
    AffineMismatchError,
    CorruptImageError,
    ImageNotFoundError,
    ShapeMismatchError,
)
from medimg_pipeline.imaging.nifti import load_nifti, validate_image_mask_pair


def test_load_nifti_missing_file(tmp_path: Path):
    with pytest.raises(ImageNotFoundError):
        load_nifti(tmp_path / "does_not_exist.nii.gz")


def test_load_nifti_corrupt_file(tmp_path: Path):
    bad = tmp_path / "corrupt.nii.gz"
    bad.write_bytes(b"not a real nifti file")
    with pytest.raises(CorruptImageError):
        load_nifti(bad)


def test_load_nifti_reads_shape_and_spacing(tmp_path: Path):
    image_path, mask_path = make_synthetic_volume_pair(tmp_path, "sub-000", shape=(16, 16, 16))
    array, img, info = load_nifti(image_path)
    assert array.shape == (16, 16, 16)
    assert info.voxel_spacing == (2.0, 2.0, 2.0)
    assert info.shape == (16, 16, 16)


def test_validate_image_mask_pair_ok(tmp_path: Path):
    image_path, mask_path = make_synthetic_volume_pair(tmp_path, "sub-000", shape=(16, 16, 16))
    _, image_img, image_info = load_nifti(image_path)
    _, mask_img, mask_info = load_nifti(mask_path)
    validate_image_mask_pair(image_info.shape, mask_info.shape, image_info.affine, mask_info.affine)


def test_validate_image_mask_pair_shape_mismatch():
    with pytest.raises(ShapeMismatchError):
        validate_image_mask_pair((16, 16, 16), (32, 32, 32), np.eye(4), np.eye(4))


def test_validate_image_mask_pair_affine_mismatch():
    affine_a = np.eye(4)
    affine_b = np.eye(4)
    affine_b[0, 0] = 5.0
    with pytest.raises(AffineMismatchError):
        validate_image_mask_pair((16, 16, 16), (16, 16, 16), affine_a, affine_b)
