"""Synthetic data generators used by tests (and by `make examples`).

None of these functions touch real patient data; everything here is
procedurally generated. Fake "identifying" values used in DICOM fixtures
(names, IDs) are obviously fabricated placeholders, used only to verify
that the anonymization module removes them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_synthetic_volume_pair(
    root: Path,
    subject: str,
    *,
    shape: tuple[int, int, int] = (32, 32, 32),
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    with_mask: bool = True,
    empty_mask: bool = False,
    label_value: int = 1,
    mask_dir_name: str = "seg",
    mask_suffix: str = "seg-total",
) -> tuple[Path, Path | None]:
    """Write a synthetic CT-like image (+ optional segmentation mask) for one
    subject, in the layout `medimg_pipeline.curation` expects by default:

        <root>/<subject>/ct/<subject>_ct.nii.gz
        <root>/<subject>/seg/<subject>_seg-total.nii.gz

    The "organ" is a solid ellipsoid blob roughly in the center of the
    volume, giving a stable, sizeable foreground region -- analogous in
    spirit (not scale) to the liver-segmentation target described in
    docs/segmentation_target_selection.md.
    """

    import nibabel as nib

    rng = np.random.default_rng(abs(hash(subject)) % (2**32))
    image = rng.normal(loc=40.0, scale=15.0, size=shape).astype(np.float32)

    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    center = np.array(shape) / 2
    radii = np.array(shape) / 4
    ellipsoid = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0
    image[ellipsoid] += 80.0  # organ is brighter than background, like contrast-enhanced tissue

    affine = np.diag([*spacing, 1.0]).astype(np.float32)

    img_dir = root / subject / "ct"
    img_dir.mkdir(parents=True, exist_ok=True)
    image_path = img_dir / f"{subject}_ct.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), str(image_path))

    if not with_mask:
        return image_path, None

    mask = np.zeros(shape, dtype=np.uint8)
    if not empty_mask:
        mask[ellipsoid] = label_value

    mask_dir = root / subject / mask_dir_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{subject}_{mask_suffix}.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))

    return image_path, mask_path


def make_synthetic_dicom_series(
    root: Path,
    *,
    n_slices: int = 3,
    patient_name: str = "Doe^Jane^FAKE",
    patient_id: str = "FAKE-PATIENT-0001",
    rows: int = 16,
    cols: int = 16,
) -> Path:
    """Write a tiny synthetic DICOM series with obviously fake PHI-like
    fields, for testing the anonymization module. Returns the directory.
    """

    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    series_dir = root / "synthetic_series"
    series_dir.mkdir(parents=True, exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()

    for i in range(n_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # --- Fake identifying fields (must be removed by anonymization) ---
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = "19800101"
        ds.PatientAddress = "123 Fake Street, Nowhere"
        ds.ReferringPhysicianName = "Smith^Robert^FAKE"
        ds.InstitutionName = "Fake General Hospital"
        ds.AccessionNumber = "ACC-FAKE-0001"
        ds.ImageComments = "Patient Jane Doe, fake note for testing"

        # --- Technical/geometry fields (must be preserved) ---
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "CT"
        ds.Manufacturer = "SyntheticVendor"
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * 2.0]
        ds.SliceLocation = float(i) * 2.0
        ds.PatientPosition = "HFS"
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        pixels = np.full((rows, cols), fill_value=100 + i, dtype=np.int16)
        ds.PixelData = pixels.tobytes()

        # Add a private tag to verify private-tag stripping.
        block = ds.private_block(0x0041, "FAKE PRIVATE CREATOR", create=True)
        block.add_new(0x01, "LO", "fake-private-value")

        ds.save_as(
            str(series_dir / f"slice_{i:03d}.dcm"),
            enforce_file_format=True,
            little_endian=True,
            implicit_vr=False,
        )

    return series_dir


def _write_dicom_series(
    directory: Path,
    *,
    n_slices: int,
    rows: int,
    cols: int,
    pixel_fn,
    study_uid: str,
    patient_id: str,
) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()

    for i in range(n_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "Synthetic^Patient^FAKE"
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "CT"
        ds.Manufacturer = "SyntheticVendor"
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * 2.0]
        ds.SliceLocation = float(i) * 2.0
        ds.PatientPosition = "HFS"
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        ds.PixelData = pixel_fn(rows, cols, i).astype(np.int16).tobytes()

        ds.save_as(
            str(directory / f"image_{i:03d}"),
            enforce_file_format=True,
            little_endian=True,
            implicit_vr=False,
        )


def make_synthetic_ircad_patient(
    root: Path, patient_id: str, *, n_slices: int = 4, rows: int = 16, cols: int = 16
) -> Path:
    """Write a synthetic patient directory in 3D-IRCADb-01's documented
    layout (see `medimg_pipeline.curation.ircad_import`):

        <root>/<patient_id>/PATIENT_DICOM/       CT series
        <root>/<patient_id>/MASKS_DICOM/liver/   matching liver mask series

    Both series share identical geometry so the imported CT and mask
    align, and the "liver" is a bright disc in a fixed quadrant so the
    mask is a meaningful, non-trivial region rather than empty.
    """

    from pydicom.uid import generate_uid

    study_uid = generate_uid()
    patient_dir = root / patient_id

    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    disc = ((yy - rows * 0.6) ** 2 + (xx - cols * 0.6) ** 2) <= (min(rows, cols) / 5) ** 2

    def ct_pixels(rows: int, cols: int, _slice_index: int) -> np.ndarray:
        base = np.full((rows, cols), 40, dtype=np.int16)
        base[disc] = 120
        return base

    def mask_pixels(rows: int, cols: int, _slice_index: int) -> np.ndarray:
        mask = np.zeros((rows, cols), dtype=np.int16)
        mask[disc] = 1
        return mask

    _write_dicom_series(
        patient_dir / "PATIENT_DICOM",
        n_slices=n_slices,
        rows=rows,
        cols=cols,
        pixel_fn=ct_pixels,
        study_uid=study_uid,
        patient_id=patient_id,
    )
    _write_dicom_series(
        patient_dir / "MASKS_DICOM" / "liver",
        n_slices=n_slices,
        rows=rows,
        cols=cols,
        pixel_fn=mask_pixels,
        study_uid=study_uid,
        patient_id=patient_id,
    )
    return patient_dir
