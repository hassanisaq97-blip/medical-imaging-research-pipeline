from pathlib import Path

from tests.helpers import make_synthetic_ircad_patient

from medimg_pipeline.curation.ircad_import import discover_ircad_patients, import_ircad_dataset


def test_discover_ircad_patients_finds_patient_dicom_folders(tmp_path: Path):
    make_synthetic_ircad_patient(tmp_path, "3Dircadb1.1")
    make_synthetic_ircad_patient(tmp_path, "3Dircadb1.2")
    (tmp_path / "not_a_patient").mkdir()

    patients = discover_ircad_patients(tmp_path)

    assert [p.name for p in patients] == ["3Dircadb1.1", "3Dircadb1.2"]


def test_import_ircad_dataset_converts_and_validates(tmp_path: Path):
    input_dir = tmp_path / "raw"
    make_synthetic_ircad_patient(input_dir, "3Dircadb1.1", n_slices=4, rows=16, cols=16)
    make_synthetic_ircad_patient(input_dir, "3Dircadb1.2", n_slices=4, rows=16, cols=16)

    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "curated"

    report = import_ircad_dataset(input_dir, staging_dir, output_dir, salt="test-salt")

    assert report.n_ok == 2
    for patient in report.patients:
        assert patient.status == "ok"
        assert Path(patient.ct_path).exists()
        assert Path(patient.mask_path).exists()

    # Converted CT and mask must actually align and the mask must be binary.
    import nibabel as nib
    import numpy as np

    for patient in report.patients:
        ct_img = nib.load(patient.ct_path)
        mask_img = nib.load(patient.mask_path)
        assert ct_img.shape == mask_img.shape
        assert np.allclose(ct_img.affine, mask_img.affine, atol=1e-3)
        mask_array = np.asanyarray(mask_img.dataobj)
        assert set(np.unique(mask_array)).issubset({0, 1})
        assert mask_array.sum() > 0  # the synthetic "liver" disc is present


def test_import_ircad_dataset_skips_patient_missing_mask(tmp_path: Path):
    input_dir = tmp_path / "raw"
    patient_dir = make_synthetic_ircad_patient(input_dir, "3Dircadb1.1", n_slices=2)
    import shutil

    shutil.rmtree(patient_dir / "MASKS_DICOM")

    report = import_ircad_dataset(input_dir, tmp_path / "staging", tmp_path / "curated", salt="s")

    assert len(report.patients) == 1
    assert report.patients[0].status == "skipped"
    assert "MASKS_DICOM" in report.patients[0].message


def test_import_ircad_dataset_no_patients_found(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    report = import_ircad_dataset(input_dir, tmp_path / "staging", tmp_path / "curated", salt="s")

    assert report.n_ok == 0
    assert report.patients == []
