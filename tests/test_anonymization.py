from pathlib import Path

import pydicom
import pytest
from tests.helpers import make_synthetic_dicom_series

from medimg_pipeline.anonymization.dicom import (
    PRESERVED_KEYWORDS,
    REMOVE_KEYWORDS,
    deidentify_directory,
    pseudonymize_identifier,
)


def test_pseudonymize_identifier_deterministic_and_salted():
    a = pseudonymize_identifier("FAKE-PATIENT-0001", salt="salt-a")
    b = pseudonymize_identifier("FAKE-PATIENT-0001", salt="salt-a")
    c = pseudonymize_identifier("FAKE-PATIENT-0001", salt="salt-b")
    assert a == b
    assert a != c


def test_deidentify_directory_removes_fake_phi(tmp_path: Path):
    input_dir = make_synthetic_dicom_series(tmp_path, n_slices=3).parent
    output_dir = tmp_path / "deidentified"

    report = deidentify_directory(input_dir, output_dir, salt="test-salt")

    assert report.n_processed == 3
    assert report.n_errors == 0

    out_files = sorted((output_dir / "synthetic_series").glob("*.dcm"))
    assert len(out_files) == 3

    for f in out_files:
        ds = pydicom.dcmread(str(f))

        for keyword in REMOVE_KEYWORDS:
            assert keyword not in ds, f"{keyword} should have been removed"

        # No fake identifying strings should survive anywhere in the dataset.
        dump = str(ds)
        assert "Jane" not in dump
        assert "Fake General Hospital" not in dump
        assert "123 Fake Street" not in dump
        assert "FAKE-PATIENT-0001" not in dump
        assert "ACC-FAKE-0001" not in dump

        # Private tags removed.
        assert (0x0041, 0x0101) not in ds

        # Technical fields preserved for scientific correctness.
        assert ds.Rows == 16
        assert ds.Columns == 16
        assert ds.Modality == "CT"
        assert list(ds.PixelSpacing) == [1.0, 1.0]
        assert ds.PixelData is not None


def test_deidentify_directory_pseudonyms_are_consistent_within_a_series(tmp_path: Path):
    input_dir = make_synthetic_dicom_series(tmp_path, n_slices=3).parent
    output_dir = tmp_path / "deidentified"
    deidentify_directory(input_dir, output_dir, salt="test-salt")

    out_files = sorted((output_dir / "synthetic_series").glob("*.dcm"))
    series_uids = {pydicom.dcmread(str(f)).SeriesInstanceUID for f in out_files}
    patient_ids = {pydicom.dcmread(str(f)).PatientID for f in out_files}

    assert len(series_uids) == 1  # all slices still belong to one pseudonymous series
    assert len(patient_ids) == 1


def test_deidentify_directory_refuses_same_dir(tmp_path: Path):
    input_dir = make_synthetic_dicom_series(tmp_path, n_slices=1).parent
    with pytest.raises(ValueError):
        deidentify_directory(input_dir, input_dir, salt="test-salt")


def test_deidentify_directory_audit_report_has_no_original_phi(tmp_path: Path):
    input_dir = make_synthetic_dicom_series(
        tmp_path, n_slices=1, patient_name="Doe^Jane^FAKE"
    ).parent
    output_dir = tmp_path / "deidentified"
    report = deidentify_directory(input_dir, output_dir, salt="test-salt")

    report_json = report.to_dict()
    dump = str(report_json)
    assert "Jane" not in dump
    assert "Doe" not in dump


def test_preserved_keywords_documented_not_empty():
    assert len(PRESERVED_KEYWORDS) > 0
