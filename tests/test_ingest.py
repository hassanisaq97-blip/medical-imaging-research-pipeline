from pathlib import Path

import pytest
from tests.helpers import make_synthetic_dicom_series

from medimg_pipeline.exceptions import MissingExternalToolError
from medimg_pipeline.imaging.ingest import ingest_dicom_directory


def test_ingest_dicom_directory_converts_series(tmp_path: Path):
    input_dir = make_synthetic_dicom_series(tmp_path, n_slices=5, rows=16, cols=16).parent
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "nifti_out"

    report = ingest_dicom_directory(
        input_dir, staging_dir, output_dir, salt="test-salt", backend="dicom2nifti"
    )

    assert report.n_series_found == 1
    assert len(report.series) == 1
    series = report.series[0]
    assert series.status == "ok"
    assert series.n_dicom_files == 5
    assert Path(series.output_nifti_path).exists()
    assert series.image_shape[2] == 5  # 5 slices stacked


def test_ingest_dicom_directory_rejects_missing_dcm2niix(tmp_path: Path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: None)
    input_dir = make_synthetic_dicom_series(tmp_path, n_slices=2).parent
    with pytest.raises(MissingExternalToolError):
        ingest_dicom_directory(
            input_dir, tmp_path / "staging", tmp_path / "out", salt="s", backend="dcm2niix"
        )
