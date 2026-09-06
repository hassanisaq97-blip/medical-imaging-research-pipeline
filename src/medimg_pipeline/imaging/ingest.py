"""Optional DICOM -> NIfTI ingestion pathway.

Pipeline (see README §6 and docs/DEPICT_TECHNICAL_RESEARCH.md):

    raw DICOM
      -> metadata de-identification (medimg_pipeline.anonymization.dicom)
      -> validation (grouping into series, checking each series is complete)
      -> DICOM-to-NIfTI conversion
      -> NIfTI QC (medimg_pipeline.imaging.nifti)
      -> curated dataset (medimg_pipeline.curation)

This project does not reimplement DICOM-to-NIfTI conversion -- that is a
well-solved problem with mature, widely validated tools. By default it
uses the pure-Python `dicom2nifti` package (already a dependency, no
external binary required). If `dcm2niix` is explicitly requested via
`backend="dcm2niix"` and is not installed, this raises
`MissingExternalToolError` with install instructions rather than failing
with a confusing subprocess error.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from medimg_pipeline.anonymization.dicom import deidentify_directory
from medimg_pipeline.exceptions import DicomConversionError, MissingExternalToolError
from medimg_pipeline.imaging.nifti import load_nifti
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("imaging.ingest")

DCM2NIIX_INSTALL_HINT = (
    "Install dcm2niix and ensure it is on PATH: "
    "`conda install -c conda-forge dcm2niix` (Linux/Mac), "
    "`brew install dcm2niix` (Mac), or see https://github.com/rordenlab/dcm2niix. "
    "Alternatively, omit --backend to use the pure-Python dicom2nifti backend, "
    "which requires no external binary."
)


@dataclass
class SeriesConversionResult:
    series_uid: str
    n_dicom_files: int
    output_nifti_path: str | None
    status: str  # "ok" | "error"
    error: str | None = None
    image_shape: list[int] | None = None
    voxel_spacing: list[float] | None = None


@dataclass
class IngestReport:
    generated_at: str
    input_dir: str
    staging_dir: str
    output_dir: str
    backend: str
    n_series_found: int
    series: list[SeriesConversionResult] = field(default_factory=list)

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _check_dcm2niix_available() -> None:
    if shutil.which("dcm2niix") is None:
        raise MissingExternalToolError("dcm2niix", DCM2NIIX_INSTALL_HINT)


def _group_series(staging_dir: Path) -> dict[str, list[Path]]:
    import pydicom

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(staging_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:  # noqa: BLE001
            continue
        series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown-series"))
        groups[series_uid].append(path)
    return dict(groups)


def _convert_series_dicom2nifti(files: list[Path], output_path: Path) -> None:
    import dicom2nifti

    with tempfile.TemporaryDirectory() as tmp:
        tmp_series_dir = Path(tmp) / "series"
        tmp_series_dir.mkdir()
        for f in files:
            (tmp_series_dir / f.name).write_bytes(f.read_bytes())
        dicom2nifti.dicom_series_to_nifti(
            str(tmp_series_dir), str(output_path), reorient_nifti=True
        )


def _convert_series_dcm2niix(files: list[Path], output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_series_dir = Path(tmp) / "series"
        tmp_series_dir.mkdir()
        for f in files:
            (tmp_series_dir / f.name).write_bytes(f.read_bytes())

        result = subprocess.run(
            [
                "dcm2niix",
                "-z",
                "y",
                "-f",
                output_path.stem.replace(".nii", ""),
                "-o",
                str(output_path.parent),
                str(tmp_series_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise DicomConversionError(
                str(tmp_series_dir), result.stderr.strip() or "dcm2niix failed"
            )


def convert_dicom_series(
    files: list[Path], output_path: Path, *, backend: str = "dicom2nifti"
) -> None:
    """Convert one DICOM series (a flat list of file paths, all belonging
    to the same series) to a single NIfTI file. Shared by
    `ingest_dicom_directory` and dataset-specific import commands (e.g.
    `medimg_pipeline.curation.ircad_import`) so there is one place that
    knows how to dispatch between backends.
    """

    if backend not in {"dicom2nifti", "dcm2niix"}:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'dicom2nifti' or 'dcm2niix'.")
    if backend == "dcm2niix":
        _check_dcm2niix_available()
        _convert_series_dcm2niix(files, output_path)
    else:
        _convert_series_dicom2nifti(files, output_path)


def ingest_dicom_directory(
    input_dir: str | Path,
    staging_dir: str | Path,
    output_dir: str | Path,
    *,
    salt: str,
    backend: str = "dicom2nifti",
) -> IngestReport:
    """Run the full ingestion pipeline on a directory of raw DICOM files.

    `staging_dir` receives the de-identified DICOM files (never the raw
    input); `output_dir` receives one NIfTI file per series plus this
    report. Raises `MissingExternalToolError` immediately if
    `backend="dcm2niix"` is requested but the binary is not on PATH,
    before any conversion is attempted.
    """

    if backend not in {"dicom2nifti", "dcm2niix"}:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'dicom2nifti' or 'dcm2niix'.")
    if backend == "dcm2niix":
        _check_dcm2niix_available()  # fail fast, before de-identifying anything

    input_dir = Path(input_dir)
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/4: de-identifying DICOM metadata -> %s", staging_dir)
    deidentify_directory(input_dir, staging_dir, salt=salt)

    logger.info("Step 2/4: grouping de-identified files into series")
    series_groups = _group_series(staging_dir)

    logger.info("Step 3/4: converting %d series to NIfTI (backend=%s)", len(series_groups), backend)
    results: list[SeriesConversionResult] = []
    for series_uid, files in series_groups.items():
        output_path = output_dir / f"series-{series_uid[-12:]}.nii.gz"
        try:
            convert_dicom_series(files, output_path, backend=backend)

            logger.info("Step 4/4: validating converted NIfTI for series %s", series_uid)
            _, _, info = load_nifti(output_path)
            results.append(
                SeriesConversionResult(
                    series_uid=series_uid,
                    n_dicom_files=len(files),
                    output_nifti_path=str(output_path),
                    status="ok",
                    image_shape=list(info.shape),
                    voxel_spacing=[round(v, 4) for v in info.voxel_spacing],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to convert series %s: %s", series_uid, exc)
            results.append(
                SeriesConversionResult(
                    series_uid=series_uid,
                    n_dicom_files=len(files),
                    output_nifti_path=None,
                    status="error",
                    error=str(exc),
                )
            )

    report = IngestReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        input_dir=str(input_dir),
        staging_dir=str(staging_dir),
        output_dir=str(output_dir),
        backend=backend,
        n_series_found=len(series_groups),
        series=results,
    )
    return report
