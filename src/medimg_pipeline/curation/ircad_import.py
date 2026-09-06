"""Import a manually downloaded 3D-IRCADb-01 checkout into this project's
standard curation layout.

3D-IRCADb-01 requires a free registration on IRCAD's own site before
download (see `docs/data_access.md`); this environment has no network
path to any external dataset host at all (confirmed by testing several
unrelated domains, not just ircad.fr -- see `docs/data_access.md`), so
this module cannot download the dataset itself. It automates everything
downstream of that one manual step instead.

## Assumed directory layout

Each patient directory is widely documented (across the many published
papers and toolkits that use this dataset) to contain:

    <patient>/PATIENT_DICOM/       one CT DICOM series
    <patient>/MASKS_DICOM/liver/   the liver ground-truth mask as a
                                    parallel DICOM series (one binary
                                    slice per CT slice)

**This layout was not independently verified against the current
official IRCAD download in this environment**, because ircad.fr is
unreachable here (see above) -- it is what the wider community
consistently describes. `discover_ircad_patients` only requires that a
`PATIENT_DICOM` folder exists somewhere under the input directory (at
any depth, under any outer naming convention), so a differently-named
top-level folder does not break discovery; if IRCAD has since changed
its internal folder names, this will surface as a clear "no
PATIENT_DICOM found" / "no MASKS_DICOM/liver found" skip reason per
patient rather than a silent wrong result.

## What this does with the data

For each discovered patient:

1. De-identifies both DICOM series (`medimg_pipeline.anonymization.dicom`)
   into a staging directory -- defense in depth. 3D-IRCADb-01 is already
   a public, research-cleared release, so this is not required by IRCAD,
   but it costs nothing and this project never writes raw DICOM to its
   working data tree unexamined.
2. Converts each de-identified series to NIfTI
   (`medimg_pipeline.imaging.ingest.convert_dicom_series`).
3. Binarizes the mask (any nonzero voxel -> foreground) and validates
   that the CT and mask share the same shape and affine
   (`medimg_pipeline.imaging.nifti.validate_image_mask_pair`) --
   catching a misaligned pair immediately rather than silently training
   on garbage.
4. Writes both into the same `<subject>/ct/*_ct.nii.gz` +
   `<subject>/seg/*_seg-*.nii.gz` layout `medimg_pipeline.curation`
   already expects by default, so `medimg_pipeline curate --data-root
   data/ircad` works with no further configuration.

Patient IDs are IRCAD's own folder names (e.g. "3Dircadb1.1"), which are
already just an ordinal, non-identifying study code, not a patient name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from medimg_pipeline.anonymization.dicom import deidentify_directory
from medimg_pipeline.imaging.ingest import convert_dicom_series
from medimg_pipeline.imaging.nifti import load_nifti, validate_image_mask_pair
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("curation.ircad_import")

CT_DICOM_SUBDIR = "PATIENT_DICOM"
MASK_DICOM_SUBDIR = "MASKS_DICOM/liver"


@dataclass
class PatientImportResult:
    patient_id: str
    status: str  # "ok" | "skipped" | "error"
    message: str
    ct_path: str | None = None
    mask_path: str | None = None


@dataclass
class ImportReport:
    generated_at: str
    dataset: str
    input_dir: str
    output_dir: str
    patients: list[PatientImportResult] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for p in self.patients if p.status == "ok")

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def discover_ircad_patients(input_dir: Path) -> list[Path]:
    """Every directory containing a PATIENT_DICOM subfolder is one patient."""

    return sorted({p.parent for p in input_dir.rglob(CT_DICOM_SUBDIR) if p.is_dir()})


def _dicom_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file())


def import_ircad_patient(
    patient_dir: Path, staging_dir: Path, output_dir: Path, *, salt: str
) -> PatientImportResult:
    patient_id = patient_dir.name
    ct_dicom_dir = patient_dir / CT_DICOM_SUBDIR
    mask_dicom_dir = patient_dir / MASK_DICOM_SUBDIR

    if not ct_dicom_dir.is_dir():
        return PatientImportResult(patient_id, "skipped", f"no {CT_DICOM_SUBDIR} folder found")
    if not mask_dicom_dir.is_dir():
        return PatientImportResult(patient_id, "skipped", f"no {MASK_DICOM_SUBDIR} folder found")

    patient_staging = staging_dir / patient_id
    try:
        deidentify_directory(ct_dicom_dir, patient_staging / "ct", salt=salt)
        deidentify_directory(mask_dicom_dir, patient_staging / "mask", salt=salt)
    except Exception as exc:  # noqa: BLE001
        return PatientImportResult(patient_id, "error", f"de-identification failed: {exc}")

    ct_out = output_dir / patient_id / "ct" / f"{patient_id}_ct.nii.gz"
    mask_out = output_dir / patient_id / "seg" / f"{patient_id}_seg-liver.nii.gz"
    ct_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        convert_dicom_series(_dicom_files(patient_staging / "ct"), ct_out)
        convert_dicom_series(_dicom_files(patient_staging / "mask"), mask_out)
    except Exception as exc:  # noqa: BLE001
        return PatientImportResult(patient_id, "error", f"DICOM->NIfTI conversion failed: {exc}")

    try:
        import nibabel as nib

        mask_img = nib.load(str(mask_out))
        binary_mask = (np.asanyarray(mask_img.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(binary_mask, mask_img.affine, mask_img.header), str(mask_out))

        _, _, ct_info = load_nifti(ct_out)
        _, _, mask_info = load_nifti(mask_out)
        validate_image_mask_pair(ct_info.shape, mask_info.shape, ct_info.affine, mask_info.affine)
    except Exception as exc:  # noqa: BLE001
        return PatientImportResult(
            patient_id, "error", f"CT/mask validation failed: {exc}", str(ct_out), str(mask_out)
        )

    return PatientImportResult(patient_id, "ok", "ok", str(ct_out), str(mask_out))


def import_ircad_dataset(
    input_dir: str | Path, staging_dir: str | Path, output_dir: str | Path, *, salt: str
) -> ImportReport:
    input_dir = Path(input_dir)
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    patients = discover_ircad_patients(input_dir)
    report = ImportReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset="ircad",
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    for patient_dir in patients:
        result = import_ircad_patient(patient_dir, staging_dir, output_dir, salt=salt)
        report.patients.append(result)
        logger.info("Patient %s: %s (%s)", result.patient_id, result.status, result.message)

    logger.info(
        "IRCAD import complete: %d/%d patients imported successfully",
        report.n_ok,
        len(patients),
    )
    return report
