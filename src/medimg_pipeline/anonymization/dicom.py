"""DICOM metadata de-identification.

## Anonymization vs. pseudonymization vs. metadata de-identification

These three terms are not interchangeable, and this module implements the
third one, which enables the second:

- **Metadata de-identification** (what this module does): removing or
  replacing identifying *DICOM header tags* (name, birth date, address,
  accession number, private tags, free-text comments, ...). This is a
  well-defined, mechanical operation on structured metadata.
- **Pseudonymization**: replacing a direct identifier (PatientID,
  UIDs) with a consistent surrogate value derived from a secret salt, so
  that records for "the same patient" still link to each other without
  exposing who they are. This module does this for `PatientID` and the
  UID triplet (Study/Series/SOPInstanceUID) — but pseudonymization is
  reversible in principle by anyone who knows (or can guess/brute-force,
  if the identifier space is small) both the salt and the original
  identifier. **The salt must be kept secret and is never written to the
  audit report.**
- **Anonymization** (in the strict, irreversible sense) is a much stronger
  claim than either of the above and is **not** what this module
  guarantees. In particular:
  - **Burned-in pixel annotations** (text rendered into the image itself
    by the scanner, e.g. on some ultrasound or secondary-capture images)
    are NOT touched by metadata scrubbing at all, because they live in
    the pixel data, not the header. A real clinical de-identification
    workflow needs separate pixel-level QC (manual review or an
    annotation-detection model) before data can be called anonymous.
    This project does not implement that step; it only scrubs metadata.
  - Sufficiently rich combinations of *technical* metadata (e.g. an exact
    acquisition timestamp plus a rare scanner configuration) can in
    principle support re-identification through linkage with an external
    record, even after this module runs.

  For these reasons, this module and its CLI are named/described as
  **de-identification**, not "anonymization", throughout this project's
  own documentation, even though the DICOM standard itself uses the term
  "Patient Identity Removed" for a related concept.

## What this module does concretely

Given a `pydicom.Dataset`:

1. Removes a fixed list of direct-identifier tags (name, birth date,
   address, referring/performing physician, institution, free-text
   comments/descriptions considered high-risk for PHI).
2. Removes all private (vendor-specific) tags.
3. Replaces `PatientID`, `AccessionNumber`, and the UID triplet with
   deterministic pseudonyms derived from a caller-supplied salt via
   SHA-256, so repeated runs with the same salt produce the same mapping
   (needed to keep a study's series/instances linked to each other).
4. Preserves every tag needed for correct scientific image processing:
   geometry (`Rows`, `Columns`, `PixelSpacing`, `SliceThickness`,
   `ImageOrientationPatient`, `ImagePositionPatient`), pixel data and its
   encoding (`PixelData`, `BitsAllocated`, `BitsStored`, `HighBit`,
   `PixelRepresentation`, `RescaleSlope`/`RescaleIntercept`,
   `PhotometricInterpretation`, `SamplesPerPixel`), and acquisition
   parameters relevant to research use (`Modality`, `Manufacturer`,
   `ManufacturerModelName`, `KVP`, `SeriesNumber`, `InstanceNumber`).
5. Never writes to the input file; always writes a new file into a
   separate output directory, and produces an audit report that lists,
   per file, which tags were removed/replaced and why — **never the
   original values**, only tag names and (for pseudonyms) the new value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from medimg_pipeline.utils.logging import get_logger

logger = get_logger("anonymization.dicom")

# Tags removed outright: direct identifiers and free-text fields with a
# realistic chance of containing PHI. Keyword form (pydicom resolves these
# to (group, element) automatically).
REMOVE_KEYWORDS: tuple[str, ...] = (
    "PatientName",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
    "PatientInsurancePlanCodeSequence",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "RequestingPhysician",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ImageComments",
    "DerivationDescription",
    "AdditionalPatientHistory",
    "MedicalRecordLocator",
    "PatientComments",
    "RequestAttributesSequence",
)

# Tags replaced with a deterministic pseudonym rather than removed,
# because removing them would either break DICOM validity (UIDs are
# required) or discard information needed to link records within one
# study/series (PatientID, AccessionNumber). Split into two groups
# because DICOM UIDs (VR "UI") are constrained to numeric, dot-separated
# strings -- a hex string is not a spec-valid UID -- while PatientID/
# AccessionNumber (VR "LO"/"SH") accept arbitrary short text.
PSEUDONYMIZE_ID_KEYWORDS: tuple[str, ...] = (
    "PatientID",
    "AccessionNumber",
)
PSEUDONYMIZE_UID_KEYWORDS: tuple[str, ...] = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
)
PSEUDONYMIZE_KEYWORDS: tuple[str, ...] = PSEUDONYMIZE_ID_KEYWORDS + PSEUDONYMIZE_UID_KEYWORDS

# Tags explicitly preserved for scientific/technical correctness. Not
# enforced programmatically (nothing here is touched unless listed
# above), this list exists as documentation and for the audit report.
PRESERVED_KEYWORDS: tuple[str, ...] = (
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelData",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "RescaleSlope",
    "RescaleIntercept",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "KVP",
    "SeriesNumber",
    "InstanceNumber",
)


def pseudonymize_identifier(value: str, salt: str) -> str:
    """Deterministically derive a pseudonymous identifier from `value`.

    Same (value, salt) always maps to the same output, so linked DICOM
    instances (same StudyInstanceUID, say) stay linked after
    de-identification. Different salts produce unlinkable outputs, so the
    salt must be treated as a secret and never committed or logged.
    """

    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return digest[:16].upper()


def pseudonymize_uid(value: str, salt: str) -> str:
    """Like `pseudonymize_identifier`, but returns a spec-valid DICOM UID
    (VR "UI": numeric components separated by dots, max 64 characters).

    Uses the "2.25." root reserved by ITU-T X.667 / ISO/IEC 9834-8 for
    UUID-derived, non-registered UIDs, followed by a deterministic large
    integer derived from SHA-256(salt, value) -- same determinism
    guarantee as `pseudonymize_identifier`, just in a format DICOM viewers
    and toolkits will accept.
    """

    digest = hashlib.sha256(f"{salt}:{value}".encode()).digest()
    as_int = int.from_bytes(digest[:16], "big")  # 128-bit -> up to 39 decimal digits
    uid = f"2.25.{as_int}"
    return uid[:64]


@dataclass
class TagAction:
    tag: str
    action: str  # "removed" | "pseudonymized" | "private_tags_removed"
    new_value: str | None = None  # only ever a pseudonym, never original PHI


@dataclass
class FileAuditRecord:
    relative_path: str
    sop_class_uid: str | None
    modality: str | None
    actions: list[TagAction] = field(default_factory=list)
    status: str = "ok"  # "ok" | "skipped_not_dicom" | "error"
    error: str | None = None


@dataclass
class AnonymizationReport:
    generated_at: str
    input_dir: str
    output_dir: str
    files: list[FileAuditRecord] = field(default_factory=list)

    @property
    def n_processed(self) -> int:
        return sum(1 for f in self.files if f.status == "ok")

    @property
    def n_skipped(self) -> int:
        return sum(1 for f in self.files if f.status == "skipped_not_dicom")

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.files if f.status == "error")

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "summary": {
                "n_files_seen": len(self.files),
                "n_processed": self.n_processed,
                "n_skipped_not_dicom": self.n_skipped,
                "n_errors": self.n_errors,
            },
            "files": [
                {
                    "relative_path": f.relative_path,
                    "sop_class_uid": f.sop_class_uid,
                    "modality": f.modality,
                    "status": f.status,
                    "error": f.error,
                    "actions": [asdict(a) for a in f.actions],
                }
                for f in self.files
            ],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _looks_like_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def deidentify_dataset(ds, salt: str) -> tuple[object, list[TagAction]]:
    """De-identify a single in-memory `pydicom.Dataset`.

    Returns the mutated dataset (mutated in place, but also returned for
    convenience) and the list of tag actions taken, for the audit report.
    """

    actions: list[TagAction] = []

    for keyword in REMOVE_KEYWORDS:
        if keyword in ds:
            delattr(ds, keyword)
            actions.append(TagAction(tag=keyword, action="removed"))

    for keyword in PSEUDONYMIZE_ID_KEYWORDS:
        if keyword in ds:
            original = str(getattr(ds, keyword))
            pseudo = pseudonymize_identifier(original, salt)
            setattr(ds, keyword, pseudo)
            actions.append(TagAction(tag=keyword, action="pseudonymized", new_value=pseudo))

    for keyword in PSEUDONYMIZE_UID_KEYWORDS:
        if keyword in ds:
            original = str(getattr(ds, keyword))
            pseudo = pseudonymize_uid(original, salt)
            setattr(ds, keyword, pseudo)
            actions.append(TagAction(tag=keyword, action="pseudonymized", new_value=pseudo))

    before = len(ds)
    ds.remove_private_tags()
    removed_private = before - len(ds)
    if removed_private > 0:
        actions.append(
            TagAction(
                tag="(private tags)", action="private_tags_removed", new_value=str(removed_private)
            )
        )

    return ds, actions


def deidentify_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    salt: str,
    allow_same_dir: bool = False,
) -> AnonymizationReport:
    """Recursively de-identify every DICOM file under `input_dir`.

    Mirrors the input directory structure under `output_dir`. Never
    writes into `input_dir` unless `allow_same_dir=True` is passed
    explicitly (it is not exposed on the CLI; this is a deliberate
    safety rail against accidentally overwriting raw source data).
    """

    import pydicom

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir.resolve() == output_dir.resolve() and not allow_same_dir:
        raise ValueError(
            "Refusing to write de-identified output over the raw input directory. "
            "Pass a different --output directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = AnonymizationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    all_files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    for src_path in all_files:
        rel = src_path.relative_to(input_dir)

        if not _looks_like_dicom(src_path):
            report.files.append(
                FileAuditRecord(
                    relative_path=str(rel),
                    sop_class_uid=None,
                    modality=None,
                    status="skipped_not_dicom",
                )
            )
            continue

        try:
            ds = pydicom.dcmread(str(src_path))
            sop_class_uid = str(getattr(ds, "SOPClassUID", "")) or None
            modality = str(getattr(ds, "Modality", "")) or None
            ds, actions = deidentify_dataset(ds, salt)

            dest_path = output_dir / rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            ds.save_as(str(dest_path), enforce_file_format=True)

            report.files.append(
                FileAuditRecord(
                    relative_path=str(rel),
                    sop_class_uid=sop_class_uid,
                    modality=modality,
                    actions=actions,
                    status="ok",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to de-identify %s: %s", rel, exc)
            report.files.append(
                FileAuditRecord(
                    relative_path=str(rel),
                    sop_class_uid=None,
                    modality=None,
                    status="error",
                    error=str(exc),
                )
            )

    logger.info(
        "De-identified %d file(s), skipped %d non-DICOM, %d error(s)",
        report.n_processed,
        report.n_skipped,
        report.n_errors,
    )
    return report
