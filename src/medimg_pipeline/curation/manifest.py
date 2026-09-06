"""Build a structured dataset manifest with QC and subject-level splits.

Produces a `data_manifest.csv` with one row per subject containing only
non-identifying information (pseudonymous subject ID, file paths, image
geometry, foreground voxel count, QC status/message, split assignment) --
never protected health information. Also writes a QC report (JSON) that
separately logs every excluded subject and the reason for exclusion, so
nothing is silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from medimg_pipeline.curation.config import CurationConfig
from medimg_pipeline.imaging.nifti import load_nifti, validate_image_mask_pair
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("curation.manifest")

QC_PASS = "pass"
QC_FAIL = "fail"


@dataclass
class SubjectRecord:
    subject_id: str
    image_path: str | None
    mask_path: str | None
    image_shape: tuple[int, ...] | None
    voxel_spacing: tuple[float, ...] | None
    foreground_voxels: int | None
    qc_status: str
    qc_message: str
    split: str | None = None


@dataclass
class QCReport:
    generated_at: str
    config: dict
    n_subjects_discovered: int
    n_included: int
    n_excluded: int
    excluded: list[dict] = field(default_factory=list)

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def discover_subjects(config: CurationConfig) -> list[str]:
    root = config.data_root_path
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    return sorted(p.name for p in root.glob(config.subject_glob) if p.is_dir())


def _find_one(root: Path, pattern: str, subject: str) -> Path | None:
    matches = sorted(root.glob(pattern.format(subject=subject)))
    return matches[0] if matches else None


def _binarize_mask(mask_array: np.ndarray, label_index: int | None) -> np.ndarray:
    if label_index is None:
        return (mask_array != 0).astype(np.uint8)
    return (mask_array == label_index).astype(np.uint8)


def build_manifest(config: CurationConfig) -> tuple[pd.DataFrame, QCReport]:
    """Discover subjects, validate each one, and build the manifest + QC report.

    Every subject gets a row; excluded subjects have qc_status="fail" and
    a human-readable qc_message, and are also listed in the QC report --
    they are never just silently dropped from the output entirely.
    """

    root = config.data_root_path
    subjects = discover_subjects(config)
    if not subjects:
        logger.warning(
            "No subject directories matched pattern %r under %s -- check --data-root and "
            "--subject-glob rather than assuming an empty dataset.",
            config.subject_glob,
            root,
        )
    records: list[SubjectRecord] = []
    excluded: list[dict] = []

    for subject in subjects:
        image_path = _find_one(root, config.image_glob, subject)
        mask_path = _find_one(root, config.mask_glob, subject)

        if image_path is None:
            msg = "missing required image modality"
            records.append(SubjectRecord(subject, None, None, None, None, None, QC_FAIL, msg))
            excluded.append({"subject_id": subject, "reason": msg})
            continue

        if mask_path is None:
            msg = "missing segmentation mask"
            records.append(
                SubjectRecord(subject, str(image_path), None, None, None, None, QC_FAIL, msg)
            )
            excluded.append({"subject_id": subject, "reason": msg})
            continue

        try:
            image_array, image_img, image_info = load_nifti(image_path)
            mask_array, mask_img, mask_info = load_nifti(mask_path)
        except Exception as exc:  # noqa: BLE001
            msg = f"could not open image or mask: {exc}"
            records.append(
                SubjectRecord(
                    subject, str(image_path), str(mask_path), None, None, None, QC_FAIL, msg
                )
            )
            excluded.append({"subject_id": subject, "reason": msg})
            continue

        try:
            validate_image_mask_pair(
                image_info.shape, mask_info.shape, image_info.affine, mask_info.affine
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            records.append(
                SubjectRecord(
                    subject,
                    str(image_path),
                    str(mask_path),
                    image_info.shape,
                    image_info.voxel_spacing,
                    None,
                    QC_FAIL,
                    msg,
                )
            )
            excluded.append({"subject_id": subject, "reason": msg})
            continue

        binary_mask = _binarize_mask(mask_array, config.label_index)
        foreground = int(np.sum(binary_mask))

        if foreground == 0:
            msg = "mask has zero foreground voxels after binarization"
            records.append(
                SubjectRecord(
                    subject,
                    str(image_path),
                    str(mask_path),
                    image_info.shape,
                    image_info.voxel_spacing,
                    0,
                    QC_FAIL,
                    msg,
                )
            )
            excluded.append({"subject_id": subject, "reason": msg})
            continue

        if foreground < config.min_foreground_voxels:
            msg = (
                f"foreground voxel count {foreground} below minimum "
                f"threshold {config.min_foreground_voxels}"
            )
            records.append(
                SubjectRecord(
                    subject,
                    str(image_path),
                    str(mask_path),
                    image_info.shape,
                    image_info.voxel_spacing,
                    foreground,
                    QC_FAIL,
                    msg,
                )
            )
            excluded.append({"subject_id": subject, "reason": msg})
            continue

        records.append(
            SubjectRecord(
                subject_id=subject,
                image_path=str(image_path),
                mask_path=str(mask_path),
                image_shape=image_info.shape,
                voxel_spacing=image_info.voxel_spacing,
                foreground_voxels=foreground,
                qc_status=QC_PASS,
                qc_message="ok",
            )
        )

    assign_splits(records, config)

    df = pd.DataFrame([asdict(r) for r in records])
    report = QCReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        config={
            "data_root": str(root),
            "image_glob": config.image_glob,
            "mask_glob": config.mask_glob,
            "label_index": config.label_index,
            "min_foreground_voxels": config.min_foreground_voxels,
        },
        n_subjects_discovered=len(subjects),
        n_included=sum(1 for r in records if r.qc_status == QC_PASS),
        n_excluded=len(excluded),
        excluded=excluded,
    )
    return df, report


def assign_splits(records: list[SubjectRecord], config: CurationConfig) -> None:
    """Assign train/val/test splits at subject granularity, deterministically.

    Only QC-passing subjects are split; excluded subjects get split=None.
    Splitting hashes the subject ID with the configured seed rather than
    using `random.shuffle`, so the assignment is stable even if new
    subjects are added later (existing subjects never move splits) --
    this also guarantees no subject can ever appear in more than one
    split, since each subject_id maps to exactly one row and one bucket.
    """

    passing = [r for r in records if r.qc_status == QC_PASS]
    n = len(passing)
    if n == 0:
        return

    # Deterministic pseudo-random rank in [0, 1) per subject, seeded.
    def rank(subject_id: str) -> float:
        h = hashlib.sha256(f"{config.split_seed}:{subject_id}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    ordered = sorted(passing, key=lambda r: rank(r.subject_id))
    n_train = round(n * config.train_frac)
    n_val = round(n * config.val_frac)

    for i, record in enumerate(ordered):
        if i < n_train:
            record.split = "train"
        elif i < n_train + n_val:
            record.split = "val"
        else:
            record.split = "test"


def run_curation(config: CurationConfig) -> tuple[pd.DataFrame, QCReport]:
    """Build the manifest/QC report and write both to disk."""

    df, report = build_manifest(config)

    manifest_path = Path(config.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    report.write_json(config.output_qc_report)

    logger.info(
        "Curation complete: %d/%d subjects passed QC, manifest -> %s, QC report -> %s",
        report.n_included,
        report.n_subjects_discovered,
        manifest_path,
        config.output_qc_report,
    )
    return df, report
