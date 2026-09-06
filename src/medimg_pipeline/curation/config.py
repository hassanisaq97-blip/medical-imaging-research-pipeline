"""Curation configuration.

The curation pipeline is deliberately glob-pattern driven rather than
hard-coded to one dataset's exact directory layout, because this
project's own test fixtures use a simplified layout while the real
Multimodal-HC dataset uses a deeper BIDS layout (see
`docs/DEPICT_TECHNICAL_RESEARCH.md` §1.3). `{subject}` in a pattern is
substituted with each discovered subject ID before globbing.

`MULTIMODAL_HC_LIVER_CONFIG` documents (but does not hard-code as
unconditional truth) the patterns and label index needed to curate a
liver-segmentation dataset from a real Multimodal-HC checkout — see the
caveat on `label_index` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurationConfig:
    data_root: str
    subject_glob: str = "sub-*"
    image_glob: str = "{subject}/ct/*_ct.nii.gz"
    mask_glob: str = "{subject}/seg/*_seg-*.nii.gz"
    # Which integer label in the mask volume is foreground. None means the
    # mask is already binary (any nonzero voxel is foreground).
    label_index: int | None = None
    min_foreground_voxels: int = 500
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 42
    output_manifest: str = "outputs/curation/data_manifest.csv"
    output_qc_report: str = "outputs/curation/qc_report.json"

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root)


# Documents the mapping needed to curate liver segmentation directly from
# a real Multimodal-HC checkout (see docs/segmentation_target_selection.md).
# `label_index=5` is TotalSegmentator v2's publicly documented "total" task
# class-map index for `liver`. This has NOT been verified against real
# Multimodal-HC derivative files (no DUA access yet, see docs/data_access.md)
# -- before relying on it, confirm the label index against the actual
# TotalSegmentator class map shipped with the installed totalsegmentator
# version, or against the BIDS sidecar JSON of a real seg-total file.
MULTIMODAL_HC_LIVER_CONFIG_TEMPLATE: dict = {
    "subject_glob": "sub-*",
    "image_glob": "{subject}/ses-quadra/ct/*br38f_ct.nii.gz",
    "mask_glob": "derivatives/totalsegmentator/{subject}/**/ct/*seg-total*.nii.gz",
    "label_index": 5,
}
