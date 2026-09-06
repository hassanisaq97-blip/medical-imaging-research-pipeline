"""Evaluate a trained checkpoint against held-out test-split subjects.

Runs inference on every QC-passing subject in a given manifest split
(default "test"), compares each prediction against that subject's own
ground-truth mask, and reports Dice and IoU per subject plus the mean --
the small results table this project's README and
docs/segmentation_target_selection.md call for. Never invents a number:
if a subject fails inference, it is reported as an error, not skipped
silently or scored as zero.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from medimg_pipeline.imaging.nifti import load_nifti
from medimg_pipeline.inference.infer import run_inference
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("inference.evaluate")


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(intersection / union)


@dataclass
class SubjectEvalResult:
    subject_id: str
    status: str  # "ok" | "error"
    dice: float | None = None
    iou: float | None = None
    foreground_voxels_gt: int | None = None
    foreground_voxels_pred: int | None = None
    error: str | None = None


@dataclass
class EvaluationReport:
    generated_at: str
    checkpoint_path: str
    manifest_path: str
    split: str
    n_subjects: int
    mean_dice: float | None
    mean_iou: float | None
    results: list[SubjectEvalResult] = field(default_factory=list)

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def write_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "subject_id",
            "status",
            "dice",
            "iou",
            "foreground_voxels_gt",
            "foreground_voxels_pred",
            "error",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))


def evaluate_test_split(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    device: str = "auto",
) -> EvaluationReport:
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    subset = df[(df.qc_status == "pass") & (df.split == split)]

    results: list[SubjectEvalResult] = []
    for row in subset.itertuples(index=False):
        try:
            inference_result = run_inference(
                row.image_path, checkpoint_path, output_dir / row.subject_id, device=device
            )
            pred_array, _, _ = load_nifti(inference_result.output_mask_path)
            gt_array, _, _ = load_nifti(row.mask_path)

            dice = dice_score(pred_array, gt_array)
            iou = iou_score(pred_array, gt_array)
            results.append(
                SubjectEvalResult(
                    subject_id=row.subject_id,
                    status="ok",
                    dice=round(dice, 4),
                    iou=round(iou, 4),
                    foreground_voxels_gt=int(gt_array.sum()),
                    foreground_voxels_pred=int(pred_array.sum()),
                )
            )
            logger.info("Subject %s: Dice=%.4f IoU=%.4f", row.subject_id, dice, iou)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Subject %s: evaluation failed: %s", row.subject_id, exc)
            results.append(
                SubjectEvalResult(subject_id=row.subject_id, status="error", error=str(exc))
            )

    ok_results = [r for r in results if r.status == "ok"]
    mean_dice = round(float(np.mean([r.dice for r in ok_results])), 4) if ok_results else None
    mean_iou = round(float(np.mean([r.iou for r in ok_results])), 4) if ok_results else None

    report = EvaluationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(manifest_path),
        split=split,
        n_subjects=len(results),
        mean_dice=mean_dice,
        mean_iou=mean_iou,
        results=results,
    )
    logger.info(
        "Evaluation complete: %d/%d subjects scored, mean Dice=%s, mean IoU=%s",
        len(ok_results),
        len(results),
        mean_dice,
        mean_iou,
    )
    return report
