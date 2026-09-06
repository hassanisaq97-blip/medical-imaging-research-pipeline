from pathlib import Path

import numpy as np
from tests.helpers import make_synthetic_volume_pair

from medimg_pipeline.curation.config import CurationConfig
from medimg_pipeline.curation.manifest import run_curation
from medimg_pipeline.inference.evaluate import dice_score, evaluate_test_split, iou_score
from medimg_pipeline.training.config import TrainConfig
from medimg_pipeline.training.train import run_training


def test_dice_score_identical_masks_is_one():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 1
    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0


def test_dice_score_disjoint_masks_is_zero():
    a = np.zeros((10, 10, 10), dtype=np.uint8)
    a[0:2, 0:2, 0:2] = 1
    b = np.zeros((10, 10, 10), dtype=np.uint8)
    b[8:10, 8:10, 8:10] = 1
    assert dice_score(a, b) == 0.0
    assert iou_score(a, b) == 0.0


def test_dice_score_both_empty_is_one():
    empty = np.zeros((5, 5, 5), dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0


def test_dice_score_partial_overlap():
    a = np.zeros((10, 1, 1), dtype=np.uint8)
    a[0:6] = 1  # 6 voxels
    b = np.zeros((10, 1, 1), dtype=np.uint8)
    b[3:9] = 1  # 6 voxels, overlap = [3,6) = 3 voxels
    # Dice = 2*3 / (6+6) = 0.5
    assert dice_score(a, b) == 0.5


def test_evaluate_test_split_end_to_end(tmp_path: Path):
    data_root = tmp_path / "data"
    for i in range(6):
        make_synthetic_volume_pair(data_root, f"sub-{i:03d}", shape=(24, 24, 24))

    curation_config = CurationConfig(
        data_root=str(data_root),
        min_foreground_voxels=10,
        output_manifest=str(tmp_path / "manifest.csv"),
        output_qc_report=str(tmp_path / "qc_report.json"),
    )
    run_curation(curation_config)

    train_config = TrainConfig(
        manifest_path=str(tmp_path / "manifest.csv"),
        output_dir=str(tmp_path / "training"),
        experiment_name="eval_smoke_test",
        patch_size=(16, 16, 16),
        batch_size=1,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=1,
        num_epochs=1,
        device="cpu",
        amp=False,
    )
    checkpoint_path = run_training(train_config)

    report = evaluate_test_split(
        tmp_path / "manifest.csv", checkpoint_path, tmp_path / "eval_out", device="cpu"
    )

    assert report.n_subjects > 0
    assert all(r.status == "ok" for r in report.results)
    assert report.mean_dice is not None
    assert 0.0 <= report.mean_dice <= 1.0

    report.write_json(tmp_path / "eval_out" / "report.json")
    report.write_csv(tmp_path / "eval_out" / "report.csv")
    assert (tmp_path / "eval_out" / "report.json").exists()
    assert (tmp_path / "eval_out" / "report.csv").exists()
