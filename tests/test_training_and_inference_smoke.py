"""End-to-end smoke test: curate -> train (tiny, CPU) -> infer -> visual QC.

Uses only synthetic volumes and a one-epoch, tiny-patch configuration.
Any metric produced here is a smoke-test result on synthetic data, never
a claim about real segmentation performance -- see docs/ for the
distinction. This test is intentionally excluded from CI's default fast
suite via the `slow` marker (see pyproject/pytest config + CI workflow),
but is still meant to be fast enough (~seconds, tiny volumes, 1 epoch,
CPU) to run locally without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import make_synthetic_volume_pair

from medimg_pipeline.curation.config import CurationConfig
from medimg_pipeline.curation.manifest import run_curation
from medimg_pipeline.imaging.nifti import load_nifti
from medimg_pipeline.inference.infer import run_inference
from medimg_pipeline.qc.visual import generate_overlay_figure
from medimg_pipeline.training.config import TrainConfig
from medimg_pipeline.training.train import run_training

pytestmark = pytest.mark.slow


def _build_tiny_dataset(root: Path, n_subjects: int = 6):
    for i in range(n_subjects):
        make_synthetic_volume_pair(
            root, f"sub-{i:03d}", shape=(24, 24, 24), spacing=(2.0, 2.0, 2.0)
        )


def test_full_pipeline_smoke(tmp_path: Path):
    data_root = tmp_path / "data"
    _build_tiny_dataset(data_root, n_subjects=6)

    curation_config = CurationConfig(
        data_root=str(data_root),
        min_foreground_voxels=10,
        output_manifest=str(tmp_path / "manifest.csv"),
        output_qc_report=str(tmp_path / "qc_report.json"),
    )
    df, report = run_curation(curation_config)
    assert report.n_included == 6

    train_config = TrainConfig(
        manifest_path=str(tmp_path / "manifest.csv"),
        output_dir=str(tmp_path / "training"),
        experiment_name="smoke_test",
        patch_size=(16, 16, 16),
        batch_size=1,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=1,
        num_epochs=1,
        val_interval=1,
        early_stopping_patience=5,
        device="cpu",
        amp=False,
    )
    checkpoint_path = run_training(train_config)
    assert checkpoint_path.exists()

    tracking_json = Path(train_config.output_dir) / "smoke_test.json"
    tracking_csv = Path(train_config.output_dir) / "smoke_test.csv"
    assert tracking_json.exists()
    assert tracking_csv.exists()

    inference_dir = tmp_path / "inference_out"
    test_image = data_root / "sub-000" / "ct" / "sub-000_ct.nii.gz"
    result = run_inference(test_image, checkpoint_path, inference_dir, device="cpu")

    assert Path(result.output_mask_path).exists()
    assert Path(result.metadata_path).exists()

    pred_array, _, pred_info = load_nifti(result.output_mask_path)
    orig_array, _, orig_info = load_nifti(test_image)
    assert pred_info.shape == orig_info.shape  # prediction resampled back to original grid

    qc_path = tmp_path / "qc_overlay.png"
    generate_overlay_figure(orig_array, pred_array, qc_path)
    assert qc_path.exists()
    assert qc_path.stat().st_size > 0
