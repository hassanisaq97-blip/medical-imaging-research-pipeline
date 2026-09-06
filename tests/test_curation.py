from pathlib import Path

from tests.helpers import make_synthetic_volume_pair

from medimg_pipeline.curation.config import CurationConfig
from medimg_pipeline.curation.manifest import build_manifest


def _make_dataset(root: Path, n_good: int = 8, n_missing_mask: int = 1, n_empty_mask: int = 1):
    subjects = []
    for i in range(n_good):
        sub = f"sub-{i:03d}"
        make_synthetic_volume_pair(root, sub, shape=(16, 16, 16))
        subjects.append(sub)

    for i in range(n_missing_mask):
        sub = f"sub-miss-{i:03d}"
        make_synthetic_volume_pair(root, sub, shape=(16, 16, 16), with_mask=False)
        subjects.append(sub)

    for i in range(n_empty_mask):
        sub = f"sub-empty-{i:03d}"
        make_synthetic_volume_pair(root, sub, shape=(16, 16, 16), empty_mask=True)
        subjects.append(sub)

    return subjects


def test_build_manifest_includes_only_qc_passing_subjects(tmp_path: Path):
    _make_dataset(tmp_path, n_good=8, n_missing_mask=1, n_empty_mask=1)
    config = CurationConfig(data_root=str(tmp_path), min_foreground_voxels=10)

    df, report = build_manifest(config)

    assert report.n_subjects_discovered == 10
    assert report.n_included == 8
    assert report.n_excluded == 2

    passing = df[df.qc_status == "pass"]
    assert len(passing) == 8
    assert (passing.foreground_voxels > 0).all()

    failing = df[df.qc_status == "fail"]
    reasons = set(failing.qc_message)
    assert any("missing segmentation mask" in r for r in reasons)
    assert any("zero foreground voxels" in r for r in reasons)


def test_manifest_has_no_leakage_across_splits(tmp_path: Path):
    _make_dataset(tmp_path, n_good=20, n_missing_mask=0, n_empty_mask=0)
    config = CurationConfig(data_root=str(tmp_path), min_foreground_voxels=10, split_seed=7)

    df, _ = build_manifest(config)
    passing = df[df.qc_status == "pass"]

    assert set(passing.split) <= {"train", "val", "test"}
    assert passing.subject_id.is_unique  # one row per subject here -> no cross-split duplication

    counts = passing.split.value_counts()
    assert counts.get("train", 0) > 0
    assert counts.get("val", 0) > 0
    assert counts.get("test", 0) > 0


def test_manifest_splits_are_stable_across_runs(tmp_path: Path):
    _make_dataset(tmp_path, n_good=12, n_missing_mask=0, n_empty_mask=0)
    config = CurationConfig(data_root=str(tmp_path), min_foreground_voxels=10, split_seed=123)

    df1, _ = build_manifest(config)
    df2, _ = build_manifest(config)

    merged = df1.merge(df2, on="subject_id", suffixes=("_1", "_2"))
    assert (merged.split_1 == merged.split_2).all()


def test_min_foreground_voxels_excludes_small_masks(tmp_path: Path):
    _make_dataset(tmp_path, n_good=4, n_missing_mask=0, n_empty_mask=0)
    config = CurationConfig(data_root=str(tmp_path), min_foreground_voxels=10**9)

    df, report = build_manifest(config)
    assert report.n_included == 0
    assert all("below minimum threshold" in m for m in df.qc_message)
