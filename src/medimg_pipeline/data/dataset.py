"""Turn a curation manifest into MONAI datasets/dataloaders.

Deliberately thin: all the interesting logic (QC, splitting, leakage
prevention) already happened in `medimg_pipeline.curation` and is baked
into the manifest CSV. This module's only job is to read that CSV and
wire it into MONAI's Dataset/DataLoader plus this project's transform
pipelines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def manifest_to_data_dicts(manifest_path: str | Path, split: str) -> list[dict]:
    """Return [{"image": ..., "label": ...}, ...] for QC-passing rows in `split`."""

    df = pd.read_csv(manifest_path)
    subset = df[(df.qc_status == "pass") & (df.split == split)]
    return [
        {"image": row.image_path, "label": row.mask_path} for row in subset.itertuples(index=False)
    ]


def build_dataloaders(
    manifest_path: str | Path,
    *,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 1,
    num_workers: int = 0,
):
    """Build train/val MONAI DataLoaders from a curation manifest.

    `num_workers=0` is the safe default: multiprocessing DataLoader
    workers can behave unpredictably on macOS/MPS and inside constrained
    containers, so this project only enables it when explicitly asked.
    """

    from monai.data import DataLoader, Dataset, list_data_collate

    from medimg_pipeline.preprocessing.transforms import get_eval_transforms, get_train_transforms

    train_dicts = manifest_to_data_dicts(manifest_path, "train")
    val_dicts = manifest_to_data_dicts(manifest_path, "val")

    # Plain Dataset, not CacheDataset: caching pre-transformed volumes in
    # memory only pays off across many epochs on a dataset small enough to
    # fit in RAM, which is not a general assumption this project makes.
    train_ds = Dataset(data=train_dicts, transform=get_train_transforms(patch_size))
    val_ds = Dataset(data=val_dicts, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader
