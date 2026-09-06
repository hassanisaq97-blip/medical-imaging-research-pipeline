"""YAML-driven training configuration.

No absolute paths are hard-coded anywhere in this project; every path
here is either relative to the working directory or supplied explicitly
in a config file / CLI flag, so the same config works on a laptop and
inside the Docker container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    manifest_path: str
    output_dir: str = "outputs/training"
    experiment_name: str = "liver_segmentation"

    patch_size: tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 1
    num_workers: int = 0

    channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    val_interval: int = 1
    early_stopping_patience: int = 15

    device: str = "auto"
    amp: bool = True
    seed: int = 42

    @staticmethod
    def from_yaml(path: str | Path) -> TrainConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "patch_size" in raw:
            raw["patch_size"] = tuple(raw["patch_size"])
        if "channels" in raw:
            raw["channels"] = tuple(raw["channels"])
        if "strides" in raw:
            raw["strides"] = tuple(raw["strides"])
        return TrainConfig(**raw)
