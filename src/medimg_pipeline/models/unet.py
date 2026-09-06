"""3D U-Net for single-organ binary segmentation, built on MONAI.

The default channel/stride configuration is deliberately small (4 levels,
max 256 channels) so a forward+backward pass fits in a few GB of RAM/VRAM
on a laptop-class CPU/MPS device, per this project's hardware
constraints. `configs/train_gpu.yaml` widens it for real GPU training.
"""

from __future__ import annotations


def build_unet(
    *,
    in_channels: int = 1,
    out_channels: int = 2,  # background + 1 foreground organ class
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    spatial_dims: int = 3,
):
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
    )
