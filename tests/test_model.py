import torch

from medimg_pipeline.models.unet import build_unet


def test_unet_forward_pass_shape():
    model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 32, 32, 32)


def test_unet_forward_pass_batch():
    model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    x = torch.randn(2, 1, 24, 24, 24)
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 2
