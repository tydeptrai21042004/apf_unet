import pytest
import torch
import torch.nn as nn

from src.models import build_model
from src.models.proposal.fourier_unet import FourierSpectralBottleneck
from src.models.registry import MODEL_REGISTRY


FOURIER_MODELS = (
    "proposal_fourier_unet",
    "fourier_unet_bounded",
    "fourier_unet_amplitude_only",
    "fourier_unet_phase_only",
    "fourier_unet_no_channel_mix",
    "fourier_unet_no_residual",
    "fourier_unet_at_encoder1",
)


def small_config() -> dict:
    return {
        "in_channels": 3,
        "num_classes": 1,
        "channels": (4, 8, 16, 32, 64),
        "fourier_expansion": 1.0,
        "fourier_init_hw": (4, 4),
        "fourier_dropout": 0.0,
        "norm": "bn",
        "act": "relu",
    }


def test_fourier_models_registered():
    for name in FOURIER_MODELS:
        assert name in MODEL_REGISTRY


def test_fourier_bottleneck_preserves_shape_and_is_identity_at_init():
    torch.manual_seed(0)
    block = FourierSpectralBottleneck(
        channels=8,
        expansion=1.0,
        init_hw=(8, 8),
        zero_init_output=True,
    ).eval()
    x = torch.randn(2, 8, 8, 8)
    with torch.no_grad():
        y = block(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)


def test_spectral_channel_mixer_is_identity_initialized():
    block = FourierSpectralBottleneck(
        channels=8,
        expansion=1.0,
        init_hw=(8, 8),
        use_channel_mixing=True,
    )
    assert isinstance(block.spectral_channel_mixer, nn.Conv2d)
    matrix = block.spectral_channel_mixer.weight[:, :, 0, 0]
    assert torch.allclose(matrix, torch.eye(8))


@pytest.mark.parametrize("name", FOURIER_MODELS)
def test_all_fourier_variants_forward_and_backward(name: str):
    model = build_model(name, small_config())
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    loss = y.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.fourier_bottleneck.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_amplitude_and_phase_ablation_flags():
    amplitude = build_model("fourier_unet_amplitude_only", small_config())
    phase = build_model("fourier_unet_phase_only", small_config())
    assert amplitude.fourier_bottleneck.use_amplitude is True
    assert amplitude.fourier_bottleneck.use_phase is False
    assert phase.fourier_bottleneck.use_amplitude is False
    assert phase.fourier_bottleneck.use_phase is True


def test_bounded_response_uses_conservative_limits():
    model = build_model("fourier_unet_bounded", small_config())
    assert model.fourier_bottleneck.amplitude_scale == pytest.approx(0.1)
    assert model.fourier_bottleneck.phase_max == pytest.approx(torch.pi / 4)


def test_no_channel_mix_variant_uses_identity_module():
    model = build_model("fourier_unet_no_channel_mix", small_config())
    assert model.fourier_bottleneck.use_channel_mixing is False
    assert isinstance(model.fourier_bottleneck.spectral_channel_mixer, nn.Identity)


def test_no_residual_variant_is_not_zero_initialized():
    model = build_model("fourier_unet_no_residual", small_config())
    block = model.fourier_bottleneck
    assert block.residual is False
    assert block.zero_init_output is False
    assert block.post.weight.abs().sum() > 0


def test_encoder1_model_uses_second_encoder_feature():
    model = build_model("fourier_unet_at_encoder1", small_config())
    assert model.fourier_stage_index == 1
    assert model.fourier_bottleneck.channels == 8


def test_legacy_model_name_and_config_keys_are_translated():
    legacy = {
        "channels": (4, 8, 16, 32, 64),
        "apf_expansion": 1.0,
        "apf_init_hw": (4, 4),
        "apf_amplitude_scale": 1.0,
        "apf_phase_max": float(torch.pi),
    }
    model = build_model("proposal_apf_unet", legacy)
    assert model.__class__.__name__ == "FourierUNet"
    assert model.fourier_bottleneck.amplitude_scale == pytest.approx(1.0)
