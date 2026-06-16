import torch

from src.models import build_model
from src.models.proposal.apf_unet import AmplitudePhaseFourierBottleneck
from src.models.registry import MODEL_REGISTRY


def test_apf_models_registered():
    for name in (
        "proposal_apf_unet",
        "proposal_apf_unet_at_encoder1",
        "apf_amplitude_only",
        "apf_phase_only",
        "fourier_unet_plain",
    ):
        assert name in MODEL_REGISTRY


def test_apf_bottleneck_preserves_shape_and_is_identity_at_init():
    torch.manual_seed(0)
    block = AmplitudePhaseFourierBottleneck(
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


def test_apf_full_model_forward_and_backward():
    model = build_model(
        "proposal_apf_unet",
        {
            "in_channels": 3,
            "num_classes": 1,
            "channels": (4, 8, 16, 32, 64),
            "apf_expansion": 1.0,
            "apf_init_hw": (4, 4),
            "apf_dropout": 0.0,
            "norm": "bn",
            "act": "relu",
        },
    )
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    y.mean().backward()
    assert model.apf_bottleneck.post.weight.grad is not None
    assert torch.isfinite(model.apf_bottleneck.post.weight.grad).all()


def test_amplitude_and_phase_ablation_flags():
    amp = build_model(
        "apf_amplitude_only",
        {"channels": (4, 8, 16, 32, 64), "apf_expansion": 1.0, "apf_init_hw": (4, 4)},
    )
    phase = build_model(
        "apf_phase_only",
        {"channels": (4, 8, 16, 32, 64), "apf_expansion": 1.0, "apf_init_hw": (4, 4)},
    )
    assert amp.apf_bottleneck.use_amplitude is True
    assert amp.apf_bottleneck.use_phase is False
    assert phase.apf_bottleneck.use_amplitude is False
    assert phase.apf_bottleneck.use_phase is True


def test_apf_encoder1_model_uses_second_encoder_feature():
    model = build_model(
        "proposal_apf_unet_at_encoder1",
        {
            "in_channels": 3,
            "num_classes": 1,
            "channels": (4, 8, 16, 32, 64),
            "apf_expansion": 1.0,
            "apf_init_hw": (8, 8),
            "apf_dropout": 0.0,
            "norm": "bn",
            "act": "relu",
        },
    )
    assert model.apf_stage_index == 1
    assert model.apf_bottleneck.channels == 8
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    y.mean().backward()
    assert model.apf_bottleneck.post.weight.grad is not None
