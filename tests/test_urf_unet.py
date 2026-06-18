from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.engine.output_utils import compute_supervised_loss, parse_model_output
from src.losses import BCEDiceLoss, StructureLoss
from src.models import build_model
from src.models.proposal.urf_unet import (
    AdaptiveRadialFourierBottleneck,
    UncertaintyRoutedLocalFourierRefiner,
)
from src.models.registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
URF_MODELS = (
    "proposal_urf_unet",
    "urf_unet_dynamic_global_only",
    "urf_unet_no_dynamic_global",
    "urf_unet_no_uncertainty",
    "urf_unet_no_boundary_supervision",
    "urf_unet_no_coarse_supervision",
)


def small_config() -> dict:
    return {
        "in_channels": 3,
        "num_classes": 1,
        "channels": (2, 4, 8, 16, 32),
        "urf_global_expansion": 1.0,
        "urf_global_num_bands": 2,
        "urf_local_decoder_index": 1,
        "urf_local_window_size": 4,
        "urf_local_num_bands": 2,
        "fourier_init_hw": (2, 2),
        "urf_global_dropout": 0.0,
        "urf_local_dropout": 0.0,
    }


def test_urf_models_are_registered():
    for name in URF_MODELS:
        assert name in MODEL_REGISTRY


def test_adaptive_global_block_is_identity_at_initialization():
    block = AdaptiveRadialFourierBottleneck(
        channels=8,
        expansion=1.0,
        num_bands=3,
        zero_init_output=True,
    ).eval()
    x = torch.randn(1, 8, 8, 8)
    with torch.no_grad():
        y = block(x)
    assert torch.equal(x, y)


def test_local_refiner_is_identity_at_initialization_and_routes_uncertainty():
    block = UncertaintyRoutedLocalFourierRefiner(
        channels=8,
        window_size=4,
        num_bands=2,
        zero_init_output=True,
    ).eval()
    x = torch.randn(1, 8, 9, 11)
    coarse = torch.zeros(1, 1, 9, 11)
    with torch.no_grad():
        y = block(x, coarse)
    assert torch.equal(x, y)
    diagnostics = block.diagnostics()
    assert diagnostics["urf/local_gate_mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", URF_MODELS)
def test_urf_variants_forward_parse_and_backward(name: str):
    torch.manual_seed(7)
    model = build_model(name, small_config())
    model.set_epoch(5)
    x = torch.randn(1, 3, 32, 32)
    masks = (torch.rand(1, 1, 32, 32) > 0.5).float()
    output = model(x)
    parsed = parse_model_output(output)
    assert parsed.main.shape == masks.shape
    assert len(parsed.aux) == 1
    assert parsed.boundary is not None
    assert parsed.boundary.shape == masks.shape
    assert torch.isfinite(parsed.main).all()

    total, _, _ = compute_supervised_loss(
        output,
        masks,
        main_loss_fn=StructureLoss(),
        boundary_loss_fn=BCEDiceLoss(),
        boundary_weight=0.2,
        use_aux_outputs=False,
        use_boundary_output=(name == "proposal_urf_unet"),
    )
    total.backward()
    gradients = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)


def test_controlled_urf_ablation_configs_build():
    config_dir = ROOT / "configs" / "urf_ablation"
    expected = {
        "proposal_fourier_unet",
        "proposal_urf_unet",
        "urf_unet_dynamic_global_only",
        "urf_unet_no_dynamic_global",
        "urf_unet_no_uncertainty",
        "urf_unet_no_boundary_supervision",
        "urf_unet_no_coarse_supervision",
    }
    assert {path.stem for path in config_dir.glob("*.yaml")} == expected

    for path in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_cfg = dict(cfg["model"])
        name = model_cfg.pop("name")
        model_cfg["channels"] = (2, 4, 8, 16, 32)
        if name == "proposal_fourier_unet":
            model_cfg["fourier_init_hw"] = (2, 2)
            model_cfg["fourier_expansion"] = 1.0
        else:
            model_cfg["urf_global_expansion"] = 1.0
            model_cfg["urf_global_num_bands"] = 2
            model_cfg["urf_local_window_size"] = 4
            model_cfg["urf_local_num_bands"] = 2
            model_cfg["fourier_init_hw"] = (2, 2)
        model = build_model(name, model_cfg).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
        parsed = parse_model_output(output)
        assert parsed.main.shape == (1, 1, 32, 32), path.name


def test_urf_ablation_training_protocol_is_controlled():
    config_dir = ROOT / "configs" / "urf_ablation"
    configs = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in config_dir.glob("*.yaml")
    }
    reference = configs["proposal_fourier_unet"]
    common_paths = [
        ("data", "augmentation"),
        ("data", "batch_size"),
        ("data", "image_size"),
        ("train", "epochs"),
        ("train", "lr"),
        ("train", "loss"),
        ("train", "optimizer"),
        ("train", "scheduler"),
        ("train", "weight_decay"),
        ("train", "grad_clip"),
        ("eval", "loss"),
        ("eval", "threshold"),
    ]
    for section, key in common_paths:
        expected = reference[section][key]
        for name, cfg in configs.items():
            assert cfg[section][key] == expected, (name, section, key)

    assert configs["proposal_urf_unet"]["train"]["use_boundary_loss"] is True
    assert configs["proposal_urf_unet"]["train"]["use_aux_outputs_loss"] is True
    assert configs["urf_unet_no_boundary_supervision"]["train"]["use_boundary_loss"] is False
    assert configs["urf_unet_no_boundary_supervision"]["train"]["use_aux_outputs_loss"] is True
    assert configs["urf_unet_no_coarse_supervision"]["train"]["use_boundary_loss"] is True
    assert configs["urf_unet_no_coarse_supervision"]["train"]["use_aux_outputs_loss"] is False
