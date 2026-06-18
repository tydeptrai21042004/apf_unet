from __future__ import annotations

import ast
from pathlib import Path

import torch
import yaml

from src.models import build_model
from src.models.builder import ALIASES
from src.models.registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
FOURIER_ABLATIONS = {
    "proposal_fourier_unet",
    "fourier_unet_bounded",
    "fourier_unet_amplitude_only",
    "fourier_unet_phase_only",
    "fourier_unet_no_channel_mix",
    "fourier_unet_no_residual",
    "fourier_unet_at_encoder1",
}


def test_only_fourier_unet_is_registered_as_the_proposal():
    proposal_names = {
        name for name in MODEL_REGISTRY if name.startswith("proposal_")
    }
    assert proposal_names == {"proposal_fourier_unet", "proposal_urf_unet"}


def test_legacy_names_are_aliases_not_duplicate_registrations():
    for legacy_name in (
        "proposal_apf_unet",
        "fourier_unet_plain",
        "apf_amplitude_only",
        "apf_phase_only",
        "proposal_apf_unet_at_encoder1",
    ):
        assert legacy_name in ALIASES
        assert legacy_name not in MODEL_REGISTRY


def test_ablation_directory_and_runner_match_exactly():
    config_names = {
        path.stem for path in (ROOT / "configs" / "ablation").glob("*.yaml")
    }
    assert config_names == FOURIER_ABLATIONS

    tree = ast.parse(
        (ROOT / "scripts" / "run_fourier_ablation.py").read_text()
    )
    runner_models = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ABLATION_MODELS":
                    runner_models = set(ast.literal_eval(node.value))
    assert runner_models == FOURIER_ABLATIONS


def test_all_fourier_ablation_configs_build_and_forward():
    for path in sorted((ROOT / "configs" / "ablation").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        model_cfg = dict(cfg["model"])
        name = model_cfg.pop("name")
        model_cfg.update(
            channels=(4, 8, 16, 32, 64),
            fourier_init_hw=(4, 4),
            fourier_expansion=1.0,
            fourier_dropout=0.0,
        )
        model = build_model(name, model_cfg).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 1, 64, 64), path.name


def test_ablation_configs_share_training_protocol():
    configs = {}
    for path in sorted((ROOT / "configs" / "ablation").glob("*.yaml")):
        configs[path.stem] = yaml.safe_load(path.read_text())

    reference = configs["proposal_fourier_unet"]
    common_paths = [
        ("data", "augmentation"),
        ("data", "batch_size"),
        ("data", "image_size"),
        ("train", "epochs"),
        ("train", "lr"),
        ("train", "weight_decay"),
        ("train", "optimizer"),
        ("train", "scheduler"),
        ("train", "loss"),
        ("train", "mixed_precision"),
        ("train", "grad_clip"),
        ("eval", "loss"),
        ("eval", "threshold"),
    ]
    for section, key in common_paths:
        expected = reference[section][key]
        for name, cfg in configs.items():
            assert cfg[section][key] == expected, (name, section, key)


def test_canonical_active_code_uses_fourier_proposal_name():
    checked = [ROOT / "configs", ROOT / "scripts", ROOT / "tools"]
    violations = []
    for base in checked:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh"}:
                text = path.read_text(errors="ignore")
                if "proposal_apf_unet" in text:
                    violations.append(str(path.relative_to(ROOT)))
    # The deprecated wrapper filename is allowed, but active experiment files
    # must use the canonical proposal_fourier_unet key.
    assert not violations, violations
