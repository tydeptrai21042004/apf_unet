from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import yaml

from src.models import build_model
from src.models.registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
APF_ABLATIONS = {
    "unet",
    "proposal_apf_unet",
    "apf_amplitude_only",
    "apf_phase_only",
    "fourier_unet_plain",
    "proposal_apf_unet_at_encoder1",
}
LEGACY_TOKENS = (
    "proposal_hf_", "proposal_hc_", "proposal_chd_", "proposal_wghc_",
    "proposal_bwghc_", "proposal_swghc_", "proposal_cwghc_", "proposal_mwghc_",
)


def test_only_apf_is_registered_as_a_proposal():
    proposal_names = {name for name in MODEL_REGISTRY if name.startswith("proposal_")}
    assert proposal_names == {"proposal_apf_unet", "proposal_apf_unet_at_encoder1"}


def test_legacy_proposal_aliases_are_rejected():
    for name in ("proposal_hf_unet", "proposal_hc_unet_no_gate", "proposal_chd_unet"):
        with pytest.raises(KeyError):
            build_model(name, {"channels": (4, 8, 16, 32, 64)})


def test_ablation_directory_and_runner_match_exactly():
    config_names = {p.stem for p in (ROOT / "configs" / "ablation").glob("*.yaml")}
    assert config_names == APF_ABLATIONS
    tree = ast.parse((ROOT / "scripts" / "run_apf_ablation.py").read_text())
    runner_models = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ABLATION_MODELS":
                    runner_models = set(ast.literal_eval(node.value))
    assert runner_models == APF_ABLATIONS


def test_all_apf_ablation_configs_build_and_forward():
    for path in sorted((ROOT / "configs" / "ablation").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        model_cfg = dict(cfg["model"])
        name = model_cfg.pop("name")
        model_cfg.update(channels=(4, 8, 16, 32, 64), apf_init_hw=(4, 4)) if name != "unet" else model_cfg.update(channels=(4, 8, 16, 32, 64))
        model = build_model(name, model_cfg).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 1, 64, 64), path.name


def test_repository_has_no_legacy_proposal_references():
    checked = [ROOT / "src", ROOT / "configs", ROOT / "scripts", ROOT / "tools"]
    violations = []
    for base in checked:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh", ".md"}:
                text = path.read_text(errors="ignore").lower()
                for token in LEGACY_TOKENS:
                    if token in text:
                        violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "\n".join(violations)


def test_apf_supports_odd_spatial_sizes_and_finite_backward():
    model = build_model("proposal_apf_unet", {
        "channels": (4, 8, 16, 32, 64), "apf_expansion": 1.0,
        "apf_init_hw": (5, 5), "norm": "gn",
    })
    x = torch.randn(2, 3, 80, 96)
    y = model(x)
    assert y.shape == (2, 1, 80, 96)
    loss = y.square().mean()
    loss.backward()
    grads = [p.grad for p in model.apf_bottleneck.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
