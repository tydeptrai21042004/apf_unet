from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.engine import Evaluator, Trainer
from src.losses import BCEDiceLoss
from src.models import build_model
from src.models.registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
BASELINES = {
    "unet", "attention_unet", "unetpp", "resunetpp", "pranet", "acsnet",
    "hardnet_mseg", "polyp_pvt", "caranet", "cfanet", "hsnet", "csca_unet",
}
ABLATIONS = {
    "unet", "proposal_apf_unet", "apf_amplitude_only", "apf_phase_only",
    "fourier_unet_plain", "proposal_apf_unet_at_encoder1",
}


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_config_root_has_exactly_three_experiment_folders():
    assert {p.name for p in CONFIGS.iterdir() if p.is_dir()} == {
        "official_faithful", "fair", "ablation"
    }
    assert not [p for p in CONFIGS.iterdir() if p.is_file() and p.suffix in {".yaml", ".yml"}]


def test_official_faithful_is_baseline_only_and_complete():
    names = {p.stem for p in (CONFIGS / "official_faithful").glob("*.yaml")}
    assert names == BASELINES
    assert not any(name.startswith("proposal_") for name in names)


def test_fair_has_same_baselines_plus_only_apf_proposal():
    names = {p.stem for p in (CONFIGS / "fair").glob("*.yaml")}
    assert names == BASELINES | {"proposal_apf_unet"}
    assert {name for name in names if name.startswith("proposal_")} == {"proposal_apf_unet"}


def test_ablation_folder_contains_only_controlled_apf_variants():
    names = {p.stem for p in (CONFIGS / "ablation").glob("*.yaml")}
    assert names == ABLATIONS
    for path in (CONFIGS / "ablation").glob("*.yaml"):
        cfg = load(path)
        assert cfg["model"]["name"] in MODEL_REGISTRY


def test_unet_cbam_is_fully_removed_as_a_selectable_baseline():
    assert "unet_cbam" not in MODEL_REGISTRY
    active_roots = [ROOT / "src", ROOT / "configs", ROOT / "scripts", ROOT / "tools"]
    hits = []
    for base in active_roots:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh"}:
                if "unet_cbam" in path.read_text(encoding="utf-8", errors="ignore").lower():
                    hits.append(path)
    assert hits == []


def test_only_one_top_level_shell_entrypoint_remains():
    assert [p.name for p in ROOT.glob("*.sh")] == ["run.sh"]
    text = (ROOT / "run.sh").read_text(encoding="utf-8")
    for command in ("fair", "faithful", "ablation", "test"):
        assert f"{command})" in text


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        return {
            "image": torch.rand(3, 32, 32, generator=g),
            "mask": (torch.rand(1, 32, 32, generator=g) > 0.5).float(),
        }


def test_train_validate_checkpoint_pipeline_for_fair_proposal(tmp_path: Path):
    cfg = load(CONFIGS / "fair" / "proposal_apf_unet.yaml")
    model_cfg = dict(cfg["model"])
    model_cfg["channels"] = [4, 8, 16, 32, 64]
    model = build_model("proposal_apf_unet", model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(TinyDataset(), batch_size=2)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=BCEDiceLoss(),
        device="cpu",
        mixed_precision=False,
    )
    train_metrics = trainer.train_one_epoch(loader, epoch=1)
    val_metrics = trainer.validate(loader)
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert 0.0 <= val_metrics["dice"] <= 1.0
    torch.save({"model": model.state_dict(), "epoch": 1, "metrics": val_metrics}, tmp_path / "pipeline.pt")
    assert (tmp_path / "pipeline.pt").exists()
    evaluator = Evaluator(device="cpu", threshold=0.5, loss_fn=BCEDiceLoss())
    metrics = evaluator.evaluate(model, loader)
    assert {"loss", "dice", "iou", "precision", "recall"} <= set(metrics)
