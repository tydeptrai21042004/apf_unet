#!/usr/bin/env python3
"""Run the controlled APF-U-Net ablation suite."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODELS = [
    "unet",
    "proposal_apf_unet",
    "apf_amplitude_only",
    "apf_phase_only",
    "fourier_unet_plain",
    "proposal_apf_unet_at_encoder1",
]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kvasir_seg")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cmd = [sys.executable, str(ROOT / "scripts" / "train_all.py"),
           "--models", ",".join(ABLATION_MODELS), "--config-dir", str(ROOT / "configs" / "ablation"),
           "--dataset", args.dataset, "--data-root", args.data_root,
           "--device", args.device, "--seed", str(args.seed)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
