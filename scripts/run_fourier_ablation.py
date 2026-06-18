#!/usr/bin/env python3
"""Run the controlled Fourier U-Net ablation suite."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODELS = [
    "proposal_fourier_unet",
    "fourier_unet_bounded",
    "fourier_unet_amplitude_only",
    "fourier_unet_phase_only",
    "fourier_unet_no_channel_mix",
    "fourier_unet_no_residual",
    "fourier_unet_at_encoder1",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the controlled Fourier U-Net architecture ablations."
    )
    parser.add_argument("--dataset", default="kvasir_seg")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-root", default="outputs_fourier_ablation")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_all.py"),
        "--models",
        ",".join(ABLATION_MODELS),
        "--config-dir",
        str(ROOT / "configs" / "ablation"),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--output-root",
        args.output_root,
    ]
    for flag, value in (
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--lr", args.lr),
        ("--num-workers", args.num_workers),
    ):
        if value is not None:
            command += [flag, str(value)]

    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
