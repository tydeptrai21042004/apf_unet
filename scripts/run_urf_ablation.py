#!/usr/bin/env python3
"""Run the controlled URF-U-Net ablation suite."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODELS = [
    "proposal_fourier_unet",
    "urf_unet_dynamic_global_only",
    "urf_unet_no_dynamic_global",
    "urf_unet_no_uncertainty",
    "urf_unet_no_boundary_supervision",
    "urf_unet_no_coarse_supervision",
    "proposal_urf_unet",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare plain Fourier U-Net with URF-U-Net and its controlled "
            "local/global/uncertainty/boundary ablations."
        )
    )
    parser.add_argument("--dataset", default="etis")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-root", default="outputs_urf_ablation")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_all.py"),
        "--models",
        ",".join(ABLATION_MODELS),
        "--config-dir",
        str(ROOT / "configs" / "urf_ablation"),
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
