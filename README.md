# Fourier U-Net and URF-U-Net

This repository contains medical-image segmentation baselines and two spectral
U-Net models:

- `proposal_fourier_unet`: **Plain Fourier U-Net**, used as the controlled
  spectral baseline;
- `proposal_urf_unet`: **URF-U-Net: Uncertainty-Routed Local--Global Fourier
  U-Net**, the new proposed model.

URF-U-Net combines image-conditioned radial global filtering at the deepest
encoder feature with uncertainty-routed local Fourier refinement in the decoder.

## Proposed method

The Fourier block performs the following operations:

1. a channel projection into a hidden feature space;
2. an orthonormal 2-D real FFT;
3. learnable frequency-dependent amplitude gain and phase shift;
4. optional cross-channel spectral mixing;
5. an inverse real FFT;
6. a residual correction added to the original bottleneck feature.

The amplitude response uses a positive exponential parameterization. The
spectral channel mixer is initialized as the identity, and the residual output
projection is initialized to zero. Therefore, the full residual block starts as
an exact identity map.

## Canonical model key

```text
proposal_fourier_unet
```

Older keys such as `proposal_apf_unet` and `fourier_unet_plain` are accepted as
backward-compatible aliases, but all new experiments and result tables should
use `proposal_fourier_unet` and the display name **Fourier U-Net**.

## Controlled Fourier ablations

The ablation suite intentionally excludes ordinary U-Net because U-Net belongs
in the main baseline comparison. The seven controlled Fourier variants are:

- `proposal_fourier_unet`: full Fourier U-Net;
- `fourier_unet_bounded`: conservative response bounds;
- `fourier_unet_amplitude_only`: no learned phase shift;
- `fourier_unet_phase_only`: no learned amplitude gain;
- `fourier_unet_no_channel_mix`: no cross-channel spectral mixer;
- `fourier_unet_no_residual`: direct feature replacement;
- `fourier_unet_at_encoder1`: Fourier block at encoder stage 1.

Every ablation configuration uses the same optimizer, learning rate, image size,
batch size, augmentation, loss, evaluation threshold, and training budget.

Run one training seed with:

```bash
python scripts/run_fourier_ablation.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --seed 42 \
  --epochs 30 \
  --batch-size 6 \
  --lr 0.0003
```

Use the supplied Kaggle multi-seed script for the complete three-seed ETIS
ablation.

## Prepare a dataset

```bash
python scripts/prepare_dataset.py \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --allow-insecure-download

python scripts/make_splits.py \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --seed 42
```

## Train Fourier U-Net

```bash
python scripts/train_one.py \
  --model proposal_fourier_unet \
  --config configs/fair/proposal_fourier_unet.yaml \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda
```

## Evaluate Fourier U-Net

```bash
python scripts/eval_one.py \
  --model proposal_fourier_unet \
  --config configs/fair/proposal_fourier_unet.yaml \
  --dataset etis \
  --split test \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --device cuda
```

## Run validation tests

```bash
python -m pytest -q \
  tests/test_fourier_unet.py \
  tests/test_fourier_repository_consistency.py \
  tests/test_pipeline_contracts.py
```

## Dataset support

The repository supports Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS,
Kvasir-Instrument, HyperKvasir segmented data, ISBI 2012, and the other dataset
keys defined in `src/datasets/registry.py`.


## URF-U-Net comparison

The focused URF ablation compares the full model directly with the stabilized
Plain Fourier U-Net baseline and five controlled variants:

```bash
bash run.sh urf-ablation
```

For the complete three-seed ETIS experiment, use
`kaggle_urf_ablation_etis_3seeds_cell.txt`. See `URF_UNET_GUIDE.md` for the
architecture, loss design, configuration keys, and interpretation rules.
