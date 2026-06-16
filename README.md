# APF-U-Net

This repository contains baseline segmentation models and one proposed method: `proposal_apf_unet` (Amplitude-Phase Fourier U-Net). Older HF, HC, CHD, and WGHC proposal families were removed to keep the code, configurations, tests, and paper experiments consistent.

## Baseline paper-alignment audit

The repository includes a source-by-source architecture audit in [`docs/BASELINE_AUDIT.md`](docs/BASELINE_AUDIT.md) and machine-readable official-source contracts in [`configs/baseline_sources.yaml`](configs/baseline_sources.yaml). The `official_faithful` configurations enforce paper-aligned backbones and major module/output contracts; `paper_fair` configurations remain controlled benchmark adaptations.

Run the complete verification suite with:

```bash
pytest -q
```


## Main proposal

`proposal_apf_unet` applies a bounded, no-gate amplitude-and-phase Fourier residual correction to the deepest encoder feature. The APF output projection is zero-initialized, so the block starts as an exact identity mapping.

## Controlled APF ablations

The directory `configs/ablation/` contains only:

- `unet`
- `proposal_apf_unet`
- `apf_amplitude_only`
- `apf_phase_only`
- `fourier_unet_plain`
- `proposal_apf_unet_at_encoder1`

Run the suite with:

```bash
python scripts/run_apf_ablation.py --dataset kvasir_seg --data-root data --device auto --seed 42
```

## Prepare data

```bash
python scripts/prepare_dataset.py --dataset kvasir_seg --data-root data --image-size 352
python scripts/make_splits.py --dataset kvasir_seg --data-root data --image-size 352 --seed 42
```

## Train the proposal

```bash
python scripts/train_one.py --model proposal_apf_unet --config configs/proposal_apf_unet.yaml --dataset kvasir_seg --data-root data --device auto
```

## Evaluate

```bash
python scripts/eval_one.py --model proposal_apf_unet --config configs/proposal_apf_unet.yaml --dataset kvasir_seg --data-root data --device auto
```

## Run tests

```bash
pytest -q
```
