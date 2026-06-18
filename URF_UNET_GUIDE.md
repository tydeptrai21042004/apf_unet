# URF-U-Net Implementation Guide

## Canonical model

- Registry key: `proposal_urf_unet`
- Manuscript name: **URF-U-Net: Uncertainty-Routed Local--Global Fourier U-Net**
- Plain comparison model: `proposal_fourier_unet`

## Architecture

URF-U-Net extends the plain Fourier U-Net with two complementary spectral paths.

### 1. Adaptive global Fourier branch

At the deepest encoder feature, the model:

1. computes an orthonormal real FFT;
2. predicts image-conditioned weights for smooth radial frequency bands;
3. forms a bounded positive amplitude response;
4. optionally performs cross-channel spectral mixing;
5. applies the inverse FFT and adds a warmed-up residual correction.

The response is resolution independent because radial bases are generated from the
runtime FFT grid rather than stored as a fixed `H x W` parameter tensor.

### 2. Uncertainty-routed local Fourier branch

After decoder block 2 by default, the model:

1. predicts coarse segmentation logits;
2. computes uncertainty `U = 4 p (1-p)`;
3. divides the decoder feature into local windows;
4. applies adaptive middle/high-frequency Fourier refinement per window;
5. gates each window by its mean uncertainty;
6. adds the local correction residually.

A routing floor prevents the branch from becoming completely inactive in highly
confident regions.

### 3. Supervision

The full model returns:

- `main`: final segmentation logits;
- `aux`: upsampled coarse logits used to make uncertainty meaningful;
- `boundary`: full-resolution boundary logits;
- `uncertainty`: a diagnostic uncertainty map.

The full configuration uses:

- Structure loss on the main prediction;
- `0.2 x` Structure loss on the coarse prediction;
- `0.2 x` BCE-Dice loss on the boundary prediction.

## Controlled ablation variants

| Registry key | Purpose |
|---|---|
| `proposal_fourier_unet` | Stabilized plain Fourier U-Net baseline |
| `urf_unet_dynamic_global_only` | Adaptive global Fourier branch without local refinement |
| `urf_unet_no_dynamic_global` | Static plain global Fourier block plus routed local refinement |
| `urf_unet_no_uncertainty` | Local Fourier refinement without uncertainty routing |
| `urf_unet_no_boundary_supervision` | Full architecture without boundary loss |
| `urf_unet_no_coarse_supervision` | Full architecture without coarse auxiliary loss |
| `proposal_urf_unet` | Complete URF-U-Net |

The ETIS ablation uses the same image size, augmentation, optimizer, learning
rate, scheduler, batch size, and main segmentation loss for every model. The two
supervision ablations isolate the additional coarse and boundary losses.

## Run one seed

```bash
python scripts/run_urf_ablation.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --seed 42 \
  --epochs 60 \
  --batch-size 6 \
  --lr 0.0001
```

## Run the complete Kaggle experiment

Use:

```text
kaggle_urf_ablation_etis_3seeds_cell.txt
```

It executes 7 variants over seeds `42,1,2`, validates all 21 runs, removes
checkpoints after evaluation, ranks methods by mean Dice, and exports a LaTeX
table.

## Important interpretation rule

The previous ETIS Plain Fourier result obtained with 30 epochs and learning rate
`3e-4` is not directly comparable with the new URF ablation. The new controlled
experiment uses 60 epochs, learning rate `1e-4`, Structure loss, response
warm-up, and bounded spectral ranges for every model. Re-run the plain Fourier
baseline in the new suite before updating manuscript claims.
