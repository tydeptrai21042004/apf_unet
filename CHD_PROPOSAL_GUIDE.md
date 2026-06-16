# CHD-U-Net proposal implementation

## Registered models

- `proposal_chd_unet`: projection-based complementary decomposition with the complementarity penalty.
- `proposal_chd_unet_no_comp`: architecture-matched ablation without the complementarity penalty.
- `proposal_hf_unet`: unchanged gated HF-U-Net reference.

## Bottleneck equations

For the deepest encoder feature `X`:

1. `G = T_HF(X)` is the exact gated HF residual branch used by HF-U-Net.
2. `P_G(X) = <X,G> / (||G||^2 + eps) * G` is calculated per sample and channel over spatial dimensions.
3. `E = X - P_G(X)` is the complementary projection residual.
4. `L = W_GHC(E)` is a channel-balanced weighted generalized h-Hartley--cosine correction.
5. `Y = X + alpha*G + beta*L`, where `beta = beta_max*tanh(raw_beta)` and starts at zero.

The training regularizer is:

`L_comp = mean(abs(cosine_similarity(G, L)))`.

The fair config uses `aux_loss_weight: 0.01`, so the total training objective is:

`L = L_seg + 0.01 * L_comp`.

## Diagnostics exported during validation and testing

- `chd/cosine_similarity`
- `chd/absolute_cosine_similarity`
- `chd/hf_energy_ratio`
- `chd/complement_energy_ratio`
- `chd/wghc_to_hf_energy_ratio`
- `chd/alpha`
- `chd/beta`

These appear in the normal evaluation metrics JSON/CSV files.

## Fair three-model comparison

```bash
bash run_chd_comparison.sh
```

One-seed ETIS screening:

```bash
DATASET=etis SEEDS=42 EPOCHS=30 BATCH_SIZE=4 IMAGE_SIZE=224 \
OUTPUT_ROOT=outputs_chd_etis_seed42 bash run_chd_comparison.sh
```

The HF reference and CHD models share the same encoder, decoder, HF settings,
augmentation, optimizer, learning rate, scheduler, segmentation loss, image size,
and training duration. CHD changes only the complementary decomposition/WGHC
branch and, for the main model, adds the declared complementarity penalty.
