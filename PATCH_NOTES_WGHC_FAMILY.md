# WGHC proposal-family patch

Added seven fair-comparison models:

- `proposal_wghc_unet`
- `proposal_bwghc_unet`
- `proposal_swghc_unet`
- `proposal_cwghc_unet`
- `proposal_mwghc_unet`
- `proposal_wghc_unet_gate`
- `axial_conv_matched`

The operator now exposes translated and reflected branches separately and supports integer displacement `p`. Balanced variants are normalized so `lambda=0.5` exactly matches the equal-weight WGHC reference. The gated ablation uses an identity-centered gate initialized at one, rather than suppressing the branch at initialization.

All `configs/paper_fair/*.yaml` files use the same data augmentation, batch size, image size, optimizer, learning rate, scheduler, loss, channels, projection, activation, dropout, alpha, and epoch count as the paper-fair baseline protocol. Only the declared architectural factor changes.

Validation completed:

- 38 focused HC/WGHC tests passed.
- 81 registry, runnability, HF-ablation, and WGHC tests passed.
- A full repository test invocation progressed normally but exceeded the execution time limit in this environment.
