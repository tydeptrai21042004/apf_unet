# APF encoder-stage-1 proposal

Added `proposal_apf_unet_at_encoder1`, which applies the no-gate amplitude--phase Fourier residual correction to the second encoder feature map (`feats[1]`) rather than to the bottleneck.

## Main additions

- model registration: `proposal_apf_unet_at_encoder1`
- builder alias: `apf_encoder1`
- configurations under `configs/`, `configs/ablation/`, `configs/paper_fair/`, and `configs/placement_ablation/`
- comparison runner: `run_apf_encoder1_comparison.sh`
- focused model test and config/registry coverage

## Validation

- 5 APF-specific tests passed
- 45 APF, model-runnability, and registry/config tests passed
