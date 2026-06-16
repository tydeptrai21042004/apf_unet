# APF-U-Net ablation suite

This directory contains only experiments related to the sole proposed method, `proposal_apf_unet`.

- `unet`: spatial-domain reference.
- `proposal_apf_unet`: full bounded amplitude-and-phase APF proposal.
- `apf_amplitude_only`: removes phase modulation.
- `apf_phase_only`: removes amplitude modulation.
- `fourier_unet_plain`: unrestricted Fourier control.
- `proposal_apf_unet_at_encoder1`: APF placement control at encoder stage 1.

All files use the same training budget and evaluation threshold. Only the stated APF design factor should differ.
