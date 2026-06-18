# Fourier U-Net controlled ablation suite

The canonical proposed method is `proposal_fourier_unet`, shown in tables as
**Fourier U-Net**.  The suite intentionally excludes the ordinary U-Net because
that baseline is evaluated in the main comparison tables.

All variants use the same dataset split, training budget, optimizer, loss,
augmentation, image size, batch size, and evaluation threshold.  Each variant
changes only one design factor relative to the full proposal:

- `proposal_fourier_unet`: full plain Fourier U-Net at the bottleneck;
- `fourier_unet_bounded`: conservative amplitude/phase response bounds;
- `fourier_unet_amplitude_only`: removes phase modulation;
- `fourier_unet_phase_only`: removes amplitude modulation;
- `fourier_unet_no_channel_mix`: removes cross-channel spectral mixing;
- `fourier_unet_no_residual`: removes the residual connection;
- `fourier_unet_at_encoder1`: moves the Fourier block to encoder stage 1.

The recommended ablation reports all seven variants over the same three seeds.
