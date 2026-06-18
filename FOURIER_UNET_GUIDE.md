# Fourier U-Net architecture and ablation guide

## Full proposal

The canonical model is `proposal_fourier_unet`.

For a bottleneck feature tensor `x`, the module computes a projected tensor
`z`, obtains `rfft2(z)`, applies a positive amplitude gain and bounded phase
shift at each frequency, mixes feature channels with an identity-initialized
1x1 spectral matrix, and reconstructs the spectral difference with `irfft2`.
The projected correction is added to `x`.

The default full proposal uses:

- amplitude scale: `1.0`;
- maximum phase shift: `pi`;
- cross-channel spectral mixing: enabled;
- residual correction: enabled;
- output projection: zero initialized;
- placement: deepest encoder feature.

## Why the old plain control became the proposal

The earlier ETIS ablation showed that the unrestricted plain Fourier variant
had the strongest overall test performance. The repository therefore promotes
that design to the canonical proposal and retains the former conservative
response as `fourier_unet_bounded`.

## Improved ablations

| Internal key | Manuscript name | Isolated factor |
|---|---|---|
| `proposal_fourier_unet` | Fourier U-Net | Full model |
| `fourier_unet_bounded` | Bounded Fourier | Response range |
| `fourier_unet_amplitude_only` | Amplitude only | Remove phase shift |
| `fourier_unet_phase_only` | Phase only | Remove amplitude gain |
| `fourier_unet_no_channel_mix` | No channel mixing | Remove spectral channel interaction |
| `fourier_unet_no_residual` | No residual | Remove identity-preserving residual path |
| `fourier_unet_at_encoder1` | Encoder-stage Fourier | Change placement |

The ordinary U-Net is not rerun in this architecture-only ablation because it
is already part of the main baseline benchmark.
