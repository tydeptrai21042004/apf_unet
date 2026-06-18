# Patch notes: plain Fourier U-Net proposal

## Main change

The former `proposal_apf_unet` method has been replaced by the canonical
`proposal_fourier_unet` method. The manuscript-facing name is **Fourier U-Net**.

The new full proposal corresponds to the previously strongest plain Fourier
setting, with amplitude scale `1.0` and phase range `pi`. It additionally uses
an identity-initialized spectral channel mixer.

## Architecture improvements

- positive exponential amplitude gains;
- bounded learnable phase shifts;
- identity-initialized cross-channel spectral mixing;
- exact identity initialization for the residual proposal;
- optional non-residual and no-channel-mixing controls;
- diagnostics for gain, phase, and channel-mixer deviation;
- backward-compatible aliases for old APF model names and configuration keys.

## Improved architecture-only ablations

The ordinary U-Net baseline was removed from the ablation folder because it is
already evaluated in the main baseline comparison. The new suite contains:

1. `proposal_fourier_unet`;
2. `fourier_unet_bounded`;
3. `fourier_unet_amplitude_only`;
4. `fourier_unet_phase_only`;
5. `fourier_unet_no_channel_mix`;
6. `fourier_unet_no_residual`;
7. `fourier_unet_at_encoder1`.

All seven configurations share the same optimization and evaluation protocol.

## Compatibility

The following old names remain accepted as aliases but are not registered as
independent methods:

- `proposal_apf_unet` -> `proposal_fourier_unet`;
- `fourier_unet_plain` -> `proposal_fourier_unet`;
- `apf_amplitude_only` -> `fourier_unet_amplitude_only`;
- `apf_phase_only` -> `fourier_unet_phase_only`;
- `proposal_apf_unet_at_encoder1` -> `fourier_unet_at_encoder1`.

## Validation

The targeted Fourier, repository consistency, and pipeline tests pass. The
remaining repository test files were also executed individually and passed.
The full ETIS GPU experiment was not rerun in this environment.
