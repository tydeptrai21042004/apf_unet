# APF-U-Net

`proposal_apf_unet` is a no-gate Fourier bottleneck that learns separate,
bounded amplitude and phase corrections. It reconstructs only the spectral
difference, so zero correction gives an exact identity residual block.

## Main equations

- `F = rFFT2(Z)`
- `A' = A * (1 + s_A tanh(U_A))`
- `Phi' = Phi + s_Phi tanh(U_Phi)`
- `DeltaF = A' exp(i Phi') - F`
- `Y = X + alpha * P_out(irFFT2(DeltaF))`

The final projection is zero-initialized. Spectral logits use a small non-zero
initialization, preventing a dead adapter while preserving exact identity.

## Fair comparison

```bash
DATASET=etis SEEDS=42 EPOCHS=30 BATCH_SIZE=4 IMAGE_SIZE=224 \
  bash run_apf_comparison.sh
```

The comparison includes the gated HF proposal, HF without gate, Fourier-only
HF ablation, plain Fourier control, amplitude-only APF, phase-only APF, and the
full APF proposal.
