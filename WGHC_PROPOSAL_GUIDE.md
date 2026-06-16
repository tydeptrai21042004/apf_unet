# WGHC proposal family

All configurations in `configs/paper_fair/` inherit the same paper-fair training protocol. They differ only in the bottleneck mechanism.

| Model | Difference |
|---|---|
| `proposal_wghc_unet` | Exact equal-weight four-term WGHC reference, no gate |
| `proposal_bwghc_unet` | One learned global translated/reflected balance |
| `proposal_swghc_unet` | One bounded learned residual-strength scalar |
| `proposal_cwghc_unet` | One translated/reflected balance per channel |
| `proposal_mwghc_unet` | Softmax fusion of displacements p=1,2,3 |
| `proposal_wghc_unet_gate` | Identity-centered channel gate, ablation only |
| `axial_conv_matched` | Ordinary translated axial control with matched scaffold |

The balanced variants use `2 * (lambda*S + (1-lambda)*R)`, so `lambda=0.5` exactly reproduces the equal-weight reference at initialization.

## Full fair comparison

```bash
bash run_wghc_comparison.sh
```

## Quick smoke test

```bash
python scripts/benchmark_multi_seed.py \
  --models "proposal_hf_unet_no_gate,proposal_wghc_unet,proposal_bwghc_unet,axial_conv_matched" \
  --dataset kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --seeds "42" \
  --epochs 2 \
  --batch-size 6 \
  --device cuda \
  --output-root outputs_wghc_smoke
```
