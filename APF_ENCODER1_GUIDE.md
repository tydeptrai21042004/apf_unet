# APF-U-Net at encoder stage 1

Model name: `proposal_apf_unet_at_encoder1`

The model applies the same no-gate amplitude--phase Fourier residual-correction block used by `proposal_apf_unet`, but to the second encoder feature map (`feats[1]`) instead of the deepest bottleneck feature.

For channels `(32,64,128,256,512)`, APF therefore operates on 64 channels. For a 224x224 input, this feature map is approximately 112x112.

## One-seed ETIS comparison

```bash
DATASET=etis SEEDS=42 EPOCHS=30 BATCH_SIZE=4 IMAGE_SIZE=224 \
  bash run_apf_encoder1_comparison.sh
```

## Direct command

```bash
python scripts/benchmark_multi_seed.py \
  --models "hf_unet_hf_at_encoder1,proposal_apf_unet,proposal_apf_unet_at_encoder1" \
  --dataset etis \
  --config-dir configs/ablation \
  --data-root data \
  --seeds "42" \
  --epochs 30 \
  --batch-size 4 \
  --image-size 224 \
  --num-workers 2 \
  --device cuda \
  --output-root outputs_apf_encoder1_etis_seed42 \
  --delete-checkpoints-after-eval
```
