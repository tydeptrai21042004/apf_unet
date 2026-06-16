# Sixteen Independent HC Sessions

The previous four large sessions are replaced by sixteen smaller, standalone sessions.
Every script can run in a fresh clone. It installs dependencies, prepares only its required dataset, runs its assigned experiment, and removes checkpoints/download archives/raw copies by default. No result ZIP is created.

| Session | Dataset(s) | Models / ablations |
|---|---|---|
| 1 | ISIC 2018 | UNet, UNet++ |
| 2 | ISIC 2018 | PraNet, ACSNet |
| 3 | ISIC 2018 | HarDNet-MSEG, proposed HC |
| 4 | Kvasir-SEG | proposed HC, HC reference |
| 5 | ISIC 2018 | Polyp-PVT, CaraNet |
| 6 | ISIC 2018 | HSNet, CFANet |
| 7 | ISIC 2018 + CVC-ClinicDB | ResUNet++, proposed HC |
| 8 | Kvasir-SEG | without-HC, shared-kernel ablations |
| 9 | Kvasir-Instrument | UNet, UNet++ |
| 10 | Kvasir-Instrument | PraNet, ACSNet |
| 11 | Kvasir-Instrument | HarDNet-MSEG, proposed HC |
| 12 | CVC-ColonDB + Kvasir-SEG | proposed HC, learnable-h ablation |
| 13 | HyperKvasir segmentation | Polyp-PVT, CaraNet |
| 14 | HyperKvasir segmentation | HSNet, CFANet |
| 15 | HyperKvasir + Kvasir-Instrument | ResUNet++, proposed HC |
| 16 | Kvasir-SEG | kernel-5, identity-projection, no-channel-expansion |

Run exactly one:

```bash
bash run_hc_session_1.sh
# ...
bash run_hc_session_16.sh
```

Disk-safe defaults:

```bash
DELETE_CHECKPOINTS_AFTER_EVAL=1
CLEAN_DOWNLOAD_ARCHIVES=1
CLEAN_RAW_DATA_AFTER_RUN=1
CLEAN_PIP_CACHE=1
CLEAN_OUTPUT_BEFORE_RUN=1
```

For a fast smoke test:

```bash
SEEDS=42 EPOCHS=1 BATCH_SIZE=2 bash run_hc_session_1.sh
```
