# URF-U-Net Patch Notes

## Added

- `src/models/proposal/urf_unet.py`
  - adaptive image-conditioned radial global Fourier bottleneck;
  - uncertainty-routed windowed local Fourier refiner;
  - coarse segmentation head for uncertainty estimation;
  - full-resolution boundary head;
  - alpha warm-up and diagnostic metrics.
- Canonical model: `proposal_urf_unet`.
- Controlled variants:
  - `urf_unet_dynamic_global_only`;
  - `urf_unet_no_dynamic_global`;
  - `urf_unet_no_uncertainty`;
  - `urf_unet_no_boundary_supervision`;
  - `urf_unet_no_coarse_supervision`.
- `configs/urf_ablation/` for comparison against `proposal_fourier_unet`.
- `configs/fair/proposal_urf_unet.yaml`.
- `scripts/run_urf_ablation.py`.
- `scripts/kaggle_urf_ablation_etis_3seeds.sh`.
- `kaggle_urf_ablation_etis_3seeds_cell.txt`.
- `tests/test_urf_unet.py`.
- `URF_UNET_GUIDE.md`.

## Controlled ETIS protocol

- Dataset key: `etis`.
- Seeds: `42,1,2`.
- Models: 7.
- Total runs: 21.
- Epochs: 60.
- Image size: 352.
- Batch size: 6.
- Learning rate: `1e-4`.
- Main loss: Structure loss.
- Full URF-U-Net:
  - coarse auxiliary weight: `0.2`;
  - boundary loss weight: `0.2`.

## Validation performed

- Python compilation completed successfully.
- Bash syntax checks completed successfully.
- 41 targeted Fourier/URF/pipeline tests passed.
- The full 21-run GPU experiment was not executed in this environment.
