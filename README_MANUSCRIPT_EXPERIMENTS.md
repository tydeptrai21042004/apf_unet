# Manuscript Experiment Reproduction Guide

This guide contains the commands required to generate the experimental results needed to update the manuscript.

## Final experimental plan

- **Existing datasets:** rerun only `proposal_hc_unet_no_gate`.
- **New datasets:** run all baselines plus `proposal_hc_unet_no_gate`.
- **Ablation study:** rerun all variants using the no-gate proposal as the reference.
- **Seeds:** `42,1,2`.
- **Epochs:** `30`.
- **Checkpoint cleanup:** performed after every experiment group.
- **Result packaging:** no ZIP packaging is required.

> Run the environment setup once at the beginning of a fresh Kaggle session. The remaining experiment blocks can be executed separately.

---

## 1. Clone the repository and prepare the environment

Run this once at the beginning of a fresh Kaggle session.

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -rf DT-unet
rm -rf outputs_hc_session_*
rm -rf data
rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface
rm -rf /root/.cache/pip /root/.cache/torch /root/.cache/huggingface

git clone --depth 1 \
  https://github.com/tydeptrai21042004/DT-unet \
  DT-unet

cd DT-unet

python -m pip uninstall -y torch torchvision torchaudio || true

python -m pip install \
  --no-cache-dir \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

grep -vE \
  '^[[:space:]]*(torch|torchvision|torchaudio)([<>=!~ ].*)?$' \
  requirements.txt \
  > /tmp/requirements_no_torch.txt

python -m pip install \
  --no-cache-dir \
  -r /tmp/requirements_no_torch.txt \
  pytest

rm -f /tmp/requirements_no_torch.txt
rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface
```

### Validate Tesla P100 compatibility

```bash
%%bash
cd /kaggle/working/DT-unet

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Compiled architectures:", torch.cuda.get_arch_list())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

if "sm_60" not in torch.cuda.get_arch_list():
    raise SystemExit("This PyTorch build does not support Tesla P100 sm_60")

x = torch.ones(8, device="cuda")
y = x * 2
torch.cuda.synchronize()

print("CUDA test passed:", y.tolist())
PY
```

---

## 2. Existing datasets: proposal no-gate only

These runs replace only the proposed-method row in the existing manuscript tables. Existing baseline models are not rerun.

### 2.1 Kvasir-SEG

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "proposal_hc_unet_no_gate" \
  --dataset kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/existing/kvasir_seg \
  --allow-insecure-download
```

### 2.2 CVC-ClinicDB

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "proposal_hc_unet_no_gate" \
  --dataset cvc_clinicdb \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/existing/cvc_clinicdb \
  --allow-insecure-download
```

### 2.3 CVC-ColonDB

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "proposal_hc_unet_no_gate" \
  --dataset cvc_colondb \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/existing/cvc_colondb \
  --allow-insecure-download
```

### Clean checkpoints after existing-dataset runs

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

find outputs_manuscript/existing \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete

rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface
df -h /kaggle/working
```

---

## 3. ISIC 2018 full experiment

Run the models in three groups to reduce peak disk usage.

### 3.1 ISIC group 1

Models: U-Net, U-Net++, PraNet, ACSNet.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "unet,unetpp,pranet,acsnet" \
  --dataset isic2018 \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/isic2018/group_1 \
  --allow-insecure-download

find outputs_manuscript/new/isic2018/group_1 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 3.2 ISIC group 2

Models: HarDNet-MSEG, Polyp-PVT, CaraNet, HSNet.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "hardnet_mseg,polyp_pvt,caranet,hsnet" \
  --dataset isic2018 \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/isic2018/group_2 \
  --allow-insecure-download

find outputs_manuscript/new/isic2018/group_2 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 3.3 ISIC group 3

Models: CFA-Net, ResUNet++, proposed HC U-Net without gate.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "cfanet,resunetpp,proposal_hc_unet_no_gate" \
  --dataset isic2018 \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/isic2018/group_3 \
  --allow-insecure-download

find outputs_manuscript/new/isic2018/group_3 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

---

## 4. Kvasir-Instrument full experiment

### 4.1 Instrument group 1

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "unet,unetpp,pranet,acsnet" \
  --dataset kvasir_instrument \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/kvasir_instrument/group_1 \
  --allow-insecure-download

find outputs_manuscript/new/kvasir_instrument/group_1 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 4.2 Instrument group 2

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "hardnet_mseg,polyp_pvt,caranet,hsnet" \
  --dataset kvasir_instrument \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/kvasir_instrument/group_2 \
  --allow-insecure-download

find outputs_manuscript/new/kvasir_instrument/group_2 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 4.3 Instrument group 3

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "cfanet,resunetpp,proposal_hc_unet_no_gate" \
  --dataset kvasir_instrument \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/kvasir_instrument/group_3 \
  --allow-insecure-download

find outputs_manuscript/new/kvasir_instrument/group_3 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

---

## 5. HyperKvasir segmentation full experiment

### 5.1 HyperKvasir group 1

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "unet,unetpp,pranet,acsnet" \
  --dataset hyper_kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/hyper_kvasir_seg/group_1 \
  --allow-insecure-download

find outputs_manuscript/new/hyper_kvasir_seg/group_1 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 5.2 HyperKvasir group 2

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "hardnet_mseg,polyp_pvt,caranet,hsnet" \
  --dataset hyper_kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/hyper_kvasir_seg/group_2 \
  --allow-insecure-download

find outputs_manuscript/new/hyper_kvasir_seg/group_2 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 5.3 HyperKvasir group 3

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "cfanet,resunetpp,proposal_hc_unet_no_gate" \
  --dataset hyper_kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/new/hyper_kvasir_seg/group_3 \
  --allow-insecure-download

find outputs_manuscript/new/hyper_kvasir_seg/group_3 \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

---

## 6. Rerun the ablation study

Use Kvasir-SEG for all ablation experiments.

### Confirm registered model names

Run this before the ablation groups:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python - <<'PY'
from src.models.registry import list_models

for name in sorted(list_models()):
    print(name)
PY
```

Do not run a model name that is absent from the registry.

### 6.1 Core proposal comparison

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "proposal_hc_unet_no_gate,proposal_hc_unet_with_gate,hc_without_hc_branch" \
  --dataset kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/ablation/core \
  --allow-insecure-download

find outputs_manuscript/ablation/core \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 6.2 Projection and channel structure

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "hc_shared_kernel,hc_identity_projection,hc_no_channel_expansion" \
  --dataset kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/ablation/channel_structure \
  --allow-insecure-download

find outputs_manuscript/ablation/channel_structure \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 6.3 Kernel and \(h\)

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "hc_kernel3,hc_kernel5,hc_fixed_h,hc_learnable_h" \
  --dataset kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/ablation/kernel_h \
  --allow-insecure-download

find outputs_manuscript/ablation/kernel_h \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

### 6.4 Transform and residual components

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models "hc_fourier_only,hc_hartley_only,hc_without_residual,hc_fixed_mixer" \
  --dataset kvasir_seg \
  --config-dir configs/paper_fair \
  --data-root data \
  --image-size 352 \
  --seeds "42,1,2" \
  --device cuda \
  --batch-size 6 \
  --epochs 30 \
  --num-workers 2 \
  --output-root outputs_manuscript/ablation/operator_components \
  --allow-insecure-download

find outputs_manuscript/ablation/operator_components \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete
```

---

## 7. Collect final CSV, JSON, and LaTeX files

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

rm -rf manuscript_results
mkdir -p manuscript_results

find outputs_manuscript \
  -type f \( \
    -name "*.csv" -o \
    -name "*.json" -o \
    -name "*.tex" \
  \) \
  -exec cp --parents {} manuscript_results/ \;

echo "Collected files:"
find manuscript_results -type f | sort
```

### Confirm that checkpoints were removed

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

remaining="$(
  find outputs_manuscript \
    -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
    | wc -l
)"

echo "Remaining checkpoint files: $remaining"

du -sh outputs_manuscript manuscript_results data 2>/dev/null || true
df -h /kaggle/working
```

---

## 8. Generate a combined result table

First inspect the repository for aggregation and table-generation scripts:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

find scripts -maxdepth 1 -type f | sort | grep -Ei \
  'aggregate|summary|table|latex|report|collect' || true
```

If `scripts/aggregate_results.py` exists:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/aggregate_results.py \
  --input-root outputs_manuscript \
  --output-csv manuscript_results/all_results.csv
```

If `scripts/export_latex_tables.py` exists:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

python scripts/export_latex_tables.py \
  --input-csv manuscript_results/all_results.csv \
  --output-dir manuscript_results/latex
```

Locate automatically generated summary files:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/DT-unet

find outputs_manuscript \
  -type f \
  \( -iname "*summary*.csv" -o \
     -iname "*aggregate*.csv" -o \
     -iname "*.tex" \) \
  | sort
```

---

## 9. Manuscript sections to update

### Existing experiment tables

Replace only the proposed HF-U-Net row for:

- Kvasir-SEG
- CVC-ClinicDB
- CVC-ColonDB

Keep the previous baseline values only when all of the following remain unchanged:

- Dataset splits
- Preprocessing
- Image resolution
- Augmentation
- Training epochs
- Loss function
- Optimizer
- Learning-rate schedule
- Batch size
- Seeds
- Evaluation threshold
- Metric implementation
- Baseline implementation
- Pretrained-weight policy

### New experiment tables

Add complete comparison tables for:

- ISIC 2018
- Kvasir-Instrument
- HyperKvasir-SEG

Each table should include:

- U-Net
- U-Net++
- PraNet
- ACSNet
- HarDNet-MSEG
- Polyp-PVT
- CaraNet
- HSNet
- CFA-Net
- ResUNet++
- Proposed HC U-Net without gate

### Ablation table

Use `proposal_hc_unet_no_gate` as the reference row.

Report the mean and standard deviation across seeds:

```text
42, 1, 2
```

### Architecture and method description

Remove or revise statements that say the final proposed model relies on a regulating gate.

The architecture description, equations, figures, algorithm, ablation discussion, abstract, and conclusion must match the final no-gate implementation.

---

## Recommended execution order

1. Environment setup and P100 validation
2. Existing proposal-only reruns
3. ISIC 2018 groups
4. Kvasir-Instrument groups
5. HyperKvasir-SEG groups
6. Ablation groups
7. Result collection
8. Result aggregation
9. Manuscript table and text updates

## Disk-safety notes

After every experiment group:

```bash
find <OUTPUT_DIRECTORY> \
  -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
  -delete

rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface
df -h /kaggle/working
```

Do not delete CSV, JSON, LaTeX, training-history, or aggregated-metric files.
