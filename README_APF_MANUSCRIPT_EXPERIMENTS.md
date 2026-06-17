# APF-U-Net Manuscript Experiment Guide

This guide updates the manuscript experiments using the cleaned APF-only repository.

The immediate objectives are:

1. Replace every old **HF-U-Net** result with a newly trained **APF-U-Net** result.
2. Add **Attention U-Net** and **CSCA U-Net** to the existing result tables.
3. Use exactly three training seeds: **42, 1, and 2**.
4. Keep the same dataset split for every model and seed.
5. Keep the supported existing datasets:
   - Kvasir-SEG
   - CVC-ClinicDB
   - CVC-ColonDB
6. Remove the unsupported **ISBI-2012-challenge** experiment.
7. Replace ISBI-2012 with the supported **ISIC 2018 Task 1** dataset.
8. Run the complete fair-comparison model set on ISIC 2018.

---

## 1. Repository experiment structure

The repository contains exactly three experiment configuration groups:

```text
configs/
├── official_faithful/
├── fair/
└── ablation/
```

For the manuscript comparison described in this guide, use:

```text
configs/fair/
```

Do not mix results from `configs/fair/` and `configs/official_faithful/` in the same table.

### Fair-comparison methods

The fair experiment contains:

```text
unet
attention_unet
unetpp
resunetpp
pranet
acsnet
hardnet_mseg
polyp_pvt
caranet
cfanet
hsnet
csca_unet
proposal_apf_unet
```

The proposed method is:

```text
proposal_apf_unet
```

The manuscript display name should be:

```text
APF-U-Net
```

---

## 2. Required experiment matrix

### 2.1 Existing manuscript datasets

Run only the three new or replaced methods on the existing supported datasets:

```text
proposal_apf_unet
attention_unet
csca_unet
```

Datasets:

```text
kvasir_seg
cvc_clinicdb
cvc_colondb
```

Seeds:

```text
42, 1, 2
```

Total:

```text
3 models × 3 datasets × 3 seeds = 27 training runs
```

These runs are used to:

- replace the old HF-U-Net row with APF-U-Net;
- add Attention U-Net;
- add CSCA U-Net.

### 2.2 Replacement for unsupported ISBI-2012

Remove the ISBI-2012 section from the manuscript.

Replace it with:

```text
ISIC 2018 Task 1
```

Run all 13 fair-comparison methods on ISIC 2018:

```text
13 models × 3 seeds = 39 training runs
```

### 2.3 Total immediate workload

```text
27 existing-dataset runs + 39 ISIC 2018 runs = 66 training runs
```

---

## 3. Fixed fair-comparison protocol

Use the same protocol for all fair-comparison methods:

```text
Image size:       352 × 352
Epochs:           30
Learning rate:    0.0003
Training seeds:   42, 1, 2
Split seed:       42
Threshold:        0.5
Config directory: configs/fair
```

The split seed and training seed are different concepts:

- **Split seed 42** determines which images belong to train, validation, and test.
- **Training seeds 42, 1, and 2** control initialization, sampling, and other training randomness.

Create each dataset split only once using split seed 42. Do not create a new split for training seeds 1 or 2.

---

# Part A — Prepare the repository

## 4. Option A: Clone from GitHub

Use this when the corrected repository has been pushed to GitHub.

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -rf DT-unet
rm -rf outputs_update_existing
rm -rf outputs_isic2018
rm -rf data

git clone --depth 1 \
  https://github.com/tydeptrai21042004/DT-unet.git \
  DT-unet

cd DT-unet

python -m pip install -q -r requirements.txt

echo "Repository:"
pwd
```

Verify that the checked-out repository is the APF-only version:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

test -f configs/fair/proposal_apf_unet.yaml
test -f configs/fair/attention_unet.yaml
test -f configs/fair/csca_unet.yaml

test ! -e configs/fair/unet_cbam.yaml

echo "APF-only fair configuration verified."
```

---

## 5. Option B: Extract the corrected ZIP on Kaggle

Change `ZIP_PATH` to the real path of the uploaded Kaggle dataset.

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -rf DT-unet
rm -rf extracted_repo
rm -rf outputs_update_existing
rm -rf outputs_isic2018
rm -rf data

ZIP_PATH="/kaggle/input/dt-unet-apf-clean/DT-unet-APF-clean-3-config-pipeline-tested.zip"

if [[ ! -f "$ZIP_PATH" ]]; then
    echo "ERROR: ZIP file not found: $ZIP_PATH"
    exit 1
fi

mkdir -p extracted_repo
unzip -q "$ZIP_PATH" -d extracted_repo

REPO_DIR="$(find extracted_repo -maxdepth 3 -type d \
  -path '*/DT-unet-main' | head -n 1)"

if [[ -z "$REPO_DIR" ]]; then
    echo "ERROR: DT-unet-main directory was not found after extraction."
    find extracted_repo -maxdepth 3 -type d
    exit 1
fi

mv "$REPO_DIR" DT-unet
rm -rf extracted_repo

cd DT-unet

python -m pip install -q -r requirements.txt

echo "Repository:"
pwd
```

---

# Part B — Validate the repository before training

## 6. Run the complete test suite

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python -m pytest -q
```

Do not begin the full experiments if tests fail.

---

## 7. Verify the three required models

This tests model construction, forward propagation, loss calculation, backward propagation, and one optimizer update without requiring a real dataset.

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/smoke_all_models.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --image-size 128 \
  --batch-size 1 \
  --device cpu
```

A GPU smoke test can also be run:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/smoke_all_models.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --image-size 128 \
  --batch-size 1 \
  --device cuda
```

---

## 8. Verify GPU availability

```bash
%%bash
set -euo pipefail

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

# Part C — Dataset preparation

## 9. General dataset-preparation rule

The repository can prepare datasets using one of these methods:

1. automatic registry-configured download;
2. a local ZIP file using `--zip-path`;
3. an extracted dataset directory using `--source-dir`.

Use automatic download first. If the download source is blocked or returns an HTTP error on Kaggle, use the local ZIP or extracted-directory command.

---

## 10. Prepare Kvasir-SEG

### Automatic download

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/prepare_dataset.py \
  --dataset kvasir_seg \
  --data-root data \
  --image-size 352
```

### Local ZIP fallback

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

KVASIR_ZIP="/kaggle/input/kvasir-seg-dataset/kvasir-seg.zip"

python scripts/prepare_dataset.py \
  --dataset kvasir_seg \
  --data-root data \
  --zip-path "$KVASIR_ZIP" \
  --image-size 352
```

### Extracted-directory fallback

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

KVASIR_DIR="/kaggle/input/kvasir-seg-dataset/Kvasir-SEG"

python scripts/prepare_dataset.py \
  --dataset kvasir_seg \
  --data-root data \
  --source-dir "$KVASIR_DIR" \
  --image-size 352
```

---

## 11. Prepare CVC-ClinicDB

### Automatic download

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/prepare_dataset.py \
  --dataset cvc_clinicdb \
  --data-root data \
  --image-size 352
```

The configured source is a public Google Drive training bundle. Google Drive downloads can sometimes fail on Kaggle because of access or quota restrictions.

### Local ZIP fallback

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

CLINICDB_ZIP="/kaggle/input/polyp-datasets/TrainDataset.zip"

python scripts/prepare_dataset.py \
  --dataset cvc_clinicdb \
  --data-root data \
  --zip-path "$CLINICDB_ZIP" \
  --image-size 352
```

### Extracted-directory fallback

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

CLINICDB_DIR="/kaggle/input/polyp-datasets/CVC-ClinicDB"

python scripts/prepare_dataset.py \
  --dataset cvc_clinicdb \
  --data-root data \
  --source-dir "$CLINICDB_DIR" \
  --image-size 352
```

---

## 12. Prepare CVC-ColonDB

### Automatic download

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/prepare_dataset.py \
  --dataset cvc_colondb \
  --data-root data \
  --image-size 352
```

### Local ZIP fallback

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

COLONDB_ZIP="/kaggle/input/polyp-datasets/TestDataset.zip"

python scripts/prepare_dataset.py \
  --dataset cvc_colondb \
  --data-root data \
  --zip-path "$COLONDB_ZIP" \
  --image-size 352
```

### Extracted-directory fallback

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

COLONDB_DIR="/kaggle/input/polyp-datasets/CVC-ColonDB"

python scripts/prepare_dataset.py \
  --dataset cvc_colondb \
  --data-root data \
  --source-dir "$COLONDB_DIR" \
  --image-size 352
```

---

## 13. Prepare ISIC 2018 Task 1

The repository uses the official ISIC Challenge training images and Task 1 ground-truth masks.

### Automatic official download

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/prepare_dataset.py \
  --dataset isic2018 \
  --data-root data \
  --image-size 352
```

### Local extracted-directory fallback

If automatic download is unavailable, upload both official archives to Kaggle, extract them, and pass their common parent directory:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

ISIC_DIR="/kaggle/input/isic-2018-task1"

python scripts/prepare_dataset.py \
  --dataset isic2018 \
  --data-root data \
  --source-dir "$ISIC_DIR" \
  --image-size 352
```

The directory should contain or include:

```text
ISIC2018_Task1-2_Training_Input/
ISIC2018_Task1_Training_GroundTruth/
```

---

# Part D — Create one fixed split per dataset

## 14. Create Kvasir-SEG split

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/make_splits.py \
  --dataset kvasir_seg \
  --data-root data \
  --image-size 352 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

---

## 15. Create CVC-ClinicDB split

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/make_splits.py \
  --dataset cvc_clinicdb \
  --data-root data \
  --image-size 352 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

---

## 16. Create CVC-ColonDB split

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/make_splits.py \
  --dataset cvc_colondb \
  --data-root data \
  --image-size 352 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

---

## 17. Create ISIC 2018 split

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/make_splits.py \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

---

## 18. Verify all split files

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

for DATASET in \
  kvasir_seg \
  cvc_clinicdb \
  cvc_colondb \
  isic2018
do
    echo "============================================================"
    echo "Dataset: $DATASET"

    for SPLIT in train val test; do
        FILE="data/splits/$DATASET/$SPLIT.txt"

        if [[ ! -f "$FILE" ]]; then
            echo "ERROR: missing split file: $FILE"
            exit 1
        fi

        COUNT="$(grep -cve '^[[:space:]]*$' "$FILE")"
        echo "$SPLIT: $COUNT samples"

        if [[ "$COUNT" -le 0 ]]; then
            echo "ERROR: empty split file: $FILE"
            exit 1
        fi
    done
done

echo "All split files are present and non-empty."
```

---

## 19. Check for overlap between splits

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python - <<'PY'
from pathlib import Path

datasets = [
    "kvasir_seg",
    "cvc_clinicdb",
    "cvc_colondb",
    "isic2018",
]

for dataset in datasets:
    root = Path("data/splits") / dataset

    values = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        values[split] = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    assert values["train"].isdisjoint(values["val"]), dataset
    assert values["train"].isdisjoint(values["test"]), dataset
    assert values["val"].isdisjoint(values["test"]), dataset

    total = sum(len(items) for items in values.values())

    print(
        f"{dataset}: "
        f"train={len(values['train'])}, "
        f"val={len(values['val'])}, "
        f"test={len(values['test'])}, "
        f"total={total}"
    )

print("No train/validation/test overlap was found.")
PY
```

---

# Part E — Update the existing three manuscript datasets

## 20. Models to run

Use:

```text
proposal_apf_unet,attention_unet,csca_unet
```

These commands run all three seeds automatically and aggregate mean and standard deviation.

The first seed performs data preparation and split creation through the full pipeline. Later seeds reuse the same processed data and split because `benchmark_multi_seed.py` automatically adds:

```text
--skip-prepare --skip-splits
```

for subsequent seeds.

Because the split files were already created manually, the first seed may recreate the same deterministic split using seed 42. The result is unchanged.

---

## 21. Run Kvasir-SEG

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --dataset kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_update_existing/kvasir_seg
```

### Disk-saving version

Use this only when checkpoints are not needed for later qualitative figures:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --dataset kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_update_existing/kvasir_seg \
  --delete-checkpoints-after-eval
```

---

## 22. Run CVC-ClinicDB

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --dataset cvc_clinicdb \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_update_existing/cvc_clinicdb
```

### Disk-saving version

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --dataset cvc_clinicdb \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_update_existing/cvc_clinicdb \
  --delete-checkpoints-after-eval
```

---

## 23. Run CVC-ColonDB

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --dataset cvc_colondb \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_update_existing/cvc_colondb
```

### Disk-saving version

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --dataset cvc_colondb \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_update_existing/cvc_colondb \
  --delete-checkpoints-after-eval
```

---

## 24. Combined command for all three existing datasets

Use this only when one Kaggle session has enough GPU time.

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

MODELS="proposal_apf_unet,attention_unet,csca_unet"
SEEDS="42,1,2"

for DATASET in kvasir_seg cvc_clinicdb cvc_colondb; do
    echo "============================================================"
    echo "Dataset: $DATASET"
    echo "Models:  $MODELS"
    echo "Seeds:   $SEEDS"
    echo "============================================================"

    python scripts/benchmark_multi_seed.py \
      --models "$MODELS" \
      --config-dir configs/fair \
      --dataset "$DATASET" \
      --data-root data \
      --image-size 352 \
      --epochs 30 \
      --lr 0.0003 \
      --device cuda \
      --num-workers 2 \
      --seeds "$SEEDS" \
      --output-root "outputs_update_existing/$DATASET"
done
```

---

# Part F — Replace ISBI-2012 with ISIC 2018

## 25. Complete ISIC 2018 fair-comparison model list

```text
unet
attention_unet
unetpp
resunetpp
pranet
acsnet
hardnet_mseg
polyp_pvt
caranet
cfanet
hsnet
csca_unet
proposal_apf_unet
```

---

## 26. Run all ISIC methods in one command

Use this only if the session has enough time and disk capacity.

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"

python scripts/benchmark_multi_seed.py \
  --models "$MODELS" \
  --config-dir configs/fair \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_isic2018/fair
```

### Disk-saving form

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"

python scripts/benchmark_multi_seed.py \
  --models "$MODELS" \
  --config-dir configs/fair \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_isic2018/fair \
  --delete-checkpoints-after-eval
```

---

# Part G — Recommended Kaggle session division

Running 66 training jobs in one notebook session is not recommended.

## 27. Session 1 — Kvasir-SEG update

```text
Models: proposal_apf_unet, attention_unet, csca_unet
Seeds: 42, 1, 2
Total: 9 runs
```

Use the command in Section 21.

## 28. Session 2 — CVC-ClinicDB update

```text
Models: proposal_apf_unet, attention_unet, csca_unet
Seeds: 42, 1, 2
Total: 9 runs
```

Use the command in Section 22.

## 29. Session 3 — CVC-ColonDB update

```text
Models: proposal_apf_unet, attention_unet, csca_unet
Seeds: 42, 1, 2
Total: 9 runs
```

Use the command in Section 23.

## 30. Session 4 — ISIC group A

```text
unet
attention_unet
unetpp
resunetpp
```

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models unet,attention_unet,unetpp,resunetpp \
  --config-dir configs/fair \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_isic2018/group_a
```

## 31. Session 5 — ISIC group B

```text
pranet
acsnet
hardnet_mseg
```

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models pranet,acsnet,hardnet_mseg \
  --config-dir configs/fair \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_isic2018/group_b
```

## 32. Session 6 — ISIC group C

```text
polyp_pvt
caranet
cfanet
```

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models polyp_pvt,caranet,cfanet \
  --config-dir configs/fair \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_isic2018/group_c
```

## 33. Session 7 — ISIC group D

```text
hsnet
csca_unet
proposal_apf_unet
```

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models hsnet,csca_unet,proposal_apf_unet \
  --config-dir configs/fair \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_isic2018/group_d
```

---

# Part H — Merge split ISIC groups

Each group produces its own aggregated CSV. To create one combined CSV:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python - <<'PY'
from pathlib import Path
import pandas as pd

roots = [
    Path("outputs_isic2018/group_a"),
    Path("outputs_isic2018/group_b"),
    Path("outputs_isic2018/group_c"),
    Path("outputs_isic2018/group_d"),
]

frames = []

for root in roots:
    candidates = list(root.rglob("multi_seed_summary.csv"))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one multi_seed_summary.csv under {root}, "
            f"found {len(candidates)}: {candidates}"
        )

    path = candidates[0]
    print("Reading:", path)
    frames.append(pd.read_csv(path))

combined = pd.concat(frames, ignore_index=True)

if "model" not in combined.columns:
    raise RuntimeError(f"Missing model column. Columns: {combined.columns.tolist()}")

duplicates = combined[combined.duplicated(subset=["model"], keep=False)]

if not duplicates.empty:
    raise RuntimeError(
        "Duplicate models found:\n"
        + duplicates[["model"]].to_string(index=False)
    )

expected = {
    "unet",
    "attention_unet",
    "unetpp",
    "resunetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "csca_unet",
    "proposal_apf_unet",
}

found = set(combined["model"].astype(str))

missing = sorted(expected - found)
unexpected = sorted(found - expected)

if missing:
    raise RuntimeError(f"Missing models: {missing}")

if unexpected:
    raise RuntimeError(f"Unexpected models: {unexpected}")

order = [
    "unet",
    "attention_unet",
    "unetpp",
    "resunetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "csca_unet",
    "proposal_apf_unet",
]

combined["model"] = pd.Categorical(
    combined["model"],
    categories=order,
    ordered=True,
)

combined = combined.sort_values("model").reset_index(drop=True)

output_dir = Path("outputs_isic2018/combined")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "multi_seed_summary.csv"
combined.to_csv(output_path, index=False)

print("Saved:", output_path)
print(combined.to_string(index=False))
PY
```

---

# Part I — Validate experiment completeness

## 34. Validate the three existing datasets

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python - <<'PY'
from pathlib import Path
import pandas as pd

expected_models = {
    "proposal_apf_unet",
    "attention_unet",
    "csca_unet",
}

datasets = [
    "kvasir_seg",
    "cvc_clinicdb",
    "cvc_colondb",
]

for dataset in datasets:
    root = Path("outputs_update_existing") / dataset
    candidates = list(root.rglob("multi_seed_summary.csv"))

    if len(candidates) != 1:
        raise RuntimeError(
            f"{dataset}: expected one summary CSV, found {len(candidates)}: "
            f"{candidates}"
        )

    path = candidates[0]
    df = pd.read_csv(path)

    if "model" not in df.columns:
        raise RuntimeError(
            f"{dataset}: missing model column in {path}. "
            f"Columns: {df.columns.tolist()}"
        )

    found_models = set(df["model"].astype(str))

    missing = expected_models - found_models
    unexpected = found_models - expected_models

    if missing:
        raise RuntimeError(f"{dataset}: missing models: {sorted(missing)}")

    if unexpected:
        raise RuntimeError(
            f"{dataset}: unexpected models: {sorted(unexpected)}"
        )

    if "num_seeds" in df.columns:
        bad = df[df["num_seeds"].astype(int) != 3]
        if not bad.empty:
            raise RuntimeError(
                f"{dataset}: some rows do not contain three seeds:\n{bad}"
            )

    print(f"{dataset}: PASS")
    print(df.to_string(index=False))
    print()

print("All existing-dataset summaries are complete.")
PY
```

---

## 35. Validate ISIC combined results

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("outputs_isic2018/combined/multi_seed_summary.csv")

if not path.exists():
    candidates = list(Path("outputs_isic2018").rglob("multi_seed_summary.csv"))
    raise FileNotFoundError(
        f"Combined ISIC summary not found at {path}. "
        f"Available summaries: {candidates}"
    )

df = pd.read_csv(path)

expected = {
    "unet",
    "attention_unet",
    "unetpp",
    "resunetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "csca_unet",
    "proposal_apf_unet",
}

found = set(df["model"].astype(str))

assert found == expected, {
    "missing": sorted(expected - found),
    "unexpected": sorted(found - expected),
}

if "num_seeds" in df.columns:
    assert (df["num_seeds"].astype(int) == 3).all()

print("ISIC 2018 result matrix is complete.")
print(df.to_string(index=False))
PY
```

---

## 36. Validate raw seed directories

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python - <<'PY'
from pathlib import Path

roots = [
    Path("outputs_update_existing/kvasir_seg"),
    Path("outputs_update_existing/cvc_clinicdb"),
    Path("outputs_update_existing/cvc_colondb"),
    Path("outputs_isic2018/group_a"),
    Path("outputs_isic2018/group_b"),
    Path("outputs_isic2018/group_c"),
    Path("outputs_isic2018/group_d"),
]

expected_seed_dirs = {"seed_42", "seed_1", "seed_2"}

for root in roots:
    if not root.exists():
        print("SKIP missing root:", root)
        continue

    found = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("seed_")
    }

    missing = expected_seed_dirs - found

    if missing:
        raise RuntimeError(f"{root}: missing seed directories: {sorted(missing)}")

    print(f"{root}: seeds verified")

print("All available experiment roots contain seeds 42, 1, and 2.")
PY
```

---

# Part J — Result locations

## 37. Existing dataset results

Search under:

```text
outputs_update_existing/kvasir_seg/
outputs_update_existing/cvc_clinicdb/
outputs_update_existing/cvc_colondb/
```

Each root contains:

```text
seed_42/
seed_1/
seed_2/
```

and an aggregated result file produced by `aggregate_seed_results.py`.

Locate all summaries with:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

find outputs_update_existing \
  -type f \
  \( -name "*.csv" -o -name "*.json" -o -name "*.tex" \) \
  | sort
```

## 38. ISIC results

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

find outputs_isic2018 \
  -type f \
  \( -name "*.csv" -o -name "*.json" -o -name "*.tex" \) \
  | sort
```

The manually merged ISIC summary is:

```text
outputs_isic2018/combined/multi_seed_summary.csv
```

---

# Part K — Save results before ending a Kaggle session

Kaggle working storage is temporary. Create result archives before the session ends.

## 39. Archive existing-dataset results

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f outputs_update_existing.zip

zip -qr \
  outputs_update_existing.zip \
  DT-unet/outputs_update_existing

ls -lh outputs_update_existing.zip
```

## 40. Archive ISIC results

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f outputs_isic2018.zip

zip -qr \
  outputs_isic2018.zip \
  DT-unet/outputs_isic2018

ls -lh outputs_isic2018.zip
```

## 41. Archive split files separately

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f experiment_splits.zip

zip -qr \
  experiment_splits.zip \
  DT-unet/data/splits

ls -lh experiment_splits.zip
```

Keep `experiment_splits.zip` with the manuscript artifacts. The exact split files are required for reproducibility.

---

# Part L — Manuscript updates

## 42. Add the two new baseline rows

Add:

```latex
Attention U-Net~\cite{oktay2018attentionunet}
```

Add CSCA U-Net using the citation key in your bibliography:

```latex
CSCA U-Net~\cite{your_csca_unet_bib_key}
```

Do not leave the placeholder citation in the final manuscript.

---

## 43. Replace the old proposal row

Remove:

```latex
HF-U-Net
```

Add:

```latex
\textbf{APF-U-Net}
```

Use only newly generated APF-U-Net values.

Do not copy or rename the old HF-U-Net values.

---

## 44. Recommended table order

```latex
U-Net~\cite{ronneberger2015unet}
Attention U-Net~\cite{oktay2018attentionunet}
U-Net++~\cite{zhou2018unetpp}
PraNet~\cite{fan2020pranet}
ACSNet~\cite{zhang2020acsnet}
HarDNet-MSEG~\cite{huang2021hardnetmseg}
CFA-Net~\cite{zhou2023cfanet}
Polyp-PVT~\cite{dong2023polyppvt}
CaraNet~\cite{lou2023caranet}
HSNet~\cite{zhang2022hsnet}
CSCA U-Net~\cite{your_csca_unet_bib_key}
ResUNet++~\cite{jha2021resunetpp}
\textbf{APF-U-Net}
```

---

## 45. Dataset-section replacement

Keep:

```latex
\paragraph{Kvasir-SEG.}

\paragraph{CVC-ClinicDB.}

\paragraph{CVC-ColonDB.}
```

Remove:

```latex
\paragraph{ISBI-2012-challenge.}
```

Add:

```latex
\paragraph{ISIC 2018 Task 1.}
```

---

## 46. Proposal terminology replacement

Replace old proposal terminology:

| Remove | Use |
|---|---|
| HF-U-Net | APF-U-Net |
| Hartley--Fourier bottleneck | amplitude--phase Fourier bottleneck |
| Hartley--Fourier module | APF module |
| HF bottleneck | APF bottleneck |

Do not retain any numerical conclusion from the old HF-U-Net experiment until the new APF results have been calculated.

---

# Part M — Optional qualitative outputs

To save predicted masks and visualizations, add:

```text
--save-predictions
--save-visualizations
```

Example:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/DT-unet

python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet \
  --config-dir configs/fair \
  --dataset kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_apf_visuals/kvasir_seg \
  --save-predictions \
  --save-visualizations
```

Do not combine `--delete-checkpoints-after-eval` with a workflow that needs checkpoints later.

---

# Part N — Troubleshooting

## 47. CUDA out-of-memory

Reduce the batch size:

```text
--batch-size 2
```

or:

```text
--batch-size 1
```

Example:

```bash
python scripts/benchmark_multi_seed.py \
  --models csca_unet \
  --config-dir configs/fair \
  --dataset isic2018 \
  --data-root data \
  --image-size 352 \
  --batch-size 1 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_isic2018/csca_only
```

Use the same effective batch policy for all compared methods when possible. If a model requires a smaller physical batch size, document the difference.

---

## 48. DataLoader worker failure

Set:

```text
--num-workers 0
```

Example:

```bash
python scripts/benchmark_multi_seed.py \
  --models proposal_apf_unet \
  --config-dir configs/fair \
  --dataset kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 0 \
  --seeds 42,1,2 \
  --output-root outputs_update_existing/kvasir_seg
```

---

## 49. TLS certificate failure

For registry downloads only, try:

```text
--allow-insecure-download
```

Example:

```bash
python scripts/prepare_dataset.py \
  --dataset kvasir_seg \
  --data-root data \
  --image-size 352 \
  --allow-insecure-download
```

Use this only when necessary. A local ZIP or extracted source directory is preferable.

---

## 50. Google Drive download failure

For CVC-ClinicDB or CVC-ColonDB, use:

```text
--zip-path
```

or:

```text
--source-dir
```

instead of repeated automatic-download attempts.

---

## 51. Resume after a failed session

The multi-seed wrapper does not provide model-level resume logic. Restart only the missing model or group in a new output directory.

Example:

```bash
python scripts/benchmark_multi_seed.py \
  --models csca_unet \
  --config-dir configs/fair \
  --dataset cvc_colondb \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_recovery/cvc_colondb_csca
```

After completion, merge its aggregated row with the other result CSV using pandas.

---

# Part O — Final checklist

Before updating the manuscript, verify:

```text
[ ] Repository tests pass.
[ ] APF-U-Net smoke test passes.
[ ] Attention U-Net smoke test passes.
[ ] CSCA U-Net smoke test passes.
[ ] Kvasir-SEG split is fixed with seed 42.
[ ] CVC-ClinicDB split is fixed with seed 42.
[ ] CVC-ColonDB split is fixed with seed 42.
[ ] ISIC 2018 split is fixed with seed 42.
[ ] Every reported method uses training seeds 42, 1, and 2.
[ ] APF-U-Net replaces HF-U-Net using new values.
[ ] Attention U-Net is added.
[ ] CSCA U-Net is added.
[ ] ISBI-2012 is removed.
[ ] ISIC 2018 is added.
[ ] Fair and official-faithful results are not mixed.
[ ] Result CSV files are archived.
[ ] Exact split files are archived.
[ ] Table rankings are recalculated from the new values.
[ ] Old Hartley–Fourier claims are removed.
```

---

# Final recommended execution order

1. Prepare or clone the repository.
2. Install dependencies.
3. Run all tests.
4. Run model smoke tests.
5. Prepare Kvasir-SEG.
6. Prepare CVC-ClinicDB.
7. Prepare CVC-ColonDB.
8. Prepare ISIC 2018.
9. Create all splits once with seed 42.
10. Verify split counts and no overlap.
11. Run Kvasir-SEG update for APF-U-Net, Attention U-Net, and CSCA U-Net.
12. Run CVC-ClinicDB update for the same three methods.
13. Run CVC-ColonDB update for the same three methods.
14. Run ISIC group A.
15. Run ISIC group B.
16. Run ISIC group C.
17. Run ISIC group D.
18. Merge ISIC group summaries.
19. Validate all methods and all three seeds.
20. Archive results and split files.
21. Replace the manuscript rows and dataset section.
22. Recalculate rankings and rewrite the result discussion.
