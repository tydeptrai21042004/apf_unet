# APF-U-Net Kaggle Session Guide

Repository:

```text
https://github.com/tydeptrai21042004/apf_unet
```

This guide runs only the experiments still needed for the manuscript:

- **Session 1:** Kvasir-SEG — APF-U-Net, Attention U-Net, CSCA U-Net
- **Session 2:** CVC-ClinicDB and CVC-ColonDB — the same three missing methods
- **Session 3:** CVC-300 — all 13 fair-comparison methods
- **Session 4:** ETIS — all 13 fair-comparison methods
- **Session 5:** Kvasir-Instrument — all 13 fair-comparison methods
- **Session 6 or more:** HyperKvasir segmented subset — all 13 fair-comparison methods
- **Final short session:** ISBI 2012 — six APF ablation variants only

All experiments use training seeds:

```text
42, 1, 2
```

The commands do not patch or modify source files.

---

# Important Kaggle notes

1. Enable a GPU accelerator before running.
2. The commands use one CUDA process. Kaggle T4 ×2 does not automatically make one process use both GPUs.
3. Every fresh Kaggle session starts with the common setup cells below.
4. Download the result ZIP before ending each session.
5. Existing baseline results for Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, and ISBI 2012 are not rerun.

---

# Common Cell 1 — Clone and install

Run this at the beginning of every fresh Kaggle session.

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -rf apf_unet

git clone --depth 1 \
  https://github.com/tydeptrai21042004/apf_unet.git \
  apf_unet

cd apf_unet

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

echo "============================================================"
echo "Repository prepared"
echo "============================================================"
pwd
git rev-parse HEAD
```

---

# Common Cell 2 — Verify repository and GPU

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

required_files=(
  "scripts/benchmark_multi_seed.py"
  "scripts/benchmark_all.py"
  "scripts/aggregate_seed_results.py"
  "scripts/prepare_dataset.py"
  "scripts/make_splits.py"

  "configs/fair/unet.yaml"
  "configs/fair/attention_unet.yaml"
  "configs/fair/unetpp.yaml"
  "configs/fair/resunetpp.yaml"
  "configs/fair/pranet.yaml"
  "configs/fair/acsnet.yaml"
  "configs/fair/hardnet_mseg.yaml"
  "configs/fair/polyp_pvt.yaml"
  "configs/fair/caranet.yaml"
  "configs/fair/cfanet.yaml"
  "configs/fair/hsnet.yaml"
  "configs/fair/csca_unet.yaml"
  "configs/fair/proposal_apf_unet.yaml"

  "configs/ablation/unet.yaml"
  "configs/ablation/fourier_unet_plain.yaml"
  "configs/ablation/apf_amplitude_only.yaml"
  "configs/ablation/apf_phase_only.yaml"
  "configs/ablation/proposal_apf_unet_at_encoder1.yaml"
  "configs/ablation/proposal_apf_unet.yaml"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: missing required file: $file"
        exit 1
    fi
done

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable.")

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
PY

echo "Repository and GPU verification passed."
```

---

# Optional Common Cell 3 — Run tests

Run this before the first experiment session or after the repository changes.

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python -m pytest -q
```

---

# Optional Common Cell 4 — Smoke-test required methods

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/smoke_all_models.py \
  --models proposal_apf_unet,attention_unet,csca_unet \
  --config-dir configs/fair \
  --image-size 128 \
  --batch-size 1 \
  --device cuda
```

---

# Session 1 — Kvasir-SEG

## Objective

Run only:

```text
proposal_apf_unet
attention_unet
csca_unet
```

for seeds:

```text
42, 1, 2
```

## Session 1 training cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

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
  --output-root outputs_session_1/kvasir_seg \
  --delete-checkpoints-after-eval
```

The first seed prepares the dataset and creates the split. Seeds 1 and 2 reuse the same prepared data and split.

## Session 1 validation cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

SUMMARY="$(find outputs_session_1/kvasir_seg \
  -type f \
  -name 'multi_seed_summary.csv' \
  | head -n 1)"

if [[ -z "$SUMMARY" ]]; then
    echo "ERROR: summary CSV not found."
    exit 1
fi

for MODEL in proposal_apf_unet attention_unet csca_unet; do
    if ! grep -q "$MODEL" "$SUMMARY"; then
        echo "ERROR: missing model $MODEL"
        exit 1
    fi
done

echo "Session 1 validation passed."
echo "Summary: $SUMMARY"
cat "$SUMMARY"
```

## Session 1 archive cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f session_1_kvasir_seg_results.zip

zip -qr \
  session_1_kvasir_seg_results.zip \
  apf_unet/outputs_session_1 \
  apf_unet/data/splits/kvasir_seg

ls -lh session_1_kvasir_seg_results.zip
```

Download:

```text
/kaggle/working/session_1_kvasir_seg_results.zip
```

---

# Session 2 — CVC-ClinicDB and CVC-ColonDB

## Objective

Run only:

```text
proposal_apf_unet
attention_unet
csca_unet
```

on both datasets with seeds `42,1,2`.

## Session 2 training cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

MODELS="proposal_apf_unet,attention_unet,csca_unet"

for DATASET in cvc_clinicdb cvc_colondb; do
    echo "============================================================"
    echo "Running $DATASET"
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
      --seeds 42,1,2 \
      --output-root "outputs_session_2/$DATASET" \
      --delete-checkpoints-after-eval
done
```

## Session 2 validation cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

for DATASET in cvc_clinicdb cvc_colondb; do
    SUMMARY="$(find "outputs_session_2/$DATASET" \
      -type f \
      -name 'multi_seed_summary.csv' \
      | head -n 1)"

    if [[ -z "$SUMMARY" ]]; then
        echo "ERROR: summary CSV not found for $DATASET"
        exit 1
    fi

    for MODEL in proposal_apf_unet attention_unet csca_unet; do
        if ! grep -q "$MODEL" "$SUMMARY"; then
            echo "ERROR: missing $MODEL for $DATASET"
            exit 1
        fi
    done

    echo "$DATASET: PASS"
    cat "$SUMMARY"
done
```

## Session 2 archive cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f session_2_clinicdb_colondb_results.zip

zip -qr \
  session_2_clinicdb_colondb_results.zip \
  apf_unet/outputs_session_2 \
  apf_unet/data/splits/cvc_clinicdb \
  apf_unet/data/splits/cvc_colondb

ls -lh session_2_clinicdb_colondb_results.zip
```

---

# Complete fair-comparison model list

Sessions 3–6 use all 13 methods:

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

The shell variable is:

```bash
FAIR_MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"
```

---

# Session 3 — CVC-300

## Session 3 training cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

FAIR_MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"

python scripts/benchmark_multi_seed.py \
  --models "$FAIR_MODELS" \
  --config-dir configs/fair \
  --dataset cvc_300 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_session_3/cvc_300 \
  --delete-checkpoints-after-eval
```

## Session 3 validation cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

SUMMARY="$(find outputs_session_3/cvc_300 \
  -type f \
  -name 'multi_seed_summary.csv' \
  | head -n 1)"

if [[ -z "$SUMMARY" ]]; then
    echo "ERROR: CVC-300 summary not found."
    exit 1
fi

EXPECTED_MODELS=(
  unet attention_unet unetpp resunetpp pranet acsnet
  hardnet_mseg polyp_pvt caranet cfanet hsnet csca_unet
  proposal_apf_unet
)

for MODEL in "${EXPECTED_MODELS[@]}"; do
    if ! grep -q "$MODEL" "$SUMMARY"; then
        echo "ERROR: missing model $MODEL"
        exit 1
    fi
done

echo "Session 3 validation passed."
cat "$SUMMARY"
```

## Session 3 archive cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f session_3_cvc300_results.zip

zip -qr \
  session_3_cvc300_results.zip \
  apf_unet/outputs_session_3 \
  apf_unet/data/splits/cvc_300

ls -lh session_3_cvc300_results.zip
```

---

# Session 4 — ETIS-LaribPolypDB

## Session 4 training cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

FAIR_MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"

python scripts/benchmark_multi_seed.py \
  --models "$FAIR_MODELS" \
  --config-dir configs/fair \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_session_4/etis \
  --delete-checkpoints-after-eval
```

## Session 4 validation cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

SUMMARY="$(find outputs_session_4/etis \
  -type f \
  -name 'multi_seed_summary.csv' \
  | head -n 1)"

if [[ -z "$SUMMARY" ]]; then
    echo "ERROR: ETIS summary not found."
    exit 1
fi

EXPECTED_MODELS=(
  unet attention_unet unetpp resunetpp pranet acsnet
  hardnet_mseg polyp_pvt caranet cfanet hsnet csca_unet
  proposal_apf_unet
)

for MODEL in "${EXPECTED_MODELS[@]}"; do
    if ! grep -q "$MODEL" "$SUMMARY"; then
        echo "ERROR: missing model $MODEL"
        exit 1
    fi
done

echo "Session 4 validation passed."
cat "$SUMMARY"
```

## Session 4 archive cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f session_4_etis_results.zip

zip -qr \
  session_4_etis_results.zip \
  apf_unet/outputs_session_4 \
  apf_unet/data/splits/etis

ls -lh session_4_etis_results.zip
```

---

# Session 5 — Kvasir-Instrument

## Session 5 training cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

FAIR_MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"

python scripts/benchmark_multi_seed.py \
  --models "$FAIR_MODELS" \
  --config-dir configs/fair \
  --dataset kvasir_instrument \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_session_5/kvasir_instrument \
  --delete-checkpoints-after-eval
```

## Session 5 validation cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

SUMMARY="$(find outputs_session_5/kvasir_instrument \
  -type f \
  -name 'multi_seed_summary.csv' \
  | head -n 1)"

if [[ -z "$SUMMARY" ]]; then
    echo "ERROR: Kvasir-Instrument summary not found."
    exit 1
fi

EXPECTED_MODELS=(
  unet attention_unet unetpp resunetpp pranet acsnet
  hardnet_mseg polyp_pvt caranet cfanet hsnet csca_unet
  proposal_apf_unet
)

for MODEL in "${EXPECTED_MODELS[@]}"; do
    if ! grep -q "$MODEL" "$SUMMARY"; then
        echo "ERROR: missing model $MODEL"
        exit 1
    fi
done

echo "Session 5 validation passed."
cat "$SUMMARY"
```

## Session 5 archive cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f session_5_kvasir_instrument_results.zip

zip -qr \
  session_5_kvasir_instrument_results.zip \
  apf_unet/outputs_session_5 \
  apf_unet/data/splits/kvasir_instrument

ls -lh session_5_kvasir_instrument_results.zip
```

---

# Session 6 or more — HyperKvasir segmented subset

HyperKvasir may require more than one Kaggle session. The simplest full command is below.

## Session 6 training cell

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

FAIR_MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"

python scripts/benchmark_multi_seed.py \
  --models "$FAIR_MODELS" \
  --config-dir configs/fair \
  --dataset hyper_kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_session_6/hyper_kvasir_seg \
  --delete-checkpoints-after-eval
```

## Recommended split across multiple Kaggle sessions

### HyperKvasir group A

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/benchmark_multi_seed.py \
  --models unet,attention_unet,unetpp,resunetpp \
  --config-dir configs/fair \
  --dataset hyper_kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_hyper/group_a \
  --delete-checkpoints-after-eval
```

### HyperKvasir group B

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/benchmark_multi_seed.py \
  --models pranet,acsnet,hardnet_mseg \
  --config-dir configs/fair \
  --dataset hyper_kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_hyper/group_b \
  --delete-checkpoints-after-eval
```

### HyperKvasir group C

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/benchmark_multi_seed.py \
  --models polyp_pvt,caranet,cfanet \
  --config-dir configs/fair \
  --dataset hyper_kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_hyper/group_c \
  --delete-checkpoints-after-eval
```

### HyperKvasir group D

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/benchmark_multi_seed.py \
  --models hsnet,csca_unet,proposal_apf_unet \
  --config-dir configs/fair \
  --dataset hyper_kvasir_seg \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_hyper/group_d \
  --delete-checkpoints-after-eval
```

Archive each group before ending its session:

```bash
%%bash
set -euo pipefail

cd /kaggle/working

GROUP_NAME="group_a"

rm -f "hyper_kvasir_${GROUP_NAME}.zip"

zip -qr \
  "hyper_kvasir_${GROUP_NAME}.zip" \
  "apf_unet/outputs_hyper/${GROUP_NAME}" \
  apf_unet/data/splits/hyper_kvasir_seg

ls -lh "hyper_kvasir_${GROUP_NAME}.zip"
```

Change `GROUP_NAME` to `group_b`, `group_c`, or `group_d` as needed.

---

# Final short session — ISBI 2012 ablation

## Ablation variants

```text
unet
fourier_unet_plain
apf_amplitude_only
apf_phase_only
proposal_apf_unet_at_encoder1
proposal_apf_unet
```

ISBI 2012 is used only for ablation and not for the main fair-comparison table.

## ISBI Cell 1 — Prepare dataset

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/prepare_dataset.py \
  --dataset isbi2012 \
  --data-root data \
  --image-size 352
```

## ISBI Cell 2 — Create fixed contiguous split

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/make_splits.py \
  --dataset isbi2012 \
  --data-root data \
  --image-size 352 \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --strategy contiguous \
  --seed 42

for SPLIT in train val test; do
    FILE="data/splits/isbi2012/${SPLIT}.txt"

    if [[ ! -f "$FILE" ]]; then
        echo "ERROR: missing $FILE"
        exit 1
    fi

    echo "$SPLIT: $(grep -cve '^[[:space:]]*$' "$FILE")"
done
```

Expected:

```text
train: 18
val: 6
test: 6
```

## ISBI Cell 3 — Run six variants for three seeds

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

ABLATION_MODELS="unet,fourier_unet_plain,apf_amplitude_only,apf_phase_only,proposal_apf_unet_at_encoder1,proposal_apf_unet"

for SEED in 42 1 2; do
    echo "============================================================"
    echo "ISBI 2012 ablation — seed $SEED"
    echo "============================================================"

    python scripts/benchmark_all.py \
      --models "$ABLATION_MODELS" \
      --config-dir configs/ablation \
      --dataset isbi2012 \
      --data-root data \
      --image-size 352 \
      --epochs 30 \
      --lr 0.0003 \
      --device cuda \
      --num-workers 2 \
      --seed "$SEED" \
      --output-root "outputs_ablation/isbi2012/seed_${SEED}" \
      --skip-prepare \
      --skip-splits

    find "outputs_ablation/isbi2012/seed_${SEED}" \
      -type d \
      -name checkpoints \
      -prune \
      -exec rm -rf {} +
done
```

## ISBI Cell 4 — Aggregate the three seeds

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/aggregate_seed_results.py \
  --output-root outputs_ablation/isbi2012 \
  --seeds 42,1,2 \
  --save-name multi_seed_summary
```

## ISBI Cell 5 — Validate result

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

SUMMARY="$(find outputs_ablation/isbi2012 \
  -type f \
  -name 'multi_seed_summary.csv' \
  | head -n 1)"

if [[ -z "$SUMMARY" ]]; then
    echo "ERROR: ISBI ablation summary not found."
    exit 1
fi

EXPECTED_MODELS=(
  unet
  fourier_unet_plain
  apf_amplitude_only
  apf_phase_only
  proposal_apf_unet_at_encoder1
  proposal_apf_unet
)

for MODEL in "${EXPECTED_MODELS[@]}"; do
    if ! grep -q "$MODEL" "$SUMMARY"; then
        echo "ERROR: missing ablation model $MODEL"
        exit 1
    fi
done

echo "ISBI 2012 ablation validation passed."
cat "$SUMMARY"
```

## ISBI Cell 6 — Archive results

```bash
%%bash
set -euo pipefail

cd /kaggle/working

rm -f final_isbi2012_ablation_results.zip

zip -qr \
  final_isbi2012_ablation_results.zip \
  apf_unet/outputs_ablation/isbi2012 \
  apf_unet/data/splits/isbi2012

ls -lh final_isbi2012_ablation_results.zip
```

---

# Recovery after a session interruption

The multi-seed runner does not automatically resume a partially completed model matrix. If one model is missing, rerun only that model in a separate output directory.

Example:

```bash
%%bash
set -euo pipefail

cd /kaggle/working/apf_unet

python scripts/benchmark_multi_seed.py \
  --models csca_unet \
  --config-dir configs/fair \
  --dataset cvc_300 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_recovery/cvc_300_csca \
  --delete-checkpoints-after-eval
```

---

# Final output checklist

After all sessions, you should have:

```text
session_1_kvasir_seg_results.zip
session_2_clinicdb_colondb_results.zip
session_3_cvc300_results.zip
session_4_etis_results.zip
session_5_kvasir_instrument_results.zip
hyper_kvasir_group_a.zip
hyper_kvasir_group_b.zip
hyper_kvasir_group_c.zip
hyper_kvasir_group_d.zip
final_isbi2012_ablation_results.zip
```

The complete new workload is:

```text
Session 1:  3 models × 3 seeds = 9 runs
Session 2:  3 models × 2 datasets × 3 seeds = 18 runs
Session 3: 13 models × 3 seeds = 39 runs
Session 4: 13 models × 3 seeds = 39 runs
Session 5: 13 models × 3 seeds = 39 runs
Session 6: 13 models × 3 seeds = 39 runs
Ablation:   6 models × 3 seeds = 18 runs

Total: 201 runs
```
