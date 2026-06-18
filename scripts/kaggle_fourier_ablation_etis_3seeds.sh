#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FOURIER U-NET CONTROLLED ABLATION — ETIS, THREE SEEDS
#
# Canonical proposal:
#   proposal_fourier_unet  -> manuscript name: Fourier U-Net
#
# Architecture-only ablations (ordinary U-Net is intentionally excluded):
#   1. proposal_fourier_unet
#   2. fourier_unet_bounded
#   3. fourier_unet_amplitude_only
#   4. fourier_unet_phase_only
#   5. fourier_unet_no_channel_mix
#   6. fourier_unet_no_residual
#   7. fourier_unet_at_encoder1
#
# Dataset: ETIS-LaribPolypDB (repository key: etis)
# Seeds: 42, 1, 2
# Expected runs: 7 variants x 3 seeds = 21
# ============================================================

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

WORK_ROOT="/kaggle/working"
REPO_DIR="${WORK_ROOT}/fourier_unet"
REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/apf_unet.git}"
# Optional: set REPO_ARCHIVE to the corrected ZIP uploaded as a Kaggle input.
REPO_ARCHIVE="${REPO_ARCHIVE:-}"

OUTPUT_ROOT="outputs_fourier_ablation/etis"
SUMMARY_PATH="${OUTPUT_ROOT}/results/tables/multi_seed_summary.csv"
TRAINING_SUMMARY_PATH="${OUTPUT_ROOT}/results/tables/ablation_training_summary.csv"
RANKING_PATH="${OUTPUT_ROOT}/results/tables/ablation_ranking.csv"
LATEX_PATH="${OUTPUT_ROOT}/results/tables/fourier_ablation_etis.tex"

MODELS="proposal_fourier_unet,fourier_unet_bounded,fourier_unet_amplitude_only,fourier_unet_phase_only,fourier_unet_no_channel_mix,fourier_unet_no_residual,fourier_unet_at_encoder1"
SEEDS="42,1,2"

EPOCHS=30
IMAGE_SIZE=352
BATCH_SIZE=6
LEARNING_RATE=0.0003
NUM_WORKERS=2
EXPECTED_MODEL_COUNT=7
EXPECTED_SEED_COUNT=3
EXPECTED_RUN_COUNT=$((EXPECTED_MODEL_COUNT * EXPECTED_SEED_COUNT))

printf '%s\n' \
  "============================================================" \
  "FOURIER U-NET CONTROLLED ABLATION" \
  "============================================================" \
  "Dataset:        etis (ETIS-LaribPolypDB)" \
  "Models:         ${MODELS}" \
  "Seeds:          ${SEEDS}" \
  "Epochs:         ${EPOCHS}" \
  "Image size:     ${IMAGE_SIZE}" \
  "Batch size:     ${BATCH_SIZE}" \
  "Learning rate:  ${LEARNING_RATE}" \
  "Expected runs:  ${EXPECTED_RUN_COUNT}" \
  "============================================================"

# ------------------------------------------------------------
# 1. Obtain the corrected repository
# ------------------------------------------------------------
cd "${WORK_ROOT}"
rm -rf "${REPO_DIR}" "${WORK_ROOT}/_fourier_repo_extract"

if [[ -z "${REPO_ARCHIVE}" ]]; then
    REPO_ARCHIVE="$(find /kaggle/input -type f \
      \( -iname 'fourier_unet_corrected.zip' \
       -o -iname 'fourier-unet-corrected.zip' \
       -o -iname 'apf_unet-main*.zip' \) \
      -print -quit 2>/dev/null || true)"
fi

if [[ -n "${REPO_ARCHIVE}" && -f "${REPO_ARCHIVE}" ]]; then
    echo "Using uploaded repository archive: ${REPO_ARCHIVE}"
    mkdir -p "${WORK_ROOT}/_fourier_repo_extract"
    unzip -q "${REPO_ARCHIVE}" -d "${WORK_ROOT}/_fourier_repo_extract"
    SOURCE_ROOT="$(find "${WORK_ROOT}/_fourier_repo_extract" \
      -type f -path '*/src/models/proposal/fourier_unet.py' \
      -printf '%h\n' | sed 's#/src/models/proposal$##' | head -n 1)"
    if [[ -z "${SOURCE_ROOT}" || ! -d "${SOURCE_ROOT}" ]]; then
        echo "ERROR: the uploaded ZIP does not contain the corrected Fourier U-Net repository."
        exit 1
    fi
    mv "${SOURCE_ROOT}" "${REPO_DIR}"
    rm -rf "${WORK_ROOT}/_fourier_repo_extract"
else
    echo "No corrected ZIP was found; cloning: ${REPO_URL}"
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
echo "Repository directory: $(pwd)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git commit: $(git rev-parse HEAD)"
fi

# Fail early if the remote repository has not yet been updated.
required_files=(
  "requirements.txt"
  "src/models/proposal/fourier_unet.py"
  "scripts/benchmark_multi_seed.py"
  "scripts/run_fourier_ablation.py"
  "scripts/aggregate_seed_results.py"
  "configs/ablation/proposal_fourier_unet.yaml"
  "configs/ablation/fourier_unet_bounded.yaml"
  "configs/ablation/fourier_unet_amplitude_only.yaml"
  "configs/ablation/fourier_unet_phase_only.yaml"
  "configs/ablation/fourier_unet_no_channel_mix.yaml"
  "configs/ablation/fourier_unet_no_residual.yaml"
  "configs/ablation/fourier_unet_at_encoder1.yaml"
)
for file in "${required_files[@]}"; do
    if [[ ! -f "${file}" ]]; then
        echo "ERROR: missing corrected repository file: ${file}"
        echo "Upload fourier_unet_corrected.zip to Kaggle or push the corrected code to REPO_URL."
        exit 1
    fi
    echo "OK: ${file}"
done

# ------------------------------------------------------------
# 2. Install dependencies and validate CUDA
# ------------------------------------------------------------
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q -r requirements.txt
python -m pip install -q pytest certifi

python - <<'PY'
import platform
import sys
import torch

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: enable a GPU accelerator in Kaggle.")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}; {props.total_memory / 2**30:.2f} GiB")
torch.cuda.set_device(0)
x = torch.ones((32, 32), device="cuda")
assert float((x @ x).mean()) > 0
print("CUDA computation test passed.")
PY
nvidia-smi

# ------------------------------------------------------------
# 3. Audit the controlled ablation configurations
# ------------------------------------------------------------
python - <<'PY'
from pathlib import Path
import yaml

root = Path("configs/ablation")
models = [
    "proposal_fourier_unet",
    "fourier_unet_bounded",
    "fourier_unet_amplitude_only",
    "fourier_unet_phase_only",
    "fourier_unet_no_channel_mix",
    "fourier_unet_no_residual",
    "fourier_unet_at_encoder1",
]
configs = {}
for model in models:
    path = root / f"{model}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["model"]["name"] != model:
        raise SystemExit(f"ERROR: {path} has the wrong model.name")
    configs[model] = cfg

common_paths = [
    ("data", "augmentation"),
    ("data", "batch_size"),
    ("data", "image_size"),
    ("data", "pin_memory"),
    ("train", "epochs"),
    ("train", "lr"),
    ("train", "weight_decay"),
    ("train", "optimizer"),
    ("train", "scheduler"),
    ("train", "t_max"),
    ("train", "eta_min"),
    ("train", "mixed_precision"),
    ("train", "grad_clip"),
    ("train", "loss"),
    ("train", "threshold"),
    ("train", "use_aux_outputs_loss"),
    ("train", "use_boundary_loss"),
    ("train", "gradient_accumulation_steps"),
    ("eval", "loss"),
    ("eval", "threshold"),
]
reference = configs["proposal_fourier_unet"]
for section, key in common_paths:
    expected = reference[section].get(key)
    for model, cfg in configs.items():
        actual = cfg[section].get(key)
        if actual != expected:
            raise SystemExit(
                f"ERROR: unfair setting: {model} {section}.{key}={actual!r}; "
                f"expected {expected!r}"
            )

allowed_model_differences = {
    "fourier_unet_bounded": {
        "name", "fourier_amplitude_scale", "fourier_phase_max"
    },
    "fourier_unet_amplitude_only": {"name", "fourier_use_phase"},
    "fourier_unet_phase_only": {"name", "fourier_use_amplitude"},
    "fourier_unet_no_channel_mix": {"name", "fourier_use_channel_mixing"},
    "fourier_unet_no_residual": {
        "name", "fourier_residual", "fourier_zero_init_output"
    },
    "fourier_unet_at_encoder1": {"name"},
}
reference_model = reference["model"]
for model, allowed in allowed_model_differences.items():
    current = configs[model]["model"]
    changed = {
        key for key in set(reference_model) | set(current)
        if reference_model.get(key) != current.get(key)
    }
    unexpected = changed - allowed
    if unexpected:
        raise SystemExit(
            f"ERROR: {model} has unintended model changes: "
            + ", ".join(sorted(unexpected))
        )
print("Controlled Fourier ablation audit passed.")
PY

# ------------------------------------------------------------
# 4. Run targeted repository tests
# ------------------------------------------------------------
python -m pytest -q \
  tests/test_fourier_unet.py \
  tests/test_fourier_repository_consistency.py \
  tests/test_pipeline_contracts.py

# ------------------------------------------------------------
# 5. Clean stale ETIS outputs/download fragments
# ------------------------------------------------------------
rm -rf "${OUTPUT_ROOT}"
rm -f \
  data/downloads/etis.zip.part \
  data/downloads/ETIS.zip.part \
  data/downloads/ETIS-LaribPolypDB.zip.part \
  data/downloads/TestDataset.zip.part \
  data/downloads/testdataset.zip.part
mkdir -p "${OUTPUT_ROOT}"

# ------------------------------------------------------------
# 6. Run 7 variants x 3 seeds on one fixed ETIS split
# ------------------------------------------------------------
python scripts/benchmark_multi_seed.py \
  --models "${MODELS}" \
  --config-dir configs/ablation \
  --dataset etis \
  --data-root data \
  --image-size "${IMAGE_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --device cuda \
  --num-workers "${NUM_WORKERS}" \
  --seeds "${SEEDS}" \
  --output-root "${OUTPUT_ROOT}" \
  --allow-insecure-download \
  --delete-checkpoints-after-eval

# ------------------------------------------------------------
# 7. Validate all 21 test/training records and export summaries
# ------------------------------------------------------------
python - \
  "${SUMMARY_PATH}" \
  "${TRAINING_SUMMARY_PATH}" \
  "${RANKING_PATH}" \
  "${LATEX_PATH}" \
  "${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
training_summary_path = Path(sys.argv[2])
ranking_path = Path(sys.argv[3])
latex_path = Path(sys.argv[4])
output_root = Path(sys.argv[5])

models = [
    "proposal_fourier_unet",
    "fourier_unet_bounded",
    "fourier_unet_amplitude_only",
    "fourier_unet_phase_only",
    "fourier_unet_no_channel_mix",
    "fourier_unet_no_residual",
    "fourier_unet_at_encoder1",
]
expected_models = set(models)
expected_seeds = {"1", "2", "42"}
expected_dataset = "etis"
metrics = ("dice", "iou", "precision", "recall", "mae", "loss")
names = {
    "proposal_fourier_unet": "Fourier U-Net",
    "fourier_unet_bounded": "Bounded Fourier",
    "fourier_unet_amplitude_only": "Amplitude only",
    "fourier_unet_phase_only": "Phase only",
    "fourier_unet_no_channel_mix": "No channel mixing",
    "fourier_unet_no_residual": "No residual",
    "fourier_unet_at_encoder1": "Fourier at encoder stage 1",
}

if not summary_path.is_file():
    raise SystemExit(f"ERROR: missing aggregate summary: {summary_path}")
with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    aggregate_rows = list(reader)
    headers = reader.fieldnames or []
rows_by_model = {row.get("model", "").strip(): row for row in aggregate_rows}
if set(rows_by_model) != expected_models:
    raise SystemExit(
        f"ERROR: aggregate models={sorted(rows_by_model)}; "
        f"expected={sorted(expected_models)}"
    )
if len(aggregate_rows) != 7:
    raise SystemExit(f"ERROR: expected 7 aggregate rows; found {len(aggregate_rows)}")

for model in models:
    row = rows_by_model[model]
    if row.get("dataset", "").strip() != expected_dataset:
        raise SystemExit(f"ERROR: wrong dataset for {model}")
    if row.get("split", "test").strip() != "test":
        raise SystemExit(f"ERROR: wrong split for {model}")
    if int(row.get("num_seeds", 0)) != 3:
        raise SystemExit(f"ERROR: {model} does not contain exactly three seeds")
    found_seeds = set(re.findall(r"-?\d+", row.get("seeds", "")))
    if found_seeds != expected_seeds:
        raise SystemExit(f"ERROR: wrong seed set for {model}: {found_seeds}")
    for metric in metrics:
        for suffix in ("mean", "std", "mean_pm_std"):
            column = f"{metric}_{suffix}"
            if column not in headers or not row.get(column, "").strip():
                raise SystemExit(f"ERROR: missing {column} for {model}")
        mean = float(row[f"{metric}_mean"])
        std = float(row[f"{metric}_std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise SystemExit(f"ERROR: invalid aggregate {metric} for {model}")

# Validate every per-seed test JSON.
per_seed = {}
for seed in expected_seeds:
    seed_root = output_root / f"seed_{seed}"
    if not seed_root.is_dir():
        raise SystemExit(f"ERROR: missing {seed_root}")
    files = sorted((seed_root / "results" / "tables").rglob("*_metrics.json"))
    if not files:
        files = sorted(seed_root.rglob("metrics_*.json"))
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("model", "")).strip()
        if model not in expected_models:
            continue
        if str(payload.get("dataset", "")).strip() != expected_dataset:
            raise SystemExit(f"ERROR: wrong dataset in {path}")
        values = payload.get("metrics", {})
        for metric in metrics:
            value = float(values[metric])
            if not math.isfinite(value):
                raise SystemExit(f"ERROR: non-finite {metric} in {path}")
        key = (seed, model)
        if key in per_seed:
            raise SystemExit(f"ERROR: duplicate record for {key}")
        per_seed[key] = path
expected_keys = {(seed, model) for seed in expected_seeds for model in models}
if set(per_seed) != expected_keys:
    missing = expected_keys - set(per_seed)
    raise SystemExit(f"ERROR: missing per-seed records: {sorted(missing)}")

# Aggregate best training/validation checkpoints.
training_records = []
for seed in expected_seeds:
    for model in models:
        path = output_root / f"seed_{seed}" / model / "results" / "summary.json"
        if not path.is_file():
            raise SystemExit(f"ERROR: missing training summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get("best_record", {})
        values = {
            "train_loss": record.get("train/loss"),
            "val_loss": record.get("val/loss"),
            "val_dice": record.get("val/dice"),
            "val_iou": record.get("val/iou"),
            "val_precision": record.get("val/precision"),
            "val_recall": record.get("val/recall"),
            "val_mae": record.get("val/mae"),
            "best_epoch": payload.get("best_epoch"),
        }
        normalized = {"model": model, "seed": seed}
        for key, raw in values.items():
            value = float(raw)
            if not math.isfinite(value):
                raise SystemExit(f"ERROR: non-finite {key} in {path}")
            normalized[key] = value
        training_records.append(normalized)

training_metrics = [
    "train_loss", "val_loss", "val_dice", "val_iou",
    "val_precision", "val_recall", "val_mae", "best_epoch",
]
training_rows = []
for model in models:
    items = [item for item in training_records if item["model"] == model]
    row = {
        "model": model,
        "method": names[model],
        "dataset": expected_dataset,
        "num_seeds": len(items),
        "seeds": "1,2,42",
    }
    for metric in training_metrics:
        values = [float(item[metric]) for item in items]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_mean_pm_std"] = f"{mean:.4f} ± {std:.4f}"
    training_rows.append(row)
training_summary_path.parent.mkdir(parents=True, exist_ok=True)
with training_summary_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(training_rows[0]))
    writer.writeheader()
    writer.writerows(training_rows)

# Rank primarily by Dice, then IoU, then lower loss.
ranked = sorted(
    aggregate_rows,
    key=lambda row: (
        -float(row["dice_mean"]),
        -float(row["iou_mean"]),
        float(row["loss_mean"]),
    ),
)
ranking_rows = []
for rank, row in enumerate(ranked, start=1):
    item = dict(row)
    item = {"rank": rank, "method": names[item["model"]], **item}
    ranking_rows.append(item)
with ranking_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(ranking_rows[0]))
    writer.writeheader()
    writer.writerows(ranking_rows)

# Manuscript-ready LaTeX table.
lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Controlled Fourier U-Net ablation on ETIS-LaribPolypDB. Values are mean $\pm$ standard deviation over three seeds. Higher is better for Dice, IoU, Precision, and Recall; lower is better for MAE and loss.}",
    r"\label{tab:fourier_ablation_etis}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Method & Dice & IoU & Precision & Recall & MAE & Loss \\",
    r"\midrule",
]
for row in ranking_rows:
    method = row["method"].replace("_", r"\_")
    lines.append(
        f"{method} & ${row['dice_mean_pm_std']}$ & ${row['iou_mean_pm_std']}$ "
        f"& ${row['precision_mean_pm_std']}$ & ${row['recall_mean_pm_std']}$ "
        f"& ${row['mae_mean_pm_std']}$ & ${row['loss_mean_pm_std']}$ \\\\" 
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

best = ranking_rows[0]
print("============================================================")
print("FOURIER ABLATION VALIDATION PASSED")
print("============================================================")
print(f"Validated per-seed test records: {len(per_seed)}")
print(f"Validated training records:      {len(training_records)}")
print(f"Best method by mean Dice:         {best['method']}")
print(f"Best Dice:                        {best['dice_mean_pm_std']}")
print(f"Best IoU:                         {best['iou_mean_pm_std']}")
print(f"Test summary:                     {summary_path}")
print(f"Training summary:                 {training_summary_path}")
print(f"Ranking:                          {ranking_path}")
print(f"LaTeX table:                      {latex_path}")
PY

# ------------------------------------------------------------
# 8. Display the final ranking
# ------------------------------------------------------------
python - "${RANKING_PATH}" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8-sig")))
columns = [
    "rank", "method", "dice_mean_pm_std", "iou_mean_pm_std",
    "precision_mean_pm_std", "recall_mean_pm_std",
    "mae_mean_pm_std", "loss_mean_pm_std",
]
widths = {
    column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
    for column in columns
}
print(" | ".join(column.ljust(widths[column]) for column in columns))
print("-+-".join("-" * widths[column] for column in columns))
for row in rows:
    print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
PY

# ------------------------------------------------------------
# 9. Final cleanup verification
# ------------------------------------------------------------
CHECKPOINT_COUNT="$(find "${OUTPUT_ROOT}" -type d -name checkpoints | wc -l)"
echo "Remaining checkpoint directories: ${CHECKPOINT_COUNT}"
if [[ "${CHECKPOINT_COUNT}" -ne 0 ]]; then
    echo "ERROR: checkpoint directories remain after evaluation cleanup."
    find "${OUTPUT_ROOT}" -type d -name checkpoints -print
    exit 1
fi

echo "Generated summary files:"
find "${OUTPUT_ROOT}/results/tables" -maxdepth 1 -type f -print | sort
du -sh "${OUTPUT_ROOT}"

echo "============================================================"
echo "FOURIER U-NET ETIS ABLATION COMPLETED SUCCESSFULLY"
echo "============================================================"
