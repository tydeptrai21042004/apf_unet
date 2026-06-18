#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# URF-U-NET CONTROLLED ABLATION — ETIS, THREE SEEDS
#
# Comparison:
#   1. proposal_fourier_unet          (Plain Fourier U-Net)
#   2. urf_unet_dynamic_global_only   (dynamic global only)
#   3. urf_unet_no_dynamic_global     (plain global + routed local)
#   4. urf_unet_no_uncertainty        (local Fourier without routing)
#   5. urf_unet_no_boundary_supervision
#   6. urf_unet_no_coarse_supervision
#   7. proposal_urf_unet              (full URF-U-Net)
#
# Dataset: ETIS-LaribPolypDB
# Seeds: 42, 1, 2
# Expected runs: 7 x 3 = 21
# ============================================================

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

WORK_ROOT="/kaggle/working"
REPO_DIR="${WORK_ROOT}/urf_unet"
REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/apf_unet.git}"
REPO_ARCHIVE="${REPO_ARCHIVE:-}"

OUTPUT_ROOT="outputs_urf_ablation/etis"
SUMMARY_PATH="${OUTPUT_ROOT}/results/tables/multi_seed_summary.csv"
TRAINING_SUMMARY_PATH="${OUTPUT_ROOT}/results/tables/urf_training_summary.csv"
RANKING_PATH="${OUTPUT_ROOT}/results/tables/urf_ablation_ranking.csv"
LATEX_PATH="${OUTPUT_ROOT}/results/tables/urf_ablation_etis.tex"

MODELS="proposal_fourier_unet,urf_unet_dynamic_global_only,urf_unet_no_dynamic_global,urf_unet_no_uncertainty,urf_unet_no_boundary_supervision,urf_unet_no_coarse_supervision,proposal_urf_unet"
SEEDS="42,1,2"
EPOCHS=60
IMAGE_SIZE=352
BATCH_SIZE=6
LEARNING_RATE=0.0001
NUM_WORKERS=2
EXPECTED_MODEL_COUNT=7
EXPECTED_SEED_COUNT=3
EXPECTED_RUN_COUNT=$((EXPECTED_MODEL_COUNT * EXPECTED_SEED_COUNT))

printf '%s\n' \
  "============================================================" \
  "URF-U-NET CONTROLLED ABLATION" \
  "============================================================" \
  "Dataset:        ETIS-LaribPolypDB" \
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
rm -rf "${REPO_DIR}" "${WORK_ROOT}/_urf_repo_extract"

if [[ -z "${REPO_ARCHIVE}" ]]; then
    REPO_ARCHIVE="$(find /kaggle/input -type f \
      \( -iname 'urf_unet_corrected.zip' \
       -o -iname 'urf-unet-corrected.zip' \
       -o -iname 'fourier_unet_corrected.zip' \
       -o -iname 'apf_unet-main*.zip' \) \
      -print -quit 2>/dev/null || true)"
fi

if [[ -n "${REPO_ARCHIVE}" && -f "${REPO_ARCHIVE}" ]]; then
    echo "Using uploaded repository archive: ${REPO_ARCHIVE}"
    mkdir -p "${WORK_ROOT}/_urf_repo_extract"
    unzip -q "${REPO_ARCHIVE}" -d "${WORK_ROOT}/_urf_repo_extract"
    SOURCE_ROOT="$(find "${WORK_ROOT}/_urf_repo_extract" \
      -type f -path '*/src/models/proposal/urf_unet.py' \
      -printf '%h\n' | sed 's#/src/models/proposal$##' | head -n 1)"
    if [[ -z "${SOURCE_ROOT}" || ! -d "${SOURCE_ROOT}" ]]; then
        echo "ERROR: uploaded ZIP does not contain the corrected URF-U-Net repository."
        exit 1
    fi
    mv "${SOURCE_ROOT}" "${REPO_DIR}"
    rm -rf "${WORK_ROOT}/_urf_repo_extract"
else
    echo "No corrected URF ZIP found; cloning ${REPO_URL}"
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
echo "Repository directory: $(pwd)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git commit: $(git rev-parse HEAD)"
fi

required_files=(
  "requirements.txt"
  "src/models/proposal/fourier_unet.py"
  "src/models/proposal/urf_unet.py"
  "scripts/benchmark_multi_seed.py"
  "scripts/aggregate_seed_results.py"
  "scripts/run_urf_ablation.py"
  "configs/urf_ablation/proposal_fourier_unet.yaml"
  "configs/urf_ablation/urf_unet_dynamic_global_only.yaml"
  "configs/urf_ablation/urf_unet_no_dynamic_global.yaml"
  "configs/urf_ablation/urf_unet_no_uncertainty.yaml"
  "configs/urf_ablation/urf_unet_no_boundary_supervision.yaml"
  "configs/urf_ablation/urf_unet_no_coarse_supervision.yaml"
  "configs/urf_ablation/proposal_urf_unet.yaml"
)
for file in "${required_files[@]}"; do
    [[ -f "${file}" ]] || { echo "ERROR: missing ${file}"; exit 1; }
    echo "OK: ${file}"
done

# ------------------------------------------------------------
# 2. Install and validate environment
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
print("CUDA devices:", torch.cuda.device_count())
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
# 3. Audit controlled configurations
# ------------------------------------------------------------
python - <<'PY'
from pathlib import Path
import yaml

expected = [
    "proposal_fourier_unet",
    "urf_unet_dynamic_global_only",
    "urf_unet_no_dynamic_global",
    "urf_unet_no_uncertainty",
    "urf_unet_no_boundary_supervision",
    "urf_unet_no_coarse_supervision",
    "proposal_urf_unet",
]
root = Path("configs/urf_ablation")
configs = {}
for name in expected:
    path = root / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["model"]["name"] != name:
        raise SystemExit(f"ERROR: {path} has wrong model.name")
    configs[name] = cfg

common = [
    ("data", "augmentation"), ("data", "batch_size"),
    ("data", "image_size"), ("train", "epochs"),
    ("train", "lr"), ("train", "loss"),
    ("train", "optimizer"), ("train", "scheduler"),
    ("train", "weight_decay"), ("train", "grad_clip"),
    ("train", "mixed_precision"), ("eval", "loss"),
    ("eval", "threshold"),
]
reference = configs["proposal_fourier_unet"]
for section, key in common:
    value = reference[section].get(key)
    for name, cfg in configs.items():
        if cfg[section].get(key) != value:
            raise SystemExit(
                f"ERROR: unfair protocol: {name} {section}.{key}="
                f"{cfg[section].get(key)!r}; expected {value!r}"
            )

full = configs["proposal_urf_unet"]["train"]
if full.get("use_boundary_loss") is not True or full.get("use_aux_outputs_loss") is not True:
    raise SystemExit("ERROR: full URF-U-Net must enable coarse and boundary supervision")
no_boundary = configs["urf_unet_no_boundary_supervision"]["train"]
if no_boundary.get("use_boundary_loss") is not False or no_boundary.get("use_aux_outputs_loss") is not True:
    raise SystemExit("ERROR: no-boundary variant must retain coarse supervision only")
no_coarse = configs["urf_unet_no_coarse_supervision"]["train"]
if no_coarse.get("use_boundary_loss") is not True or no_coarse.get("use_aux_outputs_loss") is not False:
    raise SystemExit("ERROR: no-coarse variant must retain boundary supervision only")
print("Controlled URF ablation configuration audit passed.")
PY

# ------------------------------------------------------------
# 4. Run targeted tests
# ------------------------------------------------------------
python -m pytest -q \
  tests/test_urf_unet.py \
  tests/test_fourier_unet.py \
  tests/test_fourier_repository_consistency.py \
  tests/test_pipeline_contracts.py

# ------------------------------------------------------------
# 5. Clean stale outputs and partial downloads
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
# 6. Run 6 variants x 3 seeds on one fixed ETIS split
# ------------------------------------------------------------
python scripts/benchmark_multi_seed.py \
  --models "${MODELS}" \
  --config-dir configs/urf_ablation \
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
# 7. Validate results and aggregate training summaries
# ------------------------------------------------------------
python - \
  "${SUMMARY_PATH}" \
  "${TRAINING_SUMMARY_PATH}" \
  "${RANKING_PATH}" \
  "${LATEX_PATH}" \
  "${OUTPUT_ROOT}" \
  "${EXPECTED_RUN_COUNT}" <<'PY'
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
training_path = Path(sys.argv[2])
ranking_path = Path(sys.argv[3])
latex_path = Path(sys.argv[4])
output_root = Path(sys.argv[5])
expected_run_count = int(sys.argv[6])

models = [
    "proposal_fourier_unet",
    "urf_unet_dynamic_global_only",
    "urf_unet_no_dynamic_global",
    "urf_unet_no_uncertainty",
    "urf_unet_no_boundary_supervision",
    "urf_unet_no_coarse_supervision",
    "proposal_urf_unet",
]
seeds = ["42", "1", "2"]
metrics = ["dice", "iou", "precision", "recall", "mae", "loss"]
names = {
    "proposal_fourier_unet": "Plain Fourier U-Net",
    "urf_unet_dynamic_global_only": "Dynamic global only",
    "urf_unet_no_dynamic_global": "No dynamic global",
    "urf_unet_no_uncertainty": "No uncertainty routing",
    "urf_unet_no_boundary_supervision": "No boundary supervision",
    "urf_unet_no_coarse_supervision": "No coarse supervision",
    "proposal_urf_unet": "URF-U-Net",
}

if not summary_path.is_file():
    raise SystemExit(f"ERROR: missing {summary_path}")
with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))
rows_by_model = {row["model"]: row for row in rows}
if set(rows_by_model) != set(models):
    raise SystemExit(
        f"ERROR: aggregate models={sorted(rows_by_model)}; expected={sorted(models)}"
    )
for model in models:
    row = rows_by_model[model]
    if row.get("dataset") != "etis" or row.get("split") != "test":
        raise SystemExit(f"ERROR: wrong dataset/split for {model}")
    if int(row.get("num_seeds", 0)) != 3:
        raise SystemExit(f"ERROR: {model} does not contain 3 seeds")
    for metric in metrics:
        mean = float(row[f"{metric}_mean"])
        std = float(row[f"{metric}_std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise SystemExit(f"ERROR: invalid {metric} for {model}")

training_records = []
metric_files = []
for seed in seeds:
    seed_root = output_root / f"seed_{seed}"
    if not seed_root.is_dir():
        raise SystemExit(f"ERROR: missing {seed_root}")
    metric_files.extend((seed_root / "results" / "tables").rglob("*_metrics.json"))
    for model in models:
        path = seed_root / model / "results" / "summary.json"
        if not path.is_file():
            raise SystemExit(f"ERROR: missing {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get("best_record", {})
        item = {"model": model, "seed": seed}
        values = {
            "train_loss": record.get("train/loss"),
            "val_loss": record.get("val/loss"),
            "val_dice": record.get("val/dice"),
            "val_iou": record.get("val/iou"),
            "best_epoch": payload.get("best_epoch"),
        }
        for key, raw in values.items():
            value = float(raw)
            if not math.isfinite(value):
                raise SystemExit(f"ERROR: non-finite {key} in {path}")
            item[key] = value
        training_records.append(item)

# Validate exactly one test record per model and seed.
seen = set()
for path in metric_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = str(payload.get("model", ""))
    seed = str(payload.get("seed", ""))
    if model not in models:
        continue
    key = (seed, model)
    if key in seen:
        raise SystemExit(f"ERROR: duplicate metric record {key}")
    seen.add(key)
    for metric in metrics:
        value = float(payload["metrics"][metric])
        if not math.isfinite(value):
            raise SystemExit(f"ERROR: non-finite {metric} in {path}")
if len(seen) != expected_run_count:
    raise SystemExit(f"ERROR: expected {expected_run_count} metric records, got {len(seen)}")

training_rows = []
for model in models:
    items = [item for item in training_records if item["model"] == model]
    output = {
        "model": model,
        "method": names[model],
        "dataset": "etis",
        "num_seeds": len(items),
        "seeds": "1,2,42",
    }
    for metric in ["train_loss", "val_loss", "val_dice", "val_iou", "best_epoch"]:
        values = [float(item[metric]) for item in items]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        output[f"{metric}_mean"] = mean
        output[f"{metric}_std"] = std
        output[f"{metric}_mean_pm_std"] = f"{mean:.4f} ± {std:.4f}"
    training_rows.append(output)
training_path.parent.mkdir(parents=True, exist_ok=True)
with training_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(training_rows[0]))
    writer.writeheader(); writer.writerows(training_rows)

ranking = []
for model in models:
    row = dict(rows_by_model[model])
    row["method"] = names[model]
    ranking.append(row)
ranking.sort(key=lambda row: (-float(row["dice_mean"]), float(row["mae_mean"])))
for index, row in enumerate(ranking, start=1):
    row["rank"] = index
with ranking_path.open("w", encoding="utf-8", newline="") as file:
    fields = ["rank", "model", "method", "dice_mean_pm_std", "iou_mean_pm_std", "precision_mean_pm_std", "recall_mean_pm_std", "mae_mean_pm_std", "loss_mean_pm_std"]
    writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(ranking)

lines = [
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Controlled comparison of Plain Fourier U-Net and URF-U-Net variants on ETIS. Values are mean $\\pm$ standard deviation over three seeds.}",
    "\\label{tab:urf_etis_ablation}",
    "\\scriptsize",
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    "Method & Dice & IoU & Precision & Recall & MAE & Loss " + r"\\",
    "\\midrule",
]
for row in ranking:
    method = row["method"].replace("_", "\\_")
    values = [row[f"{metric}_mean_pm_std"].replace("±", "$\\pm$") for metric in metrics]
    lines.append(method + " & " + " & ".join(values) + " " + r"\\")
lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("============================================================")
print("URF ABLATION VALIDATION PASSED")
print("============================================================")
print(f"Validated per-seed test records: {len(seen)}")
print(f"Validated training records: {len(training_records)}")
print(f"Best by mean Dice: {ranking[0]['method']} ({ranking[0]['dice_mean_pm_std']})")
print(f"Test summary: {summary_path}")
print(f"Training summary: {training_path}")
print(f"Ranking: {ranking_path}")
print(f"LaTeX: {latex_path}")
PY

# ------------------------------------------------------------
# 8. Display ranking and verify cleanup
# ------------------------------------------------------------
echo
echo "============================================================"
echo "FINAL URF ABLATION RANKING"
echo "============================================================"
python - "${RANKING_PATH}" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8-sig")))
for row in rows:
    print(
        f"{int(row['rank']):>2}. {row['method']:<28} "
        f"Dice={row['dice_mean_pm_std']:<18} "
        f"IoU={row['iou_mean_pm_std']:<18} "
        f"MAE={row['mae_mean_pm_std']}"
    )
PY

CHECKPOINT_COUNT="$(find "${OUTPUT_ROOT}" -type d -name checkpoints | wc -l)"
echo "Remaining checkpoint directories: ${CHECKPOINT_COUNT}"
if [[ "${CHECKPOINT_COUNT}" -ne 0 ]]; then
    find "${OUTPUT_ROOT}" -type d -name checkpoints -print
    exit 1
fi

echo "Output size: $(du -sh "${OUTPUT_ROOT}" | cut -f1)"
echo "URF-U-Net ablation completed successfully."
echo "Summary: ${REPO_DIR}/${SUMMARY_PATH}"
echo "LaTeX:  ${REPO_DIR}/${LATEX_PATH}"
