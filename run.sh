#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-kvasir_seg}"; DATA_ROOT="${DATA_ROOT:-data}"; DEVICE="${DEVICE:-auto}"; OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"; SEED="${SEED:-42}"
FAIR_MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_apf_unet"
FAITHFUL_MODELS="unet,attention_unet,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet"
case "${1:-help}" in
 install) "$PYTHON_BIN" -m pip install -r requirements.txt ;;
 prepare) shift; "$PYTHON_BIN" scripts/prepare_dataset.py --dataset "$DATASET" --data-root "$DATA_ROOT" "$@" ;;
 fair) "$PYTHON_BIN" scripts/benchmark_all.py --models "$FAIR_MODELS" --config-dir configs/fair --dataset "$DATASET" --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$OUTPUT_ROOT" ;;
 faithful) "$PYTHON_BIN" scripts/benchmark_all.py --models "$FAITHFUL_MODELS" --config-dir configs/official_faithful --dataset "$DATASET" --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$OUTPUT_ROOT" ;;
 ablation) "$PYTHON_BIN" scripts/run_apf_ablation.py --dataset "$DATASET" --data-root "$DATA_ROOT" --device "$DEVICE" --seed "$SEED" ;;
 test) "$PYTHON_BIN" -m pytest -q ;;
 *) echo "Usage: bash run.sh {install|prepare|fair|faithful|ablation|test}" ;;
esac
