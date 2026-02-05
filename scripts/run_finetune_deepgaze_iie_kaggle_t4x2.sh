#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_finetune_deepgaze_iie_kaggle_t4x2.sh [config_path] [fold_idx ...]
# Examples:
#   bash scripts/run_finetune_deepgaze_iie_kaggle_t4x2.sh
#   bash scripts/run_finetune_deepgaze_iie_kaggle_t4x2.sh configs/finetune_deepgaze_iie_kaggle.yaml 0 1 2 3

CONFIG_PATH="${1:-configs/finetune_deepgaze_iie_kaggle.yaml}"
shift || true
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-/kaggle/working/outputs/finetune/logs}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
mkdir -p "${LOG_DIR}"

if [[ $# -gt 0 ]]; then
  FOLDS=("$@")
else
  FOLDS=(0 1 2 3)
fi

LAST_PID=""

run_fold() {
  local gpu="$1"
  local fold="$2"
  local log_file="${LOG_DIR}/fold_${fold}_gpu_${gpu}.log"
  echo "[launch] gpu=${gpu} fold=${fold} log=${log_file}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/finetune_deepgaze_iie.py \
    --config "${CONFIG_PATH}" \
    --cv_fold_index "${fold}" >"${log_file}" 2>&1 &
  LAST_PID="$!"
  echo "[launch] pid=${LAST_PID} fold=${fold}" >&2
}

i=0
while [[ $i -lt ${#FOLDS[@]} ]]; do
  fold_a="${FOLDS[$i]}"
  run_fold 0 "${fold_a}"
  pid_a="${LAST_PID}"

  pid_b=""
  if [[ $((i + 1)) -lt ${#FOLDS[@]} ]]; then
    fold_b="${FOLDS[$((i + 1))]}"
    run_fold 1 "${fold_b}"
    pid_b="${LAST_PID}"
  fi

  wait "${pid_a}" || {
    echo "[error] fold=${fold_a} failed (pid=${pid_a})"
    exit 1
  }
  if [[ -n "${pid_b}" ]]; then
    wait "${pid_b}" || {
      echo "[error] fold=${fold_b} failed (pid=${pid_b})"
      exit 1
    }
  fi

  i=$((i + 2))
  echo "[done] completed up to index ${i} of ${#FOLDS[@]} folds"
done

echo "[done] all requested folds completed"
