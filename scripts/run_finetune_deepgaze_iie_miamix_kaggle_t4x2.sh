#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_finetune_deepgaze_iie_miamix_kaggle_t4x2.sh [config_path] [fold_idx ...]
# Examples:
#   bash scripts/run_finetune_deepgaze_iie_miamix_kaggle_t4x2.sh
#   bash scripts/run_finetune_deepgaze_iie_miamix_kaggle_t4x2.sh configs/finetune_deepgaze_iie_miamix_kaggle.yaml 0 1 2 3

CONFIG_PATH="${1:-configs/finetune_deepgaze_iie_miamix_kaggle.yaml}"
shift || true
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-/kaggle/working/outputs/finetune_miamix/logs}"
TORCH_HOME="${TORCH_HOME:-/kaggle/working/.cache/torch}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
export TORCH_HOME
mkdir -p "${LOG_DIR}"
mkdir -p "${TORCH_HOME}"

if [[ $# -gt 0 ]]; then
  FOLDS=("$@")
else
  FOLDS=(0 1 2 3)
fi

LAST_PID=""

warm_torch_hub_cache() {
  local warmup_log="${LOG_DIR}/torch_hub_warmup.log"
  echo "[warmup] initializing DeepGazeIIE cache at TORCH_HOME=${TORCH_HOME}" >&2
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -c "from deepgaze_pytorch.deepgaze2e import DeepGazeIIE; DeepGazeIIE(pretrained=False); print('warmup_ok')" \
    >"${warmup_log}" 2>&1 || {
      echo "[error] DeepGaze cache warmup failed. Tail of ${warmup_log}:" >&2
      tail -n 120 "${warmup_log}" >&2 || true
      exit 1
    }
  echo "[warmup] cache ready" >&2
}

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

report_failure() {
  local fold="$1"
  local pid="$2"
  local gpu="$3"
  local log_file="${LOG_DIR}/fold_${fold}_gpu_${gpu}.log"
  echo "[error] fold=${fold} failed (pid=${pid})"
  if [[ -f "${log_file}" ]]; then
    echo "[error] tail of ${log_file}:"
    tail -n 120 "${log_file}"
  else
    echo "[error] log file not found: ${log_file}"
  fi
}

i=0
warm_torch_hub_cache
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
    report_failure "${fold_a}" "${pid_a}" 0
    exit 1
  }
  if [[ -n "${pid_b}" ]]; then
    wait "${pid_b}" || {
      report_failure "${fold_b}" "${pid_b}" 1
      exit 1
    }
  fi

  i=$((i + 2))
  echo "[done] completed up to index ${i} of ${#FOLDS[@]} folds"
done

echo "[done] all requested folds completed"
