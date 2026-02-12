#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash AugSal/scripts/run_pipeline_kaggle_t4x2.sh [config_path]
# Example:
#   bash AugSal/scripts/run_pipeline_kaggle_t4x2.sh AugSal/configs/kaggle_diffusers_lowmem.yaml
#
# Notes:
#   - If any shard fails, this script retries automatically as single-GPU unless
#     AUTO_FALLBACK_SINGLE=0.

CONFIG_PATH="${1:-AugSal/configs/kaggle_diffusers_lowmem.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

NUM_SHARDS="${NUM_SHARDS:-2}"
if [[ "${NUM_SHARDS}" -ne 2 ]]; then
  echo "[warn] NUM_SHARDS=${NUM_SHARDS}; this launcher is optimized for T4x2 (2 shards)." >&2
fi

SHARDS_ROOT="${SHARDS_ROOT:-/kaggle/working/AugSal/shards}"
MERGED_ROOT="${MERGED_ROOT:-/kaggle/working/AugSal/augmented_data}"
LOG_DIR="${LOG_DIR:-/kaggle/working/AugSal/logs}"
COPY_MODE="${COPY_MODE:-copy}"  # copy | hardlink
AUTO_FALLBACK_SINGLE="${AUTO_FALLBACK_SINGLE:-1}"  # 1 -> retry as single-GPU run if any shard fails
FALLBACK_GPU="${FALLBACK_GPU:-0}"

mkdir -p "${SHARDS_ROOT}" "${LOG_DIR}"

LAST_PID=""
run_shard() {
  local gpu="$1"
  local shard_idx="$2"
  local shard_out="${SHARDS_ROOT}/shard_${shard_idx}"
  local log_file="${LOG_DIR}/augsal_shard_${shard_idx}_gpu_${gpu}.log"
  echo "[launch] shard=${shard_idx} gpu=${gpu} out=${shard_out}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" AugSal/pipeline.py \
    --config "${CONFIG_PATH}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard_idx}" \
    --output_root "${shard_out}" \
    >"${log_file}" 2>&1 &
  LAST_PID="$!"
  echo "[launch] pid=${LAST_PID}" >&2
}

report_failure() {
  local shard_idx="$1"
  local pid="$2"
  local gpu="$3"
  local log_file="${LOG_DIR}/augsal_shard_${shard_idx}_gpu_${gpu}.log"
  echo "[error] shard=${shard_idx} failed (pid=${pid})" >&2
  if [[ -f "${log_file}" ]]; then
    tail -n 120 "${log_file}" >&2
  fi
}

run_single_fallback() {
  local log_file="${LOG_DIR}/augsal_single_fallback_gpu_${FALLBACK_GPU}.log"
  echo "[warn] running single-GPU fallback on gpu=${FALLBACK_GPU}" >&2
  CUDA_VISIBLE_DEVICES="${FALLBACK_GPU}" "${PYTHON_BIN}" AugSal/pipeline.py \
    --config "${CONFIG_PATH}" \
    --num_shards 1 \
    --shard_index 0 \
    --output_root "${MERGED_ROOT}" \
    >"${log_file}" 2>&1
}

run_shard 0 0
pid0="${LAST_PID}"

run_shard 1 1
pid1="${LAST_PID}"

status0=0
status1=0
wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?

if [[ "${status0}" -eq 0 && "${status1}" -eq 0 ]]; then
  "${PYTHON_BIN}" AugSal/scripts/merge_shards.py \
    --shards_root "${SHARDS_ROOT}" \
    --out_root "${MERGED_ROOT}" \
    --copy_mode "${COPY_MODE}" \
    --overwrite
  echo "[done] merged dataset: ${MERGED_ROOT}" >&2
  exit 0
fi

if [[ "${status0}" -ne 0 ]]; then
  report_failure 0 "${pid0}" 0
fi
if [[ "${status1}" -ne 0 ]]; then
  report_failure 1 "${pid1}" 1
fi

if [[ "${AUTO_FALLBACK_SINGLE}" == "1" ]]; then
  run_single_fallback || {
    fallback_log="${LOG_DIR}/augsal_single_fallback_gpu_${FALLBACK_GPU}.log"
    echo "[error] single-GPU fallback failed" >&2
    if [[ -f "${fallback_log}" ]]; then
      tail -n 120 "${fallback_log}" >&2
    fi
    exit 1
  }
  echo "[done] single-GPU fallback output: ${MERGED_ROOT}" >&2
  exit 0
fi

echo "[error] shard run failed and AUTO_FALLBACK_SINGLE=${AUTO_FALLBACK_SINGLE}" >&2
exit 1
