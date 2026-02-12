#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash AugSal/scripts/run_pipeline_kaggle_t4x2.sh [config_path]
# Example:
#   bash AugSal/scripts/run_pipeline_kaggle_t4x2.sh AugSal/configs/kaggle_diffusers_img2img.yaml

CONFIG_PATH="${1:-AugSal/configs/kaggle_diffusers_img2img.yaml}"
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

run_shard 0 0
pid0="${LAST_PID}"

run_shard 1 1
pid1="${LAST_PID}"

wait "${pid0}" || {
  report_failure 0 "${pid0}" 0
  exit 1
}
wait "${pid1}" || {
  report_failure 1 "${pid1}" 1
  exit 1
}

"${PYTHON_BIN}" AugSal/scripts/merge_shards.py \
  --shards_root "${SHARDS_ROOT}" \
  --out_root "${MERGED_ROOT}" \
  --copy_mode "${COPY_MODE}" \
  --overwrite

echo "[done] merged dataset: ${MERGED_ROOT}" >&2
