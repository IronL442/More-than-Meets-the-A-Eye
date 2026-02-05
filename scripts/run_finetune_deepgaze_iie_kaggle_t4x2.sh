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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ $# -gt 0 ]]; then
  FOLDS=("$@")
else
  FOLDS=(0 1 2 3)
fi

run_fold() {
  local gpu="$1"
  local fold="$2"
  echo "[launch] gpu=${gpu} fold=${fold}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/finetune_deepgaze_iie.py \
    --config "${CONFIG_PATH}" \
    --cv_fold_index "${fold}" &
  echo $!
}

i=0
while [[ $i -lt ${#FOLDS[@]} ]]; do
  fold_a="${FOLDS[$i]}"
  pid_a="$(run_fold 0 "${fold_a}")"

  pid_b=""
  if [[ $((i + 1)) -lt ${#FOLDS[@]} ]]; then
    fold_b="${FOLDS[$((i + 1))]}"
    pid_b="$(run_fold 1 "${fold_b}")"
  fi

  wait "${pid_a}"
  if [[ -n "${pid_b}" ]]; then
    wait "${pid_b}"
  fi

  i=$((i + 2))
  echo "[done] completed up to index ${i} of ${#FOLDS[@]} folds"
done

echo "[done] all requested folds completed"
