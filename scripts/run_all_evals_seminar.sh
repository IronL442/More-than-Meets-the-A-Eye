#!/usr/bin/env bash
set -euo pipefail

# Runs all seminar evaluation configs sequentially:
# 1) Baseline (pretrained)
# 2) Finetuned 4-fold checkpoints
# 3) MiaMix 4-fold checkpoints
# 4) AugSal medium 4-fold checkpoints
# 5) AugSal strong 4-fold checkpoints

PYTHON_BIN="${PYTHON_BIN:-python}"

run_eval() {
  local cfg="$1"
  echo "==> Running eval config: ${cfg}"
  "${PYTHON_BIN}" -m saliency_bench.core.runner --config "${cfg}"
}

echo "==> [1/5] Baseline seminar"
run_eval "configs/baseline_seminar.yaml"

echo "==> [2/5] Finetuned folds"
for fold in 01 02 03 04; do
  run_eval "configs/eval_deepgaze_iie_finetuned_fold_${fold}.yaml"
done

echo "==> [3/5] MiaMix folds"
for fold in 01 02 03 04; do
  run_eval "configs/miamix_evaluation_ft_fold_${fold}.yaml"
done

echo "==> [4/5] AugSal (medium) folds"
for fold in 01 02 03 04; do
  run_eval "configs/augsal_evaluation_ft_fold_${fold}.yaml"
done

echo "==> [5/5] AugSal (strong) folds"
for fold in 01 02 03 04; do
  run_eval "configs/augsal_evaluation_strong_ft_fold_${fold}.yaml"
done

echo "==> Aggregating fold metrics"
"${PYTHON_BIN}" scripts/aggregate_fold_metrics.py \
  --pattern "outputs/non_augmented/eval/deepgaze_iie_finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/non_augmented/eval/deepgaze_iie_finetuned_cv_summary.csv"

"${PYTHON_BIN}" scripts/aggregate_fold_metrics.py \
  --pattern "outputs/miamix/eval/finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/miamix/eval/deepgaze_iie_finetuned_cv_summary.csv"

"${PYTHON_BIN}" scripts/aggregate_fold_metrics.py \
  --pattern "outputs/augsal/eval/finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/augsal/eval/deepgaze_iie_finetuned_cv_summary.csv"

"${PYTHON_BIN}" scripts/aggregate_fold_metrics.py \
  --pattern "outputs/augsal_strong/eval/finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/augsal_strong/eval/deepgaze_iie_finetuned_cv_summary.csv"

echo "==> Building overview tables/plots"
"${PYTHON_BIN}" scripts/make_results_overview.py --out_dir "outputs/overview"

echo "==> Done. Output roots:"
echo "  - outputs/eval/baseline_seminar"
echo "  - outputs/non_augmented/eval"
echo "  - outputs/miamix/eval"
echo "  - outputs/augsal/eval"
echo "  - outputs/augsal_strong/eval"
echo "  - outputs/overview"
