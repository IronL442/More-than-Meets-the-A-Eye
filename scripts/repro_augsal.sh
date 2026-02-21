#!/usr/bin/env bash
set -euo pipefail

python AugSal/pipeline.py --config AugSal/configs/default.yaml

python scripts/finetune_deepgaze_iie.py --config configs/finetune_deepgaze_iie_augsal.yaml

for fold in 01 02 03 04; do
  python -m saliency_bench.core.runner --config "configs/augsal_evaluation_ft_fold_${fold}.yaml"
done

python scripts/aggregate_fold_metrics.py \
  --pattern "outputs/augsal/eval/finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/augsal/eval/deepgaze_iie_finetuned_cv_summary.csv"
