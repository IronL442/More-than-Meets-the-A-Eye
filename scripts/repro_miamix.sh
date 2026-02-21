#!/usr/bin/env bash
set -euo pipefail

python MiaMix/MiaMix.py \
  --include_list splits/trainval.txt \
  --exclude_list splits/test.txt \
  --output_root MiaMix/augmented_images \
  --seed 1337 \
  --deterministic

python scripts/finetune_deepgaze_iie.py --config configs/finetune_deepgaze_iie_miamix.yaml

for fold in 01 02 03 04; do
  python -m saliency_bench.core.runner --config "configs/miamix_evaluation_ft_fold_${fold}.yaml"
done

python scripts/aggregate_fold_metrics.py \
  --pattern "outputs/miamix/eval/finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/miamix/eval/deepgaze_iie_finetuned_cv_summary.csv"
