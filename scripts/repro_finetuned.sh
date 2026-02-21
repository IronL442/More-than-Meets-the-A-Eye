#!/usr/bin/env bash
set -euo pipefail

python scripts/make_holdout_split.py \
  --root data/seminar_data \
  --test_count 30 \
  --seed 1337 \
  --out_dir splits

python scripts/precompute_gt_mean.py --config configs/precompute_seminar_gt_mean.yaml

python scripts/finetune_deepgaze_iie.py --config configs/finetune_deepgaze_iie.yaml

for fold in 01 02 03 04; do
  python -m saliency_bench.core.runner --config "configs/eval_deepgaze_iie_finetuned_fold_${fold}.yaml"
done

python scripts/aggregate_fold_metrics.py \
  --pattern "outputs/non_augmented/eval/deepgaze_iie_finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/non_augmented/eval/deepgaze_iie_finetuned_cv_summary.csv"
