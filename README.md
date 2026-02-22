# Saliency Reproducibility Benchmark

This repository contains a reproducible pipeline for saliency training and evaluation with:

- `Finetuned` (DeepGaze II-E on non-augmented training split)
- `AugSal` (caption-guided synthetic augmentation + pseudo labels)
- `MiaMix` (mix-based augmentation)

The codebase is organized around:

- training: `scripts/finetune_deepgaze_iie.py`
- evaluation: `saliency_bench/core/runner.py`
- augmentation: `AugSal/pipeline.py` and `MiaMix/MiaMix.py`

## Environment Setup

Use Python `3.10` or `3.11`.

Canonical reproducible install (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.lock.txt
```

`requirements.lock.txt` already includes DeepGaze + CLIP and experiment extras used in this repo.

If you want a leaner install instead, use `requirements.txt` and then add only the extras you need:

```bash
# Lean base install
python -m pip install -r requirements.txt

# W&B logging
python -m pip install -r requirements.optional-wandb.txt

# Diffusers backend for AugSal
python -m pip install -r requirements.optional-diffusers.txt

# DeepGaze II-E dependency
python -m pip install -r requirements.optional-deepgaze.txt
```

## Dataset Layout

Expected local layout:

```text
data/seminar_data/
  images/*.jpg|png
  gt_maps/                # participant-level GT maps (e.g. PXX_IMG...png)
  gt_maps_mean/*.npy      # pre-aggregated GT maps for evaluation/training
  image_captions.json     # required by AugSal
```

Split files:

```text
splits/test.txt
splits/trainval.txt
```

Regenerate splits deterministically if needed:

```bash
python scripts/make_holdout_split.py --root data/seminar_data --test_count 30 --seed 1337 --out_dir splits
```

## Reproducibility Controls

- Training and augmentation configs include deterministic settings under `determinism`.
- Fine-tuning writes `run_metadata.json` per fold with config hash + seed.
- AugSal writes reproducibility metadata in `run_summary.json`.
- Prediction cache keys include dataset/model/config hash to avoid fold/checkpoint collisions.
- MiaMix defaults to leakage-safe filtering:
  - include: `splits/trainval.txt`
  - exclude: `splits/test.txt`

## Canonical End-to-End Runs

### 1) Finetuned (non-augmented)

```bash
bash scripts/repro_finetuned.sh
```

Uses:

- training config: `configs/finetune_deepgaze_iie.yaml`
- eval configs: `configs/eval_deepgaze_iie_finetuned_fold_01.yaml` ... `_04.yaml`

### 2) AugSal

```bash
bash scripts/repro_augsal.sh
```

Uses:

- augmentation config: `AugSal/configs/default.yaml`
- training config: `configs/finetune_deepgaze_iie_augsal.yaml`
- eval configs: `configs/augsal_evaluation_ft_fold_01.yaml` ... `_04.yaml`

### 3) MiaMix

```bash
bash scripts/repro_miamix.sh
```

Uses:

- augmentation entrypoint: `MiaMix/MiaMix.py`
- training config: `configs/finetune_deepgaze_iie_miamix.yaml`
- eval configs: `configs/miamix_evaluation_ft_fold_01.yaml` ... `_04.yaml`

## Manual Evaluation and Aggregation

Run one evaluation:

```bash
python -m saliency_bench.core.runner --config configs/eval_deepgaze_iie_pretrained.yaml
```

Aggregate 4-fold results:

```bash
python scripts/aggregate_fold_metrics.py \
  --pattern "outputs/non_augmented/eval/deepgaze_iie_finetuned_fold_*/ALL_SUMMARY.csv" \
  --out "outputs/non_augmented/eval/deepgaze_iie_finetuned_cv_summary.csv"
```

Equivalent aggregation can be done for AugSal and MiaMix by changing `--pattern` and `--out`.

Generate a combined seminar overview (table + plots):

```bash
python scripts/make_results_overview.py --out_dir outputs/overview
```

## Key Config Files

- base finetune CV: `configs/finetune_deepgaze_iie.yaml`
- AugSal finetune CV: `configs/finetune_deepgaze_iie_augsal.yaml`
- MiaMix finetune CV: `configs/finetune_deepgaze_iie_miamix.yaml`
- baseline eval: `configs/eval_deepgaze_iie_pretrained.yaml`

Kaggle-specific configs are provided separately:

- `configs/finetune_deepgaze_iie_kaggle.yaml`
- `configs/finetune_deepgaze_iie_augsal_kaggle.yaml`
- `configs/finetune_deepgaze_iie_miamix_kaggle.yaml`
- `AugSal/configs/kaggle_*.yaml`

## Output Paths

- fine-tune checkpoints:
  - `outputs/finetune/deepgaze_iie/fold_XX/final.pth`
  - `outputs/finetune_augsal/deepgaze_iie/fold_XX/final.pth`
  - `outputs/finetune_miamix/deepgaze_iie/fold_XX/final.pth`
- eval summaries:
  - `outputs/*/eval/*/ALL_SUMMARY.csv`
- fold aggregates:
  - produced by `scripts/aggregate_fold_metrics.py`
