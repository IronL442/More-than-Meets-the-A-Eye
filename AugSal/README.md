# AugSal Pipeline (Repo-Local)

This folder contains a self-contained, AugSal-like data generation workflow for this repository.

It builds synthetic training pairs from:
- source images (`data/seminar_data/images`)
- source mean GT maps (`data/seminar_data/gt_maps_mean`)
- image captions (`data/seminar_data/image_captions.json`)

The generated dataset is exported in the same shape your fine-tuning code already expects:

```text
AugSal/augmented_data/
  images/*.jpg
  gt_maps/*.npy
  change_maps/*.npy
  metadata.csv
  metadata.jsonl
  run_summary.json
```

## What this implements

- Caption-conditioned prompt construction (`augsal/prompting.py`)
- Pluggable augmentation backends (`augsal/backends.py`)
  - `opencv_caption_style` (default, no extra deps)
  - `diffusers_img2img` (optional, requires `diffusers` + model access)
- Optional diffusers cross-attention map capture and saliency-guided token-map selection
- Pseudo-label generation using image-difference attention + original mean GT (`augsal/pseudo_label.py`)
- End-to-end dataset build runner (`pipeline.py`)

## Quickstart

Generate augmented training data (default backend = OpenCV):

```bash
python3 AugSal/pipeline.py --config AugSal/configs/default.yaml
```

Run a smoke test first:

```bash
python3 AugSal/pipeline.py --config AugSal/configs/default.yaml --max_images 5
```

Use the wrapper script:

```bash
bash AugSal/scripts/run_pipeline.sh AugSal/configs/default.yaml --max_images 20
```

Useful runtime overrides:

```bash
python3 AugSal/pipeline.py --config AugSal/configs/default.yaml \
  --num_shards 2 --shard_index 0 --output_root /tmp/augsal_shard0
```

## Optional Diffusers backend

Use config:

```bash
python3 AugSal/pipeline.py --config AugSal/configs/diffusers_img2img.yaml
```

Or override backend from CLI:

```bash
python3 AugSal/pipeline.py --config AugSal/configs/default.yaml --backend diffusers_img2img
```

If you use diffusers, ensure `torch` + `diffusers` are installed and model download is available.

### Diffusers cross-attention mode

`AugSal/configs/diffusers_img2img.yaml` enables:
- `generation.diffusers.cross_attention.enabled: true` to capture token maps from UNet cross-attention.
- `cross_attention.enabled: true` to use selected token maps in pseudo-labeling.
- `cross_attention.blend_weight` to mix pixel-change maps with selected attention maps.

When enabled, metadata includes:
- `cross_attention_used`
- `cross_attention_num_maps`
- `selected_token`
- `selected_token_index`
- `selected_token_score`

And selected maps can be saved to `selected_attention_maps/` if `cross_attention.save_selected_maps: true`.

## Kaggle-ready usage

Kaggle configs are included:
- `AugSal/configs/kaggle_opencv.yaml`
- `AugSal/configs/kaggle_diffusers_img2img.yaml`
- `AugSal/configs/kaggle_diffusers_lowmem.yaml`

### Single-GPU run

```bash
# install once per session if needed
python -m pip install diffusers transformers accelerate safetensors

python AugSal/pipeline.py --config AugSal/configs/kaggle_diffusers_img2img.yaml
```

### T4x2 parallel run (sharded)

This launches two shard jobs (one per GPU), then merges outputs:

```bash
bash AugSal/scripts/run_pipeline_kaggle_t4x2.sh AugSal/configs/kaggle_diffusers_lowmem.yaml
```

The launcher now auto-falls back to a single-GPU run if a shard fails (for example, OOM).
You can disable fallback by setting `AUTO_FALLBACK_SINGLE=0`.
If you still see CUDA OOM, force single-shard directly:

```bash
NUM_SHARDS=1 bash AugSal/scripts/run_pipeline_kaggle_t4x2.sh
```

Merged output is written to:
- `/kaggle/working/AugSal/augmented_data`

Shard outputs are written to:
- `/kaggle/working/AugSal/shards/shard_0`
- `/kaggle/working/AugSal/shards/shard_1`

You can merge manually with:

```bash
python AugSal/scripts/merge_shards.py \
  --shards_root /kaggle/working/AugSal/shards \
  --out_root /kaggle/working/AugSal/augmented_data \
  --overwrite
```

## Fine-tuning with generated data

Use this config directly with your existing trainer:

```bash
python3 scripts/finetune_deepgaze_iie.py --config configs/finetune_deepgaze_iie_augsal.yaml
```

Use both Kaggle T4 GPUs by running CV folds in parallel:

```bash
bash scripts/run_finetune_deepgaze_iie_augsal_kaggle_t4x2.sh configs/finetune_deepgaze_iie_augsal.yaml
```

Then evaluate on the original non-augmented test images with all four finetuned folds:

```bash
python3 -m saliency_bench.core.runner --config AugSal/configs/augsal_evaluation_ft_fold_01.yaml
python3 -m saliency_bench.core.runner --config AugSal/configs/augsal_evaluation_ft_fold_02.yaml
python3 -m saliency_bench.core.runner --config AugSal/configs/augsal_evaluation_ft_fold_03.yaml
python3 -m saliency_bench.core.runner --config AugSal/configs/augsal_evaluation_ft_fold_04.yaml
```

## Important defaults

- `default.yaml` only augments IDs from `splits/trainval.txt` to avoid test leakage.
- `copy_originals: true` keeps original train images in the generated dataset alongside synthetic ones.
- `num_augs_per_image: 2` by default.

## Tuning knobs

Primary controls are in `AugSal/configs/default.yaml`:
- `generation.num_augs_per_image`
- `generation.backend`
- `pseudo_label.diff_weight`
- `pseudo_label.min_change_threshold`
- `output.copy_originals`
