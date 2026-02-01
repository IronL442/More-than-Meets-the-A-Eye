# Saliency Benchmark Playground

Modular saliency benchmarking toolkit that lets you mix-and-match dataset adapters and model adapters, cache predictions, compute metrics, and visualize results (static PNGs or interactive notebooks). This repository ships with toy datasets/models so you can smoke-test the pipeline immediately, and provides hooks for dropping in real-world saliency corpora (e.g., CAT2000) or deep networks (e.g., MSI-Net, DeepGaze V3, TensorFlow ports).

---

## 1. Quickstart

```bash
# 1) create the virtual environment (Python 3.10/3.11 recommended)
#for Windows without python3
python3 -m venv .venv
source .venv/bin/activate
#for Windows
.venv\Scripts\activate

# 2) install deps (PyTorch + OpenCV, etc.)
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
pip install git+https://github.com/matthias-k/DeepGaze.git # for DeepGaze IIE


# 3) (optional) synthesize a tiny toy dataset
python scripts/make_toy_data.py

# 4) run the benchmark
python -m saliency_bench.core.runner --config configs/baseline.yaml
```

Outputs land in:

* `predictions/` – cached `.npy` saliency maps per dataset/model/image
* `outputs/*.csv` – per-image metrics
* `outputs/*summary.csv` – per model×dataset aggregates + `ALL_SUMMARY.csv`
* `outputs/heatmaps/<dataset>/<model>/*.png` – optional PNG heatmaps and overlays

---

## 2. Project Layout

```
configs/                  # YAML experiment configs
datasets/                 # dataset adapters (folder, CAT2000, csv_indexed, …)
models/                   # model adapters (center bias, blur, MSI-Net, DeepGaze, …)
saliency_bench/core/      # registry, runner, shared interfaces, torch base class
saliency_bench/utils/     # image ops, fixation→density kernel, heatmap writer
metrics/metrics.py        # AUC-Judd, sAUC, NSS, CC, KL, SIM
scripts/make_toy_data.py  # generates 3 sample images + GT/fixations
scripts/create_saliency_notebook.py  # builds notebooks/saliency_viewer.ipynb
tests/                    # smoke tests for models/datasets/metrics/imports
```

---

## 3. Configuration (configs/baseline.yaml)

```yaml
experiment: "baseline_benchmark"
output_dir: "outputs"
prediction_cache_dir: "predictions"
cache_predictions: true

limit_images: 20          # optional: cap images processed per dataset
predict_only: true        # skip metric computation; useful for quick caches

models:
  - "center_bias"
  - "blur_baseline"
  - "msi_net"
  - "deepgaze_v3"
  - "msi_net_tf"

# Optional: specify which models to actually run (if omitted, all models run)
active_models:
  - "deepgaze_v3"

model_kwargs:
  center_bias:
    requires_size: [224, 224]
    sigma_ratio: 0.3
  blur_baseline:
    ksize: 41
  msi_net:
    requires_size: [256, 256]
    weights_path: "weights/msinet/msinet.pt"
  deepgaze_v3:
    requires_size: [384, 384]
    weights_path: "weights/deepgaze_v3/dg3.pt"
  msi_net_tf:
    repo_id: "alexanderkroner/MSI-Net"

datasets:
  - name: "folder"
    root: "data/yourset"
    split: "test"
  - name: "CAT2000"
    root: "data/CAT2000"
    split: "test"
    sigma_px: 19
    provide_sauc_nonfix: true
  - name: "csv_indexed"
    index_csv: "data/MIT_like/index.csv"
    split: "val"
    sigma_px: 15

heatmaps_png:
  enabled: true            # emit PNGs for each prediction
  overlay: true            # also save original+heat overlay
  normalize: "max"         # "max" | "sum" | "none"
  colormap: "jet"
  alpha: 0.5               # overlay opacity
  out_dir: "outputs/heatmaps"
```

**Key flags**

| Flag | Purpose |
| --- | --- |
| `limit_images` | cap how many samples to iterate per dataset (debugging, sanity runs). |
| `active_models` | optional list of model names to run; if omitted, all models in `models` list run. |
| `predict_only` | skip metric computation/aggregation (still caches predictions, saves PNGs). |
| `heatmaps_png` | control PNG outputs (enable overlay, color map, normalization, output dir). |

---

## 4. Adding Datasets

All dataset adapters inherit `SaliencyDataset` (`saliency_bench/core/interfaces.py`) and register via `@register("dataset", "<name>")`.

### 4.1 FolderDataset (`datasets/folder_dataset.py`)

Layout:
```
root/
  images/*.jpg|png|bmp
  gt_maps/<stem>.npy      (optional float32 H×W, sum=1)
  fixations/<stem>.npy    (optional uint8 H×W points)
```

* GT/fixations optional. If both missing, the runner uses a uniform map.
* If only fixations exist, a Gaussian `fixations_to_density` fallback synthesizes a GT map.
* Configure via:
  ```yaml
  - name: "folder"
    root: "data/mydataset"
    sigma_px: 15   # optional, for fixation→density blur
  ```

### 4.2 CAT2000 Adapter (`datasets/cat2000.py`)

* Expects the canonical CAT2000 layout (images, fixations, gt_maps). Missing GTs can be generated from fixations.
* `provide_sauc_nonfix: true` optionally samples non-fix maps for sAUC evaluation.

### 4.3 CSV-Indexed Adapter (`datasets/csv_indexed.py`)

* Provide a CSV with columns: `image_id,image_path,fix_path(optional),map_path(optional)`.
* Great for datasets with custom storage or varied folder structures.

### 4.4 Creating Your Own Dataset

1. Create `datasets/<name>.py`.
2. Implement a class deriving from `SaliencyDataset` that yields `{"image_id","image","gt_map","fixations", ...}`.
3. Decorate with `@register("dataset", "<name>")`.
4. Add to your config under `datasets`.

---

## 5. Adding Models

All models inherit `SaliencyModel`. You can:

* Use simple numpy-based baselines (see `models/center_bias.py`, `models/blur_baseline.py`).
* Extend the Torch helper (`saliency_bench/core/torch_base.py`) to auto-handle ImageNet normalization, resizing, device placement, etc. (see `models/msi_net.py`, `models/deepgaze_v3.py`).
* Integrate TensorFlow models (example placeholder: `models/msi_net_tf.py`).

**Steps**
1. Create `models/<model>.py`.
2. Implement `preprocess()` (resize/normalize), `predict()` (return float32 map), optionally override `postprocess()`.
3. Register with `@register("model", "<name>")`.
4. Provide any custom kwargs via `model_kwargs` in the config.

When the runner starts, it auto-imports all `datasets.*` and `models.*` modules, so new adapters are picked up without manual changes.

---

## 5.1 DeepGaze II-E Fine-Tuning (blocks + freeze semantics)

The fine-tuning script (`scripts/finetune_deepgaze_iie.py`) uses a **named blocks** API to control what trains.
The meaning of `freeze` depends on the `mode`:

* `mode: "train_blocks"` → **only the listed blocks are trainable; everything else is frozen**
* `mode: "freeze_blocks"` → **the listed blocks are frozen; everything else is trainable**

Available block names:
`backbone`, `saliency`, `fixation_selection`, `finalizer`, `head`, `all`

`head` is shorthand for `saliency + fixation_selection + finalizer`.

You can also optionally unfreeze the **last N backbone modules** in addition to the blocks via
`backbone_last_n_modules` (useful for staged fine-tuning).

Example (train head only, then head + last 2 backbone modules):

```yaml
training:
  stages:
    - name: "feature_extraction"
      freeze:
        mode: "train_blocks"
        blocks: ["head"]
    - name: "fine_tune"
      freeze:
        mode: "train_blocks"
        blocks: ["head"]
        backbone_last_n_modules: 2
```

### Multi-participant GT aggregation

If your dataset provides multiple ground-truth maps per image (e.g., one per participant),
the fine-tuning loader can aggregate them before training. Configure via:

```yaml
data:
  gt_aggregate: "mean"   # "mean" | "median" | "random"
  gt_resize_interp: "bilinear"  # "bilinear" | "bicubic"
  gt_cache_dir: "data/img_bin/gt_maps_mean"
```

`mean`/`median` aggregate all participant maps into a single target map.  
`random` picks one participant map per image per epoch (stochastic target).

If `gt_cache_dir` is set and `gt_aggregate: "mean"`, precompute mean maps once:

```bash
python scripts/precompute_gt_mean.py --config configs/finetune_deepgaze_iie.yaml
```

This writes `<gt_cache_dir>/<image_id>.npy`, which the fine-tuning loader will reuse.

### Weights & Biases tracking

Enable W&B logging in your fine-tune config:

```yaml
wandb:
  enabled: true
  project: "deepgaze-finetune"
  entity: null
  name: null
  group: null
  tags: ["deepgaze_iie"]
  notes: null
  dir: null
  mode: null
```

If GT map resolution differs from image resolution, maps are resized to image size
using the configured interpolation and then renormalized (sum=1).

Loss options for fine-tuning:
* `kl`: KL divergence between GT map and prediction  
* `nll_fixations`: negative log-likelihood of fixation maps

---

## 6. Visualization & Inspection

### 6.1 PNG Heatmaps & Overlays
Enable via the `heatmaps_png` section. PNGs are stored at:
```
outputs/heatmaps/<dataset>/<model>/<image_id>.png
outputs/heatmaps/<dataset>/<model>/<image_id>_overlay.png
```

### 6.2 Jupyter Notebook Viewer

Generate/update the notebook:
```bash
python scripts/create_saliency_notebook.py
```
Open `notebooks/saliency_viewer.ipynb` in JupyterLab/Notebook to:
* Browse models, datasets, and cached predictions.
* Toggle colormaps/normalization modes.
* Adjust overlay alpha (live).
* Save ad-hoc PNGs from the UI.

### 6.3 Metrics Quicklook

Each run creates per-image and aggregate CSVs. For a quick sanity check:
```bash
ls outputs/*.csv
```
or load via pandas:
```python
import pandas as pd
df = pd.read_csv("outputs/folder__center_bias.csv")
print(df.head())
```

---

## 7. Testing & Linting

* Smoke tests are provided under `tests/` (dataset/model contract checks + metric edge cases):
  ```bash
  pytest -q
  ```
* Optional formatting/linting (not enforced): install `black`, `isort`, `flake8` and run accordingly.

---

## 8. Extending / Best Practices

* **Caching** – Keep `cache_predictions: true` to avoid recomputing heavy models while iterating on metrics or visualization.
* **GPU/MPS** – Torch models auto-select `cuda` if available, otherwise CPU. Adjust `device` via `model_kwargs` if needed (e.g., `"cuda:0"`/`"mps"`).
* **Sampling** – Use `limit_images` + `predict_only` for fast, non-metric debugging. Example:
  ```yaml
  limit_images: 30
  predict_only: true
  heatmaps_png:
    enabled: true
    overlay: true
  ```
* **Dataset Adapters** – Always normalize GT maps (sum=1). Provide optional `fixations` for NSS/AUC; include `nonfix` for sAUC if available.
* **Model Outputs** – Ensure predictions are non-negative before postprocess; the default `postprocess` resizes to dataset resolution and renormalizes to sum=1.
* **Visualization** – Use the notebook viewer or PNG outputs to QA saliency qualitatively alongside quantitative metrics.

---

## 9. Troubleshooting

| Issue | Fix |
| --- | --- |
| `KeyError: "dataset 'foo' not found"` | Ensure your dataset adapter file is in `datasets/` and decorated with `@register("dataset","foo")`, and that `__init__.py` exists (already handled). |
| CSV aggregation fails (`KeyError: 'dataset'`) | Happens only if no rows were produced (e.g., `limit_images=0`). Increase limit or ensure dataset paths exist. |
| PNG alpha slider in notebook doesn’t change opacity | Regenerate the notebook after updating `scripts/create_saliency_notebook.py`; blended overlays now use OpenCV’s weighted sum. |
| PyTorch MPS warnings | Informational; on macOS you can force CPU by passing `device: "cpu"` in `model_kwargs`. |

---

## 10. Kaggle Evaluation Workflow (DeepGaze IIE)

This repo is runnable on Kaggle, with a few environment-specific details:

* Internet may be disabled by default. If you need DeepGaze from GitHub, enable internet or vendor the package.
* Kaggle paths are typically `/kaggle/input/...` for datasets and `/kaggle/working/...` for writable outputs.
* With cross-validation enabled, each fold writes its own weights to `.../fold_XX/final.pth`.

### 10.1 Setup (once per notebook)

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
pip install git+https://github.com/matthias-k/DeepGaze.git
```

### 10.2 Create the holdout split

```bash
python scripts/make_holdout_split.py \
  --root /kaggle/input/img_bin \
  --test_count 30 \
  --out_dir /kaggle/working/splits
```

### 10.3 Fine-tune with 4-fold CV

Config: `configs/finetune_deepgaze_iie_kaggle.yaml`

```bash
python scripts/finetune_deepgaze_iie.py --config configs/finetune_deepgaze_iie_kaggle.yaml
```

Outputs:
```
/kaggle/working/outputs/finetune/deepgaze_iie/fold_01/final.pth
/kaggle/working/outputs/finetune/deepgaze_iie/fold_02/final.pth
/kaggle/working/outputs/finetune/deepgaze_iie/fold_03/final.pth
/kaggle/working/outputs/finetune/deepgaze_iie/fold_04/final.pth
```

### 10.4 Evaluate (pretrained baseline + each finetuned fold)

Pretrained baseline:
```bash
python -m saliency_bench.core.runner --config configs/eval_deepgaze_iie_pretrained.yaml
```

Finetuned folds:
```bash
python -m saliency_bench.core.runner --config configs/eval_deepgaze_iie_finetuned_fold_01.yaml
python -m saliency_bench.core.runner --config configs/eval_deepgaze_iie_finetuned_fold_02.yaml
python -m saliency_bench.core.runner --config configs/eval_deepgaze_iie_finetuned_fold_03.yaml
python -m saliency_bench.core.runner --config configs/eval_deepgaze_iie_finetuned_fold_04.yaml
```

### 10.5 Aggregate fold metrics

```bash
python scripts/aggregate_fold_metrics.py
```

This writes:
```
outputs/eval/deepgaze_iie_finetuned_cv_summary.csv
```

---

Happy benchmarking! Plug in your favorite saliency datasets and models to compare them in a unified pipeline.
