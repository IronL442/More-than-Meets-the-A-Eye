import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["cells"] = []


def md(s):
    return nbf.v4.new_markdown_cell(s)


def py(s):
    return nbf.v4.new_code_cell(s)


nb["cells"].append(
    md(
        "# Saliency Viewer\n"
        "Browse cached saliency predictions (NPY heatmaps), visualize as PNG heatmaps or overlays, and optionally inspect metrics.\n\n"
        "**Assumptions**\n"
        "- Predictions are stored under `predictions/{model}/{dataset}/{image_id}.npy`.\n"
        "- Original images are provided by the dataset adapter; for CAT2000 test set, you can point to the Stimuli folder.\n"
        "- PNG saving is optional; this notebook is primarily for interactive exploration.\n"
    )
)

nb["cells"].append(
    py(
        """# --- imports ---
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output
import cv2

# --- user-configurable defaults ---
PRED_DIR = "predictions"              # where .npy heatmaps are cached
DATASETS_IMG_ROOTS = {
    # Adjust per dataset. For CAT2000 official test set:
    "CAT2000": "data/CAT2000/testSet/testSet/Stimuli",
    # If you have a flat images folder for another dataset:
    # "folder": "data/yourset/images",
}
DEFAULT_DATASET = "CAT2000"

# Colormap options (matplotlib)
COLORMAPS = ["gray", "jet", "turbo", "hot", "viridis", "inferno", "magma", "plasma"]

# Normalize modes for saliency display
NORM_MODES = ["max", "sum", "none"]

# If your metrics CSVs exist (runner outputs), we can load them
METRICS_DIR = "outputs"
"""
    )
)

nb["cells"].append(
    py(
        """# --- utilities ---
def list_models(pred_root=PRED_DIR):
    if not os.path.isdir(pred_root):
        return []
    return sorted([d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))])

def list_datasets_for_model(model, pred_root=PRED_DIR):
    path = os.path.join(pred_root, model)
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def list_image_ids(model, dataset, pred_root=PRED_DIR):
    p = os.path.join(pred_root, model, dataset)
    if not os.path.isdir(p):
        return []
    files = sorted(glob.glob(os.path.join(p, "*.npy")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]

def load_pred(model, dataset, image_id, pred_root=PRED_DIR):
    f = os.path.join(pred_root, model, dataset, image_id + ".npy")
    m = np.load(f).astype(np.float32)
    return m

def find_image_file(dataset, image_id):
    # try CAT2000 official first: Stimuli/<Category>/<Filename>.* where image_id might be 'Category_Filename'
    stim_root = DATASETS_IMG_ROOTS.get(dataset)
    if stim_root and os.path.isdir(stim_root):
        # attempt parse Category_Filename
        if "_" in image_id:
            cat, stem = image_id.split("_", 1)
            for ext in [".jpg",".jpeg",".png",".bmp"]:
                p = os.path.join(stim_root, cat, stem + ext)
                if os.path.exists(p):
                    return p
        # fallback: recursive find by stem
        stem = image_id
        for ext in [".jpg",".jpeg",".png",".bmp"]:
            cand = glob.glob(os.path.join(stim_root, "*", stem + ext))
            if cand:
                return cand[0]

    # fallback flat images dir: data/<dataset>/images
    flat = os.path.join("data", dataset, "images", image_id + ".jpg")
    if os.path.exists(flat): return flat
    flatp = os.path.join("data", dataset, "images", image_id + ".png")
    if os.path.exists(flatp): return flatp
    return None

def normalize_map(m, mode="max"):
    m = m.astype(np.float32)
    if mode == "max":
        vmax = float(m.max())
        if vmax > 0: m = m / vmax
    elif mode == "sum":
        s = float(m.sum())
        if s > 0:
            m = m / s
            vmax = float(m.max())
            if vmax > 0: m = m / vmax
    m = np.clip(m, 0, 1)
    return m

def overlay_rgb(image_rgb, sal, alpha=0.5, cmap="jet", norm="max"):
    if image_rgb is None:
        return None
    sal = normalize_map(sal, norm)
    sal_u8 = (sal * 255).round().astype(np.uint8)
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO if hasattr(cv2, "COLORMAP_INFERNO") else cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA if hasattr(cv2, "COLORMAP_MAGMA") else cv2.COLORMAP_JET,
        "plasma": cv2.COLORMAP_PLASMA if hasattr(cv2, "COLORMAP_PLASMA") else cv2.COLORMAP_JET,
        "gray": cv2.COLORMAP_BONE,
    }
    cm_code = cmap_map.get(cmap.lower(), cv2.COLORMAP_JET)
    heat_bgr = cv2.applyColorMap(sal_u8, cm_code)
    img_bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = 1.0 - alpha
    blended = cv2.addWeighted(img_bgr, beta, heat_bgr, alpha, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""
    )
)

nb["cells"].append(
    py(
        """# --- widgets ---
models = list_models()
w_model = W.Dropdown(options=models, description="Model:", disabled=False)
w_dataset = W.Dropdown(options=[], description="Dataset:", disabled=False)
w_image = W.Dropdown(options=[], description="Image ID:", disabled=False)

w_cmap = W.Dropdown(options=COLORMAPS, value="jet", description="Colormap:")
w_norm = W.Dropdown(options=NORM_MODES, value="max", description="Normalize:")
w_alpha = W.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="Overlay α:")
w_show_overlay = W.Checkbox(value=True, description="Show overlay")
w_save_png = W.Checkbox(value=False, description="Save PNGs on save click")
w_out_dir = W.Text(value="outputs/notebook_exports", description="Out dir:")
w_refresh = W.Button(description="Refresh lists", button_style="")
w_prev = W.Button(description="<< Prev", button_style="")
w_next = W.Button(description="Next >>", button_style="")
w_save = W.Button(description="Save current", button_style="info")

left = W.VBox([w_model, w_dataset, w_image, w_cmap, w_norm, w_alpha, w_show_overlay, w_save_png, w_out_dir,
               W.HBox([w_prev, w_next, w_refresh, w_save])])
display(left)

out = W.Output()
display(out)

def refresh_datasets(*_):
    dss = list_datasets_for_model(w_model.value) if w_model.value else []
    w_dataset.options = dss
    if DEFAULT_DATASET in dss:
        w_dataset.value = DEFAULT_DATASET
    refresh_images()

def refresh_images(*_):
    ids = list_image_ids(w_model.value, w_dataset.value) if (w_model.value and w_dataset.value) else []
    w_image.options = ids
    if ids:
        w_image.value = ids[0]
    render()

def step_image(delta):
    ids = list(w_image.options)
    if not ids: return
    try:
        idx = ids.index(w_image.value)
    except ValueError:
        idx = 0
    idx = (idx + delta) % len(ids)
    w_image.value = ids[idx]
    render()

def save_current(pred, img, base_out):
    os.makedirs(base_out, exist_ok=True)
    image_id = w_image.value
    # heatmap only (normalized, colored)
    plt.imsave(os.path.join(base_out, f"{image_id}_heatmap.png"),
               normalize_map(pred, w_norm.value), cmap=w_cmap.value)
    if img is not None:
        ov = overlay_rgb(img, pred, alpha=w_alpha.value, cmap=w_cmap.value, norm=w_norm.value)
        plt.imsave(os.path.join(base_out, f"{image_id}_overlay.png"), ov)

def render(*_):
    out.clear_output(wait=True)
    if not (w_model.value and w_dataset.value and w_image.value):
        with out: print("Select model, dataset, and image.")
        return
    try:
        pred = load_pred(w_model.value, w_dataset.value, w_image.value)
    except Exception as e:
        with out: print("Failed to load prediction:", e)
        return

    img_path = find_image_file(w_dataset.value, w_image.value)
    img = imread_rgb(img_path) if img_path else None

    with out:
        clear_output(wait=True)
        ncols = 2 if (w_show_overlay.value and img is not None) else 1
        fig, axes = plt.subplots(1, ncols, figsize=(10 if ncols==1 else 16, 6))

        if ncols == 1:
            ax = axes
            ax.set_title("Heatmap")
            ax.imshow(normalize_map(pred, w_norm.value), cmap=w_cmap.value)
            ax.axis("off")
        else:
            ax0, ax1 = axes
            ax0.set_title("Heatmap")
            ax0.imshow(normalize_map(pred, w_norm.value), cmap=w_cmap.value)
            ax0.axis("off")

            ax1.set_title("Overlay")
            ov = overlay_rgb(img, pred, alpha=w_alpha.value, cmap=w_cmap.value, norm=w_norm.value)
            ax1.imshow(ov)
            ax1.axis("off")
        plt.show()

def on_refresh_clicked(_): refresh_datasets()
def on_prev_clicked(_): step_image(-1)
def on_next_clicked(_): step_image(+1)
def on_save_clicked(_):
    if not (w_model.value and w_dataset.value and w_image.value): return
    pred = load_pred(w_model.value, w_dataset.value, w_image.value)
    img_path = find_image_file(w_dataset.value, w_image.value)
    img = imread_rgb(img_path) if img_path else None
    base_out = os.path.join(w_out_dir.value, w_dataset.value, w_model.value)
    save_current(pred, img, base_out)
    if w_save_png.value:
        print(f"Saved PNGs to: {base_out}")

w_model.observe(lambda ch: refresh_datasets(), names="value")
w_dataset.observe(lambda ch: refresh_images(), names="value")
w_image.observe(lambda ch: render(), names="value")
w_cmap.observe(lambda ch: render(), names="value")
w_norm.observe(lambda ch: render(), names="value")
w_alpha.observe(lambda ch: render(), names="value")
w_show_overlay.observe(lambda ch: render(), names="value")

w_refresh.on_click(on_refresh_clicked)
w_prev.on_click(on_prev_clicked)
w_next.on_click(on_next_clicked)
w_save.on_click(on_save_clicked)

# initial population
refresh_datasets()
"""
    )
)

nb["cells"].append(
    md(
        "## (Optional) Metrics quicklook\n"
        "If you ran with metrics enabled, this cell will try to load summary CSVs from `outputs/*summary.csv` and display them."
    )
)
nb["cells"].append(
    py(
        """import glob, pandas as pd
from IPython.display import display

def load_summaries(outputs_dir=METRICS_DIR):
    csvs = glob.glob(os.path.join(outputs_dir, "*__summary.csv"))
    dfs = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
            df["__file"] = os.path.basename(c)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        print("No summary CSVs found.")
        return None
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

summ = load_summaries()
if summ is not None:
    display(summ.head(20))
"""
    )
)

os.makedirs("notebooks", exist_ok=True)
out_path = "notebooks/saliency_viewer.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path}")

print("Created scripts/create_saliency_notebook.py")
