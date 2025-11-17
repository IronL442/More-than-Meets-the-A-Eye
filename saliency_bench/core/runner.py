import os
from importlib import import_module
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from metrics.metrics import auc_judd, cc, kl_div, nss, sauc, sim_score
from saliency_bench.core.registry import build
from saliency_bench.utils.image_ops import renorm_prob
from saliency_bench.utils.heatmap_viz import save_heatmap_png, save_overlay_png


def _import_adapters():
    import pkgutil

    for pkg_name in ("datasets", "models"):
        pkg = import_module(pkg_name)
        for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__):
            import_module(f"{pkg_name}.{mod_name}")


def compute_metrics(pred: np.ndarray, sample: Dict[str, Any]) -> Dict[str, float]:
    gt = sample["gt_map"].astype(np.float32)
    fix = sample.get("fixations", None)
    out = {
        "AUC_Judd": auc_judd(pred, fix) if fix is not None else np.nan,
        "NSS": nss(pred, fix) if fix is not None else np.nan,
        "CC": cc(pred, gt),
        "KL": kl_div(pred, gt),
        "SIM": sim_score(pred, gt),
    }
    if "nonfix" in sample and sample["nonfix"] is not None and fix is not None:
        out["sAUC"] = sauc(pred, fix, sample["nonfix"])
    else:
        out["sAUC"] = np.nan
    return out


def run_experiment(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    limit = cfg.get("limit_images", None)
    predict_only = cfg.get("predict_only", False)
    viz_cfg = cfg.get("heatmaps_png", {})
    save_png = bool(viz_cfg.get("enabled", False))
    save_overlay = bool(viz_cfg.get("overlay", False))
    png_norm = str(viz_cfg.get("normalize", "max"))
    png_cmap = str(viz_cfg.get("colormap", "jet"))
    png_alpha = float(viz_cfg.get("alpha", 0.5))
    png_dir = str(viz_cfg.get("out_dir", "outputs/heatmaps"))

    _import_adapters()

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = cfg.get("prediction_cache_dir", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    all_rows = []
    for ds_cfg in cfg["datasets"]:
        ds = build("dataset", ds_cfg["name"], **{k: v for k, v in ds_cfg.items() if k != "name"})
        for model_name in cfg["models"]:
            model_kwargs = cfg.get("model_kwargs", {}).get(model_name, {})
            model = build("model", model_name, **model_kwargs)
            rows = []
            count = 0
            for sample in tqdm(ds, desc=f"{ds.name}:{model_name}"):
                if limit is not None and count >= limit:
                    break
                count += 1

                img = sample["image"]
                H, W = sample["gt_map"].shape
                pred_path = os.path.join(
                    pred_dir, f"{model_name}__{ds.name}__{sample['image_id']}.npy"
                )
                if cfg.get("cache_predictions", True) and os.path.exists(pred_path):
                    pred = np.load(pred_path).astype(np.float32)
                else:
                    inp = model.preprocess(img)
                    pred_model = model.predict(inp)
                    pred = model.postprocess(pred_model, (H, W))
                    pred = renorm_prob(pred)
                    if cfg.get("cache_predictions", True):
                        np.save(pred_path, pred)

                if save_png:
                    base_dir = os.path.join(png_dir, ds.name, model.name)
                    os.makedirs(base_dir, exist_ok=True)
                    png_path = os.path.join(base_dir, f"{sample['image_id']}.png")
                    save_heatmap_png(
                        png_path,
                        pred,
                        normalize=png_norm,
                        colormap=png_cmap,
                    )
                    if save_overlay:
                        overlay_path = os.path.join(
                            base_dir, f"{sample['image_id']}_overlay.png"
                        )
                        save_overlay_png(
                            overlay_path,
                            sample["image"],
                            pred,
                            alpha=png_alpha,
                            normalize=png_norm,
                            colormap=png_cmap,
                        )

                if predict_only:
                    row = {
                        "dataset": ds.name,
                        "split": getattr(ds, "split", "NA"),
                        "model": model.name,
                        "image_id": sample["image_id"],
                    }
                else:
                    metrics = compute_metrics(pred, sample)
                    row = {
                        "dataset": ds.name,
                        "split": getattr(ds, "split", "NA"),
                        "model": model.name,
                        "image_id": sample["image_id"],
                        **metrics,
                    }
                rows.append(row)

            df = pd.DataFrame(rows)
            part_csv = os.path.join(out_dir, f"{ds.name}__{model.name}.csv")
            df.to_csv(part_csv, index=False)

            if not df.empty:
                agg = (
                    df.drop(columns=["image_id"], errors="ignore")
                    .groupby(["dataset", "split", "model"])
                    .mean(numeric_only=True)
                    .reset_index()
                )
                agg_csv = os.path.join(out_dir, f"{ds.name}__{model.name}__summary.csv")
                agg.to_csv(agg_csv, index=False)
                all_rows.append(agg)

    if all_rows:
        final = pd.concat(all_rows, ignore_index=True)
        final.to_csv(os.path.join(out_dir, "ALL_SUMMARY.csv"), index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    run_experiment(args.config)
