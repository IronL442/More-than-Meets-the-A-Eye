import argparse
import os
import sys
from typing import List

import cv2
import numpy as np
import yaml

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from saliency_bench.utils.image_ops import renorm_prob


def _load_image_paths(root: str) -> List[str]:
    import glob
    img_dir = os.path.join(root, "images")
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(sorted([p for p in glob.glob(os.path.join(img_dir, ext))]))
    return paths


def _find_gt_paths(gt_dir: str, stem: str) -> List[str]:
    if not os.path.isdir(gt_dir):
        return []
    import glob
    exts = (".npy", ".png", ".jpg", ".jpeg", ".bmp")
    direct = os.path.join(gt_dir, stem + ".npy")
    if os.path.exists(direct):
        return [direct]
    paths: List[str] = []
    for ext in exts:
        pattern = os.path.join(gt_dir, f"*_{stem}{ext}")
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def _resize_map(arr: np.ndarray, H: int, W: int, interp: str) -> np.ndarray:
    if arr.shape == (H, W):
        return arr.astype(np.float32)
    if interp == "bilinear":
        interp_flag = cv2.INTER_LINEAR
    elif interp == "bicubic":
        interp_flag = cv2.INTER_CUBIC
    else:
        raise ValueError(f"Unknown gt_resize_interp: {interp}")
    return cv2.resize(arr.astype(np.float32), (W, H), interpolation=interp_flag)


def _load_gt_map(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read GT map: {path}")
    return img.astype(np.float32)


def precompute(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "data/img_bin")
    gt_dir = os.path.join(root, "gt_maps")
    gt_cache_dir = data_cfg.get("gt_cache_dir", None)
    gt_resize_interp = str(data_cfg.get("gt_resize_interp", "bilinear")).lower()

    if not gt_cache_dir:
        raise ValueError("data.gt_cache_dir is required for precompute.")

    os.makedirs(gt_cache_dir, exist_ok=True)

    img_paths = _load_image_paths(root)
    if not img_paths:
        raise FileNotFoundError(f"No images found under {root}/images")

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_paths = _find_gt_paths(gt_dir, stem)
        if not gt_paths:
            raise FileNotFoundError(f"No GT maps found for {stem} under {gt_dir}")

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        H, W = img_bgr.shape[:2]

        maps = []
        for path in gt_paths:
            arr = _load_gt_map(path)
            arr = _resize_map(arr, H, W, gt_resize_interp)
            arr = np.clip(arr, 0.0, None)
            maps.append(arr)

        mean_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
        mean_map = renorm_prob(mean_map)
        entropy = -np.sum(mean_map * np.log(mean_map + 1e-12))
        print(f"{stem}: entropy={entropy:.3f}")

        np.save(os.path.join(gt_cache_dir, stem + ".npy"), mean_map)

    print(f"Saved mean GT maps to: {gt_cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    precompute(args.config)
