import os

import cv2
import numpy as np


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _normalize_map(m: np.ndarray, mode: str = "max") -> np.ndarray:
    """
    Normalize saliency map (float32 HxW) into uint8 [0..255].
    mode: "max" (divide by max), "sum" (divide by sum * H*W), or "none".
    """
    m = np.asarray(m, dtype=np.float32)
    if mode == "max":
        vmax = float(m.max())
        if vmax > 0:
            m = m / vmax
    elif mode == "sum":
        s = float(m.sum())
        if s > 0:
            m = m / s
            vmax = float(m.max())
            if vmax > 0:
                m = m / vmax
    m = np.clip(m, 0.0, 1.0)
    m = (m * 255.0).round().astype(np.uint8)
    return m


def save_heatmap_png(
    out_path: str,
    saliency_map: np.ndarray,
    normalize: str = "max",
    colormap: str = "gray",
):
    _ensure_dir(out_path)
    m8 = _normalize_map(saliency_map, mode=normalize)

    if colormap == "gray":
        cv2.imwrite(out_path, m8)
        return

    cmap_dict = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
    }
    cm_code = cmap_dict.get(colormap.lower(), cv2.COLORMAP_JET)
    color_bgr = cv2.applyColorMap(m8, cm_code)
    cv2.imwrite(out_path, color_bgr)


def save_overlay_png(
    out_path: str,
    image_rgb_uint8: np.ndarray,
    saliency_map: np.ndarray,
    alpha: float = 0.5,
    normalize: str = "max",
    colormap: str = "jet",
):
    _ensure_dir(out_path)
    m8 = _normalize_map(saliency_map, mode=normalize)

    cmap_dict = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
    }
    cm_code = cmap_dict.get(colormap.lower(), cv2.COLORMAP_JET)
    heat_bgr = cv2.applyColorMap(m8, cm_code)

    img_bgr = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = 1.0 - alpha
    blended = cv2.addWeighted(img_bgr, beta, heat_bgr, alpha, 0.0)

    cv2.imwrite(out_path, blended)
