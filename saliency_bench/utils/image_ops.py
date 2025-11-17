import cv2
import numpy as np


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 3 and img.shape[2] in (3, 4)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_keep_aspect(img: np.ndarray, target_short: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) == target_short:
        return img
    scale = target_short / min(h, w)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def resize_exact(img: np.ndarray, hw: tuple[int, int], interp=cv2.INTER_LINEAR) -> np.ndarray:
    H, W = hw
    return cv2.resize(img, (W, H), interpolation=interp)


def resize_map_bilinear(map_f: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    m = map_f.astype(np.float32)
    m = resize_exact(m, hw, cv2.INTER_LINEAR)
    m[m < 0] = 0
    return m


def renorm_prob(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    s = float(m.sum())
    if s < eps:
        H, W = m.shape
        return np.full((H, W), 1.0 / (H * W), dtype=np.float32)
    return (m / s).astype(np.float32)


def zscore(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(m.mean())
    sd = float(m.std())
    return (m - mu) / (sd + eps)

