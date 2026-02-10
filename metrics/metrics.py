from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


def _normalize_prob(
    m: np.ndarray,
    *,
    eps: float = 0.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    arr = np.asarray(m, dtype=dtype)
    arr = np.clip(arr, 0.0, None)
    if eps > 0.0:
        arr = arr + np.asarray(eps, dtype=dtype)
    s = float(arr.sum())
    if s <= 0.0:
        h, w = arr.shape
        return np.full((h, w), 1.0 / (h * w), dtype=dtype)
    return (arr / s).astype(dtype, copy=False)


def cc(pred: np.ndarray, gt_map: np.ndarray, eps: float = 1e-8) -> float:
    p = _normalize_prob(pred, dtype=np.float32)
    g = _normalize_prob(gt_map, dtype=np.float32)
    p = p - p.mean()
    g = g - g.mean()
    denom = float(p.std() * g.std() + eps)
    return float((p * g).mean() / denom)


def kl_div(
    pred: np.ndarray,
    gt_map: np.ndarray,
    *,
    eps: float = 1e-7,
    use_float64: bool = True,
) -> float:
    # KL(GT || Pred)
    dtype = np.float64 if use_float64 else np.float32
    p = _normalize_prob(pred, eps=eps, dtype=dtype)
    g = _normalize_prob(gt_map, eps=eps, dtype=dtype)
    return float(np.sum(g * (np.log(g) - np.log(p)), dtype=dtype))


def _emd_1d_from_marginals(p: np.ndarray, g: np.ndarray) -> float:
    p_x = p.sum(axis=0)
    g_x = g.sum(axis=0)
    p_y = p.sum(axis=1)
    g_y = g.sum(axis=1)
    cdf_dx = np.abs(np.cumsum(p_x) - np.cumsum(g_x)).sum()
    cdf_dy = np.abs(np.cumsum(p_y) - np.cumsum(g_y)).sum()
    return float(np.sqrt(cdf_dx * cdf_dx + cdf_dy * cdf_dy))


def _resize_map(m: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    arr = np.asarray(m, dtype=np.float32)
    if arr.shape == (h, w):
        return arr
    if cv2 is not None:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)

    # Fallback if OpenCV is unavailable: nearest-grid resampling.
    h0, w0 = arr.shape
    ys = np.linspace(0, h0 - 1, h).round().astype(np.int32)
    xs = np.linspace(0, w0 - 1, w).round().astype(np.int32)
    return arr[np.ix_(ys, xs)].astype(np.float32)


def emd_wasserstein(
    pred: np.ndarray,
    gt_map: np.ndarray,
    *,
    downsample_hw: tuple[int, int] = (40, 40),
    eps: float = 1e-7,
) -> float:
    h, w = downsample_hw
    p = _resize_map(pred, (h, w))
    g = _resize_map(gt_map, (h, w))
    p = _normalize_prob(p, eps=eps, dtype=np.float32)
    g = _normalize_prob(g, eps=eps, dtype=np.float32)

    ys, xs = np.indices((h, w), dtype=np.float32)
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    sig_p = np.concatenate([p.reshape(-1, 1), coords], axis=1).astype(np.float32)
    sig_g = np.concatenate([g.reshape(-1, 1), coords], axis=1).astype(np.float32)

    if cv2 is not None and hasattr(cv2, "EMD"):
        dist, _, _ = cv2.EMD(sig_p, sig_g, cv2.DIST_L2)
        return float(dist)
    return _emd_1d_from_marginals(p, g)
