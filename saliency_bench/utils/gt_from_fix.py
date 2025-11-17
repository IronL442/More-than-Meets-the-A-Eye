import cv2
import numpy as np

from saliency_bench.utils.image_ops import renorm_prob


def fixations_to_density(fix_bin: np.ndarray, sigma_px: int = 19) -> np.ndarray:
    if fix_bin is None:
        raise ValueError("fix_bin is None")
    k = int(max(3, sigma_px * 6) | 1)
    m = cv2.GaussianBlur(fix_bin.astype(np.float32), (k, k), sigma_px)
    return renorm_prob(m)

