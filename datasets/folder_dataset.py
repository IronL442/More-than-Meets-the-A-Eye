import glob
import os
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np

from saliency_bench.core.interfaces import SaliencyDataset
from saliency_bench.core.registry import register
from saliency_bench.utils.gt_from_fix import fixations_to_density
from saliency_bench.utils.image_ops import renorm_prob, to_rgb_uint8


@register("dataset", "folder")
class FolderDataset(SaliencyDataset):
    """
    Generic folder-based adapter.

    Expected layout (GT/fixations OPTIONAL):
      root/images/*.jpg|png
      root/gt_maps/<stem>.npy        (float32 H×W, probability map)   [optional]
      root/fixations/<stem>.npy      (uint8 H×W, 1s at fix locations) [optional]

    If neither GT nor fixations exist, a uniform GT is used so inference still works.
    """

    def __init__(self, root: str, split: str = "test", sigma_px: int = 15):
        self.name = "folder"
        self.root = root
        self.split = split
        self.sigma_px = sigma_px
        self.img_dir = os.path.join(root, "images")
        self.gt_dir = os.path.join(root, "gt_maps")
        self.fix_dir = os.path.join(root, "fixations")

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(self.img_dir, e)))
        self.img_paths = sorted(paths)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for p in self.img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = to_rgb_uint8(img)
            H, W = img.shape[:2]

            gt_path = os.path.join(self.gt_dir, stem + ".npy")
            gt: Optional[np.ndarray] = None
            if os.path.exists(gt_path):
                gt = np.load(gt_path).astype(np.float32)

            fix_path = os.path.join(self.fix_dir, stem + ".npy")
            fix: Optional[np.ndarray] = None
            if os.path.exists(fix_path):
                fix = np.load(fix_path).astype(np.uint8)
                if gt is None:
                    try:
                        gt = fixations_to_density(fix, sigma_px=self.sigma_px)
                    except Exception:
                        gt = None

            if gt is None:
                gt = np.full((H, W), 1.0 / (H * W), dtype=np.float32)

            yield {
                "image_id": stem,
                "image": img,
                "gt_map": renorm_prob(gt),
                "fixations": fix,
            }
