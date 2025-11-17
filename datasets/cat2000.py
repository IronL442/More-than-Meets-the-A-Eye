import glob
import os
import random
from typing import Any, Dict, Iterator, Optional, List, Tuple

import cv2
import numpy as np

from saliency_bench.core.interfaces import SaliencyDataset
from saliency_bench.core.registry import register
from saliency_bench.utils.gt_from_fix import fixations_to_density
from saliency_bench.utils.image_ops import renorm_prob, to_rgb_uint8


@register("dataset", "CAT2000")
class CAT2000(SaliencyDataset):
    """
    CAT2000 adapter for the official folder structure, e.g.
      root/testSet/testSet/Stimuli/<Category>/<Image>.jpg|png|jpeg

    Notes:
    - The official *testSet* does NOT include fixations/GT maps.
      We therefore yield a uniform gt_map so inference/caching works.
      Metrics depending on fixations (AUC/NSS/sAUC) will be NaN.
    - If you later place *.npy fixations or gt_maps under root/fixations or root/gt_maps
      with IDs matching the generated 'image_id', they will be used automatically.
    """

    def __init__(
        self,
        root: str = "data/CAT2000",
        split: str = "test",
        sigma_px: int = 19,
        provide_sauc_nonfix: bool = True,
    ):
        self.name = "CAT2000"
        self.split = split
        self.root = root
        self.sigma_px = sigma_px
        self.provide_sauc_nonfix = provide_sauc_nonfix

        # Official tree for stimuli
        self.stim_dir = os.path.join(root, "testSet", "testSet", "Stimuli")

        # Optional fix/gt folders (only used if you provide your own .npy files)
        self.fix_dir = os.path.join(root, "fixations")
        self.map_dir = os.path.join(root, "gt_maps")

        # Collect all images recursively by category; support common extensions
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths: List[str] = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(self.stim_dir, "*", ext)))
        paths = sorted(paths)

        # Build stable IDs that avoid collisions across categories
        # image_id pattern: "<Category>_<Stem>"
        def make_id(p: str) -> str:
            cat = os.path.basename(os.path.dirname(p))
            stem = os.path.splitext(os.path.basename(p))[0]
            return f"{cat}_{stem}"

        self.items: List[Tuple[str, str]] = [(make_id(p), p) for p in paths]
        self.ids = [iid for iid, _ in self.items]  # keep for sAUC pool logic

    # --- loaders ---

    def _load_img(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return to_rgb_uint8(img)

    def _load_fix(self, img_id: str) -> Optional[np.ndarray]:
        # If you provide your own fixations under root/fixations/<image_id>.npy
        p = os.path.join(self.fix_dir, img_id + ".npy")
        return np.load(p).astype(np.uint8) if os.path.exists(p) else None

    def _load_map(self, img_id: str) -> Optional[np.ndarray]:
        # If you provide your own continuous maps under root/gt_maps/<image_id>.npy
        p = os.path.join(self.map_dir, img_id + ".npy")
        return np.load(p).astype(np.float32) if os.path.exists(p) else None

    # --- iterator ---

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if len(self.items) == 0:
            # Nothing found under official tree; also allow flat 'images' dir as a fallback
            flat_dir = os.path.join(self.root, "images")
            for p in sorted(glob.glob(os.path.join(flat_dir, "*"))):
                img_id = os.path.splitext(os.path.basename(p))[0]
                self.items.append((img_id, p))

        # Preload pool for sAUC (only meaningful if fixations exist)
        pool = None
        if self.provide_sauc_nonfix:
            pool = [self._load_fix(i) for i in self.ids]

        for i, (img_id, path) in enumerate(self.items):
            img = self._load_img(path)
            fix = self._load_fix(img_id)
            gt = self._load_map(img_id)

            if gt is None and fix is not None:
                # Build a density map from fixations if available
                gt = fixations_to_density(fix, sigma_px=self.sigma_px)

            if gt is None:
                # CAT2000 test set has no GT; use uniform so the pipeline runs
                H, W = img.shape[:2]
                gt = np.full((H, W), 1.0 / (H * W), dtype=np.float32)

            sample = {
                "image_id": img_id,
                "image": img,
                "gt_map": renorm_prob(gt),
                "fixations": fix,
            }

            if pool is not None and fix is not None:
                # sAUC: pick non-fixations from a random other image of same shape
                nonfix = None
                for _ in range(10):
                    j = random.randrange(0, len(self.ids))
                    if j == i:
                        continue
                    cand = pool[j]
                    if cand is not None and cand.shape == fix.shape:
                        nonfix = cand
                        break
                sample["nonfix"] = nonfix

            yield sample
