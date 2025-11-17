import os
from typing import Any, Dict, Iterator

import cv2
import numpy as np
import pandas as pd

from saliency_bench.core.interfaces import SaliencyDataset
from saliency_bench.core.registry import register
from saliency_bench.utils.gt_from_fix import fixations_to_density
from saliency_bench.utils.image_ops import renorm_prob, to_rgb_uint8


@register("dataset", "csv_indexed")
class CSVIndexedDataset(SaliencyDataset):
    def __init__(self, index_csv: str, split: str = "test", sigma_px: int = 15):
        self.name = "csv_indexed"
        self.split = split
        self.sigma_px = sigma_px
        self.df = pd.read_csv(index_csv)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _, row in self.df.iterrows():
            img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = to_rgb_uint8(img)

            fix = None
            gt = None
            fix_path = row.get("fix_path", None)
            if isinstance(fix_path, str) and os.path.exists(fix_path):
                fix = np.load(fix_path).astype(np.uint8)
                gt = fixations_to_density(fix, sigma_px=self.sigma_px)
            map_path = row.get("map_path", None)
            if isinstance(map_path, str) and os.path.exists(map_path):
                gt = np.load(map_path).astype(np.float32)
            if gt is None:
                H, W = img.shape[:2]
                gt = np.full((H, W), 1.0 / (H * W), dtype=np.float32)

            yield {
                "image_id": str(row["image_id"]),
                "image": img,
                "gt_map": renorm_prob(gt),
                "fixations": fix,
            }

