from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np


class SaliencyModel(ABC):
    name: str
    requires_size: Optional[Tuple[int, int]] = None  # (H, W) or None
    device: str = "cpu"

    @abstractmethod
    def preprocess(self, image_np: np.ndarray) -> Any:
        ...

    @abstractmethod
    def predict(self, model_input: Any) -> np.ndarray:
        """Returns (H_m, W_m) float32 map"""
        ...

    def postprocess(self, pred_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        from saliency_bench.utils.image_ops import renorm_prob, resize_map_bilinear

        out = resize_map_bilinear(pred_map, target_hw)
        return renorm_prob(out)


class SaliencyDataset(ABC):
    name: str
    split: str

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields dict with keys:
        - image_id: str
        - image: np.ndarray(H, W, 3) uint8 RGB
        - gt_map: np.ndarray(H, W) float32, sum=1
        - fixations: Optional[np.ndarray(H, W)) binary {0,1}
        """
        ...

