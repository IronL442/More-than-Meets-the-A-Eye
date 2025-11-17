import cv2
import numpy as np

from saliency_bench.core.interfaces import SaliencyModel
from saliency_bench.core.registry import register


@register("model", "blur_baseline")
class BlurBaseline(SaliencyModel):
    def __init__(self, device: str = "cpu", requires_size=None, ksize: int = 31):
        self.name = "blur_baseline"
        self.device = device
        self.requires_size = requires_size
        self.ksize = ksize

    def preprocess(self, image_np: np.ndarray):
        img = image_np.astype(np.float32) / 255.0
        lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return lum

    def predict(self, model_input) -> np.ndarray:
        k = self.ksize | 1
        return cv2.GaussianBlur(model_input, (k, k), 0).astype(np.float32)

