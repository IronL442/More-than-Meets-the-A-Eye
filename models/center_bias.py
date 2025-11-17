import numpy as np

from saliency_bench.core.interfaces import SaliencyModel
from saliency_bench.core.registry import register


@register("model", "center_bias")
class CenterBias(SaliencyModel):
    def __init__(self, device: str = "cpu", requires_size=(224, 224), sigma_ratio: float = 0.25):
        self.name = "center_bias"
        self.device = device
        self.requires_size = requires_size
        self.sigma_ratio = sigma_ratio

    def preprocess(self, image_np: np.ndarray):
        Hm, Wm = self.requires_size
        return (Hm, Wm)

    def predict(self, model_input) -> np.ndarray:
        Hm, Wm = model_input
        y = np.linspace(-1, 1, Hm)[:, None]
        x = np.linspace(-1, 1, Wm)[None, :]
        r2 = x**2 + y**2
        sigma2 = self.sigma_ratio**2
        m = np.exp(-0.5 * r2 / (sigma2 + 1e-8)).astype(np.float32)
        return m
