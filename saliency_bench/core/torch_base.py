import numpy as np
import torch
from typing import Optional, Tuple

from saliency_bench.core.interfaces import SaliencyModel
from saliency_bench.utils.image_ops import resize_exact

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)  # C,H,W
    return x


class TorchSaliencyModel(SaliencyModel):
    def __init__(
        self,
        device: Optional[str] = None,
        requires_size: Optional[Tuple[int, int]] = None,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.requires_size = requires_size
        self.mean = torch.tensor(mean)[:, None, None].to(self.device)
        self.std = torch.tensor(std)[:, None, None].to(self.device)

    def preprocess(self, image_np: np.ndarray) -> torch.Tensor:
        Hm, Wm = self.requires_size if self.requires_size else image_np.shape[:2]
        img_resized = resize_exact(image_np, (Hm, Wm))
        x = _to_tensor(img_resized).to(self.device)
        x = (x - self.mean) / self.std
        return x.unsqueeze(0)  # 1,C,H,W

    @torch.no_grad()
    def predict(self, model_input: torch.Tensor) -> np.ndarray:
        self.net.eval()
        y = self.net(model_input)  # 1,1,Hm,Wm or 1,Hm,Wm
        if y.ndim == 4:
            y = y[:, 0]
        y = torch.relu(y)  # non-negative map
        return y.squeeze(0).float().cpu().numpy()

