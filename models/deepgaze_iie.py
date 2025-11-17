import os
import numpy as np
import torch

from scipy.ndimage import zoom
from scipy.special import logsumexp

import deepgaze_pytorch

from saliency_bench.core.interfaces import SaliencyModel
from saliency_bench.core.registry import register
from saliency_bench.utils.image_ops import renorm_prob


@register("model", "deepgaze_iie")
class DeepGazeIIEAdapter(SaliencyModel):
    """
    DeepGaze II-E adapter that implements exactly the usage from the official repo:

        image = ...
        centerbias_template = np.load('centerbias_mit1003.npy')  # or zeros
        centerbias = zoom(..., order=0, mode='nearest')
        centerbias -= logsumexp(centerbias)
        image_tensor = torch.tensor([image.transpose(2, 0, 1)])
        centerbias_tensor = torch.tensor([centerbias])
        log_density_prediction = model(image_tensor, centerbias_tensor)

    - Input from dataset: uint8 RGB (H,W,3)
    - Output to benchmark: saliency map (H,W) float32, sum~1
    """

    def __init__(
        self,
        centerbias_path: str = "data/centerbias/centerbias_mit1003.npy",
        device: str = None,
        use_uniform_centerbias: bool = False,
    ):
        self.name = "deepgaze_iie"

        # device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # DeepGaze II-E model with pretrained weights
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(self.device)
        self.model.eval()

        # centerbias template (2D array)
        if use_uniform_centerbias:
            # exact suggestion from the README: zeros -> uniform prior
            cb = np.zeros((1024, 1024), dtype=np.float32)
        else:
            if not os.path.exists(centerbias_path):
                raise FileNotFoundError(
                    f"Centerbias file not found at {centerbias_path}. "
                    "Download it from the DeepGaze repo releases."
                )
            cb = np.load(centerbias_path).astype(np.float32)  # [H0, W0]

        self.centerbias_template = cb  # keep as numpy; we will zoom per image

    # ---------- SaliencyModel API ----------

    def preprocess(self, image_np: np.ndarray):
        """
        Dataset gives uint8 RGB (H,W,3).

        We:
          - store the original image (for shape)
          - build the centerbias for this H,W (zoom + renorm)
          - create tensors exactly as in the README example:
                image_tensor:    [1,3,H,W]
                centerbias_tensor:[1,H,W]
        """
        assert image_np.ndim == 3 and image_np.shape[2] == 3
        img = image_np.astype(np.float32)  # [H,W,3] in [0..255]
        H, W = img.shape[:2]

        # --- build centerbias for this image size (exact repo logic) ---
        cb_tmpl = self.centerbias_template  # [H0, W0]
        zoom_factors = (
            H / cb_tmpl.shape[0],
            W / cb_tmpl.shape[1],
        )
        centerbias = zoom(
            cb_tmpl,
            zoom_factors,
            order=0,
            mode="nearest",
        )  # [H,W]

        # renormalize log-density
        # (this is exactly `centerbias -= logsumexp(centerbias)`)
        centerbias = centerbias - logsumexp(centerbias)

        # --- build tensors as in the example ---
        # image_tensor: [1,3,H,W]
        image_tensor = torch.tensor(
            [img.transpose(2, 0, 1)],  # [3,H,W] -> [1,3,H,W]
            dtype=torch.float32,
            device=self.device,
        )

        # centerbias_tensor: [1,H,W]
        centerbias_tensor = torch.tensor(
            [centerbias],  # [H,W] -> [1,H,W]
            dtype=torch.float32,
            device=self.device,
        )

        return (image_tensor, centerbias_tensor)

    def predict(self, model_input) -> np.ndarray:
        """
        Run DeepGazeIIE and return a saliency map in model space (H,W).
        DeepGaze outputs log-density; we exponentiate to get density.
        """
        image_tensor, centerbias_tensor = model_input
        with torch.no_grad():
            log_pred = self.model(image_tensor, centerbias_tensor)  # [1,1,H,W]
            density = torch.exp(log_pred[0, 0])  # [H,W]
        return density.cpu().numpy().astype(np.float32)

    def postprocess(self, pred_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """
        DeepGaze outputs are already at original H×W, so we only renormalize
        to obtain a proper probability map (sum=1).
        """
        return renorm_prob(pred_map)
