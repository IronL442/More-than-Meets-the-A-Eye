import os
import numpy as np
import torch
import torch.nn.functional as F
 
import deepgaze_pytorch
 
from saliency_bench.core.interfaces import SaliencyModel
from saliency_bench.core.registry import register
from saliency_bench.utils.image_ops import renorm_prob
 
 
@register("model", "deepgaze_iie")
class DeepGazeIIEAdapter(SaliencyModel):
    """
    Adapter for DeepGaze II-E (deepgaze_pytorch.DeepGazeIIE).
 
    - Input:  uint8 RGB image (H,W,3) from dataset
    - Output: saliency map (H,W) as float32, non-negative, sum ~ 1
 
    Uses the official MIT1003 centerbias template, resized to the image size.
    """
 
    def __init__(
        self,
        centerbias_path: str = "data/centerbias/centerbias_mit1003.npy",
        device: str = None,
    ):
        self.name = "deepgaze_iie"
 
        # device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
 
        # load DG-IIE model with pretrained weights
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(self.device)
        self.model.eval()
 
        # load centerbias template (log-density)
        if not os.path.exists(centerbias_path):
            raise FileNotFoundError(
                f"Centerbias file not found at {centerbias_path}. "
                "Download it from the DeepGaze repo releases."
            )
        cb = np.load(centerbias_path).astype(np.float32)  # log-density
        # store as [1,1,H0,W0] tensor on device
        self.cb_template = torch.from_numpy(cb)[None].to(self.device)
 
    def _make_centerbias(self, H: int, W: int) -> torch.Tensor:

        """

        Resize the template log centerbias to (H,W) and renormalize.
 
        DeepGaze expects centerbias as [B,H,W], not [B,1,H,W].

        Returns: tensor [1,H,W] on self.device.

        """

        # cb_template: [1,H0,W0]

        cb = self.cb_template  # [1,H0,W0]

        # interpolate in log-space; use unsqueeze to match [B,1,H0,W0] for F.interpolate

        cb_4d = cb[:, None, ...]                         # [1,1,H0,W0]

        cb_resized = F.interpolate(

            cb_4d, size=(H, W), mode="bilinear", align_corners=False

        )                                               # [1,1,H,W]

        cb_resized = cb_resized[:, 0, :, :]             # [1,H,W]
 
        # renormalize log-density: subtract log-sum-exp over spatial dims

        flat = cb_resized.view(1, -1)                   # [1, H*W]

        logZ = torch.logsumexp(flat, dim=1, keepdim=True)  # [1,1]

        cb_resized = cb_resized - logZ.view(1, 1, 1)    # broadcast to [1,H,W]
 
        return cb_resized  
 
    
    # ---- SaliencyModel API ----
    def preprocess(self, image_np: np.ndarray):
        """
        Dataset gives uint8 RGB (H,W,3).
        We:
          - normalize to [0,1]
          - convert to tensor [1,3,H,W]
          - build matching centerbias [1,1,H,W]
        """
        assert image_np.ndim == 3 and image_np.shape[2] == 3
        img = image_np.astype(np.float32) / 255.0
        H, W = img.shape[:2]
        img_t = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)  # [1,3,H,W]
        cb_t = self._make_centerbias(H, W)  # [1,1,H,W]
        # store H,W if needed later
        self._last_hw = (H, W)
        return (img_t, cb_t)
 
    def predict(self, model_input) -> np.ndarray:
        """
        Run DeepGazeIIE and return a saliency map in model space (H,W).
 
        We exponentiate the predicted log-density to get density values.
        """
        img_t, cb_t = model_input
        with torch.no_grad():
            log_pred = self.model(img_t, cb_t)  # [1,1,H,W]
            density = torch.exp(log_pred[0, 0])  # [H,W]
        return density.cpu().numpy().astype(np.float32)
 
    def postprocess(self, pred_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """
        DeepGaze outputs already match the original H×W, so we just renormalize
        to obtain a proper probability map (sum=1).
        """
        return renorm_prob(pred_map)