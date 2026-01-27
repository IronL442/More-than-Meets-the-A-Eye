import os
import numpy as np
import tensorflow as tf
import keras
from huggingface_hub import snapshot_download

from saliency_bench.core.interfaces import SaliencyModel
from saliency_bench.core.registry import register
from saliency_bench.utils.image_ops import renorm_prob


@register("model", "msi_net_tf")
class MSINetTF(SaliencyModel):
    """
    TensorFlow/Keras adapter for MSI-Net (SavedModel) hosted on Hugging Face.
    Loads via huggingface_hub.snapshot_download + keras.layers.TFSMLayer.
    Follows the preprocessing from the model card: resize to closest of
    (320,320), (240,320), (320,240) with preserve_aspect_ratio=True, then pad.
    """

    def __init__(self, repo_id: str = "alexanderkroner/MSI-Net"):
        self.name = "msi_net_tf"
        self.repo_id = repo_id
        # Download/cached under ~/.cache/huggingface
        # Disable symlinks by default on Windows to avoid WinError 1314.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        self.hf_dir = snapshot_download(repo_id=self.repo_id)

        # Keras 3: wrap SavedModel for inference
        try:
            self.layer = keras.layers.TFSMLayer(self.hf_dir, call_endpoint="serving_default")
        except Exception:
            # Some exports use a different signature name
            self.layer = keras.layers.TFSMLayer(self.hf_dir, call_endpoint="predict")

        # context dict to carry info between preprocess and postprocess
        self._ctx = {}

    # ---------- helpers ----------
    @staticmethod
    def _best_target_shape(orig_hw):
        """
        Choose the closest among (320,320), (240,320), (320,240) by aspect ratio.
        """
        h, w = orig_hw
        ar = h / float(w)
        # reference aspect ratios
        choices = [(320, 320), (240, 320), (320, 240)]
        def ar_diff(hw):
            return abs((hw[0] / float(hw[1])) - ar)
        return min(choices, key=ar_diff)

    @staticmethod
    def _pad_to_exact(t, target_hw):
        """
        Pad H,W to the target with symmetric padding; returns padded tensor and paddings.
        t: (1,H,W,3), target_hw: (Ht,Wt)
        """
        th, tw = target_hw
        vh = th - t.shape[1]
        vw = tw - t.shape[2]
        v1, v2 = int(vh // 2), int(vh - vh // 2)
        h1, h2 = int(vw // 2), int(vw - vw // 2)
        t = tf.pad(t, [[0, 0], [v1, v2], [h1, h2], [0, 0]])
        return t, (v1, v2), (h1, h2)

    # ---------- interface methods ----------
    def preprocess(self, image_np: np.ndarray):
        """
        Input: uint8 RGB (H,W,3)
        Output: model_input tensor (1,Ht,Wt,3 float32), while stashing:
          - original_hw
          - paddings (vpad, hpad)
        """
        assert image_np.ndim == 3 and image_np.shape[2] == 3
        img = image_np.astype(np.float32)  # keep 0..255 as per HF example
        H, W = img.shape[:2]
        target_hw = self._best_target_shape((H, W))  # (Ht, Wt)

        t = tf.expand_dims(img, axis=0)  # (1,H,W,3)
        t = tf.image.resize(t, target_hw, preserve_aspect_ratio=True)
        t, vpad, hpad = self._pad_to_exact(t, target_hw)

        # stash context for postprocess
        self._ctx["orig_hw"] = (H, W)
        self._ctx["vpad"] = vpad
        self._ctx["hpad"] = hpad
        return t

    def predict(self, model_input) -> np.ndarray:
        """
        Returns saliency map in the *model space* (Ht×Wt) as float32 non-negative.
        """
        t = model_input  # (1,Ht,Wt,3)
        out = self.layer(t)  # dict or tensor depending on SavedModel signature

        # Normalize outputs into a single HxW float32 array
        if isinstance(out, dict):
            for k in ("output", "saliency", "default", "Identity", "Identity_1"):
                if k in out:
                    y = out[k]
                    break
            else:
                y = next(iter(out.values()))
        else:
            y = out

        y = tf.nn.relu(y)
        if len(y.shape) == 4:  # (1,H,W,1)
            y = y[:, :, :, 0]
        y = tf.squeeze(y, axis=0)  # (H,W)
        return y.numpy().astype(np.float32)

    def postprocess(self, pred_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """
        Crop away padding, resize back to original H×W, renormalize to sum=1.
        target_hw is ignored; we use the stashed original size for evaluation.
        """
        H_orig, W_orig = self._ctx["orig_hw"]
        vpad, hpad = self._ctx["vpad"], self._ctx["hpad"]

        t = tf.convert_to_tensor(pred_map[None, ..., None])  # (1,Ht,Wt,1)
        t = t[:, vpad[0]: t.shape[1] - vpad[1], hpad[0]: t.shape[2] - hpad[1], :]
        t = tf.image.resize(t, (H_orig, W_orig))
        out = tf.squeeze(t, axis=[0, -1]).numpy().astype(np.float32)
        return renorm_prob(out)
