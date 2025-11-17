import torch
import torch.nn as nn

from saliency_bench.core.registry import register
from saliency_bench.core.torch_base import TorchSaliencyModel


class DummyDeepGaze(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 32
        self.body = nn.Sequential(
            nn.Conv2d(3, ch, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, x):
        return self.body(x)


@register("model", "deepgaze_v3")
class DeepGazeV3(TorchSaliencyModel):
    def __init__(
        self,
        device=None,
        requires_size=(384, 384),
        weights_path="weights/deepgaze_v3/dg3.pt",
    ):
        super().__init__(device=device, requires_size=requires_size)
        self.name = "deepgaze_v3"
        self.net = DummyDeepGaze().to(self.device)
        try:
            sd = torch.load(weights_path, map_location=self.device)
            self.net.load_state_dict(sd, strict=False)
        except Exception:
            pass

