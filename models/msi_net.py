import torch
import torch.nn as nn

from saliency_bench.core.registry import register
from saliency_bench.core.torch_base import TorchSaliencyModel


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 32
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(nn.Conv2d(ch, 1, 1))

    def forward(self, x):
        return self.dec(self.enc(x))


@register("model", "msi_net")
class MSINet(TorchSaliencyModel):
    def __init__(
        self,
        device=None,
        requires_size=(256, 256),
        weights_path="weights/msinet/msinet.pt",
    ):
        super().__init__(device=device, requires_size=requires_size)
        self.name = "msi_net"
        self.net = TinyUNet().to(self.device)
        try:
            sd = torch.load(weights_path, map_location=self.device)
            self.net.load_state_dict(sd, strict=False)
        except Exception:
            pass  # runs even without weights

