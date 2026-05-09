import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import INIT_TEMP


class CLIPContrastiveLoss(nn.Module):
    """Symmetric InfoNCE with a learnable temperature.

    Returns (loss, acc, None) — third slot is reserved for diagnostics so the
    contract matches AlignUniformLoss and the training loop can unpack
    uniformly.
    """

    def __init__(self, init_temp: float = INIT_TEMP, label_smoothing: float = 0.1):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))
        self.label_smoothing = label_smoothing

    def forward(self, z_spec, z_pep, z_decoy=None, **_):
        temp = self.log_temp.clamp(min=math.log(0.04), max=math.log(0.5)).exp()
        logits = (z_spec @ z_pep.T) / temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)) / 2
        acc = ((logits.argmax(dim=1) == labels).float().mean() +
               (logits.argmax(dim=0) == labels).float().mean()) / 2
        return loss, acc, None
