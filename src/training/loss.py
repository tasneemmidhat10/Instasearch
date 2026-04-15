import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils.config import INIT_TEMP

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, init_temp=INIT_TEMP):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(self, z_spec, z_pep):
        temp = self.log_temp.exp()
        logits = (z_spec @ z_pep.T) / temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc