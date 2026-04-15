import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils.config import INIT_TEMP

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, init_temp=INIT_TEMP, label_smoothing = 0.1):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))
        self.label_smoothing = label_smoothing # to prevent the model from becoming overconfident, adds a regularization term and acts as a calibration for the accuracy scores
        
    def forward(self, z_spec, z_pep):
        temp = self.log_temp.clamp(min=math.log(0.04), max=math.log(0.5)).exp()
        logits = (z_spec @ z_pep.T) / temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) + F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)) / 2
        acc = ((logits.argmax(dim=1) == labels).float().mean() + (logits.argmax(dim=0) == labels).float().mean()) / 2 # COMPUTE ACC ROW-WISE (SPEC -> PEP) AND COLUMN-WISE (PEP -> SPEC)
        return loss, acc
