"""Spectrum encoder that wraps a pretrained, frozen InstaNovo backbone.

Lifted from `dual_encoders_alignmentand_UniformityLoss_withPTMs.ipynb`.
The backbone is held as a full `InstaNovo` module so that
`instanovo._encoder(x, precursors, x_mask)` is callable directly.

Note: requires `precursors` shaped (B, 3) = [precursor_mass, precursor_charge,
precursor_mz] — the InstaNovo convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import EMBED_DIM, DROPOUT


class InstaSearchSpectrumEncoder(nn.Module):
    def __init__(self, instanovo_model, d_instanovo: int,
                 embed_dim: int = EMBED_DIM, freeze_encoder: bool = True,
                 pool_mode: str = "mean", dropout: float = DROPOUT):
        super().__init__()
        self.instanovo = instanovo_model
        self.freeze_encoder = freeze_encoder
        self.pool_mode = pool_mode

        if freeze_encoder:
            for p in self.instanovo.parameters():
                p.requires_grad = False
            self.instanovo.eval()

        self.proj_head = nn.Sequential(
            nn.Linear(d_instanovo, d_instanovo * 2),
            nn.GELU(),
            nn.LayerNorm(d_instanovo * 2),
            nn.Dropout(dropout),
            nn.Linear(d_instanovo * 2, d_instanovo * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_instanovo * 2, embed_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.instanovo.eval()
        return self

    def _run_backbone(self, x, precursors, x_mask):
        if self.freeze_encoder:
            with torch.no_grad():
                return self.instanovo._encoder(x, precursors, x_mask)
        return self.instanovo._encoder(x, precursors, x_mask)

    def forward(self, x, precursors, return_peak_representations: bool = False):
        """x: [B, MAX_PEAKS, 2]    precursors: [B, 3] (mass, charge, mz)."""
        # InstaNovo's charge_encoder indexes by `charge.int() - 1`; clamp to >=1.
        precursors = precursors.clone()
        precursors[:, 1] = precursors[:, 1].clamp(min=1)

        # Build padding mask from zero-peak rows (matches our preprocessing).
        x_mask = (x.abs().sum(dim=-1) == 0)

        peak_repr, pad_mask = self._run_backbone(x, precursors, x_mask)

        if self.pool_mode == "mean":
            valid = (~pad_mask).unsqueeze(-1).to(peak_repr.dtype)
            pooled = (peak_repr * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        elif self.pool_mode == "latent":
            pooled = peak_repr[:, 1, :]
        elif self.pool_mode == "precursor":
            pooled = peak_repr[:, 0, :]
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        embedding = F.normalize(self.proj_head(pooled), p=2, dim=-1)

        if return_peak_representations:
            return embedding, peak_repr, pad_mask
        return embedding
