import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'c:/Users/tasne/Desktop/InstaSearch Project/InstaNovo')
from instanovo.transformer.layers import MultiScalePeakEmbedding
from .joint_model import TransformerEncoderBlock, ProjectionHead
from ..utils.config import D_MODEL, N_HEADS, D_FF, N_LAYERS, EMBED_DIM, DROPOUT

class SpectrumEncoder(nn.Module):
    """
    Encodes an (mz, intensity) peak matrix together with precursor information
    into a fixed-size embedding.

    Precursor encoding (FIXED):
        - Peak encoder (MultiScalePeakEmbedding) is reused to embed the
          precursor mz/intensity pair as a single-peak tensor.
        - Charge is embedded via a dedicated nn.Embedding lookup.
        - The two representations are concatenated then projected to D_MODEL
          via a single Linear layer to form the precursor embedding.
        - This replaces the original bug where a flat Linear(2, D_MODEL) was
          applied directly to the raw [mz, charge] floats — losing the
          multi-scale mz information provided by the peak encoder.
    """
    # Maximum charge state we expect to see (1-based; charges 0..MAX_CHARGE).
    MAX_CHARGE: int = 4

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, n_layers=N_LAYERS, embed_dim=EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        self.peak_encoder = MultiScalePeakEmbedding(d_model, dropout=dropout)

        # ── Precursor branch (fixed) ─────────────────────────────────────
        # The peak encoder produces [B, 1, d_model] for a single peak.
        # Squeeze to [B, d_model], then concat with charge embedding.
        self.charge_embedding = nn.Embedding(
            num_embeddings=self.MAX_CHARGE + 1,   # indices 0 … MAX_CHARGE
            embedding_dim=d_model
        )
        # After concat: [B, 2 * d_model]  →  [B, d_model]
        self.precursor_proj = nn.Linear(2 * d_model, d_model)
        # ────────────────────────────────────────────────────────────────

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.proj_head = ProjectionHead(d_model, d_model * 2, embed_dim)

    def _encode_precursor(self, precursors: torch.Tensor) -> torch.Tensor:
        """
        precursors: [B, 2]  where [:, 0] = precursor_mz, [:, 1] = charge (int)

        Returns: [B, d_model] precursor embedding.
        """
        # --- mz branch via peak encoder ---
        # Reshape to [B, 1, 2] so peak_encoder sees one "peak" per spectrum.
        mz_peak = precursors[:, :1].unsqueeze(-1)          # [B, 1, 1]
        # Pad to [B, 1, 2] with a unit intensity so the encoder has valid input.
        ones    = torch.ones_like(mz_peak)
        mz_peak = torch.cat([mz_peak, ones], dim=-1)       # [B, 1, 2]
        mz_emb  = self.peak_encoder(mz_peak).squeeze(1)    # [B, d_model]

        # --- charge branch via nn.Embedding ---

        charge_raw = precursors[:, 1]
        charge_raw = torch.nan_to_num(charge_raw, nan=0.0, posinf=0.0, neginf=0.0)
        charge = charge_raw.long().clamp(0, self.MAX_CHARGE)
        charge_emb  = self.charge_embedding(charge)                       # [B, d_model]

        # --- concat + project ---
        combined    = torch.cat([mz_emb, charge_emb], dim=-1)  # [B, 2*d_model]
        return self.precursor_proj(combined)                     # [B, d_model]

    def forward(self, x, precursors):
        """
        x:          [B, MAX_PEAKS, 2]
        precursors: [B, 2]  (mz, charge)
        Returns:    [B, EMBED_DIM]  L2-normalised embeddings
        """
        B = x.size(0)

        # Padding mask: True where peak row is all-zero (padded)
        is_pad   = (x.abs().sum(dim=-1) == 0)              # [B, MAX_PEAKS]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_mask, is_pad], dim=1)    # [B, MAX_PEAKS+1]

        # Peak embeddings
        x = self.peak_encoder(x)                           # [B, MAX_PEAKS, d_model]

        # Precursor embedding injected into CLS
        pre_emb = self._encode_precursor(precursors).unsqueeze(1)  # [B, 1, d_model]
        cls     = self.cls_token.expand(B, -1, -1) + pre_emb       # [B, 1, d_model]
        x       = torch.cat([cls, x], dim=1)               # [B, MAX_PEAKS+1, d_model]

        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        return self.proj_head(self.final_norm(x[:, 0, :]))