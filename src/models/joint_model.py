import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)