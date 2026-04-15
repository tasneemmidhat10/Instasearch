import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'c:/Users/tasne/Desktop/InstaSearch Project/InstaNovo')
from instanovo.transformer.layers import PositionalEncoding
from .joint_model import TransformerEncoderBlock, ProjectionHead
from ..utils.config import D_MODEL, N_HEADS, D_FF, N_LAYERS, EMBED_DIM, DROPOUT, MAX_PEPTIDE_LEN

AA_VOCAB = {aa: idx for idx, aa in enumerate([
    "<PAD>", "A", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W",
    "Y", "B", "Z", "X", "U", "O"
])}
NUM_AA = len(AA_VOCAB)

class PeptideEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, n_layers=N_LAYERS, embed_dim=EMBED_DIM, max_len=MAX_PEPTIDE_LEN, dropout=DROPOUT):
        super().__init__()
        # NUM_AA = 26 tokens (indices 0-25, with 0 = <PAD>)  — correct size
        self.aa_embed     = nn.Embedding(NUM_AA, d_model, padding_idx=0)
        # max_len = MAX_PEPTIDE_LEN + 1 to account for the prepended CLS token
        self.aa_pos_embed = PositionalEncoding(d_model, dropout=dropout,
                                               max_len=max_len + 1)

        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers     = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.proj_head  = ProjectionHead(d_model, d_model * 2, embed_dim)

    def forward(self, tokens):
        """
        tokens: [B, MAX_PEPTIDE_LEN]  (int64, 0 = pad)
        Returns: [B, EMBED_DIM]  L2-normalised embeddings
        """
        B = tokens.size(0)

        # Padding mask: True where token == 0 (PAD); prepend False for CLS
        is_pad   = (tokens == 0)                                   # [B, seq_len]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        pad_mask = torch.cat([cls_mask, is_pad], dim=1)            # [B, seq_len+1]

        # Amino-acid embeddings + positional encoding
        # PositionalEncoding expects (seq_len, B, d_model) — transpose in/out
        x = self.aa_embed(tokens)                                  # [B, seq_len, d]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                    # [B, 1, d]
        x   = torch.cat([cls, x], dim=1)                          # [B, seq_len+1, d]

        x = self.aa_pos_embed(x.transpose(0, 1)).transpose(0, 1)  # [B, seq_len, d]


        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        return self.proj_head(self.final_norm(x[:, 0, :]))