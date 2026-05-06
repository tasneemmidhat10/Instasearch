# Architecture

## Role in the dual-encoder pipeline

The system is a CLIP-style dual encoder for peptide / mass-spectrum retrieval:

```
spectrum  --> InstaSearchSpectrumEncoder --> z_spec  \
                                                      >-- CLIPContrastiveLoss
peptide   --> PeptideEncoder            --> z_pep   /
```

Both towers produce L2-normalised embeddings of shape `[B, EMBED_DIM]` that live on the unit hypersphere. The contrastive loss pulls matching `(spectrum, peptide)` pairs together and pushes mismatched pairs apart in that shared space.

## What the module wraps

The backbone is a full `instanovo.transformer.model.InstaNovo` module loaded from a pretrained checkpoint (`instanovo-v1.1.0` by default). We only use its `_encoder(x, precursors, x_mask)` path — the autoregressive peptide decoder half is never called. The encoder returns:

- `peak_repr`: `[B, L, d_model]`, where `L = 1 + 1 + n_peaks`
  - index 0 = **precursor** token (a learned summary of precursor m/z and charge)
  - index 1 = **latent** spectrum token (InstaNovo's sequence-level summary slot)
  - indices 2..L-1 = one token per input peak
- `pad_mask`: `[B, L]` boolean, `True` at padded positions.

## Freezing contract

`freeze_encoder=True` flips `requires_grad=False` on every InstaNovo parameter and calls `self.instanovo.eval()` in `__init__`. To keep the backbone in eval mode even when the outer wrapper is switched to train mode, `train()` is overridden to re-assert `instanovo.eval()`. This matters because the backbone contains dropout and LayerNorm stats that must stay fixed while the projection head is trained — otherwise the frozen features would drift across batches.

The backbone forward pass additionally runs inside `torch.no_grad()` so the autograd graph does not retain intermediate activations through the entire InstaNovo stack. This is a pure memory optimisation: it is valid precisely because no parameter upstream of the projection head receives gradients.

## Data flow through `forward`

```
x: [B, MAX_PEAKS, 2]   # (mz, intensity) rows, zero-padded
precursors: [B, 2]      # (precursor_mz, charge)

1. Clamp charge >= 1                  # InstaNovo indexes charge by charge.int()-1
2. Build x_mask from zero-peak rows   # True == padded
3. peak_repr, pad_mask = instanovo._encoder(x, precursors, x_mask)  [frozen, no_grad]
4. pool peak_repr along L             # mean / latent / precursor
5. proj_head(pooled): d_instanovo -> 2*d_instanovo -> EMBED_DIM
6. F.normalize(..., p=2, dim=-1)      # unit-norm output

returns: [B, EMBED_DIM]
```

## Projection head

A 2-layer MLP that expands to `2 * d_instanovo` before collapsing to `EMBED_DIM`:

```
Linear(d_instanovo, 2 * d_instanovo)
GELU
LayerNorm(2 * d_instanovo)
Dropout
Linear(2 * d_instanovo, EMBED_DIM)
```

This is the only part of the spectrum tower that learns. It has to do three jobs simultaneously: (1) reduce dimensionality from InstaNovo's `d_model` down to `EMBED_DIM`, (2) discard whatever parts of the pretrained representation are de-novo-decoding-specific rather than retrieval-useful, and (3) align the geometry with the peptide tower so matching pairs share a direction on the unit sphere.

## Pooling modes

`pool_mode` controls which slice of `peak_repr` becomes the pooled vector fed to the projection head:

| Mode        | Selection                         | Intuition |
|-------------|-----------------------------------|-----------|
| `"mean"`    | mean over non-padded positions    | averages information across precursor, latent, and all peaks — default, most robust |
| `"latent"`  | `peak_repr[:, 1, :]`              | uses InstaNovo's learned sequence-summary token directly |
| `"precursor"` | `peak_repr[:, 0, :]`            | uses only the precursor-conditioned token; ignores peak detail |

The current notebook uses `"mean"`.
