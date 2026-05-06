# API reference

All methods belong to `class InstaSearchSpectrumEncoder(nn.Module)` in notebook cell `7265d906`.

## `__init__(instanovo_model, d_instanovo, embed_dim=EMBED_DIM, freeze_encoder=True, pool_mode="mean", dropout=DROPOUT)`

Stores the pretrained backbone and builds the projection head.

| Argument          | Type          | Notes |
|-------------------|---------------|-------|
| `instanovo_model` | `InstaNovo`   | A full `InstaNovo` module returned by `InstaNovo.from_pretrained(...)`. Stored on `self.instanovo`. |
| `d_instanovo`     | `int`         | The backbone's `d_model`. Read from `instanovo_config["dim_model"]`. Drives the projection head input size. |
| `embed_dim`       | `int`         | Output dimensionality of the contrastive embedding. Must match the peptide tower's `embed_dim`. |
| `freeze_encoder`  | `bool`        | If `True`, sets `requires_grad=False` on all backbone params and puts the backbone in `eval()`. |
| `pool_mode`       | `str`         | `"mean"`, `"latent"`, or `"precursor"`. See [architecture.md](architecture.md) for semantics. |
| `dropout`         | `float`       | Dropout inside the projection head. Does not affect the frozen backbone. |

Side effects when `freeze_encoder=True`:
- Every `instanovo.parameters()` entry becomes non-trainable.
- `self.instanovo.eval()` is called so dropout / LayerNorm running stats stay fixed.

## `train(mode=True)`

Overrides `nn.Module.train`. It still toggles the outer wrapper (so the projection head's dropout respects train/eval), but if `freeze_encoder` is set it forces `self.instanovo.eval()` back on immediately. Without this override, `model_spec.train()` at the top of each training epoch would re-enable dropout and BN-style running-stat updates inside the frozen backbone and silently corrupt the features.

Returns `self` (standard `nn.Module.train` contract).

## `_run_backbone(x, precursors, x_mask)`

Thin indirection that calls `self.instanovo._encoder(x, precursors, x_mask)`. When `freeze_encoder=True`, the call is wrapped in `torch.no_grad()` so no intermediate activations are retained for autograd — a significant memory saving during training.

Returns `(peak_repr, pad_mask)` exactly as InstaNovo's `_encoder` does:
- `peak_repr`: `[B, 1 + 1 + n_peaks, d_instanovo]`
- `pad_mask`:  `[B, 1 + 1 + n_peaks]`, `True` at padded positions.

This is an internal helper; external callers use `forward`.

## `forward(x, precursors, return_peak_representations=False)`

The public entry point. Signature matches the old `SpectrumEncoder.forward` so the training loop didn't have to change.

**Inputs**
- `x`: `FloatTensor [B, MAX_PEAKS, 2]` — `[:, :, 0]` is m/z, `[:, :, 1]` is intensity. Short spectra are zero-padded.
- `precursors`: `FloatTensor [B, 2]` — `[:, 0]` precursor m/z, `[:, 1]` charge.
- `return_peak_representations`: if `True`, also returns the raw backbone output for downstream reuse.

**What it does, step by step**

1. `precursors[:, 1].clamp(min=1)` — InstaNovo's charge embedding is indexed by `charge.int() - 1`, so a charge of 0 would wrap to `-1` and read from the wrong embedding slot. Clamping to `>=1` protects against rows where charge was missing (preprocessing falls back to 0).
2. `x_mask = (x.abs().sum(dim=-1) == 0)` — build the padding mask directly from zero rows. This matches our preprocessing convention and avoids relying on a pad token ID.
3. `peak_repr, pad_mask = self._run_backbone(x, precursors, x_mask)` — frozen forward pass (no-grad when frozen).
4. Pool `peak_repr` along dim=1 according to `pool_mode`:
   - `"mean"`: masked mean using `~pad_mask`. Division is `clamp(min=1)`'d to avoid div-by-zero on all-pad rows.
   - `"latent"`: `peak_repr[:, 1, :]`.
   - `"precursor"`: `peak_repr[:, 0, :]`.
5. Run `self.proj_head(pooled)` to project `d_instanovo -> EMBED_DIM`.
6. `F.normalize(..., p=2, dim=-1)` for cosine-similarity-friendly unit vectors.

**Outputs**
- Default: `embedding` of shape `[B, EMBED_DIM]`, each row a unit vector.
- With `return_peak_representations=True`: `(embedding, peak_repr, pad_mask)`. The raw `peak_repr` is exposed so a future rescoring or cross-attention head can reuse the frozen features without running InstaNovo again.
