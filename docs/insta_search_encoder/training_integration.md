# How the encoder is used in training

This page walks through exactly where `InstaSearchSpectrumEncoder` plugs into the notebook.

## 1. Loading the pretrained backbone

Notebook cell `67ee7bd2`:

```python
from instanovo.transformer.model import InstaNovo

INSTANOVO_CHECKPOINT = "instanovo-v1.1.0"
instanovo_model, instanovo_config = InstaNovo.from_pretrained(INSTANOVO_CHECKPOINT)
d_instanovo = int(instanovo_config["dim_model"])
```

`InstaNovo.from_pretrained` accepts either a registry id from `InstaNovo/instanovo/models.json` (which auto-downloads the checkpoint into `~/.cache/instanovo/`) or a local `.ckpt` path. It returns the full module plus the config dict; we grab `dim_model` to tell the projection head what input size to expect.

## 2. Constructing the dual encoder

Same cell:

```python
model_spec = InstaSearchSpectrumEncoder(
    instanovo_model=instanovo_model,
    d_instanovo=d_instanovo,
    embed_dim=EMBED_DIM,
    freeze_encoder=True,
    pool_mode="mean",
    dropout=DROPOUT,
).to(DEVICE)

model_pep = PeptideEncoder(
    d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
    n_layers=N_LAYERS, embed_dim=EMBED_DIM,
).to(DEVICE)

loss_fn = CLIPContrastiveLoss(init_temp=INIT_TEMP, label_smoothing=LABEL_SMOOTHING).to(DEVICE)
```

`embed_dim` is shared between both towers — it is the dimension the contrastive loss compares. `DEVICE` can be CPU or CUDA; the frozen backbone is moved alongside the wrapper.

## 3. Building the optimiser without the frozen weights

The projection head is the only trainable part of the spectrum tower, so we filter:

```python
trainable_spec = [p for p in model_spec.parameters() if p.requires_grad]
trainable_pep  = list(model_pep.parameters())
optimizer = torch.optim.AdamW(
    trainable_spec + trainable_pep + [loss_fn.log_temp],
    lr=LR, weight_decay=1e-2,
)
```

The `requires_grad` filter is important: handing frozen parameters to AdamW would still allocate optimiser state for them (momentum buffers), wasting memory for weights that never update. `loss_fn.log_temp` is the CLIP temperature, which is trained jointly.

`torch.nn.utils.clip_grad_norm_` downstream also iterates `list(model_spec.parameters()) + list(model_pep.parameters())` — that is safe because `clip_grad_norm_` ignores parameters whose `.grad` is `None`, and the frozen backbone never receives gradients.

## 4. Inside the training loop

Notebook cell `7135fe91`:

```python
def train_epoch(model_spec, model_pep, loader, loss_fn, opt, scaler):
    model_spec.train(); model_pep.train()   # <-- backbone stays in eval() (see api.md: train())
    for specs, peps, pres in loader:
        specs, peps, pres = specs.to(DEVICE), peps.to(DEVICE), pres.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            z_spec = model_spec(specs, pres)   # forward through frozen backbone + trainable head
            z_pep  = model_pep(peps)
            loss, acc, diagnostics = loss_fn(z_spec, z_pep)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(model_spec.parameters()) + list(model_pep.parameters()),
            max_norm=1.0,
        )
        scaler.step(opt); scaler.update()
```

Key behaviours to notice:

- **`model_spec.train()` is safe.** The override in `InstaSearchSpectrumEncoder.train` re-asserts `self.instanovo.eval()` immediately afterwards, so projection-head dropout is active while backbone dropout / LayerNorm stats stay frozen.
- **Gradients only flow into the projection head** and into `z_pep`'s path through `model_pep`. The backbone contributes activations via `torch.no_grad()`, so backprop through it is skipped entirely.
- **`autocast` is compatible** because the `no_grad` context does not interfere with mixed-precision — the backbone just runs in fp16/bf16 and emits the pooled features into fp32 inside the projection head.
- **Validation (`validate`)** runs with `model_spec.eval(); model_pep.eval()` inside `@torch.no_grad()` — in this regime even the projection head has no gradients and dropout is off in both halves.

## 5. Interface contract with the training loop

`model_spec(specs, pres)` is used exactly like the old custom `SpectrumEncoder` — same two positional arguments, same `[B, EMBED_DIM]` unit-norm output. That is why no change was required inside `train_epoch` or `validate` when the spectrum backbone was swapped to InstaNovo. The old `return_peak_representations` argument is optional and unused by the training loop today.

## 6. Memory and compute implications

- Forward cost is dominated by InstaNovo's encoder. This is fixed per sample; batch size is limited mainly by that forward pass plus the autocast activation cache on the peptide tower.
- Backward cost is small: gradients flow through the projection head (~hundreds of thousands of params) and the peptide encoder only. The frozen backbone contributes zero backward compute.
- Optimiser state size is proportional to trainable params only, thanks to the `requires_grad` filter.
