import torch


def _infer_device(model):
    return next(model.parameters()).device


def _accumulate_diag(agg: dict | None, diag: dict | None) -> dict | None:
    if diag is None:
        return agg
    if agg is None:
        return {k: float(v) for k, v in diag.items()}
    for k, v in diag.items():
        agg[k] = agg.get(k, 0.0) + float(v)
    return agg


def _finalize_diag(agg: dict | None, n: int) -> dict | None:
    if agg is None or n == 0:
        return agg
    return {k: v / n for k, v in agg.items()}


def _split_batch(batch, device):
    """Move tensors to device and split into (specs, peps, pres, decoy_or_None)."""
    if len(batch) == 4:
        specs, peps, pres, decoy = [t.to(device) for t in batch]
        return specs, peps, pres, decoy
    specs, peps, pres = [t.to(device) for t in batch]
    return specs, peps, pres, None


def train_epoch(model_spec, model_pep, loader, loss_fn, opt, scaler):
    model_spec.train(); model_pep.train()
    device = _infer_device(model_pep)  # peptide encoder is always trainable
    total_loss, total_acc = 0.0, 0.0
    diag_agg: dict | None = None

    params = [p for p in model_spec.parameters() if p.requires_grad] \
             + list(model_pep.parameters())

    for batch in loader:
        specs, peps, pres, decoy = _split_batch(batch, device)
        opt.zero_grad()

        if device.type == "cuda" and scaler is not None:
            with torch.amp.autocast("cuda"):
                z_spec = model_spec(specs, pres)
                z_pep  = model_pep(peps)
                z_decoy = model_pep(decoy) if decoy is not None else None
                loss, acc, diag = loss_fn(z_spec, z_pep, z_decoy=z_decoy)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            z_spec = model_spec(specs, pres)
            z_pep  = model_pep(peps)
            z_decoy = model_pep(decoy) if decoy is not None else None
            loss, acc, diag = loss_fn(z_spec, z_pep, z_decoy=z_decoy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

        total_loss += loss.item()
        total_acc  += acc.item()
        diag_agg = _accumulate_diag(diag_agg, diag)

    n = max(1, len(loader))
    return total_loss / n, total_acc / n, _finalize_diag(diag_agg, n)


@torch.no_grad()
def validate(model_spec, model_pep, loader, loss_fn):
    model_spec.eval(); model_pep.eval()
    device = _infer_device(model_pep)
    total_loss, total_acc = 0.0, 0.0
    diag_agg: dict | None = None

    if len(loader) == 0:
        return 0.0, 0.0, None

    for batch in loader:
        specs, peps, pres, decoy = _split_batch(batch, device)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                z_spec = model_spec(specs, pres)
                z_pep  = model_pep(peps)
                z_decoy = model_pep(decoy) if decoy is not None else None
                loss, acc, diag = loss_fn(z_spec, z_pep, z_decoy=z_decoy)
        else:
            z_spec = model_spec(specs, pres)
            z_pep  = model_pep(peps)
            z_decoy = model_pep(decoy) if decoy is not None else None
            loss, acc, diag = loss_fn(z_spec, z_pep, z_decoy=z_decoy)

        total_loss += loss.item()
        total_acc  += acc.item()
        diag_agg = _accumulate_diag(diag_agg, diag)

    n = len(loader)
    return total_loss / n, total_acc / n, _finalize_diag(diag_agg, n)
