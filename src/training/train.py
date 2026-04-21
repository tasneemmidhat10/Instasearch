import torch
from .loss import CLIPContrastiveLoss

def _infer_device(model):
    return next(model.parameters()).device

def train_epoch(model_spec, model_pep, loader, loss_fn, opt, scaler):
    model_spec.train(); model_pep.train()
    device = _infer_device(model_spec)
    total_loss, total_acc = 0.0, 0.0

    params = list(model_spec.parameters()) + list(model_pep.parameters())

    for specs, peps, pres in loader:
        specs, peps, pres = specs.to(device), peps.to(device), pres.to(device)
        opt.zero_grad()

        if device.type == "cuda" and scaler is not None:
            with torch.amp.autocast("cuda"):
                z_spec = model_spec(specs, pres)
                z_pep  = model_pep(peps)
                loss, acc = loss_fn(z_spec, z_pep)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            z_spec = model_spec(specs, pres)
            z_pep = model_pep(peps)
            loss, acc = loss_fn(z_spec, z_pep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

        total_loss += loss.item(); total_acc += acc.item()
    return total_loss/len(loader), total_acc/len(loader)

@torch.no_grad()
def validate(model_spec, model_pep, loader, loss_fn):
    model_spec.eval(); model_pep.eval()
    device = _infer_device(model_spec)
    total_loss, total_acc = 0.0, 0.0

    if len(loader) == 0:
        return 0.0, 0.0

    for specs, peps, pres in loader:
        specs, peps, pres = specs.to(device), peps.to(device), pres.to(device)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                z_spec = model_spec(specs, pres)
                z_pep = model_pep(peps)
                loss, acc = loss_fn(z_spec, z_pep)
        else:
            z_spec = model_spec(specs, pres)
            z_pep = model_pep(peps)
            loss, acc = loss_fn(z_spec, z_pep)

        total_loss += loss.item(); total_acc += acc.item()
    return total_loss/len(loader), total_acc/len(loader)
