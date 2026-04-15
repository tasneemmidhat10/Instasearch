import torch
from .loss import CLIPContrastiveLoss
from ..utils.constants import DEVICE

def train_epoch(model_spec, model_pep, loader, loss_fn, opt, scaler):
    model_spec.train(); model_pep.train()
    total_loss, total_acc = 0.0, 0.0

    for specs, peps, pres in loader:
        specs, peps, pres = specs.to(DEVICE), peps.to(DEVICE), pres.to(DEVICE)
        opt.zero_grad()

        if DEVICE.type == "cuda" and scaler is not None:
            # FIX: guard scaler usage — scaler is None on CPU
            with torch.amp.autocast("cuda"):
                z_spec = model_spec(specs, pres)
                z_pep  = model_pep(peps)
                loss, acc = loss_fn(z_spec, z_pep)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm(list(model_spec.parameters() + list(model_pep.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            z_spec = model_spec(specs, pres)
            z_pep = model_pep(peps)
            loss, acc = loss_fn(z_spec, z_pep)
            loss.backward()
            opt.step()

        total_loss += loss.item(); total_acc += acc.item()
    return total_loss/len(loader), total_acc/len(loader)

@torch.no_grad()
def validate(model_spec, model_pep, loader, loss_fn):
    model_spec.eval(); model_pep.eval()
    total_loss, total_acc = 0.0, 0.0

    if len(loader) == 0:
        return 0.0, 0.0

    for specs, peps, pres in loader:
        specs, peps, pres = specs.to(DEVICE), peps.to(DEVICE), pres.to(DEVICE)

        if DEVICE.type == "cuda":
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
