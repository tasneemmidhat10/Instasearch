#!/usr/bin/env python3
"""Hydra entry point for dual-encoder training.

Examples:
    # CLIP contrastive (default)
    python train_script.py

    # Alignment + uniformity loss
    python train_script.py loss=align_uniform

    # AlignUniform with decoy margin
    python train_script.py loss=align_uniform loss.use_decoy=true

    # Override anything from the CLI
    python train_script.py loss=clip dataset.split='train[:2000]' \\
        optim.num_epochs=2 dataset.batch_size=64
"""
import sys
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, random_split

# Make `src` and the vendored InstaNovo importable.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "InstaNovo"))

from instanovo.utils.residues import ResidueSet  # noqa: E402

from src.data.dataset import SpectraPeptideDataset  # noqa: E402
from src.data.decoys import make_shuffled_decoys  # noqa: E402
from src.data.preprocess import preprocess_dataset  # noqa: E402
from src.models.insta_search_spectrum_encoder import InstaSearchSpectrumEncoder  # noqa: E402
from src.models.peptide_encoder import PeptideEncoder  # noqa: E402
from src.training.train import train_epoch, validate  # noqa: E402
from src.utils.instanovo_loader import load_instanovo_backbone  # noqa: E402


def _pick_device(device_cfg):
    if device_cfg:
        return torch.device(device_cfg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_scheduler(optimizer, optim_cfg):
    warmup = max(1, int(optim_cfg.warmup_epochs))
    warmup_sched = LambdaLR(optimizer,
                            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=optim_cfg.num_epochs)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                        milestones=[warmup])


def _save_ckpt(path, epoch, model_spec, model_pep, loss_fn, optimizer, history, cfg):
    torch.save({
        'epoch': epoch,
        'model_spec_state_dict': model_spec.state_dict(),
        'model_pep_state_dict':  model_pep.state_dict(),
        'loss_fn_state_dict':    loss_fn.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'history': history,
        'config':  OmegaConf.to_container(cfg, resolve=True),
    }, path)


def _format_diag(diag):
    if not diag:
        return ""
    return " | " + " ".join(f"{k}={v:.4f}" for k, v in diag.items())


@hydra.main(version_base=None,
            config_path=str(_PROJECT_ROOT / "configs"),
            config_name="dual_encoder")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    device = _pick_device(cfg.device)
    print(f"Using device: {device}")

    # --- ResidueSet (PTM-aware vocabulary) -------------------------------
    residue_set = ResidueSet(
        residue_masses=OmegaConf.to_container(cfg.residues.residues, resolve=True)
    )
    print(f"ResidueSet vocab size: {len(residue_set.vocab)} "
          f"(PAD={residue_set.PAD_INDEX})")

    # --- Frozen InstaNovo backbone ---------------------------------------
    print(f"Loading InstaNovo backbone: {cfg.model.instanovo_checkpoint}")
    instanovo_model, _instanovo_cfg, d_instanovo = load_instanovo_backbone(
        cfg.model.instanovo_checkpoint, device, freeze=cfg.model.freeze_backbone,
    )
    print(f"InstaNovo dim_model = {d_instanovo}")

    # --- Encoders ---------------------------------------------------------
    model_spec = InstaSearchSpectrumEncoder(
        instanovo_model=instanovo_model,
        d_instanovo=d_instanovo,
        embed_dim=cfg.model.embed_dim,
        freeze_encoder=cfg.model.freeze_backbone,
        pool_mode=cfg.model.pool_mode,
        dropout=cfg.model.spec_dropout,
    ).to(device)

    pep_kwargs = OmegaConf.to_container(cfg.model.pep_args, resolve=True)
    model_pep = PeptideEncoder(
        vocab_size=len(residue_set.vocab),
        pad_idx=residue_set.PAD_INDEX,
        **pep_kwargs,
    ).to(device)

    # --- Data -------------------------------------------------------------
    print(f"Loading dataset: {cfg.dataset.name} [{cfg.dataset.split}]")
    ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    df = ds.to_pandas()
    print(f"Preprocessing {len(df)} rows ...")
    specs, peps, pres = preprocess_dataset(
        df,
        residue_set=residue_set,
        max_len=cfg.model.pep_args.max_len,
        max_peaks=cfg.dataset.max_peaks,
    )

    use_decoy = bool(cfg.loss.get("use_decoy", False))
    decoys = (make_shuffled_decoys(peps, residue_set.PAD_INDEX, seed=cfg.seed)
              if use_decoy else None)

    dataset = SpectraPeptideDataset(specs, peps, pres, decoy_peps=decoys)

    n = len(dataset)
    train_n = int(cfg.split.train * n)
    val_n   = int(cfg.split.val   * n)
    test_n  = n - train_n - val_n
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    print(f"Split: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    train_loader = DataLoader(train_ds, batch_size=cfg.dataset.batch_size,
                              shuffle=True, num_workers=cfg.dataset.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=cfg.dataset.batch_size,
                              shuffle=False, num_workers=cfg.dataset.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=cfg.dataset.batch_size,
                              shuffle=False, num_workers=cfg.dataset.num_workers)

    # --- Loss (Hydra-instantiated; supports any registered _target_) -----
    # Strip non-loss keys before instantiation.
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    loss_cfg.pop("name", None)
    loss_cfg.pop("use_decoy", None)
    loss_fn = hydra.utils.instantiate(loss_cfg).to(device)
    print(f"Loss: {cfg.loss.name}  (use_decoy={use_decoy})")

    # --- Optimizer (only trainable params) -------------------------------
    trainable = [p for p in model_spec.parameters() if p.requires_grad] \
              + list(model_pep.parameters()) \
              + [p for p in loss_fn.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.optim.lr,
                      weight_decay=cfg.optim.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg.optim)

    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model_spec.instanovo.parameters())
    print(f"Trainable params: {n_train:,}  |  Frozen InstaNovo: {n_frozen:,}")

    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and cfg.amp) else None

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [],
               'diag': [], 'val_diag': []}

    print(f"Starting training: {cfg.optim.num_epochs} epochs")
    epoch = -1
    try:
        for epoch in range(cfg.optim.num_epochs):
            l, a, diag       = train_epoch(model_spec, model_pep, train_loader,
                                           loss_fn, optimizer, scaler)
            vl, va, vdiag    = validate(model_spec, model_pep, val_loader, loss_fn)
            scheduler.step()

            history['loss'].append(l);     history['acc'].append(a)
            history['val_loss'].append(vl); history['val_acc'].append(va)
            history['diag'].append(diag);   history['val_diag'].append(vdiag)

            print(f"Epoch {epoch+1}/{cfg.optim.num_epochs} | "
                  f"loss={l:.4f} acc={a:.4f} | val_loss={vl:.4f} val_acc={va:.4f}"
                  f"{_format_diag(diag)}")

            if (epoch + 1) % cfg.save_every == 0:
                ckpt = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
                _save_ckpt(ckpt, epoch + 1, model_spec, model_pep, loss_fn,
                           optimizer, history, cfg)
                print(f"  saved {ckpt}")

    except KeyboardInterrupt:
        print("Training interrupted; saving checkpoint ...")
        _save_ckpt(output_dir / "interrupted_checkpoint.pt", epoch + 1,
                   model_spec, model_pep, loss_fn, optimizer, history, cfg)

    tl, ta, tdiag = validate(model_spec, model_pep, test_loader, loss_fn)
    print(f"Test | loss={tl:.4f} acc={ta:.4f}{_format_diag(tdiag)}")

    final_path = output_dir / "final_model.pt"
    _save_ckpt(final_path, epoch + 1, model_spec, model_pep, loss_fn,
               optimizer, history, cfg)
    print(f"Saved final model: {final_path}")


if __name__ == '__main__':
    main()
