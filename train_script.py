#!/usr/bin/env python3
"""
Command-line interface for training the Dual Encoder model for proteomics.
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from datasets import load_dataset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.spectrum_encoder import SpectrumEncoder
from src.models.peptide_encoder import PeptideEncoder
from src.training.loss import CLIPContrastiveLoss
from src.training.train import train_epoch, validate
from src.data.preprocess import preprocess_dataset
from src.data.dataset import SpectraPeptideDataset
from src.utils.config import SEED, DEVICE
from src.utils.constants import DEVICE as DEVICE_CONST

def main():
    parser = argparse.ArgumentParser(description="Train Dual Encoder for Proteomics")
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='InstaDeepAI/ms_ninespecies_benchmark',
                       help='HuggingFace dataset name')
    parser.add_argument('--dataset_split', type=str, default='train[:20000]',
                       help='Dataset split to use')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                       help='Weight decay')
    parser.add_argument('--init_temp', type=float, default=0.1,
                       help='Initial temperature for contrastive loss')
    parser.add_argument('--warmup_epochs', type=float, default=5,
                       help='Sets the warmup epochs for the learning rate')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=SEED,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detect if None')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = DEVICE_CONST
    
    # Set seed
    torch.manual_seed(args.seed)
    
    print(f"Using device: {device}")
    print("Loading dataset...")
    
    # Load dataset
    ds = load_dataset(args.dataset, split=args.dataset_split)
    df = ds.to_pandas()
    
    print("Preprocessing...")

    specs, peps, pres = preprocess_dataset(df)
    
    # Split dataset
    total_num = len(specs)
    train_size = int(0.8 * total_num)
    val_size = int(0.1 * total_num)
    test_size = num_valid - train_size - val_size

    assert len(specs) == len(peps) == len(pres), "Mismatched dataset lengths after preprocessing"
    
    dataset = SpectraPeptideDataset(specs, peps, pres)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], 
                                           generator=torch.Generator().manual_seed(args.seed))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset split: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    
    # Create models
    model_spec = SpectrumEncoder(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_layers=args.n_layers, embed_dim=args.embed_dim
    ).to(device)
    
    model_pep = PeptideEncoder(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_layers=args.n_layers, embed_dim=args.embed_dim
    ).to(device)
    
    loss_fn = CLIPContrastiveLoss(init_temp=args.init_temp).to(device)
    
    optimizer = AdamW(
        list(model_spec.parameters()) + list(model_pep.parameters()) + [loss_fn.log_temp],
        lr=args.learning_rate, weight_decay=args.weight_decay
    )

    warmup_scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs)
    Cosine_scheduler = CosineAnnealingLR(optimizer, T_max = args.num_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, Cosine_scheduler], milestones=[args.warmup_epochs])

    
    # Scaler for mixed precision
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    print(f"Starting training ({args.num_epochs} epochs)...")

    try:
        for epoch in range(args.num_epochs):
            l, a = train_epoch(model_spec, model_pep, train_loader, loss_fn, optimizer, scaler)
            vl, va = validate(model_spec, model_pep, val_loader, loss_fn)
            
            history['loss'].append(l)
            history['acc'].append(a)
            history['val_loss'].append(vl)
            history['val_acc'].append(va)
    
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {l:.4f} | Acc: {a:.4f} | Val Loss: {vl:.4f} | Val Acc: {va:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_spec_state_dict': model_spec.state_dict(),
                    'model_pep_state_dict': model_pep.state_dict(),
                    'loss_fn_state_dict': loss_fn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'args': args
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        torch.save({
                    'epoch': epoch + 1,
                    'model_spec_state_dict': model_spec.state_dict(),
                    'model_pep_state_dict': model_pep.state_dict(),
                    'loss_fn_state_dict': loss_fn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'args': args
                }, os.path.join(args.output_dir, 'interrupted_checkpoint.pt'))
    
    # Final evaluation
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    tl, ta = validate(model_spec, model_pep, test_loader, loss_fn)
    print(f"\nTest Results | Loss: {tl:.4f} | Acc: {ta:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model_spec_state_dict': model_spec.state_dict(),
        'model_pep_state_dict': model_pep.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'args': vars(args),
        'history': history
    }, final_path)
    print(f"Saved final model: {final_path}")

if __name__ == '__main__':
    main()
