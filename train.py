import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import random
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch_xla.amp import autocast, GradScaler
from data_preparation import GambiaDataProcessor, GambiaDroughtDataset
from Models.MSTSN import EnhancedMSTSN

# os.environ['XLA_USE_BF16'] = '1'  # Use bfloat16 where possible
os.environ['XLA_TENSOR_ALLOCATOR_MAX_BYTES'] = '3221225472'  # 3GB buffer


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class ImprovedDroughtLoss(nn.Module):
    def __init__(self, base_loss=nn.HuberLoss(delta=0.5), alpha=3.0, gamma=2.0):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Ensure shapes match [batch, nodes]
        assert pred.shape == target.shape, f"Shapes must match: pred {pred.shape} vs target {target.shape}"
        
        base = self.base_loss(pred, target)
        drought_mask = (target < -0.5).float()
        error = torch.abs(pred - target)
        focal_weight = (1 - torch.exp(-error)) ** self.gamma
        drought_err = focal_weight * error * drought_mask
        return base + self.alpha * torch.mean(drought_err)
class DroughtMetrics:
    @staticmethod
    def calculate(y_true, y_pred):
        drought_mask = y_true < -0.5
        metrics = {
            'drought_rmse': np.sqrt(mean_squared_error(y_true[drought_mask], y_pred[drought_mask])),
            'false_alarm': np.mean((y_pred < -0.5) & ~drought_mask),
            'detection_rate': np.mean((y_pred[drought_mask] < -0.5))
        }
        return {k: 0.0 if np.isnan(v) else v for k,v in metrics.items()}

def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced MSTSN for Drought Prediction')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                      default='/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz')
    parser.add_argument('--seq_len', type=int, default=12)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)  # Increased for TPU
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    
    # System parameters
    parser.add_argument('--results_dir', type=str, 
                      default='/kaggle/working/MSTSN/enhanced_results')
    
    return parser.parse_args()

def collate_fn(batch):
    features, targets = torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch])
    return features.half(), targets.half()  # FP16 for memory savings

def compute_metrics(y_true, y_pred, loss=None):
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        **DroughtMetrics.calculate(y_true, y_pred)
    }
    if loss is not None:
        metrics['loss'] = loss
    return metrics
    
def evaluate(model, dataloader, device, loss_fn=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            if loss_fn is not None:
                total_loss += loss_fn(preds, y).item() * x.size(0)
    
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    loss = total_loss / len(dataloader.dataset) if loss_fn is not None else None
    return compute_metrics(y_true, y_pred, loss=loss)

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    # TPU Setup
    device = xm.xla_device()
    early_stopper = EarlyStopper()
    # Force FP16 across all operations
    torch.set_float32_matmul_precision('medium')  # For TPU efficiency
    # Data loading
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    
    # Dataset setup
    full_dataset = GambiaDroughtDataset(
        features=features,
        targets=targets,
        valid_pixels=processor.valid_pixels,
        seq_len=args.seq_len
    )
    
    # Temporal split
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))
    
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    train_set.dataset.training = True
    val_set.dataset.training = False

    # TPU-Optimized DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Reduced from 2 to 0 for TPU stability
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False  # Add this
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Reduced from 2 to 0
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )
    
    # Wrap with TPU parallel loader
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)

    # Model initialization
    model = EnhancedMSTSN(num_nodes=processor.adj_matrix.shape[0]).to(device)
    model = model.to(torch.bfloat16)
    # model.use_checkpoint = True  # Enable checkpointing
    print(f"Model memory: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6}MB")
    print(f"Input dtype: {next(model.parameters()).dtype}")  # Should show torch.bfloat16
    # print(f"Checkpointing enabled: {model.use_checkpoint}")
    # Define checkpoint function
    def checkpoint_forward(x):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward
        
        # Checkpoint each major component separately
        x = checkpoint(create_custom_forward(model.spatial_processor), x)
        x = x.reshape(-1, args.seq_len, x.size(-1))  # Reshape for temporal
        x = checkpoint(create_custom_forward(model.temporal_processor), x)
        return x
    # Optimizer - Using AdamW instead of Adafactor for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    
        
    loss_fn = ImprovedDroughtLoss(alpha=3.0, gamma=2.0)
    scaler = GradScaler('xla')

    # Training loop
    best_val_rmse = float('inf')
    history = {'train': [], 'val': []}
    grad_accum_steps = 4
    print("\n=== Starting TPU Training ===")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        optimizer.zero_grad()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")
                
                optimizer.zero_grad()
                
            with autocast(xm.xla_device()):
                pred = model(x)  # Checkpointing now handled inside model
                loss = loss_fn(pred, y)
                
                scaler.scale(loss).backward()
                xm.mark_step()
                if (i + 1) % grad_accum_steps == 0:
                    xm.optimizer_step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                    
                train_loss += loss.item() * x.size(0)
                train_preds.append(pred.detach().cpu().numpy())
                train_targets.append(y.detach().cpu().numpy())
                tepoch.set_postfix(loss=loss.item())

        # Training metrics
        train_metrics = compute_metrics(
            np.concatenate(train_targets),
            np.concatenate(train_preds),
            loss=train_loss / len(train_loader.dataset)
        )
        history['train'].append(train_metrics)

        # Validation
        val_metrics = evaluate(model, val_loader, device, loss_fn=loss_fn)
        history['val'].append(val_metrics)

        # Early stopping check
        if early_stopper(val_metrics['rmse']):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save best model
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            xm.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': vars(args)
            }, f"{args.results_dir}/best_model.pth")

        # Epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print("  Training - " + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in train_metrics.items()]))
        print("  Validation - " + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()]))
    
    # Final save
    xm.save(model.state_dict(), f"{args.results_dir}/final_model.pth")
    torch.save(history, f"{args.results_dir}/training_history.pt")
    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")

if __name__ == "__main__":
    main()
