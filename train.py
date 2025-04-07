import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import random
from data_preparation import GambiaDataProcessor, GambiaDroughtDataset
from Models.MSTSN import MSTSN_Gambia

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

class DroughtAwareLoss(nn.Module):
    def __init__(self, base_loss=nn.HuberLoss(delta=0.5), alpha=3.0):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha

    def forward(self, pred, target):
        base = self.base_loss(pred, target)
        drought_mask = (target < -0.5).float()
        drought_err = torch.abs(pred - target) * drought_mask
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    
    # Model parameters
    parser.add_argument('--gcn_dim1', type=int, default=128)
    parser.add_argument('--gcn_dim2', type=int, default=256)
    parser.add_argument('--gru_dim', type=int, default=192)
    parser.add_argument('--gru_layers', type=int, default=3)
    
    # System parameters
    parser.add_argument('--results_dir', type=str, 
                      default='/kaggle/working/MSTSN/enhanced_results')
    
    return parser.parse_args()

def collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return features, targets

def compute_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        **DroughtMetrics.calculate(y_true, y_pred)
    }

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return compute_metrics(y_true, y_pred)

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopper = EarlyStopper()

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

    # Data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Model initialization
    model = MSTSN_Gambia(
        adj_matrix=processor.adj_matrix,
        gcn_dim1=args.gcn_dim1,
        gcn_dim2=args.gcn_dim2,
        gru_dim=args.gru_dim,
        gru_layers=args.gru_layers
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    loss_fn = DroughtAwareLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr*2,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )

    # Training loop
    best_val_rmse = float('inf')
    history = {'train': [], 'val': []}
    
    print("\n=== Starting Enhanced Training ===")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * x.size(0)
                train_preds.append(pred.detach().cpu().numpy())
                train_targets.append(y.detach().cpu().numpy())
                tepoch.set_postfix(loss=loss.item())

        # Training metrics
        train_metrics = compute_metrics(
            np.concatenate(train_targets),
            np.concatenate(train_preds)
        )
        train_metrics['loss'] = train_loss / len(train_loader.dataset)
        history['train'].append(train_metrics)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        history['val'].append(val_metrics)

        # Early stopping check
        if early_stopper(val_metrics['rmse']):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save best model
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': vars(args)
            }, f"{args.results_dir}/best_model.pth")

        # Epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"Train Loss: {train_metrics['loss']:.4f} | RMSE: {train_metrics['rmse']:.4f} | R²: {train_metrics['r2']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f} | DR: {val_metrics['detection_rate']:.2%} | FA: {val_metrics['false_alarm']:.2%}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Final save
    torch.save(model.state_dict(), f"{args.results_dir}/final_model.pth")
    torch.save(history, f"{args.results_dir}/training_history.pt")
    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")

if __name__ == "__main__":
    main()
