import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from data_preparation import GambiaDataProcessor, GambiaDroughtDataset
from Models.MSTSN import MSTSN_Gambia

def parse_args():
    parser = argparse.ArgumentParser(description='Train MSTSN for Drought Prediction')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                       default='/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz',
                       help='Path to input NPZ file')
    parser.add_argument('--seq_len', type=int, default=12,
                       help='Input sequence length in months')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 penalty)')
    
    # Model parameters
    parser.add_argument('--gcn_dim1', type=int, default=64,
                       help='First GCN layer output dimension')
    parser.add_argument('--gcn_dim2', type=int, default=32,
                       help='Second GCN layer output dimension')
    parser.add_argument('--gru_dim', type=int, default=64,
                       help='GRU hidden dimension')
    parser.add_argument('--gru_layers', type=int, default=2,
                       help='Number of GRU layers')
    
    # System parameters
    parser.add_argument('--results_dir', type=str, 
                       default='/kaggle/working/MSTSN/results',
                       help='Directory to save results')
    
    return parser.parse_args()

def collate_fn(batch):
    features = [item[0] for item in batch]  # List of (seq_len, 1, 1, 3)
    targets = [item[1] for item in batch]
    
    # Stack with added batch dimension
    return (
        torch.stack(features, dim=0),  # (batch_size, seq_len, 1, 1, 3)
        torch.stack(targets, dim=0)
    )

def compute_metrics(y_true, y_pred):
    """Calculate all regression metrics"""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds)
            all_targets.append(y)
    
    # Ensure consistent shapes
    y_true = torch.cat(all_targets).flatten()  # Flatten to [batch_size]
    y_pred = torch.cat(all_preds).flatten()
    
    return compute_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy())
class DroughtLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), alpha=2.0):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha  # Weight for drought periods

    def forward(self, pred, target):
        base = self.base_loss(pred, target)
        # Extra penalty for drought mispredictions
        drought_mask = (target < -0.5).float()
        drought_err = (pred - target).abs() * drought_mask
        return base + self.alpha * drought_err.mean()
def main():
    args = parse_args()
    
    # Setup directories and device
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Results will be saved to: {args.results_dir}")

    # Load and prepare data
    print("\n=== Loading Data ===")
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()  # Sets valid_pixels and num_nodes
    
    dataset = GambiaDroughtDataset(
        features=features,
        targets=targets,
        valid_pixels=processor.valid_pixels,
        seq_len=args.seq_len
    )
    
    # Split dataset using random indices instead of random_split
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    
    train_set = torch.utils.data.Subset(dataset, indices[:split])
    val_set = torch.utils.data.Subset(dataset, indices[split:])
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,  # Critical for time series
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # For validation - also no shuffling
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    # Initialize model
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
    loss_fn = DroughtLoss(base_loss=nn.HuberLoss(delta=0.5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )
    # Training loop
    best_val_rmse = float('inf')
    history = {'train': [], 'val': []}
    
    print("\n=== Starting Training ===")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")
                
                x, y = batch  # Direct unpacking
                x, y = x.to(device), y.to(device)  
                y = y.flatten()
                optimizer.zero_grad()
                
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.append(pred.detach().cpu().numpy())
                train_targets.append(y.detach().cpu().numpy())
                tepoch.set_postfix(loss=loss.item())
        
        # Calculate training metrics
        train_metrics = compute_metrics(
            np.concatenate(train_targets),
            np.concatenate(train_preds)
        )
        train_metrics['loss'] = train_loss / len(train_loader)
        history['train'].append(train_metrics)
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device)
        history['val'].append(val_metrics)
        scheduler.step(val_metrics['rmse'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training - Loss: {train_metrics['loss']:.4f} | "
              f"MSE: {train_metrics['mse']:.4f} | "
              f"RMSE: {train_metrics['rmse']:.4f} | "
              f"MAE: {train_metrics['mae']:.4f} | "
              f"R²: {train_metrics['r2']:.4f}")
        print(f"  Validation - MSE: {val_metrics['mse']:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | "
              f"MAE: {val_metrics['mae']:.4f} | "
              f"R²: {val_metrics['r2']:.4f}")
        
        # Save best model
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            model_path = os.path.join(args.results_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': vars(args)
            }, model_path)
            print(f"  Saved new best model with validation RMSE: {best_val_rmse:.4f}")
    
    # Save final model and history
    final_model_path = os.path.join(args.results_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    history_path = os.path.join(args.results_dir, 'training_history.pt')
    torch.save(history, history_path)
    
    print("\n=== Training Complete ===")
    print(f"Best Validation RMSE: {best_val_rmse:.4f}")
    print(f"Models saved to: {args.results_dir}")

if __name__ == "__main__":
    main()
