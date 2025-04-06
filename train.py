import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from data_preparation import GambiaDataProcessor, GambiaDroughtDataset
from models.MSTSN import MSTSN_Gambia
from tqdm import tqdm

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

def main():
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"\nSaving results to: {args.results_dir}")
    
    # Initialize data
    print("\n=== Loading Data ===")
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    
    # Create datasets
    dataset = GambiaDroughtDataset(
        features=features,
        targets=targets,
        valid_pixels=processor.valid_pixels,
        seq_len=args.seq_len
    )
    
    # Split datasets
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(
        dataset, 
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print("\n=== Starting Training ===")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")
                
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.results_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'args': vars(args)
            }, model_path)
            print(f"Saved new best model with val loss: {val_loss:.4f}")
    
    # Save training history
    history_path = os.path.join(args.results_dir, 'training_history.pt')
    torch.save(history, history_path)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(args.results_dir, 'best_model.pth')}")
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    main()
