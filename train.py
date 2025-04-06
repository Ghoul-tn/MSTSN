import torch
from torch.utils.data import DataLoader, random_split
from data_preparation import GambiaDataProcessor, GambiaDroughtDataset
from models.MSTSN import MSTSN_Gambia

def main():
    # Initialize data processor
    processor = GambiaDataProcessor(
        '/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz'
    )
    features, targets = processor.process_data()
    
    # Create dataset
    dataset = GambiaDroughtDataset(
        features=features,
        targets=targets,
        valid_pixels=processor.valid_pixels,
        seq_len=12
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(
        dataset, 
        [train_size, len(dataset) - train_size]
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSTSN_Gambia(processor.adj_matrix).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    
    # Training loop
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Early stopping check can be added here

if __name__ == "__main__":
    main()
