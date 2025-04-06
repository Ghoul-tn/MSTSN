import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse
from tqdm import tqdm  # For progress bars

class GambiaDataProcessor:
    def __init__(self, data_path):
        print(f"\nInitializing DataProcessor with data from: {data_path}")
        self.data = np.load(data_path)
        self.valid_pixels = None
        self.adj_matrix = None
        
    def process_data(self):
        """Step 1: Data Preparation with verbose logging"""
        print("\n=== Loading and Processing Raw Data ===")
        
        # Load data with shape verification
        print("\nLoading variables from .npz file:")
        ndvi = self.data['NDVI'][:, :, :, 0]
        soil = self.data['SoilMoisture'][:, :, :, 0]
        spi = self.data['SPI'][:, :, :, 0]
        lst = self.data['LST'][:, :, :, 0]
        
        print(f"Original shapes - NDVI: {ndvi.shape}, Soil: {soil.shape}, "
              f"SPI: {spi.shape}, LST: {lst.shape}")

        # Create valid pixel mask
        print("\nCreating valid pixel mask...")
        self.valid_pixels = np.where(~np.isnan(spi).all(axis=0))
        valid_count = len(self.valid_pixels[0])
        print(f"Found {valid_count} valid pixels (out of {spi.shape[1]*spi.shape[2]} total)")
        
        # Create adjacency matrix
        print("\nBuilding adjacency matrix (5km neighborhood)...")
        coords = np.column_stack(self.valid_pixels)
        distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(-1))
        adj = (distances <= 10).astype(float)
        np.fill_diagonal(adj, 0)
        
        # Calculate connectivity statistics
        connections_per_node = adj.sum(axis=1)
        print(f"Adjacency matrix stats - Mean connections: {connections_per_node.mean():.1f}, "
              f"Max: {connections_per_node.max()}, Min: {connections_per_node.min()}")
        
        self.adj_matrix = scipy.sparse.csr_matrix(adj)
        print(f"Adjacency matrix size: {self.adj_matrix.shape} "
              f"(sparsity: {100*(1-self.adj_matrix.nnz/(adj.shape[0]*adj.shape[1])):.2f}%)")
        
        # Normalization
        print("\nNormalizing features...")
        def normalize(x):
            mean = np.nanmean(x)
            std = np.nanstd(x)
            print(f"Normalizing {x.shape} - mean: {mean:.3f}, std: {std:.3f}")
            return (x - mean) / std
            
        features = np.stack([
            normalize(ndvi), 
            normalize(soil), 
            normalize(lst)
        ], axis=-1)
        targets = normalize(spi)
        
        print("\nFinal processed shapes:")
        print(f"Features array: {features.shape} (timesteps × height × width × channels)")
        print(f"Targets array: {targets.shape} (timesteps × height × width)")
        
        return features, targets

class GambiaDroughtDataset(Dataset):
    """Step 2: PyTorch Dataset with progress tracking"""
    def __init__(self, features, targets, valid_pixels, seq_len=12):
        print(f"\nInitializing Dataset with sequence length: {seq_len}")
        self.features = features
        self.targets = targets
        self.valid_pixels = valid_pixels
        self.seq_len = seq_len
        self.pixel_coords = list(zip(*valid_pixels))
        
        # Pre-compute valid indices
        self.valid_indices = []
        print("Generating valid sample indices...")
        for pixel_idx in tqdm(range(len(self.pixel_coords))):
            max_time_idx = features.shape[0] - seq_len
            self.valid_indices.extend([
                (pixel_idx, time_idx) 
                for time_idx in range(max_time_idx)
            ])
        
        print(f"Created {len(self)} total samples "
              f"({len(self.pixel_coords)} pixels × {max_time_idx} time steps)")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        pixel_idx, time_idx = self.valid_indices[idx]
        y, x = self.pixel_coords[pixel_idx]
        
        # Verify shapes during first access
        if idx == 0:
            print("\nFirst sample shapes:")
            print(f"Input sequence: {self.features[time_idx:time_idx+self.seq_len, y, x, :].shape}")
            print(f"Target value: {self.targets[time_idx+self.seq_len, y, x].shape}")
        
        x_seq = self.features[time_idx:time_idx+self.seq_len, y, x, :]
        y_target = self.targets[time_idx+self.seq_len, y, x]
        
        return torch.FloatTensor(x_seq), torch.FloatTensor([y_target])

# Example usage with verbose output
if __name__ == "__main__":
    print("=== Running Data Preparation Test ===")
    processor = GambiaDataProcessor(
        '/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz'
    )
    features, targets = processor.process_data()
    
    print("\nCreating dataset...")
    dataset = GambiaDroughtDataset(
        features=features,
        targets=targets,
        valid_pixels=processor.valid_pixels,
        seq_len=12
    )
    
    # Test one batch
    sample_x, sample_y = dataset[0]
    print(f"\nSample batch shapes - X: {sample_x.shape}, y: {sample_y.shape}")
