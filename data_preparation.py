import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse

class GambiaDataProcessor:
    def __init__(self, data_path):
        self.data = np.load(data_path)
        self.valid_pixels = None
        self.adj_matrix = None
        
    def process_data(self):
        """Step 1: Data Preparation"""
        # Load and mask data
        ndvi = self.data['NDVI'][:, :, :, 0]
        soil = self.data['SoilMoisture'][:, :, :, 0]
        spi = self.data['SPI'][:, :, :, 0]
        lst = self.data['LST'][:, :, :, 0]
        
        # Create valid pixel mask
        self.valid_pixels = np.where(~np.isnan(spi).all(axis=0))
        valid_count = len(self.valid_pixels[0])
        
        # Create adjacency matrix (5km threshold)
        coords = np.column_stack(self.valid_pixels)
        distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(-1))
        adj = (distances <= 5).astype(float)
        np.fill_diagonal(adj, 0)
        self.adj_matrix = scipy.sparse.csr_matrix(adj)  # Sparse format
        
        # Normalize and stack features
        def normalize(x):
            return (x - np.nanmean(x)) / np.nanstd(x)
            
        features = np.stack([normalize(ndvi), normalize(soil), normalize(lst)], axis=-1)
        targets = normalize(spi)
        
        return features, targets

class GambiaDroughtDataset(Dataset):
    """Step 2: PyTorch Dataset"""
    def __init__(self, features, targets, valid_pixels, seq_len=12):
        self.features = features  # (T, H, W, C)
        self.targets = targets    # (T, H, W)
        self.valid_pixels = valid_pixels
        self.seq_len = seq_len
        self.pixel_coords = list(zip(*valid_pixels))
        
    def __len__(self):
        return len(self.pixel_coords) * (self.features.shape[0] - self.seq_len)
    
    def __getitem__(self, idx):
        pixel_idx = idx // (self.features.shape[0] - self.seq_len)
        time_idx = idx % (self.features.shape[0] - self.seq_len)
        
        y, x = self.pixel_coords[pixel_idx]
        x_seq = self.features[time_idx:time_idx+self.seq_len, y, x, :]
        y_target = self.targets[time_idx+self.seq_len, y, x]
        
        return torch.FloatTensor(x_seq), torch.FloatTensor([y_target])
