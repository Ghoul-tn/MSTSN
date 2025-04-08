import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from scipy.interpolate import interp1d
import random

class GambiaDataProcessor:
    def __init__(self, data_path):
        print(f"\nInitializing DataProcessor with data from: {data_path}")
        self.data = np.load(data_path)
        self.valid_pixels = None
        self.adj_matrix = None
        self.scalers = {
            'ndvi': MinMaxScaler(feature_range=(0, 1)),
            'soil': StandardScaler(),
            'lst': StandardScaler(),
            'spi': QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        }

    def process_data(self):
        """Process and normalize all input data"""
        print("\n=== Loading and Processing Raw Data ===")
        
        ndvi = self.data['NDVI']
        soil = self.data['SoilMoisture']
        spi = self.data['SPI']
        lst = self.data['LST']

        # Create valid pixel mask
        valid_mask = (
            (~np.isnan(ndvi).all(axis=0)) & 
            (~np.isnan(soil).all(axis=0)) & 
            (~np.isnan(spi).all(axis=0)) & 
            (~np.isnan(lst).all(axis=0))
        )
        self.valid_pixels = np.where(valid_mask)
        num_nodes = len(self.valid_pixels[0])

        # Initialize arrays
        features = np.zeros((287, num_nodes, 3), dtype=np.float32)
        targets = np.zeros((287, num_nodes), dtype=np.float32)

        # Extract valid pixels
        y_idx, x_idx = self.valid_pixels
        for t in range(287):
            features[t, :, 0] = ndvi[t, y_idx, x_idx]
            features[t, :, 1] = soil[t, y_idx, x_idx]
            features[t, :, 2] = lst[t, y_idx, x_idx]
            targets[t, :] = spi[t, y_idx, x_idx]

        # LST interpolation
        for node in range(num_nodes):
            lst_series = features[:, node, 2]
            valid_mask = ~np.isnan(lst_series)
            if valid_mask.sum() > 1:
                interp_func = interp1d(
                    np.arange(287)[valid_mask],
                    lst_series[valid_mask],
                    kind='linear',
                    fill_value="extrapolate"
                )
                features[:, node, 2] = interp_func(np.arange(287))
            else:
                features[:, node, 2] = np.nan_to_num(lst_series, nan=np.nanmean(lst_series))

        # Normalizations
        features[:, :, 0] = self.scalers['ndvi'].fit_transform(features[:, :, 0].reshape(-1, 1)).reshape(287, num_nodes)
        features[:, :, 1] = self.scalers['soil'].fit_transform(features[:, :, 1].reshape(-1, 1)).reshape(287, num_nodes)
        features[:, :, 2] = self.scalers['lst'].fit_transform(features[:, :, 2].reshape(-1, 1)).reshape(287, num_nodes)
        targets = self.scalers['spi'].fit_transform(targets.reshape(-1, 1)).reshape(287, num_nodes)

        # Adjacency matrix
        coords = np.column_stack(self.valid_pixels)
        distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(-1))
        self.adj_matrix = (distances <= 10).astype(float)
        np.fill_diagonal(self.adj_matrix, 0)
        self.adj_matrix = scipy.sparse.csr_matrix(self.adj_matrix)

        return features, targets

class GambiaDroughtDataset(Dataset):
    def __init__(self, features, targets, valid_pixels, seq_len=12):
        y_idx, x_idx = valid_pixels
        self.features = features[:, y_idx, x_idx, :]
        self.targets = targets[:, y_idx, x_idx]
        self.seq_len = seq_len
        self.training = False
        
        # Generate indices
        self.valid_indices = []
        num_pixels = len(valid_pixels[0])
        max_time = features.shape[0] - seq_len
        
        for pixel_idx in tqdm(range(num_pixels), desc="Generating sequences"):
            for time_idx in range(max_time):
                self.valid_indices.append((pixel_idx, time_idx))
        
        self.valid_indices.sort(key=lambda x: x[1])

    def __getitem__(self, idx):
        start_idx = max(0, idx - self.seq_len)
        x_seq = self.features[start_idx:idx]
        
        if len(x_seq) < self.seq_len:
            padding = np.zeros((self.seq_len - len(x_seq), *x_seq.shape[1:]), dtype=np.float32)
            x_seq = np.concatenate([padding, x_seq])
        
        if self.training:
            mask = np.random.rand(*x_seq.shape) < 0.1
            x_seq[mask] = 0
        
        return torch.tensor(x_seq, dtype=torch.float16), torch.tensor(self.targets[idx], dtype=torch.float16)

    def __len__(self):
        return len(self.features) - self.seq_len

if __name__ == "__main__":
    processor = GambiaDataProcessor("/kaggle/input/gambia-data.npz")
    features, targets = processor.process_data()
    dataset = GambiaDroughtDataset(features, targets, processor.valid_pixels)
    sample_x, sample_y = dataset[0]
    print(f"\nSample shapes - X: {sample_x.shape}, y: {sample_y.shape}")
