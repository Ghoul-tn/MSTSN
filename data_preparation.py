import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from scipy.interpolate import interp1d

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
        
        # Load raw data
        ndvi = self.data['NDVI']          # (287, 41, 84)
        soil = self.data['SoilMoisture']  # (287, 41, 84)
        spi = self.data['SPI']            # (287, 41, 84)
        lst = self.data['LST']            # (287, 41, 84)

        print(f"Original shapes - NDVI: {ndvi.shape}, Soil: {soil.shape}, "
              f"SPI: {spi.shape}, LST: {lst.shape}")

        # Create strict valid pixel mask
        print("\nCreating valid pixel mask...")
        valid_mask = (
            (~np.isnan(ndvi).all(axis=0) & 
            (~np.isnan(soil).all(axis=0) & 
            (~np.isnan(spi).all(axis=0) & 
            (~np.isnan(lst).all(axis=0)
        )
        self.valid_pixels = np.where(valid_mask)
        num_nodes = len(self.valid_pixels[0])
        print(f"Using {num_nodes} valid pixels (based on complete data availability)")

        # Initialize arrays for valid data
        features = np.zeros((287, num_nodes, 3))  # (time, nodes, features)
        targets = np.zeros((287, num_nodes))      # (time, nodes)

        # Extract valid pixels
        print("\nExtracting valid pixels...")
        y_idx, x_idx = self.valid_pixels
        for t in range(287):
            features[t, :, 0] = ndvi[t, y_idx, x_idx]  # NDVI
            features[t, :, 1] = soil[t, y_idx, x_idx]  # Soil
            features[t, :, 2] = lst[t, y_idx, x_idx]   # LST
            targets[t, :] = spi[t, y_idx, x_idx]       # SPI

        # Handle LST NaNs with interpolation
        print("\nInterpolating missing LST values...")
        for node in range(num_nodes):
            lst_series = features[:, node, 2]
            valid_mask = ~np.isnan(lst_series)
            if np.sum(valid_mask) > 1:
                interp_func = interp1d(
                    np.arange(287)[valid_mask],
                    lst_series[valid_mask],
                    kind='linear',
                    fill_value="extrapolate"
                )
                features[:, node, 2] = interp_func(np.arange(287))
            else:
                features[:, node, 2] = np.nan_to_num(lst_series, nan=np.nanmean(lst_series))

        # Apply expert normalizations
        print("\nApplying feature-specific normalizations:")
        # NDVI - MinMax
        features[:, :, 0] = self.scalers['ndvi'].fit_transform(
            features[:, :, 0].reshape(-1, 1)
        ).reshape(287, num_nodes)
        print("- NDVI: MinMax [0,1] applied")
        
        # Soil - Standard
        features[:, :, 1] = self.scalers['soil'].fit_transform(
            features[:, :, 1].reshape(-1, 1)
        ).reshape(287, num_nodes)
        print("- Soil Moisture: StandardScaler applied")
        
        # LST - Standard
        features[:, :, 2] = self.scalers['lst'].fit_transform(
            features[:, :, 2].reshape(-1, 1)
        ).reshape(287, num_nodes)
        print("- LST: StandardScaler applied")
        
        # SPI - Quantile
        targets = self.scalers['spi'].fit_transform(
            targets.reshape(-1, 1)
        ).reshape(287, num_nodes)
        print("- SPI: QuantileTransformer (Gaussian) applied")

        # Create adjacency matrix
        print("\nBuilding adjacency matrix (5km neighborhood)...")
        coords = np.column_stack(self.valid_pixels)
        distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(-1))
        self.adj_matrix = (distances <= 5).astype(float)
        np.fill_diagonal(self.adj_matrix, 0)
        self.adj_matrix = scipy.sparse.csr_matrix(self.adj_matrix)

        # Reshape for MSTSN (time, height, width, channels)
        print("\nReshaping for MSTSN architecture...")
        feature_grid = np.full((287, 41, 84, 3), np.nan)
        target_grid = np.full((287, 41, 84), np.nan)
        
        for t in range(287):
            feature_grid[t, y_idx, x_idx, :] = features[t]
            target_grid[t, y_idx, x_idx] = targets[t]

        print("\nFinal processed shapes:")
        print(f"- Features: {feature_grid.shape} (time, height, width, channels)")
        print(f"- Targets: {target_grid.shape} (time, height, width)")
        print(f"- Adjacency matrix: {self.adj_matrix.shape} (sparsity: {100*(1-self.adj_matrix.nnz/(num_nodes*num_nodes)):.2f}%)")
        
        return feature_grid, target_grid

class GambiaDroughtDataset(Dataset):
    def __init__(self, features, targets, valid_pixels, seq_len=12):
        # Validate inputs
        assert len(valid_pixels) == 2, "valid_pixels must be (y_indices, x_indices) tuple"
        assert features.shape[:3] == targets.shape, "Feature/target spatial dimensions mismatch"
        
        self.features = features  # (287, 41, 84, 3)
        self.targets = targets    # (287, 41, 84)
        self.valid_pixels = valid_pixels
        self.seq_len = seq_len
        
        # Generate valid indices (pixel_idx, time_idx)
        self.valid_indices = []
        num_pixels = len(valid_pixels[0])
        max_time = features.shape[0] - seq_len
        
        print(f"\nGenerating {seq_len}-month sequences for {num_pixels} pixels...")
        for pixel_idx in tqdm(range(num_pixels)):
            for time_idx in range(max_time):
                self.valid_indices.append((pixel_idx, time_idx))
        
        # Sort by time_idx to maintain temporal order
        self.valid_indices.sort(key=lambda x: x[1])
        print(f"Created {len(self)} total samples ({num_pixels} pixels × {max_time} time steps)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        pixel_idx, time_idx = self.valid_indices[idx]
        y, x = self.valid_pixels[0][pixel_idx], self.valid_pixels[1][pixel_idx]
        
        # Get sequence and target (add 1x1 spatial dim)
        x_seq = self.features[time_idx:time_idx+self.seq_len, y, x, :]  # (seq_len, 3)
        y_target = self.targets[time_idx+self.seq_len, y, x]
        
        return (
            torch.FloatTensor(x_seq).unsqueeze(1).unsqueeze(1),  # (seq_len, 1, 1, 3)
            torch.FloatTensor([y_target]).squeeze()               # scalar
        )

if __name__ == "__main__":
    # Test data processing
    processor = GambiaDataProcessor("/kaggle/input/gambia-data.npz")
    features, targets = processor.process_data()
    
    # Test dataset
    dataset = GambiaDroughtDataset(features, targets, processor.valid_pixels)
    sample_x, sample_y = dataset[0]
    print(f"\nSample shapes - X: {sample_x.shape}, y: {sample_y.shape}")
