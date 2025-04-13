import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from scipy.interpolate import interp1d
import scipy.sparse

class GambiaDataProcessor:
    def __init__(self, data_path):
        print(f"\nInitializing DataProcessor with data from: {data_path}")
        self.data = np.load(data_path)
        self.valid_pixels = None
        self.num_nodes = None
        self.scalers = {
            'ndvi': MinMaxScaler(feature_range=(0, 1)),
            'soil': StandardScaler(),
            'lst': StandardScaler(),
            'spi': QuantileTransformer(output_distribution='normal')
        }

    def process_data(self):
        """Process and normalize all input data, returns NumPy arrays"""
        print("\n=== Loading and Processing Raw Data ===")
        
        # Load raw data
        ndvi = self.data['NDVI']
        soil = self.data['SoilMoisture']
        spi = self.data['SPI']
        lst = self.data['LST']

        # Create valid pixel mask from SPI (same as your transformer code)
        valid_mask = ~np.isnan(spi[0])  # Use SPI's mask
        self.valid_pixels = np.where(valid_mask)
        self.num_nodes = valid_mask.sum()  # Should be 2152
        print(f"Found {self.num_nodes} valid pixels for processing")

        # Initialize arrays - [time, nodes, features]
        features = np.zeros((287, self.num_nodes, 3), dtype=np.float32)
        targets = np.zeros((287, self.num_nodes), dtype=np.float32)

        # Extract valid pixels
        y_idx, x_idx = self.valid_pixels
        
        # Handle missing values in soil moisture - interpolate if necessary
        if np.isnan(soil).any():
            # Find the indices of valid soil moisture pixels
            soil_valid_mask = ~np.isnan(soil[0])
            soil_valid_pixels = np.where(soil_valid_mask)
            
            # If soil has different valid pixels than SPI, we need to interpolate
            if soil_valid_pixels[0].shape[0] != self.num_nodes:
                print("Interpolating missing soil moisture data...")
                # Create interpolated soil moisture for all valid SPI pixels
                for t in range(287):
                    # Get valid soil values at this time step
                    valid_y, valid_x = np.where(~np.isnan(soil[t]))
                    valid_values = soil[t, valid_y, valid_x]
                    
                    # Only interpolate if we have valid values
                    if len(valid_values) > 0:
                        # Create position arrays for interpolation
                        valid_pos = np.column_stack([valid_y, valid_x])
                        query_pos = np.column_stack([y_idx, x_idx])
                        
                        # Use nearest neighbor interpolation 
                        from scipy.spatial import cKDTree
                        tree = cKDTree(valid_pos)
                        dist, idx = tree.query(query_pos, k=1)
                        
                        # Apply interpolation
                        features[t, :, 1] = valid_values[idx]
                    else:
                        # Use zeros for completely missing time steps
                        features[t, :, 1] = 0.0
            else:
                # Simply extract valid pixels if the masks are the same
                for t in range(287):
                    features[t, :, 1] = np.nan_to_num(soil[t, y_idx, x_idx], nan=0.0)
        else:
            # No NaNs in soil data, process normally
            for t in range(287):
                features[t, :, 1] = soil[t, y_idx, x_idx]

        # Process NDVI and LST which have same valid pixels as SPI
        for t in range(287):
            features[t, :, 0] = np.nan_to_num(ndvi[t, y_idx, x_idx], nan=0.0)
            features[t, :, 2] = np.nan_to_num(lst[t, y_idx, x_idx], nan=0.0)
            targets[t, :] = spi[t, y_idx, x_idx]
        print("\nPre-preprocessing statistics:")
        print(f"NDVI range: {features[:,:,0].min():.2f} to {features[:,:,0].max():.2f}")
        print(f"Soil range: {features[:,:,1].min():.2f} to {features[:,:,1].max():.2f}") 
        print(f"LST range: {features[:,:,2].min():.2f} to {features[:,:,2].max():.2f}")
        print(f"Target SPI range: {targets.min():.2f} to {targets.max():.2f}")
        # Apply normalizations
        features[:, :, 0] = (ndvi_processed - 0.15) / 0.35  # NDVI-specific scaling
        features[:, :, 1] = soil_processed / 100.0          # Soil moisture [0-100] to [0-1]
        features[:, :, 2] = (lst_processed + 10) / 50.0     # Celsius scaling (assuming -10°C to 40°C range)
        features[:, :, 0] = self.scalers['ndvi'].fit_transform(features[:, :, 0].reshape(-1, 1)).reshape(287, self.num_nodes)
        features[:, :, 1] = self.scalers['soil'].fit_transform(features[:, :, 1].reshape(-1, 1)).reshape(287, self.num_nodes)
        features[:, :, 2] = self.scalers['lst'].fit_transform(features[:, :, 2].reshape(-1, 1)).reshape(287, self.num_nodes)
        targets = self.scalers['spi'].fit_transform(targets.reshape(-1, 1)).reshape(287, self.num_nodes)
        print("\nPost-preprocessing statistics:")
        print(f"NDVI range: {features[:,:,0].min():.2f} to {features[:,:,0].max():.2f}")
        print(f"Soil range: {features[:,:,1].min():.2f} to {features[:,:,1].max():.2f}") 
        print(f"LST range: {features[:,:,2].min():.2f} to {features[:,:,2].max():.2f}")
        print(f"Target SPI range: {targets.min():.2f} to {targets.max():.2f}")
        print("Data statistics after processing:")
        print(f"Features - Min: {features.min():.2f}, Max: {features.max():.2f}")
        print(f"Targets - Min: {targets.min():.2f}, Max: {targets.max():.2f}")
        print(f"Feature means: {np.mean(features, axis=(0,1))}")
        print(f"Target mean: {np.mean(targets):.2f}")
        return features, targets
        
    def create_adjacency_on_demand(self, threshold=10):
        """Create adjacency matrix only when explicitly requested"""
        if self.valid_pixels is None:
            raise ValueError("Must call process_data before creating adjacency matrix")
            
        # Create adjacency matrix if needed for analysis or visualization
        coords = np.column_stack(self.valid_pixels)
        distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(-1))
        adj_matrix = (distances <= threshold).astype(np.float32)
        # Normalize by node degree
        degree = adj_matrix.sum(axis=1, keepdims=True)
        adj_matrix = adj_matrix / (degree + 1e-7)  # Avoid division by zero
        return scipy.sparse.csr_matrix(adj_matrix)

def create_tf_dataset(features, targets, seq_len=12, batch_size=16, training=False):
    """Creates TensorFlow dataset with TPU-optimized pipeline"""
    
    def generator():
        max_time = features.shape[0] - seq_len
        for idx in range(max_time):
            start_idx = max(0, idx)
            end_idx = start_idx + seq_len
            x_seq = features[start_idx:end_idx]
            y_target = targets[end_idx-1]  # Predict last step in sequence

            # Pad if sequence is too short
            if len(x_seq) < seq_len:
                padding = np.zeros((seq_len - len(x_seq), *x_seq.shape[1:]), dtype=np.float32)
                x_seq = np.concatenate([padding, x_seq])

            # Apply random masking augmentation
            if training:
                mask = np.random.rand(*x_seq.shape) < 0.1
                x_seq[mask] = 0

            yield x_seq, y_target

    output_signature = (
        tf.TensorSpec(shape=(seq_len, None, 3), dtype=tf.float32),  # [seq_len, nodes, features]
        tf.TensorSpec(shape=(None,), dtype=tf.float32)              # [nodes]
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    dataset = dataset.repeat() 
    # TPU-optimized pipeline with proper repeat
    return dataset \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

def train_val_split(features, targets, train_ratio=0.8):
    """Temporal split of dataset"""
    total_samples = features.shape[0] - 12  # seq_len=12
    split_idx = int(total_samples * train_ratio)
    
    train_features = features[:split_idx+12]  # Include sequence window
    train_targets = targets[:split_idx+12]
    
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]
    
    return (train_features, train_targets), (val_features, val_targets)
