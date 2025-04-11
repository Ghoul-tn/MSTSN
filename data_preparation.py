import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from scipy.interpolate import interp1d

# Modified GambiaDataProcessor in data_preparation.py
class GambiaDataProcessor:
    def __init__(self, data_path):
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
        # Load raw data
        ndvi = self.data['NDVI']
        soil = self.data['SoilMoisture']
        spi = self.data['SPI']
        lst = self.data['LST']

        # Create base mask from soil moisture (2139 valid pixels)
        base_mask = (~np.isnan(soil).all(axis=0)
        
        # Identify LST NaNs within base mask
        lst_nan_mask = np.isnan(lst).any(axis=0)
        combined_mask = base_mask & ~lst_nan_mask
        
        # Final valid pixels (2139 - LST NaNs)
        self.valid_pixels = np.where(combined_mask)
        self.num_nodes = combined_mask.sum()
        print(f"Final valid pixels considering Soil+LST: {self.num_nodes}")

        # Initialize arrays
        features = np.zeros((287, self.num_nodes, 3), dtype=np.float32)  # NDVI, Soil, LST
        targets = np.zeros((287, self.num_nodes), dtype=np.float32)  # SPI

        # Extract valid pixels
        y_idx, x_idx = self.valid_pixels
        for t in range(287):
            # Direct assignment for NDVI and Soil
            features[t, :, 0] = ndvi[t, y_idx, x_idx]
            features[t, :, 1] = soil[t, y_idx, x_idx]
            
            # LST with interpolation
            lst_slice = lst[t, y_idx, x_idx]
            if np.isnan(lst_slice).any():
                # Temporal interpolation for LST
                valid_times = [i for i in range(287) if not np.isnan(lst[i, y_idx, x_idx]).any()]
                f = interp1d(valid_times, lst[valid_times, y_idx, x_idx], 
                            axis=0, fill_value="extrapolate")
                lst_slice = f(t)
            features[t, :, 2] = lst_slice

            # Targets (SPI)
            targets[t, :] = spi[t, y_idx, x_idx]

        # Apply normalizations
        features[:, :, 0] = self.scalers['ndvi'].fit_transform(features[:, :, 0].reshape(-1, 1)).reshape(287, -1)
        features[:, :, 1] = self.scalers['soil'].fit_transform(features[:, :, 1].reshape(-1, 1)).reshape(287, -1)
        features[:, :, 2] = self.scalers['lst'].fit_transform(features[:, :, 2].reshape(-1, 1)).reshape(287, -1)
        targets = self.scalers['spi'].fit_transform(targets.reshape(-1, 1)).reshape(287, -1)

        return features, targets

    def create_adjacency_on_demand(self, threshold=10):
        """Create adjacency matrix only when explicitly requested"""
        if self.valid_pixels is None:
            raise ValueError("Must call process_data before creating adjacency matrix")
            
        # Create adjacency matrix if needed for analysis or visualization
        coords = np.column_stack(self.valid_pixels)
        distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(-1))
        adj_matrix = (distances <= threshold).astype(float)
        np.fill_diagonal(adj_matrix, 0)
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

    # TPU-optimized pipeline
    return dataset \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE) \
        .cache()

def train_val_split(features, targets, train_ratio=0.8):
    """Temporal split of dataset"""
    total_samples = features.shape[0] - 12  # seq_len=12
    split_idx = int(total_samples * train_ratio)
    
    train_features = features[:split_idx+12]  # Include sequence window
    train_targets = targets[:split_idx+12]
    
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]
    
    return (train_features, train_targets), (val_features, val_targets)
