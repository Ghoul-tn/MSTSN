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
            'spi': QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        }

    def process_data(self):
        """Optimized data processing with memory-efficient operations"""
        print("\n=== Processing Data ===")
        
        # Load data with memory mapping
        with np.load(self.data.files[0]) as data:
            ndvi = data['NDVI']
            soil = data['SoilMoisture']
            spi = data['SPI']
            lst = data['LST']

        # Vectorized valid pixel detection
        valid_mask = (~np.isnan(ndvi)).all(axis=0) & \
                    (~np.isnan(soil)).all(axis=0) & \
                    (~np.isnan(spi)).all(axis=0) & \
                    (~np.isnan(lst)).all(axis=0)
        self.valid_pixels = np.where(valid_mask)
        self.num_nodes = len(self.valid_pixels[0])

        # Pre-allocate arrays
        features = np.zeros((287, self.num_nodes, 3), dtype=np.float32)
        targets = np.zeros((287, self.num_nodes), dtype=np.float32)

        # Vectorized data extraction
        y_idx, x_idx = self.valid_pixels
        features[:, :, 0] = ndvi[:, y_idx, x_idx]  # NDVI
        features[:, :, 1] = soil[:, y_idx, x_idx]   # Soil
        features[:, :, 2] = lst[:, y_idx, x_idx]    # LST
        targets[:, :] = spi[:, y_idx, x_idx]        # SPI

        # Optimized interpolation
        valid_lst = ~np.isnan(features[:, :, 2])
        for n in range(self.num_nodes):
            if np.sum(valid_lst[:, n]) > 1:
                interp_func = interp1d(
                    np.arange(287)[valid_lst[:, n]],
                    features[valid_lst[:, n], n, 2],
                    kind='linear',
                    fill_value="extrapolate"
                )
                features[:, n, 2] = interp_func(np.arange(287))
            else:
                features[:, n, 2] = np.nan_to_num(features[:, n, 2])

        # Parallel normalization
        for i, scaler in enumerate(self.scalers.values()):
            if i < 3:  # Features
                features[:, :, i] = scaler.fit_transform(features[:, :, i].reshape(-1, 1)).reshape(287, self.num_nodes)
            else:       # Targets
                targets = scaler.fit_transform(targets.reshape(-1, 1)).reshape(287, self.num_nodes)

        return features, targets

def create_tf_dataset(features, targets, seq_len=12, batch_size=32, training=False):
    """TPU-optimized dataset pipeline with vectorized operations"""
    
    def generator():
        max_time = features.shape[0] - seq_len
        for i in range(max_time):
            x = features[i:i+seq_len]
            y = targets[i+seq_len-1]
            
            if training and np.random.rand() < 0.3:
                mask = np.random.rand(*x.shape) < 0.1
                x[mask] = 0
                
            yield x.astype(np.float32), y.astype(np.float32)
    
    output_signature = (
        tf.TensorSpec(shape=(seq_len, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    ).batch(batch_size, drop_remainder=True) \
     .prefetch(tf.data.AUTOTUNE) \
     .cache()
