import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = '/usr/share/grpc/roots.pem'
import argparse
import numpy as np
import tensorflow as tf
from data_preparation import GambiaDataProcessor, create_tf_dataset, train_val_split
from mstsn import EnhancedMSTSN


class DroughtMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='drought_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.drought_rmse = self.add_weight(name='drmse', initializer='zeros')
        self.false_alarm = self.add_weight(name='fa', initializer='zeros')
        self.detection_rate = self.add_weight(name='dr', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        drought_mask = tf.cast(y_true < -0.5, tf.float32)
        safe_mask = 1.0 - drought_mask
        
        # Drought RMSE
        squared_errors = tf.square(y_true - y_pred) * drought_mask
        sum_squared_errors = tf.reduce_sum(squared_errors)
        total_drought = tf.maximum(tf.reduce_sum(drought_mask), 1e-7)
        self.drought_rmse.assign_add(tf.sqrt(sum_squared_errors / total_drought))
        
        # False alarm rate
        false_alarms = tf.cast((y_pred < -0.5) & (y_true >= -0.5), tf.float32) * safe_mask
        self.false_alarm.assign_add(tf.reduce_sum(false_alarms) / tf.maximum(tf.reduce_sum(safe_mask), 1e-7))
        
        # Detection rate
        correct_detections = tf.cast((y_pred < -0.5) & (y_true < -0.5), tf.float32) * drought_mask
        self.detection_rate.assign_add(tf.reduce_sum(correct_detections) / tf.maximum(total_drought, 1e-7))
        
        self.count.assign_add(1.0)

    def result(self):
        return {
            'drought_rmse': self.drought_rmse / self.count,
            'false_alarm': self.false_alarm / self.count,
            'detection_rate': self.detection_rate / self.count
        }

    def reset_state(self):
        for var in self.variables:
            var.assign(0.0)

@tf.function
def drought_loss(y_true, y_pred, alpha=3.0, gamma=2.0):
    base_loss = tf.keras.losses.Huber(delta=0.5)(y_true, y_pred)
    
    drought_mask = tf.cast(y_true < -0.5, tf.float32)
    error = tf.abs(y_pred - y_true)
    focal_weight = tf.pow(1.0 - tf.exp(-error), gamma)
    drought_err = tf.reduce_mean(focal_weight * error * drought_mask) * alpha
    
    return base_loss + drought_err

def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced MSTSN for Drought Prediction (TensorFlow)')
    
    # Data parameters
    parser.add_argument('--data_path', type=str,
                      default='/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz')
    parser.add_argument('--seq_len', type=int, default=12)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4)  # Reduced batch size for small dataset
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=3.0,
                      help='Weight for drought-specific loss component')
    parser.add_argument('--gamma', type=float, default=2.0,
                      help='Focal weight exponent for drought samples')
    
    # System parameters
    parser.add_argument('--results_dir', type=str,
                      default='/kaggle/working/MSTSN/enhanced_results_tf')
    parser.add_argument('--mixed_precision', action='store_true',
                      help='Enable bfloat16 mixed precision training')
    parser.add_argument('--use_tpu', action='store_true', default=False,
                      help='Attempt to use TPU if available')
    
    return parser.parse_args()

def configure_distribute_strategy(use_tpu=False):
    """Configure appropriate distribution strategy based on available hardware and user preference"""
    if use_tpu:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print(f"Running on TPU: {tpu.master()}")
            return strategy, True
        except Exception as e:
            print(f"TPU initialization failed: {e}")
            print("Falling back to CPU/GPU strategy")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s). Using MirroredStrategy.")
        strategy = tf.distribute.MirroredStrategy()
        return strategy, False
    else:
        print("No GPUs found. Using default strategy (CPU).")
        strategy = tf.distribute.get_strategy()
        return strategy, False

# Modified create_tf_dataset function to handle small datasets better
def create_robust_tf_dataset(features, targets, seq_len=12, batch_size=16, training=False, repeat=True):
    """Creates TensorFlow dataset with better handling for small datasets"""
    
    print(f"Creating dataset with {features.shape[0]} time points and {features.shape[1]} nodes")
    
    # Calculate effective dataset size
    max_time = features.shape[0] - seq_len
    dataset_size = max(0, max_time)
    print(f"Dataset contains {dataset_size} effective samples")
    
    # Adjust batch size if needed
    if dataset_size < batch_size and dataset_size > 0:
        orig_batch = batch_size
        batch_size = max(1, dataset_size)
        print(f"WARNING: Dataset too small for batch size {orig_batch}. Reduced to {batch_size}")
    
    def generator():
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
    
    # Make sure dataset isn't empty
    if dataset_size == 0:
        print("WARNING: Empty dataset detected. Creating dummy data for testing.")
        dummy_x = np.zeros((seq_len, features.shape[1], 3), dtype=np.float32)
        dummy_y = np.zeros((features.shape[1],), dtype=np.float32)
        dataset = tf.data.Dataset.from_tensors((dummy_x, dummy_y))

    # Apply repeat to prevent StopIteration errors during TPU training
    if repeat:
        dataset = dataset.repeat()
    
    # Optimize pipeline
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, dataset_size

# Learning rate scheduler for transformer training
def get_lr_schedule(initial_lr, warmup_steps=1000):
    def lr_schedule(step):
        # Linear warmup followed by cosine decay
        if step < warmup_steps:
            return initial_lr * (step / warmup_steps)
        else:
            decay_steps = 100000  # Adjust as needed
            step_after_warmup = step - warmup_steps
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * step_after_warmup / decay_steps))
            return initial_lr * cosine_decay
    
    return lr_schedule

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Get strategy and TPU status
    strategy, using_tpu = configure_distribute_strategy(args.use_tpu)

    # Data loading
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    
    print(f"Data loaded: Features shape {features.shape}, Targets shape {targets.shape}")
    
    # Check for minimum viable dataset size
    if processor.num_nodes < 1:
        print("ERROR: No valid pixels found in dataset. Please check your data.")
        return
    
    print(f"Found {processor.num_nodes} nodes for processing")
    
    # Adjust batch size for distribution strategy
    global_batch_size = args.batch_size
    if using_tpu:
        global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    
    # Mixed precision
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_jit(True)
        print('Mixed precision enabled')
    
    # Split datasets
    (train_feat, train_targ), (val_feat, val_targ) = train_val_split(features, targets)
    
    # Create datasets with improved handling for small datasets
    with strategy.scope():
        train_ds, train_size = create_robust_tf_dataset(
            train_feat, train_targ,
            seq_len=args.seq_len,
            batch_size=global_batch_size,
            training=True
        )
        
        val_ds, val_size = create_robust_tf_dataset(
            val_feat, val_targ,
            seq_len=args.seq_len,
            batch_size=global_batch_size,
            repeat=False  # Don't repeat validation dataset
        )
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, train_size // global_batch_size)
    validation_steps = max(1, val_size // global_batch_size)
    
    print(f"Training with {steps_per_epoch} steps per epoch, {validation_steps} validation steps")
    
    # Apply learning rate schedule with warmup
    lr_schedule = get_lr_schedule(args.lr)
    
    # Model configuration
    with strategy.scope():
        # Check processor.num_nodes exists and is valid
        if processor.num_nodes is None or processor.num_nodes <= 0:
            print("ERROR: Invalid number of nodes detected. Check data processing.")
            return
            
        model = EnhancedMSTSN(num_nodes=processor.num_nodes)
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay
        )
        
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true,y_pred: drought_loss(y_true, y_pred, args.alpha, args.gamma),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                DroughtMetrics()
            ],
            steps_per_execution=4 if using_tpu else 1  # Adjust for dataset size
        )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            min_delta=0.005,
            monitor='val_rmse',
            mode='min',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, 'best_model.h5'),
            monitor='val_rmse',
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.results_dir, 'training_log.csv')
        )
    ]
    
    # Train model with explicit steps
    try:
        history = model.fit(
            train_ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        # Save final model
        model.save(os.path.join(args.results_dir, 'final_model.h5'))
        
        # Print final metrics
        print("\nTraining completed. Final metrics:")
        if history.history.get('val_rmse'):
            print(f"Best validation RMSE: {min(history.history['val_rmse']):.4f}")
        if history.history.get('val_mae'):
            print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)
    main()
