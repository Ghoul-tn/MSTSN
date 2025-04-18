import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import argparse
import numpy as np
import tensorflow as tf
from data_preparation import GambiaDataProcessor, create_tf_dataset, train_val_split
from mstsn import EnhancedMSTSN


def get_metrics():
    return [
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.MeanSquaredError(name='mse')
    ]

# Simplified loss function
@tf.function
def drought_loss(y_true, y_pred):
    # Create mask for valid entries (non-NaN in either tensor)
    mask = tf.math.logical_not(
        tf.math.logical_or(
            tf.math.is_nan(y_true),
            tf.math.is_nan(y_pred)
        )
    )
    
    # Apply mask using TensorFlow operations only
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    
    # Compute MSE loss with numerical stability
    squared_errors = tf.square(y_true_masked - y_pred_masked)
    sum_squared = tf.reduce_sum(squared_errors)
    count = tf.cast(tf.size(y_true_masked), tf.float32) + tf.keras.backend.epsilon()
    if tf.math.is_nan(sum_squared) or tf.math.is_inf(sum_squared):
        tf.print("Warning: NaN or Inf in loss computation!")
        return 0.1  # Return a small placeholder loss instead of NaN
    return sum_squared / count

def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced MSTSN for Drought Prediction (TensorFlow/GPU)')
    
    # Data parameters
    parser.add_argument('--data_path', type=str,
                      default='/kaggle/input/time-series-gambia-wuli/time_series_wuli/Gambia,_The_Wuli/Gambia,_The_wuli_combined.npz')
    parser.add_argument('--seq_len', type=int, default=12)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)  # Good default for ~2000 pixels
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=3.0,
                      help='Weight for drought-specific loss component')
    parser.add_argument('--gamma', type=float, default=2.0,
                      help='Focal weight exponent for drought samples')
    
    # System parameters
    parser.add_argument('--results_dir', type=str,
                      default='/kaggle/working/MSTSN/enhanced_results_tf')
    parser.add_argument('--mixed_precision', action='store_true',
                      help='Enable mixed precision training')
    
    return parser.parse_args()

def configure_gpu():
    """Configure GPU and memory growth to prevent TF from allocating all memory"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth needs to be set before GPUs have been initialized
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            
            # Set visible devices to avoid using unwanted GPUs
            tf.config.set_visible_devices(gpus, 'GPU')
            
            # Get GPU information
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  GPU {i}: {details.get('device_name', 'Unknown')}")
            
            return tf.distribute.MirroredStrategy()
        except Exception as e:
            print(f"Error configuring GPUs: {e}")
            print("Falling back to default strategy")
    else:
        print("No GPUs found. Using default strategy (CPU).")
    
    return tf.distribute.get_strategy()

def debug_dataset(ds, steps=2):
    """Manually inspect a few batches of the dataset to verify data shapes"""
    print("\n=== Debugging Dataset ===")
    
    # Extract and print a few samples
    iterator = iter(ds)
    for i in range(steps):
        try:
            x_batch, y_batch = next(iterator)
            print(f"Batch {i+1}:")
            print(f"  X shape: {x_batch.shape}")
            print(f"  Y shape: {y_batch.shape}")
            print(f"  X min/max: {tf.reduce_min(x_batch):.4f}/{tf.reduce_max(x_batch):.4f}")
            print(f"  Y min/max: {tf.reduce_min(y_batch):.4f}/{tf.reduce_max(y_batch):.4f}")
            print(f"  Y mean/std: {tf.reduce_mean(y_batch):.4f}/{tf.math.reduce_std(y_batch):.4f}")
        except StopIteration:
            print("  Dataset exhausted!")
            break
    print("=== End Debug ===\n")

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps=1000, decay_steps=100000):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        
    def __call__(self, step):
        # Convert to float
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        # Linear warmup
        warmup_factor = tf.minimum(1.0, step / warmup_steps)
        
        # Cosine decay after warmup
        step_after_warmup = tf.maximum(0.0, step - warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(
            tf.constant(np.pi) * step_after_warmup / decay_steps
        ))
        
        # Combine both parts
        return self.initial_lr * tf.where(
            step < warmup_steps,
            warmup_factor,
            cosine_decay
        )
    
    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps
        }

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Configure GPU and get strategy
    strategy = configure_gpu()
    
    # Data loading
    print(f"\nLoading data from: {args.data_path}")
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    
    # Add after processor.process_data()
    print(f"NaN check - Features: {np.isnan(features).any()}, Targets: {np.isnan(targets).any()}")
    if np.isnan(features).any() or np.isnan(targets).any():
        print(f"Feature NaNs: {np.isnan(features).sum()}, Target NaNs: {np.isnan(targets).sum()}")    
    print(f"Data loaded: Features shape {features.shape}, Targets shape {targets.shape}")
    print(f"NaN in features: {np.isnan(features).sum()}/{features.size}")
    print(f"NaN in targets: {np.isnan(targets).sum()}/{targets.size}")
    print(f"Feature range: {features.min()} to {features.max()}")
    print(f"Target range: {targets.min()} to {targets.max()}")
    
    # Create static adjacency matrix
    adj_matrix = processor.create_adjacency_on_demand(threshold=10)
    
    # Check for minimum viable dataset size
    if processor.num_nodes < 1:
        print("ERROR: No valid pixels found in dataset. Please check your data.")
        return
    
    print(f"Working with {processor.num_nodes} valid pixels for processing")
    
    # Adjust batch size for distribution strategy if using multiple GPUs
    global_batch_size = args.batch_size
    if strategy.num_replicas_in_sync > 1:
        global_batch_size = args.batch_size * strategy.num_replicas_in_sync
        print(f"Using global batch size of {global_batch_size} for {strategy.num_replicas_in_sync} GPUs")
    
    # Mixed precision
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')  # Use float16 for GPU
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled (float16)')
    
    # Split datasets
    (train_feat, train_targ), (val_feat, val_targ) = train_val_split(features, targets)
    print(f"Training data shape: {train_feat.shape}, Validation data shape: {val_feat.shape}")
    
    # Create datasets
    with strategy.scope():
        train_ds = create_tf_dataset(
            train_feat, train_targ,
            seq_len=args.seq_len,
            batch_size=global_batch_size,
            training=True
        )
        
        val_ds = create_tf_dataset(
            val_feat, val_targ,
            seq_len=args.seq_len,
            batch_size=global_batch_size
        )
        
        # Debug dataset to verify shapes
        debug_dataset(train_ds)
    
    # Calculate steps per epoch
    time_steps_train = train_feat.shape[0] - args.seq_len
    steps_per_epoch = max(1, (time_steps_train * processor.num_nodes) // global_batch_size)
    
    # Calculate validation steps
    time_steps_val = val_feat.shape[0] - args.seq_len
    validation_steps = max(1, (time_steps_val * processor.num_nodes) // global_batch_size)
    
    print(f"Training with {steps_per_epoch} steps per epoch, {validation_steps} validation steps")
    
    # Apply learning rate schedule with warmup
    warmup_steps = min(1000, steps_per_epoch * 5)
    lr_schedule = WarmupCosineDecay(
        initial_lr=args.lr, 
        warmup_steps=warmup_steps,
        decay_steps=args.epochs * steps_per_epoch
    )
    
    # Model configuration
    with strategy.scope():
        model = EnhancedMSTSN(num_nodes=processor.num_nodes, adj_matrix=adj_matrix)
        
        # Create a small dummy input with the correct dimensions for testing
        print("Creating dummy input for model verification...")
        try:
            # Use small batch size and sequence length for testing
            dummy_batch_size = 2
            dummy_seq_len = 12
            dummy_input = tf.zeros([dummy_batch_size, dummy_seq_len, processor.num_nodes, 3], dtype=tf.float32)
            print(f"Dummy input shape: {dummy_input.shape}")
            
            # Try a forward pass
            print("Testing forward pass...")
            output = model(dummy_input)
            print(f"Forward pass successful! Output shape: {output.shape}")
            
            # Try to build the model explicitly
            model.build(input_shape=(None, args.seq_len, processor.num_nodes, 3))
            
            # Try model summary with error handling
            try:
                model.summary()
            except Exception as e:
                print(f"Warning: Could not generate model summary due to: {e}")
                print("This is likely a TensorFlow/Keras issue and won't affect training.")
        except Exception as e:
            print(f"Error during model verification: {e}")
            import traceback
            traceback.print_exc()
            print("Model couldn't be initialized correctly. Please fix the errors above.")
            return
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            global_clipnorm=0.5,  # Reduced from 1.0 to 0.5
            epsilon=1e-5  # Increased from 1e-7 for better stability
        )        
        
        if args.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Use single steps_per_execution for GPU - less aggressive batching compared to TPU
        steps_per_execution = 1
        print(f"Using steps_per_execution: {steps_per_execution}")
      
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),  # Use standard MSE
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae')
            ],
            steps_per_execution=steps_per_execution,
            jit_compile=True  # Enable XLA compilation for better GPU performance
        )

    # Callbacks - updated to use .keras extension
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            min_delta=0.005,
            monitor='val_rmse',
            mode='min',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, 'best_model.keras'),
            monitor='val_rmse',
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.results_dir, 'training_log.csv')
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_rmse',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Add TensorBoard for visualization
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.results_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    try:
        print("\nStarting model training...")
        history = model.fit(
            train_ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        try:
            model.save(os.path.join(args.results_dir, 'final_model.keras'))
            print(f"Model saved to {os.path.join(args.results_dir, 'final_model.keras')}")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Try saving weights only if full model save fails
            model.save_weights(os.path.join(args.results_dir, 'final_model_weights.keras'))
            print(f"Model weights saved to {os.path.join(args.results_dir, 'final_model_weights.keras')}")
        
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
    # Enable XLA compilation for better performance
    tf.config.optimizer.set_jit(True)
    
    # Use deterministic operations for reproducibility
    tf.config.experimental.enable_op_determinism()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    main()
