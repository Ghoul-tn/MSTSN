import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = '/usr/share/grpc/roots.pem'
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
    parser = argparse.ArgumentParser(description='Train Enhanced MSTSN for Drought Prediction (TensorFlow)')
    
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
                      help='Enable bfloat16 mixed precision training')
    parser.add_argument('--use_tpu', action='store_true', default=True,
                      help='Attempt to use TPU if available')
    
    return parser.parse_args()

def configure_distribute_strategy(use_tpu=True):
    """Configure appropriate distribution strategy based on available hardware and user preference"""
    if use_tpu:
        try:
            print("Attempting to initialize TPU...")
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print(f"Running on TPU: {tpu.master()}")
            print(f"TPU cores: {strategy.num_replicas_in_sync}")
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


# # Learning rate scheduler for transformer training
# def get_lr_schedule(initial_lr, warmup_steps=1000):
#     def lr_schedule(step):
#         # Linear warmup followed by cosine decay
#         if step < warmup_steps:
#             return initial_lr * (step / warmup_steps)
#         else:
#             decay_steps = 100000  # Adjust as needed
#             step_after_warmup = step - warmup_steps
#             cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * step_after_warmup / decay_steps))
#             return initial_lr * cosine_decay
    
#     return lr_schedule
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
    
def configure_tpu_options():
    """Configure TF to handle unsupported ops by running them on CPU"""
    # Enable soft device placement to allow ops without TPU implementation to run on CPU
    tf.config.set_soft_device_placement(True)
    print("Enabled soft device placement for TPU compatibility")

# class NanGradientTerminator(tf.keras.callbacks.Callback):
#     def on_batch_end(self, batch, logs=None):
#         if np.isnan(logs.get('loss')):
#             print(f"\nNaN loss detected at batch {batch}, terminating training")
#             self.model.stop_training = True

class GradientNormMonitor(tf.keras.callbacks.Callback):
    def __init__(self, log_frequency=10, threshold=100.0):
        """
        Monitor gradient norms during training to catch potential explosions.
        
        Args:
            log_frequency: Log gradient norms every N batches
            threshold: Alert when gradient norm exceeds this value
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.threshold = threshold
        self.batch_gradients = []
        self.gradient_norms = []
        self.layer_norms = {}
        
    def on_train_begin(self, logs=None):
        # Store the original train_step function
        self.original_train_step = self.model.train_step
        
        # Define a custom train_step to capture gradients
        def train_step_with_grad_logging(data):
            with tf.GradientTape() as tape:
                # Use the model's normal training logic
                result = self.original_train_step(data)
                
            # Get the trainable weights
            weights = self.model.trainable_weights
            
            # Compute gradients
            with tape.stop_recording():
                loss = self.model.compiled_loss(
                    y_pred=result['predictions'] if isinstance(result, dict) else result,
                    y_true=data[1],
                    regularization_losses=self.model.losses
                )
            
            # Get gradients and calculate norm
            gradients = tape.gradient(loss, weights)
            gradient_norm = tf.linalg.global_norm(gradients)
            
            # Store for later analysis
            self.current_batch_gradients = gradients
            self.current_gradient_norm = gradient_norm
            
            # Check for NaN/Inf in gradients
            any_nan = any(tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None)
            any_inf = any(tf.reduce_any(tf.math.is_inf(g)) for g in gradients if g is not None)
            
            if any_nan or any_inf or gradient_norm > self.threshold:
                print(f"\n⚠️ WARNING: Found {'NaN' if any_nan else 'Inf' if any_inf else 'Large'} gradients!")
                print(f"  Gradient norm: {gradient_norm.numpy():.4f}")
                
                # Log per-layer gradient norms to identify the problematic layer
                if gradient_norm > self.threshold:
                    print("\nPer-layer gradient norms:")
                    for i, (w, g) in enumerate(zip(weights, gradients)):
                        if g is not None:
                            layer_name = w.name
                            g_norm = tf.linalg.global_norm([g]).numpy()
                            print(f"  {layer_name}: {g_norm:.4f}")
            
            return result
        
        # Replace the train_step with our custom version
        self.model.train_step = train_step_with_grad_logging
    
    def on_batch_end(self, batch, logs=None):
        # Only log every N batches to avoid too much output
        if hasattr(self, 'current_gradient_norm') and batch % self.log_frequency == 0:
            norm = self.current_gradient_norm.numpy()
            self.gradient_norms.append(norm)
            print(f"\nBatch {batch} - Gradient Norm: {norm:.4f}")
            
            # Store for later analysis
            if hasattr(self, 'current_batch_gradients'):
                self.batch_gradients.append(batch)
    
    def on_train_end(self, logs=None):
        # Restore the original train_step
        if hasattr(self, 'original_train_step'):
            self.model.train_step = self.original_train_step
        
        # Print summary statistics
        if self.gradient_norms:
            print("\n=== Gradient Norm Summary ===")
            print(f"Min: {min(self.gradient_norms):.4f}")
            print(f"Max: {max(self.gradient_norms):.4f}")
            print(f"Mean: {sum(self.gradient_norms)/len(self.gradient_norms):.4f}")
            print(f"Exceeded threshold ({self.threshold}) in {sum(1 for n in self.gradient_norms if n > self.threshold)} batches")

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Enable soft device placement for TPU compatibility
    configure_tpu_options()
    # Get strategy and TPU status
    strategy, using_tpu = configure_distribute_strategy(args.use_tpu)
    
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
    # Create static adjacency matrix - this is the key change
    adj_matrix = processor.create_adjacency_on_demand(threshold=10)
    # Check for minimum viable dataset size
    if processor.num_nodes < 1:
        print("ERROR: No valid pixels found in dataset. Please check your data.")
        return
    
    print(f"Working with {processor.num_nodes} valid pixels for processing")
    
    # Adjust batch size for distribution strategy
    global_batch_size = args.batch_size
    if using_tpu:
        global_batch_size = args.batch_size * strategy.num_replicas_in_sync
        print(f"Using global batch size of {global_batch_size} for TPU")
    
    # Mixed precision
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_jit(True)
        print('Mixed precision enabled')
    
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
    
    # Apply learning rate schedule with warmup - USING FIXED VERSION
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
        except Exception as e:
            print(f"Error during model verification: {e}")
            import traceback
            traceback.print_exc()
            print("Model couldn't be initialized correctly. Please fix the errors above.")
            return
        
        # If we get here, model initialization succeeded
        model.summary()
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            global_clipnorm=0.5,  # REDUCE from 1.0 to 0.5
            epsilon=1e-5  # INCREASE from 1e-7 for better stability
        )        
        if args.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=2**12)
        # Compile with reasonable steps_per_execution for TPU
        steps_per_execution = min(16, max(1, steps_per_epoch // 10)) if using_tpu else 1
        print(f"Using steps_per_execution: {steps_per_execution}")
      
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),  # Use standard MSE
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae')
            ],
            steps_per_execution=steps_per_execution,
            jit_compile=False
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
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_rmse',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # NanGradientTerminator(),
        GradientNormMonitor(log_frequency=5, threshold=50.0)         
    ]
    # callbacks.append(NanGradientTerminator())
    # Train model with explicit steps
    try:
        print("\nStarting model training...")
        history = model.fit(
            train_ds,  # Important: repeat dataset for TPU training
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,  # Important: repeat validation dataset too
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        # Save final model
        try:
            model.save(os.path.join(args.results_dir, 'final_model.h5'))
            print(f"Model saved to {os.path.join(args.results_dir, 'final_model.h5')}")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Try saving weights only if full model save fails
            model.save_weights(os.path.join(args.results_dir, 'final_model_weights.h5'))
            print(f"Model weights saved to {os.path.join(args.results_dir, 'final_model_weights.h5')}")
        
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
    main()
