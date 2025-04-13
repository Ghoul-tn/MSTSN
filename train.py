import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = '/usr/share/grpc/roots.pem'
import argparse
import numpy as np
import tensorflow as tf
from data_preparation import GambiaDataProcessor, create_tf_dataset, train_val_split
from mstsn import EnhancedMSTSN

def _calculate_drought_loss(y_true, y_pred, alpha=3.0, gamma=2.0):
    # Original loss calculation with valid values only
    base_loss = tf.keras.losses.Huber(delta=0.5)(y_true, y_pred)
    
    drought_mask = tf.cast(y_true < -0.5, tf.float32)
    error = tf.abs(y_pred - y_true)
    focal_weight = tf.pow(1.0 - tf.exp(-error), gamma)
    drought_err = tf.reduce_mean(focal_weight * error * drought_mask) * alpha
    
    # Check for NaN in result and replace with small constant
    result = base_loss + drought_err
    return tf.where(tf.math.is_nan(result), tf.constant(0.1, dtype=tf.float32), result)

@tf.function
def drought_loss(y_true, y_pred, alpha=3.0, gamma=2.0):
    # Check for NaNs
    mask = tf.logical_not(tf.math.is_nan(y_true) | tf.math.is_nan(y_pred))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    
    # Use tf.cond instead of if
    return tf.cond(
        tf.equal(tf.size(y_true), 0),
        lambda: tf.constant(0.1, dtype=tf.float32),
        lambda: _calculate_drought_loss(y_true, y_pred, alpha, gamma)
    )

class DroughtMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='drought_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        # Create separate metric objects to track results
        self.drought_rmse_metric = tf.keras.metrics.Mean(name='drought_rmse')
        self.false_alarm_metric = tf.keras.metrics.Mean(name='false_alarm')
        self.detection_rate_metric = tf.keras.metrics.Mean(name='detection_rate')

    @tf.function(experimental_relax_shapes=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle NaNs if present
        valid_mask = tf.logical_not(tf.math.is_nan(y_true) | tf.math.is_nan(y_pred))
        y_true = tf.boolean_mask(y_true, valid_mask)
        y_pred = tf.boolean_mask(y_pred, valid_mask)
        
        # Skip calculation if no valid data
        if tf.equal(tf.size(y_true), 0):
            return
        
        # Drought mask (SPI < -0.5 indicates drought conditions)
        drought_mask = tf.cast(y_true < -0.5, tf.float32)
        safe_mask = 1.0 - drought_mask
        
        # Total drought and non-drought pixels for calculations
        total_drought = tf.reduce_sum(drought_mask)
        total_safe = tf.reduce_sum(safe_mask)
        
        # Update drought RMSE - if there are drought pixels
        if tf.greater(total_drought, 0):
            # Drought RMSE (errors only for drought pixels)
            squared_errors = tf.square(y_true - y_pred) * drought_mask
            sum_squared_errors = tf.reduce_sum(squared_errors)
            batch_drought_rmse = tf.sqrt(sum_squared_errors / total_drought)
            self.drought_rmse_metric.update_state(batch_drought_rmse)
            
            # Detection rate (correctly predicted droughts / actual droughts)
            correct_detections = tf.reduce_sum(
                tf.cast(tf.logical_and(y_pred < -0.5, y_true < -0.5), tf.float32)
            )
            batch_detection_rate = correct_detections / total_drought
            self.detection_rate_metric.update_state(batch_detection_rate)
        
        # Update false alarm rate - if there are non-drought pixels
        if tf.greater(total_safe, 0):
            # False alarm rate (wrongly predicted droughts / non-drought pixels)
            false_alarms = tf.reduce_sum(
                tf.cast(tf.logical_and(y_pred < -0.5, y_true >= -0.5), tf.float32) * safe_mask
            )
            batch_false_alarm = false_alarms / total_safe
            self.false_alarm_metric.update_state(batch_false_alarm)

    def result(self):
        return {
            'drought_rmse': self.drought_rmse_metric.result(),
            'false_alarm': self.false_alarm_metric.result(),
            'detection_rate': self.detection_rate_metric.result()
        }

    def reset_state(self):
        self.drought_rmse_metric.reset_state()
        self.false_alarm_metric.reset_state()
        self.detection_rate_metric.reset_state()

class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        # Create state variables
        self.sum_squared_residuals = self.add_weight(
            name='sum_squared_residuals', initializer='zeros')
        self.sum_squared_total = self.add_weight(
            name='sum_squared_total', initializer='zeros')
    
    @tf.function(experimental_relax_shapes=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle NaNs if present
        valid_mask = tf.logical_not(tf.math.is_nan(y_true) | tf.math.is_nan(y_pred))
        y_true_valid = tf.boolean_mask(y_true, valid_mask)
        y_pred_valid = tf.boolean_mask(y_pred, valid_mask)
        
        # Skip calculation if no valid data
        if tf.equal(tf.size(y_true_valid), 0):
            return
        
        # Calculate mean of y_true
        y_mean = tf.reduce_mean(y_true_valid)
        
        # Sum of squared residuals (predicted vs actual)
        squared_residuals = tf.reduce_sum(tf.square(y_true_valid - y_pred_valid))
        
        # Sum of squared total (actual vs mean)
        squared_total = tf.reduce_sum(tf.square(y_true_valid - y_mean))
        
        # Update state variables
        self.sum_squared_residuals.assign_add(squared_residuals)
        self.sum_squared_total.assign_add(squared_total)
        
    def result(self):
        # Avoid division by zero
        denominator = tf.maximum(self.sum_squared_total, 1e-10)
        r2 = 1.0 - (self.sum_squared_residuals / denominator)
        
        # Clip to avoid unreasonable values if sum_squared_total is very small
        return tf.clip_by_value(r2, -1.0, 1.0)
            
    def reset_state(self):
        self.sum_squared_residuals.assign(0.0)
        self.sum_squared_total.assign(0.0)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced MSTSN for Drought Prediction (TensorFlow)')
    
    # Data parameters
    parser.add_argument('--data_path', type=str,
                      default='/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz')
    parser.add_argument('--seq_len', type=int, default=12)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)  # Good default for ~2000 pixels
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
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
    
def configure_tpu_options():
    """Configure TF to handle unsupported ops by running them on CPU"""
    # Enable soft device placement to allow ops without TPU implementation to run on CPU
    tf.config.set_soft_device_placement(True)
    print("Enabled soft device placement for TPU compatibility")
    
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
    
    # Create datasets - FIX: Don't add .repeat() here
    with strategy.scope():
        train_ds = create_tf_dataset(
            train_feat, train_targ,
            seq_len=args.seq_len,
            batch_size=global_batch_size,
            training=True  # The generator will handle repeating internally
        )
        
        val_ds = create_tf_dataset(
            val_feat, val_targ,
            seq_len=args.seq_len,
            batch_size=global_batch_size,
            training=False  # No infinite repeat for validation
        )
        
        # Debug dataset to verify shapes
        debug_dataset(train_ds)
    
    # Calculate steps per epoch - with proper estimation for your dataset size
    time_steps = features.shape[0] - args.seq_len
    train_time_steps = train_feat.shape[0] - args.seq_len
    val_time_steps = val_feat.shape[0] - args.seq_len
    
    # Calculate steps based on actual available samples
    steps_per_epoch = max(1, train_time_steps // global_batch_size)
    validation_steps = max(1, val_time_steps // global_batch_size)
    
    print(f"Training with {steps_per_epoch} steps per epoch, {validation_steps} validation steps")
    
    # Apply learning rate schedule with warmup
    warmup_steps = min(1000, steps_per_epoch * 5)
    lr_schedule = WarmupCosineDecay(
        initial_lr=args.lr, 
        warmup_steps=warmup_steps,
        decay_steps=100000
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
            clipnorm=1.0  
        )
        
        # Use smaller steps_per_execution to improve stability on TPU
        steps_per_execution = min(9, max(1, steps_per_epoch // 10)) if using_tpu else 1
        print(f"Using steps_per_execution: {steps_per_execution}")
        
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true,y_pred: drought_loss(y_true, y_pred, args.alpha, args.gamma),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanSquaredError(name='mse'),  # Added MSE
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                R2Score(name='r2'),  # Added R²
                DroughtMetrics()
            ],
            steps_per_execution=steps_per_execution
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
        # Add TensorBoard if you want visualization
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.results_dir, 'logs'),
            update_freq='epoch'
        )
    ]
    
    # Train model
    try:
        print("\nStarting model training...")
        history = model.fit(
            train_ds,  # NOTE: No .repeat() here since our generator will loop infinitely
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,  # No .repeat() needed
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
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
        if history.history.get('val_r2'):
            print(f"Best validation R²: {max(history.history['val_r2']):.4f}")
        if history.history.get('val_mse'):
            print(f"Best validation MSE: {min(history.history['val_mse']):.4f}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Enable XLA compilation for better performance
    tf.config.optimizer.set_jit(True)
    main()
