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
    parser.add_argument('--batch_size', type=int, default=32)  # TPU-friendly batch size
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
    
    return parser.parse_args()

def configure_tpu():
    # TPU initialization with enhanced configuration
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"Running on TPU: {tpu.master()}")
        return strategy, True  # Return strategy and TPU status
    except Exception as e:
        print(f"TPU initialization failed: {e}")
        strategy = tf.distribute.get_strategy()
        print("Using CPU/GPU strategy")
        return strategy, False  # Return strategy and TPU status


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
    strategy, using_tpu = configure_tpu()

    # Adjust batch size for TPU
    if using_tpu:
        args.batch_size = args.batch_size * strategy.num_replicas_in_sync
    
    # Mixed precision
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_jit(True)
        print('Mixed precision enabled')
    
    # Data loading
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    
    # Split datasets
    (train_feat, train_targ), (val_feat, val_targ) = train_val_split(features, targets)
    
    # Create datasets
    with strategy.scope():
        train_ds = create_tf_dataset(
            train_feat, train_targ,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            training=True
        )
        val_ds = create_tf_dataset(
            val_feat, val_targ,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    
    # Apply learning rate schedule with warmup
    lr_schedule = get_lr_schedule(args.lr)
    
    # Model configuration
    with strategy.scope():
        model = EnhancedMSTSN(num_nodes=processor.num_nodes)
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,  # Changed to use scheduler
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
            steps_per_execution=16  # Critical for TPU performance
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
    
    # Train model
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(args.results_dir, 'final_model.h5'))
    
    # Print final metrics
    print("\nTraining completed. Final metrics:")
    print(f"Best validation RMSE: {min(history.history['val_rmse']):.4f}")
    print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")
    
    # Additional metrics from DroughtMetrics if available
    if 'val_drought_metrics' in history.history:
        drought_metrics = history.history['val_drought_metrics']
        best_epoch = history.history['val_rmse'].index(min(history.history['val_rmse']))
        if best_epoch < len(drought_metrics):
            best_metrics = drought_metrics[best_epoch]
            print(f"Drought-specific metrics at best epoch:")
            for key, value in best_metrics.items():
                print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)
    main()
