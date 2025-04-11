import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = '/usr/share/grpc/roots.pem'
import argparse
import numpy as np
import tensorflow as tf
from data_preparation import GambiaDataProcessor, create_tf_dataset, train_val_split
from mstsn import EnhancedMSTSN

# TPU initialization with enhanced configuration
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print(f"Running on TPU: {tpu.master()}")
    USE_TPU = True
except Exception as e:
    print(f"TPU initialization failed: {e}")
    strategy = tf.distribute.get_strategy()
    USE_TPU = False
    print("Using CPU/GPU strategy")

# Enable for TPU performance
if USE_TPU:
    tf.config.optimizer.set_jit(True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

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
        sum_squared = tf.reduce_sum(squared_errors)
        total_drought = tf.maximum(tf.reduce_sum(drought_mask), 1e-7)
        self.drought_rmse.assign_add(tf.sqrt(sum_squared / total_drought))
        
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
    return base_loss + alpha * tf.reduce_mean(focal_weight * error * drought_mask)

def get_lr_schedule(initial_lr, warmup_steps=1000):
    def lr_schedule(step):
        step = tf.cast(step, tf.float32)
        if step < warmup_steps:
            return initial_lr * (step / warmup_steps)
        else:
            decay_steps = tf.constant(100000.0)
            step_after = step - warmup_steps
            return initial_lr * 0.5 * (1 + tf.cos(np.pi * step_after / decay_steps))
    return lr_schedule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz')
    parser.add_argument('--batch_size', type=int, default=4)  # Per replica
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--results_dir', default='/kaggle/working/results')
    args = parser.parse_args()

    # TPU-specific batch sizing
    if USE_TPU:
        GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    else:
        GLOBAL_BATCH_SIZE = args.batch_size
    
    # Data loading
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    (train_feat, train_targ), (val_feat, val_targ) = train_val_split(features, targets)

    # Dataset creation with TPU optimizations
    train_ds = create_tf_dataset(
        train_feat, train_targ,
        batch_size=GLOBAL_BATCH_SIZE,
        training=True
    )
    val_ds = create_tf_dataset(
        val_feat, val_targ,
        batch_size=GLOBAL_BATCH_SIZE
    )

    with strategy.scope():
        # Model with reduced dimensions for TPU
        model = EnhancedMSTSN(
            num_nodes=processor.num_nodes
        )
        # Build model with proper input signature
        input_shape = (None, 12, None, 3)  # [batch, seq_len, nodes, features]
        model.build(input_shape)
        # Initialize with example data
        example_input = tf.ones([4, 12, 2139, 3], dtype=tf.float32)  # Match your node count
        _ = model(example_input)
        # Optimizer with learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=2e-4,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-3
        )

        # Compile with TPU-specific options
        model.compile(
            optimizer=optimizer,
            loss=lambda y,t: drought_loss(y, t, alpha=3.0, gamma=2.0),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                DroughtMetrics()
            ],
            steps_per_execution=16 if USE_TPU else None,
            jit_compile=USE_TPU
        )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            min_delta=0.005,
            monitor='val_rmse',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, 'best_model.h5'),
            monitor='val_rmse',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.results_dir, 'logs')
        )
    ]
    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2 if USE_TPU else 1  # Less logging for TPU
    )

    # Final save
    model.save(os.path.join(args.results_dir, 'final_model.h5'))
    print(f"Training completed. Best validation RMSE: {min(history.history['val_rmse']):.4f}")

if __name__ == "__main__":
    # Clear previous session and set random seeds
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Configure for XLA/TPU
    tf.config.optimizer.set_jit(True)
    main()
