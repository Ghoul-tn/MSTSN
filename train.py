import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_sum = self.add_weight(name='total_sum', initializer='zeros')
        self.residual_sum = self.add_weight(name='residual_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        mean_y = tf.reduce_mean(y_true)
        total = tf.reduce_sum(tf.square(y_true - mean_y))
        residual = tf.reduce_sum(tf.square(y_true - y_pred))
        self.total_sum.assign_add(total)
        self.residual_sum.assign_add(residual)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        r2 = 1 - (self.residual_sum / tf.maximum(self.total_sum, 1e-7))
        return r2

    def reset_state(self):
        for var in self.variables:
            var.assign(0.0)

@tf.function
def drought_loss(y_true, y_pred, alpha=3.0, gamma=2.0):
    mask = tf.logical_not(tf.math.is_nan(y_true) | tf.math.is_nan(y_pred))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    
    if tf.equal(tf.size(y_true), 0):
        return tf.constant(0.1, dtype=tf.float32)
    
    base_loss = tf.keras.losses.Huber(delta=0.5)(y_true, y_pred)
    drought_mask = tf.cast(y_true < -0.5, tf.float32)
    error = tf.abs(y_pred - y_true)
    focal_weight = tf.pow(1.0 - tf.exp(-error), gamma)
    drought_err = tf.reduce_mean(focal_weight * error * drought_mask) * alpha
    
    result = base_loss + drought_err
    return tf.where(tf.math.is_nan(result), tf.constant(0.1, dtype=tf.float32), result)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced MSTSN for Drought Prediction (TensorFlow)')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/gambia-upper-river-time-series/Gambia_The_combined.npz')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--results_dir', type=str, default='/kaggle/working/MSTSN/enhanced_results_tf')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--use_tpu', action='store_true', default=True)
    return parser.parse_args()

def configure_distribute_strategy(use_tpu=True):
    try:
        if use_tpu:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            return strategy, True
    except Exception as e:
        print(f"TPU initialization failed: {e}")
    
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
        return strategy, False
    else:
        strategy = tf.distribute.get_strategy()
        return strategy, False

def debug_dataset(ds, steps=2):
    print("\n=== Debugging Dataset ===")
    iterator = iter(ds)
    for i in range(steps):
        try:
            x_batch, y_batch = next(iterator)
            print(f"Batch {i+1}:")
            print(f"  X shape: {x_batch.shape}")
            print(f"  Y shape: {y_batch.shape}")
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
        step = tf.cast(step, tf.float32)
        warmup_factor = tf.minimum(1.0, step / self.warmup_steps)
        step_after_warmup = tf.maximum(0.0, step - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * step_after_warmup / self.decay_steps))
        return self.initial_lr * tf.where(step < self.warmup_steps, warmup_factor, cosine_decay)
    
    def get_config(self):
        return {"initial_lr": self.initial_lr, "warmup_steps": self.warmup_steps, "decay_steps": self.decay_steps}

def configure_tpu_options():
    tf.config.set_soft_device_placement(True)
    print("Enabled soft device placement for TPU compatibility")

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    configure_tpu_options()
    strategy, using_tpu = configure_distribute_strategy(args.use_tpu)

    # Data loading and processing
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    adj_matrix = processor.create_adjacency_on_demand(threshold=10)

    # Calculate proper steps
    (train_feat, train_targ), (val_feat, val_targ) = train_val_split(features, targets)
    time_steps_train = train_feat.shape[0] - args.seq_len
    time_steps_val = val_feat.shape[0] - args.seq_len
    steps_per_epoch = max(1, time_steps_train // args.batch_size)
    validation_steps = max(1, time_steps_val // args.batch_size)

    # Dataset creation
    with strategy.scope():
        train_ds = create_tf_dataset(train_feat, train_targ, args.seq_len, args.batch_size, training=True)
        val_ds = create_tf_dataset(val_feat, val_targ, args.seq_len, args.batch_size)
        debug_dataset(train_ds)

    # Model configuration
    with strategy.scope():
        model = EnhancedMSTSN(num_nodes=processor.num_nodes, adj_matrix=adj_matrix)
        lr_schedule = WarmupCosineDecay(args.lr)
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            global_clipnorm=1.0
        )

        model.compile(
            optimizer=optimizer,
            loss=lambda y_true,y_pred: drought_loss(y_true, y_pred, args.alpha, args.gamma),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.MeanSquaredError(name='mse'),
                R2Score(),
                DroughtMetrics()
            ],
            steps_per_execution=16 if using_tpu else 1
        )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, min_delta=0.005, monitor='val_rmse'),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.results_dir, 'best_model.h5'), save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(args.results_dir, 'training_log.csv')),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.5, patience=5)
    ]

    # Training
    try:
        history = model.fit(
            train_ds.repeat(args.epochs),
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds.repeat(args.epochs),
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        model.save(os.path.join(args.results_dir, 'final_model.h5'))
    except Exception as e:
        print(f"Training failed: {e}")
        model.save_weights(os.path.join(args.results_dir, 'final_weights.h5'))

if __name__ == "__main__":
    tf.config.optimizer.set_jit(True)
    main()
