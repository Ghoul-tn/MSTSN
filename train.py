import os
import argparse
import tensorflow as tf
from data_preparation import GambiaDataProcessor, create_tf_dataset
from mstsn import EnhancedMSTSN

def configure_tpu():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(resolver)
        print("Running on TPU:", resolver.master())
        return strategy, True
    except:
        strategy = tf.distribute.get_strategy()
        print("Running on GPU/CPU")
        return strategy, False

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
        self.drought_rmse.assign_add(tf.sqrt(tf.reduce_sum(squared_errors) / tf.maximum(tf.reduce_sum(drought_mask), 1e-7))
        
        # False alarms
        false_alarms = tf.cast((y_pred < -0.5) & (y_true >= -0.5), tf.float32)
        self.false_alarm.assign_add(tf.reduce_sum(false_alarms * safe_mask) / tf.maximum(tf.reduce_sum(safe_mask), 1e-7))
        
        # Detection rate
        correct_detections = tf.cast((y_pred < -0.5) & (y_true < -0.5), tf.float32)
        self.detection_rate.assign_add(tf.reduce_sum(correct_detections) / tf.maximum(tf.reduce_sum(drought_mask), 1e-7))
        
        self.count.assign_add(1.0)

    def result(self):
        return {
            'drought_rmse': self.drought_rmse / self.count,
            'false_alarm': self.false_alarm / self.count,
            'detection_rate': self.detection_rate / self.count
        }

@tf.function
def drought_loss(y_true, y_pred, alpha=3.0, gamma=2.0):
    base_loss = tf.keras.losses.Huber(delta=0.5)(y_true, y_pred)
    drought_mask = tf.cast(y_true < -0.5, tf.float32)
    error = tf.abs(y_pred - y_true)
    focal_weight = tf.pow(1.0 - tf.exp(-error), gamma)
    return base_loss + alpha * tf.reduce_mean(focal_weight * error * drought_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/content/drive/MyDrive/Gambia Upper River/Gambia_The_combined.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--results_dir', default='/content/results')
    args = parser.parse_args()

    strategy, using_tpu = configure_tpu()
    if using_tpu:
        args.batch_size = args.batch_size * strategy.num_replicas_in_sync

    # Enable mixed precision and XLA
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    tf.config.optimizer.set_jit(True)

    # Data pipeline
    processor = GambiaDataProcessor(args.data_path)
    features, targets = processor.process_data()
    train_ds = create_tf_dataset(features[:200], targets[:200], training=True)
    val_ds = create_tf_dataset(features[200:], targets[200:])

    # Model training
    with strategy.scope():
        model = EnhancedMSTSN(processor.num_nodes)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(2e-4),
            loss=lambda y,t: drought_loss(y,t, alpha=3.0, gamma=2.0),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
                DroughtMetrics()
            ],
            steps_per_execution=16 if using_tpu else None
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, min_delta=0.005),
            tf.keras.callbacks.ModelCheckpoint(f'{args.results_dir}/best_model.h5'),
            tf.keras.callbacks.CSVLogger(f'{args.results_dir}/history.csv')
        ]
    )

    model.save(f'{args.results_dir}/final_model.h5')

if __name__ == "__main__":
    main()
