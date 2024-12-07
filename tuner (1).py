import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

class Tuner:
    def build_model(self, hp):
        model = tf.keras.Sequential()

        # Input layer
        model.add(layers.Input((32, 1)))

        # Add LSTM layers with a tunable number of layers and units
        num_lstm_layers = hp.Int('num_lstm_layers', 1, 4)
        for i in range(num_lstm_layers):
            model.add(layers.LSTM(
                units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                return_sequences=True if i < num_lstm_layers - 1 else False,
                kernel_regularizer=regularizers.l2(1e-6)
            ))

        # Add Dense layers with tunable units and number of layers
        num_dense_layers = hp.Int('num_dense_layers', 1, 3)
        for i in range(num_dense_layers):
            model.add(layers.Dense(
                units=hp.Int(f'dense_units_{i}', min_value=16, max_value=128, step=16),
                activation='relu'
            ))
        # Add a Dropout layer with a tunable dropout rate
        model.add(layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(layers.Dense(1)) # Output layer

        model.compile(
            loss='mean_absolute_error',
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            metrics=['mean_absolute_error']
        )
        return model
