import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        # [0, 255] to [0, 1.0] range.
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        # (28, 28) -> (28, 28, 1)
        layers.Reshape(target_shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    # ground truth class label is an integer index
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
)

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

# model.fit(np.random.randint(256, size=(1000, 28, 28)), np.random.randint(10, size=(1000, 1)),
#          epochs=1, batch_size=128, validation_split=0.1)


