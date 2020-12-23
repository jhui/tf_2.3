import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

model = keras.Sequential(
    [
        layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy: ", test_accuracy)

predictions = model.predict(x_test[:10])
label = np.argmax(predictions[0])

print("label ", label)

# model.fit(np.random.randint(256, size=(1000, 28, 28)), np.random.randint(10, size=(1000, 1)),
#          epochs=1, batch_size=128, validation_split=0.1)


# model = keras.Sequential(
#     [
#         keras.Input(shape=(28, 28)),
#         # [0, 255] to [0, 1.0] range.
#         layers.experimental.preprocessing.Rescaling(1.0 / 255),
#         # (28, 28) -> (28, 28, 1)
#         layers.Reshape(target_shape=(28, 28, 1)),
#         layers.Conv2D(32, 3, activation="relu"),
#         layers.MaxPooling2D(2),
#         layers.Conv2D(32, 3, activation="relu"),
#         layers.MaxPooling2D(2),
#         layers.Conv2D(32, 3, activation="relu"),
#         layers.Flatten(),
#         layers.Dense(128, activation="relu"),
#         layers.Dropout(0.2),
#         layers.Dense(10),
#     ]
# )