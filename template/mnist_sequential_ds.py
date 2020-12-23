import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

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
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(train_ds, epochs=10)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
print("Test accuracy: ", test_accuracy)

predictions = model.predict(x_test[:10])
label = np.argmax(predictions[0])

print("label ", label)

probabilities = tf.nn.softmax(predictions).numpy()

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

results = probability_model(x_test[:5])
