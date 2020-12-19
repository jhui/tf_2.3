import tensorflow as tf
import numpy as np
from tensorflow import keras

# [0, 1.0)
data = np.random.random_sample(size=(100, 28, 28)).astype(np.float32)

data = np.random.randint(256, size=(100, 28, 28))

# Model
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

