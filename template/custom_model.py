import tensorflow as tf


# Create an override model to classify pictures
class SequentialModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SequentialModel, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(10)

    # @tf.function
    def call(self, x):
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


input_data = tf.random.uniform([60, 28, 28])

model = SequentialModel()
result = model(input_data)
print(result)

input_data = tf.random.uniform([60, 28, 28])
labels = tf.random.uniform([60])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(run_eagerly=False, loss=loss_fn)
model.fit(input_data, labels, epochs=3)

print("Running eagerly")
model.compile(run_eagerly=True, loss=loss_fn)
model.fit(input_data, labels, epochs=1)
