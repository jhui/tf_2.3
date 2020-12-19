import tensorflow as tf
import numpy as np

ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])  # Get 1, 2, ... 4
for d in ds:
    print(d.numpy())

ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
# sample shape (10,)
ds = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
# sample shape  (scalar, (100,))
ds2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
     tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

ds = tf.data.Dataset.zip((ds, ds2))

# Full example using ds
train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images / 255.0
labels = labels.astype(np.int32)

fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(fmnist_train_ds, epochs=2)

# limits to 20 steps in each epochs
model.fit(fmnist_train_ds.repeat(), epochs=2, steps_per_epoch=20)


loss, accuracy = model.evaluate(fmnist_train_ds)
print("Loss :", loss)
print("Accuracy :", accuracy)

# label is ignored in model.predict
result = model.predict(fmnist_train_ds, steps=10)
print(result.shape)

result = model(images[0:1])


### Fake data
# [0, 1.0)
data = np.random.random_sample(size=(100, 28, 28)).astype(np.float32)

model.fit(np.random.randint(256, size=(100, 28, 28)),
          np.random.randint(10, size=(100,)), epochs=2)
predictions = model.predict(np.random.randint(256, size=(100, 28, 28)))
result = model(np.random.randint(256, size=(1, 28, 28)))


