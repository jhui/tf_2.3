import tensorflow as tf
import numpy as np

a = tf.constant(
        [[1., 2.],
         [3., 4.],
         [5., 6.]]
        )

print(a.shape)           # (3, 2)
print(a[1, :])           # [3., 4.]
print(a[:, 0:1])         # [[1.], [3.], [5.]]
print(a[:, -1])          # [2., 4., 6.]
print(a[0:2, 0:1])       # [[1.], [3.]]
print(a[2:0:-1, 1:2])    # [[6.], [4.]]

a = tf.constant(
        [[[1., 2.],
          [3., 4.],
          [5., 6.]],
         [[7., 8.],
          [9., 10.],
          [11., 12.]]
        ]
)

print(a[:, :, -1])       # [[ 2.  4.  6.]
                         #  [ 8. 10. 12.]]
print(a[..., 0])         # [[1., 3., 5.],
                         #  [7., 9., 11.]]
print(a.shape)           # (2, 3, 2)
a1 = a[:, tf.newaxis, :, :]
print(a1.shape)          # (2, 1, 3, 2)
print(tf.squeeze(a1).shape) # (2, 3, 2)

b = tf.reshape(a, [1, 12])
print(b)                 # [1., 2., 3., ... , 11., 12.]
# -1 means whatever in reshaping
print(tf.reshape(b, [2, -1])) # [[1. 2. 3. ... 6.]
                              #  [7. 8. 9. ... 12.]]

# Data Creation

t = tf.ones([3, 2])
t = tf.zeros([1, 200, 200, 3])

# float32, range: [0, 1)
t = tf.random.uniform([1, 5, 2])

t = tf.random.normal((2, 3))

inputs = tf.range(10.)[:, None]                  # shape (10, 1)
labels = inputs * 5. + tf.range(5.)[None, :]     # shape (10, 5)

data = [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
]
# [batch, time, features] -> [time, batch, features]
result = tf.transpose(data, [1, 0, 2])

# a = tf.ones([3, 2])
# b = tf.zeros([3, 2])
a = np.zeros((3, 2))
b = np.ones((3, 2))

r = tf.concat([a, b], axis=0)

