from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import timeit


@tf.function
def f2(n):
    print("some")


print(f2(1))
print()
print()


# print(tf.constant(1))

# print(f(tf.constant([1, 3]), tf.constant(3)))
# print(f(tf.constant([1, 3]), tf.constant(3)))

# print(f(tf.constant([1, 4]), 3))


@tf.function
def f(n):
    while n > 0:
        print("trace value ", n)
        tf.print("execution value ", n)
        n = n - 1


f(tf.constant(3))
print("next")
print("next")
print("next")

x0 = tf.Variable(3.0)
x1 = tf.Variable(0.0)

with tf.GradientTape() as tape:
  # Update x1 = x1 + x0.
  x1.assign_add(x0)
  # The tape starts recording from x1.
  y = x1**2   # y = (x1 + x0)**2

# This doesn't work.
print(tape.gradient(y, x0))   #dy/dx0 = 2*(x1 + x2)



@tf.function
def f(n):
    for i in range(n):
        print("trace value ", i)
        tf.print("execution value ", i)


f(tf.constant(3))
print("next")
f(3)