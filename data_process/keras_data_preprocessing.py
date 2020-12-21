import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data = np.array([[110, 0.2, 0.3], [80, 0.9, 1.0], [150, 1.6, 1.7], ])
layer = preprocessing.Normalization()
layer.adapt(data)
normalized_data = layer(data)

print(normalized_data)

data = [
    "ξεῖν᾽, ἦ τοι μὲν ὄνειροι ἀμήχανοι ἀκριτόμυθοι",
    "γίγνοντ᾽, οὐδέ τι πάντα τελείεται ἀνθρώποισι.",
    "δοιαὶ γάρ τε πύλαι ἀμενηνῶν εἰσὶν ὀνείρων:",
    "αἱ μὲν γὰρ κεράεσσι τετεύχαται, αἱ δ᾽ ἐλέφαντι:",
    "τῶν οἳ μέν κ᾽ ἔλθωσι διὰ πριστοῦ ἐλέφαντος,",
    "οἵ ῥ᾽ ἐλεφαίρονται, ἔπε᾽ ἀκράαντα φέροντες:",
    "οἱ δὲ διὰ ξεστῶν κεράων ἔλθωσι θύραζε,",
    "οἵ ῥ᾽ ἔτυμα κραίνουσι, βροτῶν ὅτε κέν τις ἴδηται.",
]
layer = preprocessing.TextVectorization()
layer.adapt(data)
vectorized_text = layer(data)
print(vectorized_text)

vocab = ["a", "b", "c", "d"]
data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
layer = preprocessing.StringLookup(vocabulary=vocab)
vectorized_data = layer(data)
print(vectorized_data)

### Image data augmentation (on-device)
data_augmentation = keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ]
)

input_shape = (32, 32, 3)
classes = 10
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = preprocessing.Rescaling(1.0 / 255)(x)
outputs = keras.applications.ResNet50(
    weights=None, input_shape=input_shape, classes=classes
)(x)
model = keras.Model(inputs, outputs)

### Normalizing numerical features

(x_train, y_train), _ = keras.datasets.cifar10.load_data()
x_train = x_train.reshape((len(x_train), -1))
input_shape = x_train.shape[1:]
classes = 10

normalizer = preprocessing.Normalization()
normalizer.adapt(x_train)

inputs = keras.Input(shape=input_shape)
x = normalizer(inputs)
outputs = layers.Dense(classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train)

### Encoding string categorical features via one-hot encoding
# Corpus
data = tf.constant(["a", "b", "c", "b", "c", "a"])

# Use StringLookup to build an index of the feature values
indexer = preprocessing.StringLookup()
indexer.adapt(data)

# Use CategoryEncoding to encode the integer indices to a one-hot vector
encoder = preprocessing.CategoryEncoding(output_mode="binary")
encoder.adapt(indexer(data))

# Convert new test data (which includes unknown feature values)
test_data = tf.constant(["a", "b", "c", "d", "e", ""])
encoded_data = encoder(indexer(test_data))
print(encoded_data)

### Encoding integer categorical features via one-hot encoding
data = tf.constant([10, 20, 20, 10, 30, 0])

indexer = preprocessing.IntegerLookup()
indexer.adapt(data)

encoder = preprocessing.CategoryEncoding(output_mode="binary")
encoder.adapt(indexer(data))

test_data = tf.constant([10, 10, 20, 50, 60, 0])
encoded_data = encoder(indexer(test_data))
print(encoded_data)

### Applying the hashing trick to an integer categorical feature

# Sample data: 10,000 random integers with values between 0 and 100,000
data = np.random.randint(0, 100000, size=(10000, 1))

# Use the Hashing layer to hash the values to the range [0, 64]
hasher = preprocessing.Hashing(num_bins=64, salt=1337)

# Use the CategoryEncoding layer to one-hot encode the hashed values
encoder = preprocessing.CategoryEncoding(max_tokens=64, output_mode="binary")
encoded_data = encoder(hasher(data))
print(encoded_data.shape)  # (10000, 64)

### Encoding text as a sequence of token indices

# Define some text data to adapt the layer
data = tf.constant(
    [
        "The Brain is wider than the Sky",
        "For put them side by side",
        "The one the other will contain",
        "With ease and You beside",
    ]
)
# Instantiate TextVectorization with "int" output_mode
text_vectorizer = preprocessing.TextVectorization(output_mode="int")
# Index the vocabulary via `adapt()`
text_vectorizer.adapt(data)

# You can retrieve the vocabulary we indexed via get_vocabulary()
vocab = text_vectorizer.get_vocabulary()
print("Vocabulary:", vocab)

# Create an Embedding + LSTM model
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = layers.Embedding(input_dim=len(vocab), output_dim=64)(x)
outputs = layers.LSTM(1)(x)
model = keras.Model(inputs, outputs)

# Call the model on test data (which includes unknown tokens)
test_data = tf.constant(["The Brain is deeper than the sea"])
test_output = model(test_data)

### Encoding text as a dense matrix of ngrams with multi-hot encoding
# Define some text data to adapt the layer
data = tf.constant(
    [
        "The Brain is wider than the Sky",
        "For put them side by side",
        "The one the other will contain",
        "With ease and You beside",
    ]
)
# Instantiate TextVectorization with "binary" output_mode (multi-hot)
# and ngrams=2 (index all bigrams)
text_vectorizer = preprocessing.TextVectorization(output_mode="binary", ngrams=2)
# Index the bigrams via `adapt()`
text_vectorizer.adapt(data)

vocab = text_vectorizer.get_vocabulary()
print("Vocabulary:", vocab)

vectorized_text = text_vectorizer(["The Brain is deeper than the sea"])

print(
    "Encoded text:\n",
    vectorized_text.numpy(),
    "\n",
)

# Create a Dense model
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# Call the model on test data (which includes unknown tokens)
test_data = tf.constant(["The Brain is deeper than the sea"])
test_output = model(test_data)

print("Model output:", test_output)

### Encoding text as a dense matrix of ngrams with TF-IDF weighting
# Define some text data to adapt the layer
data = tf.constant(
    [
        "The Brain is wider than the Sky",
        "For put them side by side",
        "The one the other will contain",
        "With ease and You beside",
    ]
)
# Instantiate TextVectorization with "tf-idf" output_mode
# (multi-hot with TF-IDF weighting) and ngrams=2 (index all bigrams)
text_vectorizer = preprocessing.TextVectorization(output_mode="tf-idf", ngrams=2)
# Index the bigrams and learn the TF-IDF weights via `adapt()`
text_vectorizer.adapt(data)

print(
    "Encoded text:\n",
    text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
    "\n",
)

# Create a Dense model
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# Call the model on test data (which includes unknown tokens)
test_data = tf.constant(["The Brain is deeper than the sea"])
test_output = model(test_data)
print("Model output:", test_output)





