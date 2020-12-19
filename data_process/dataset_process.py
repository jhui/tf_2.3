import tensorflow as tf
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np

ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
for d in ds:
    print(d.numpy())  # Get 1, 2, ..., 4

ds = tf.data.Dataset.from_tensor_slices([[1, 2],
                                         [3, 4],
                                         [5, 6]
                                         ])
for d in ds:
    print(d.numpy())

ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
ds = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))  # data shape (10,)
ds2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
     tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))  # data: (scalar, (100,))

# Combining two dataset, say one with label and one with data
ds = tf.data.Dataset.zip((ds, ds2))

# Dataset for SparseTensor
ds = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([20, 4]))
for d in ds.repeat().batch(2).take(3):  # contain 3 batches each 2 samples
    print("batch ", d.numpy())

### Load the Fashion MNIST as ndarray and create a dataset
train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images / 255
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

### TFRecord
# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec",
                                         "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
ds = tf.data.TFRecordDataset(filenames=[fsns_test_file])
raw_example = next(iter(ds))
# Many projects use serialized record that require an additional decode.
parsed = tf.train.Example.FromString(raw_example.numpy())

### Text DS
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']
file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

ds = tf.data.TextLineDataset(file_paths)
for line in ds.take(5):  # the dataset will contain 5 samples
    print(line.numpy())

ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = ds.interleave(tf.data.TextLineDataset, cycle_length=3)

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)


def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), "0")


# Skip the first line & find those survived only
survivors = titanic_lines.skip(1).filter(survived)

### CSV DS
df = pd.read_csv(titanic_file)
head = df.head()
print(head)

titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

# Load from disk for better performance
titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived")

for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    print("features:")
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived", select_columns=['class', 'fare', 'survived'])

for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

### Consuming sets of files for DS
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
# The root directory contains a directory for each class:
flowers_root = pathlib.Path(flowers_root)
for item in flowers_root.glob("*"):
    print(item.name)

# list_ds contains a list of files for flowers of different classes
list_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))
for f in list_ds.take(5):
    print(f.numpy())


# Read the data and extract the label from the path,
# returning (image, label) pairs:
def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label


labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())

### Batch
# 0, -1, ..., -99
# Each batch contains 4 samples
ds = tf.data.Dataset.range(0, -100, -1).batch(4)

# Pad sample with 0 so the samples in each batch has the same length
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
batch_dataset = dataset.padded_batch(4, padded_shapes=(None,))

for batch in batch_dataset.take(2):
    print(batch.numpy())
    print()

ds = dataset.batch(7, drop_remainder=True)

# Repeat a ds
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.repeat(3).batch(2)
print(list(dataset.as_numpy_iterator()))

# shuffle a dataset
ds = tf.data.Dataset.range(100).shuffle(3)

print(list(ds.take(12).as_numpy_iterator()))

### Preprocessing data
list_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-2]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    return image, label


images_ds = list_ds.map(parse_image)

file_path = next(iter(list_ds))
image, label = parse_image(file_path)


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')


show(image, label)
plt.show()

for image, label in images_ds.take(2):
    show(image, label)
plt.show()

### Use scipy to argument data
import scipy.ndimage as ndimage


def random_rotate_image(image):
    image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image


def tf_random_rotate_image(image, label):
    im_shape = image.shape
    [image, ] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image, label


rot_ds = images_ds.map(tf_random_rotate_image)

image, label = next(iter(images_ds))
image = random_rotate_image(image)
show(image, label)
plt.show()

for image, label in rot_ds.take(2):
    show(image, label)
    plt.show()

# Parsing tf.Example and create a ds
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec",
                                         "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
dataset = tf.data.TFRecordDataset(filenames=[fsns_test_file])


def tf_parse(eg):
    print("eg ", eg)
    example = tf.io.parse_example(
        eg[tf.newaxis],
        {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
        })
    return example['image/encoded'][0], example['image/text'][0]


decoded_ds = dataset.map(tf_parse)

# manually examined the features
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

feature = parsed.features.feature
raw_img = feature['image/encoded'].bytes_list.value[0]
img = tf.image.decode_png(raw_img)
plt.imshow(img)
plt.axis('off')
_ = plt.title(feature["image/text"].bytes_list.value[0])
plt.show()

### Time series
range_ds = tf.data.Dataset.range(100000)
batches = range_ds.batch(10, drop_remainder=True)

for batch in batches.take(5):
    print(batch.numpy())


def dense_1_step(batch):
    # Shift features and labels one step relative to each other.
    return batch[:-1], batch[1:]


predict_dense_1_step = batches.map(dense_1_step)

for features, label in predict_dense_1_step.take(3):
    print(features.numpy(), " => ", label.numpy())

batches = range_ds.batch(15, drop_remainder=True)


def label_next_5_steps(batch):
    return (batch[:-5],  # Take all except the last 5
            batch[-5:])  # take the remainder


predict_5_steps = batches.map(label_next_5_steps)

for features, label in predict_5_steps.take(3):
    print(features.numpy(), " => ", label.numpy())

feature_length = 10
label_length = 5

features = range_ds.batch(feature_length, drop_remainder=True)
labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:-5])

predict_5_steps = tf.data.Dataset.zip((features, labels))

for features, label in predict_5_steps.take(3):
    print(features.numpy(), " => ", label.numpy())

### Window
window_size = 5

windows = range_ds.window(window_size, shift=1)
for sub_ds in windows.take(5):
    print(sub_ds)

for x in windows.flat_map(lambda x: x).take(30):
    print(x.numpy(), end=' ')


def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)


for example in windows.flat_map(sub_to_batch).take(5):
    print(example.numpy())


def make_window_dataset(ds, window_size=5, shift=1, stride=1):
    windows = ds.window(window_size, shift=shift, stride=stride)

    def sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True)

    windows = windows.flat_map(sub_to_batch)
    return windows


ds = make_window_dataset(range_ds, window_size=10, shift=5, stride=3)

for example in ds.take(10):
    print(example.numpy())

dense_labels_ds = ds.map(dense_1_step)

for inputs, labels in dense_labels_ds.take(3):
    print(inputs.numpy(), "=>", labels.numpy())

### Resamplling
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip',
    fname='creditcard.zip',
    extract=True)

csv_path = zip_path.replace('.zip', '.csv')

creditcard_ds = tf.data.experimental.make_csv_dataset(
    csv_path, batch_size=1024, label_name="Class",
    # Set the column types: 30 floats and an int.
    column_defaults=[float()] * 30 + [int()])


# Discover the samples' classes are highly unbalanced
def count(counts, batch):
    features, labels = batch  # (labels: Tensor(1024, ) int32
    class_1 = labels == 1
    class_1 = tf.cast(class_1, tf.int32)

    class_0 = labels == 0
    class_0 = tf.cast(class_0, tf.int32)

    counts['class_0'] += tf.reduce_sum(class_0)
    counts['class_1'] += tf.reduce_sum(class_1)

    return counts


counts = creditcard_ds.take(10).reduce(
    initial_state={'class_0': 0, 'class_1': 0},
    reduce_func=count)

counts = np.array([counts['class_0'].numpy(),
                   counts['class_1'].numpy()]).astype(np.float32)

fractions = counts / counts.sum()
print(fractions)  # [0.9957 0.0043]

# Datasets sampling
# Use filter to create datasets for different classes
negative_ds = (
    creditcard_ds
        .unbatch()
        .filter(lambda features, label: label == 0)
        .repeat())
positive_ds = (
    creditcard_ds
        .unbatch()
        .filter(lambda features, label: label == 1)
        .repeat())

# Form a balanced ds
balanced_ds = tf.data.experimental.sample_from_datasets(
    [negative_ds, positive_ds], [0.5, 0.5]).batch(10)


### Rejection resampling
# Just return the label
def class_func(features, label):
    return label


# Configure a resampler containing a target distribution,
# and optionally an initial distribution estimate.
resampler = tf.data.experimental.rejection_resample(
    class_func, target_dist=[0.5, 0.5], initial_dist=fractions)

# Because creditcard_ds is batched, we need to unbatch it first
# to create a new datasorce using the resampler
resample_ds = creditcard_ds.unbatch().apply(resampler).batch(10)

# The resample ds contains (extra_label, sample)
# Since sample contains (lable, data), we can drop the extra_label
balanced_ds = resample_ds.map(lambda extra_label, features_and_label: features_and_label)

for features, labels in balanced_ds.take(10):
    print(labels.numpy())
