# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:02:38 2022

@author: lenovo
"""
# =============================================================================
# importing necessary lybraries
# =============================================================================
import pathlib 
import tensorflow as tf
import random
import os
import IPython.display as display
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import MobileNet

# =============================================================================
# load paths 
# =============================================================================
# data_root_orig
data_root = pathlib.Path("C:\\Users\\lenovo\\Downloads\\flower_photos(copy)\\flower_photos")

i=0
for item in data_root.glob("*/*"):
    if i<11:
        print(item)
    i+=1

all_image_paths = [str(lst) for lst in data_root.glob('*/*')]
random.shuffle(all_image_paths)
all_image_paths[:10]

# =============================================================================
# Attribution
# =============================================================================
for item in data_root.iterdir():
    print(item)
attribution = (data_root/'LICENSE.txt').open(encoding = 'utf-8').readlines()[4:]
# attribution = open('C:/Users/lenovo/Downloads/flower_photos(copy)/flower_photos/LICENSE.txt',encoding = 'utf-8').readlines()[4:]
attribution = [line.split(' CC-BY') for line in attribution]
attribution = dict(attribution)


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attribution[str(image_rel).replace("\\","/")].split(' - ')[:-1])

attribution['roses/466486216_ab13b55763.jpg']
for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()


# =============================================================================
# image labeling
# =============================================================================
label_names = sorted(item.name for item in data_root.glob('*//') if item.is_dir())
label_names

label_to_index = dict((name, index) for index, name in enumerate(label_names))
label_to_index

# Create a labels for all the images
all_images_label = [label_to_index[pathlib.Path(path).parent.name]  for path in all_image_paths]
i=0
for path in all_image_paths:
    if i<6 :
        print(pathlib.Path(path).parent.name)
    i+=1

# =============================================================================
# Load and formate image in tensorflow
# =============================================================================
img_raw = tf.io.read_file(all_image_paths[0])
print(repr(img_raw)[:100]+"...")

# Decoding of image path into tensor image
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.dtype)
print(img_tensor.shape)

# Resize image for the model
img_final = tf.image.resize(img_tensor,[192,192])
img_final = img_final/255.0
img_final.shape
print(img_final.numpy().max())
print(img_final.numpy().min())

# Wraping up these up(Resize, load and image plot) in simple functions for later.
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


image_path = all_image_paths[0]
label = all_images_label[0]

plt.imshow(load_and_preprocess_image(image_path))
plt.grid(False)
plt.xlabel(caption_image(image_path))
plt.title(label_names[label].title())
print()

# =============================================================================
# Build a tf.data.Dataset
# =============================================================================

# The easiest way to build a tf.data.Dataset is using the from_tensor_slices method.

# Slicing the array of strings, results in a dataset of strings:
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)

AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)


plt.figure(figsize=(8,8))
for n, image in enumerate(image_ds.take(4)):
  # plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(caption_image(all_image_paths[n]))
  plt.show()

# =============================================================================
# Building a paired dataset of images and labels
# =============================================================================
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_images_label, tf.int64))

for label in label_ds.take(10):
  print(label_names[label.numpy()])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_images_label))

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds

# =============================================================================
# Training Method basic
# =============================================================================

BATCH_SIZE = 32
image_count = len(all_image_paths)
# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds



# =============================================================================
# 
# There are a few things to note here:
# 
# The order is important.
# 
# A .shuffle after a .repeat would shuffle items across epoch boundaries (some items will be seen twice before others are seen at all).
# A .shuffle after a .batch would shuffle the order of the batches, but not shuffle the items across batches.
# You use a buffer_size the same size as the dataset for a full shuffle. Up to the dataset size, large values provide better randomization, but use more memory.
# 
# The shuffle buffer is filled before any elements are pulled from it. So a large buffer_size may cause a delay when your Dataset is starting.
# 
# The shuffeled dataset doesn't report the end of a dataset until the shuffle-buffer is completely empty. The Dataset is restarted by .repeat, causing another wait for the shuffle-buffer to be filled.
# 
# This last point can be addressed by using the tf.data.Dataset.apply method with the fused tf.data.experimental.shuffle_and_repeat function:
# =============================================================================
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

# =============================================================================
# Pipe the Dataset to the model 
# =============================================================================

# Fetch a copy of MobileNet v2 from tf.keras.applications.

# This will be used for a simple transfer learning example.

# Set the MobileNet weights to be non-trainable:


# mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

# Before you pass the input to the MobilNet model, you need to convert it from a range of [0,1] to [-1,1]:
def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)


# The MobileNet returns a 6x6 spatial grid of features for each image.

# Pass it a batch of images to see:
# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

# Build a model wrapped around MobileNet and use tf.keras.layers.GlobalAveragePooling2D to average over those space dimensions before the output tf.keras.layers.Dense layer:

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation = 'softmax')])


# Now it produces outputs of the expected shape
logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)


# Compile the model to describe the training procedure
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])


# There are 2 trainable variables - the Dense weights and bias
len(model.trainable_variables)
model.summary()

# You are ready to train the model.

# Note that for demonstration purposes you will only run 3 steps per epoch, but normally you would specify the real number of steps, as defined below, before passing it to model.fit()

steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
steps_per_epoch

model.fit(ds, epochs=1, steps_per_epoch=3)

# =============================================================================
# Performance
# =============================================================================
import time
default_timeit_steps = 2*steps_per_epoch+1

def timeit(ds, steps=default_timeit_steps):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(steps+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
  print("Total time: {}s".format(end-overall_start))


# Performance of the model is

ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds

timeit(ds)


# =============================================================================
# Cache 
# =============================================================================
ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds

timeit(ds)
# One disadvantage to using an in memory cache is that the cache must be rebuilt on each run, giving the same startup delay each time the dataset is started:

    
# if the data doesn't fit the memory then use cache file
ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)
ds

timeit(ds)
