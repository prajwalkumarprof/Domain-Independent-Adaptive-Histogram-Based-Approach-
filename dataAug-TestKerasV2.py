import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



data_dir="data/cifar/train/"
data_test="data/cifar/test/"
 
filenames = glob.glob(os.path.join(data_dir, '*/*.jpg'))
#print(filenames)


batch_size = 32
img_height = 180
img_width = 180

num_classes = 9

IMG_SIZE = 180




train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

test_ds=val_ds

plt.figure(figsize=(10, 10))
img_array=[]
labels_array=[]
for images, labels in train_ds.take(1):
   for i in range(9):
     ax = plt.subplot(3, 3, i + 1)
     plt.imshow(images[i].numpy().astype("uint8"))
     plt.title(class_names[labels[i]])
     plt.axis("off")
    
#working
#plt.show()
#print(images[0])

IMG_SIZE = 150

#resize_and_rescale = tf.keras.Sequential([
 # layers.Resizing(IMG_SIZE, IMG_SIZE),
  #layers.Rescaling(1/2)
#])

#plt.figure(figsize=(10, 10))
#result = resize_and_rescale(images)
#plt.imshow(result[0].numpy().astype("int32"))
#plt.show()

#data_augmentation = tf.keras.Sequential([
  #layers.RandomFlip("horizontal_and_vertical"),
  #layers.RandomRotation(0.2),
#])

def resize_and_rescale(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image



model = tf.keras.Sequential([
  #resize_and_rescale,
 # data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
   layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
   layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
  # Rest of your model.
])


#aug_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))
aug_ds = train_ds.map(lambda x, y: (resize_and_rescale(x ), y))
batch_size = 32


AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False, augment=False): 
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set.
 # if augment:
  #  ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
  #              num_parallel_calls=AUTOTUNE)esize_and_rescale

#train_ds = prepare(train_ds, shuffle=True, augment=True)
#val_ds = prepare(val_ds)
#test_ds = prepare(test_ds)


 
#model.compile(
 # optimizer='adam',
 ## loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 # metrics=['accuracy'])

#history =model.fit(
 # aug_ds,
#  validation_data=val_ds,
 # epochs=9
#)


print(aug_ds)

print("-------------------train--------------------")
 
#print(train_ds)

# visulazing augmentation not required to run
#image=images[0]

def random_invert_img(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = (255-x)
  else:
    x
  return x

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
 for img in images : 
  for i in range(9):
    image=img
    image = tf.cast(tf.expand_dims(image, 0), tf.float32)
    #image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=2)
    image = random_invert_img(image)
    augmented_image = resize_and_rescale(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0].numpy().astype("float32"))
    plt.axis("off")
    aug_ds.map(augmented_image)
    #train_ds.add(augmented_image)
  #plt.show()

aug_ds = aug_ds.unbatch()
images = list(aug_ds.map(lambda x, y: x))
labels = list(aug_ds.map(lambda x, y: y))

print(len(labels))
print(len(images))