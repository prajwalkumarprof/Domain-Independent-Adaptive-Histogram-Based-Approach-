import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import seaborn as sns


import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#import tensorflow_datasets as tfds
import glob

import matplotlib.pyplot as plt

print(tf.__version__)

data_dir="data/cifar/train/"
data_test="data/cifar/test/"
#image_count = len(list(data_dir.glob('*/*.jpg')))
#print(image_count)
filenames = glob.glob(os.path.join(data_dir, '*/*.jpg'))
print(filenames)


batch_size = 32
img_height = 180
img_width = 180

data_augmentation = tf.keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


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



plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    

 

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))




#---------augm start

IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

aug_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))

AUTOTUNE = tf.data.AUTOTUNE

 

#def resize_and_rescale(image, label):
 # image = tf.cast(image, tf.float32)
 # image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
 # image = (image / 255.0)
  #return image, label

#def augment(image_label, seed):
 # image, label = image_label
 # image, label = resize_and_rescale(image, label)
  #image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
 # new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
 # image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
 # image = tf.image.stateless_random_brightness( image, max_delta=0.5, seed=new_seed)
 # #image = tf.clip_by_value(image, 0, 1)
 # return image, label



 

#---------augm end


train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 9


model = tf.keras.Sequential([
  #resize_and_rescale,
 # data_augmentation,
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes),

 
])
#------------






#-----------
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history =model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)


model.summary()

epochs=3
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']



print ('ACCURECY')
print(acc)
print ('Validation ACCURECY')
print(val_acc)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("Traininggraph.png")
#plt.show()


train_predictions_baseline = model.predict(train_ds)
#test_predictions_baseline = model.predict(test_ds)
print (train_predictions_baseline)
#for i in range(9):
 # print ("prediction: class:")
 # print(np.argmax(train_predictions_baseline[i]))


#------------------- working
predicted_categories = tf.argmax(train_predictions_baseline, axis=1)

print("predicted_categories")
print(predicted_categories)
true_categories = tf.concat([y for x, y in train_ds], axis=0)
print("true_categories")
print(true_categories)
print( "confusion_matrix 1")

cmm=confusion_matrix(predicted_categories, true_categories)
print(cmm)

#correct_labels = tf.concat([item for item in true_categories], axis = 0)
#predicted_labels = tf.concat([item for item in predicted_categories], axis = 0)

print( "baseline_results model.evaluate ")


baseline_results = model.evaluate(train_ds,   verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print('')

#------------------- working befor

 
#y_pred = model.predict_classes(images)
#print(classification_report(val_ds, train_predictions_baseline))



def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions  )
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.savefig("Confusionmatrixheat.png")
  plt.show()

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))



plot_cm(predicted_categories, true_categories)
 
print("labels:")
print(labels[0])
print("train_predictions_baseline")
print( train_predictions_baseline[0])

print ( "CONFUSTION MATRIX")

train_predictions_baseline=train_predictions_baseline[train_predictions_baseline >= 0]
#print (train_predictions_baseline)
 



#cm =tf.math.confusion_matrix(labels,train_predictions_baseline)
#print (cm)
#plot_cm(class_names, train_predictions_baseline)
#-------------------
#classpredictionresult=tf.compat.v1.metrics.mean_per_class_accuracy(
 #   labels,
  #  train_predictions_baseline,
   # num_classes,
     
#)
#print("Class metric avg")
#print(classpredictionresult)
#-------------------


 

#test_loss, test_acc = model.evaluate(image_batch,  class_names, verbose=2)

#print('\nTest accuracy:', test_acc)


#print (model)





