#! /usr/bin/env python
import tensorflow as tf
import numpy as np
from tensorflow import keras

rock_path = "./images/p/p-1.jpg"
img_height = 180
img_width = 180
class_names = ['n', 'p', 'r', 's']


model = tf.keras.models.load_model('model.h5')

model.summary()

img = keras.preprocessing.image.load_img(
    rock_path, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

