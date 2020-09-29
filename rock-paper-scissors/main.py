#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

img_height = 224
img_width = 224
batch_size = 20
data_dir = "images"

def load_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def build_model():
    N_mobile = tf.keras.applications.NASNetMobile(
            input_shape=(img_width, img_height, 3), 
            include_top=False, weights='imagenet')
    N_mobile.trainable = False
    x = N_mobile.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(712, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    preds = tf.keras.layers.Dense(4,activation='softmax')(x)
    model = tf.keras.Model(inputs=N_mobile.input, outputs=preds)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    return model

if __name__ == "__main__":
    train_ds, val_ds = load_data()
    model = build_model()
    epochs=15
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      batch_size=batch_size,
      epochs=epochs)

    model.save("model.h5")

# previous model -> before I found out about NASNetMobile!
# data_augmentation = keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3 )),
#     layers.experimental.preprocessing.RandomRotation(0.1),
#     layers.experimental.preprocessing.RandomZoom(0.1),
#   ])
# 
# model = Sequential([
#   data_augmentation,
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(4)
# ])

