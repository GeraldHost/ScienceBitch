#! /usr/bin/env python
import re
import string
import matplotlib.pyplot as plt

import tensorflow as tf
from dataloader import DataLoader
from model import ImdbModel

data_loader = DataLoader()
train_ds, val_ds, test_ds = data_loader()

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = ImdbModel()
model.build((None, 250))

model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    optimizer='adam', 
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

checkpoint_path = "training_1/cp.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(
   train_ds,
   validation_data=val_ds,
   epochs=1,
   callbacks=[cp_callback])

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

