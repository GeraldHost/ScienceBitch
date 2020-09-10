#! /usr/bin/env python
import numpy as np
import tensorflow as tf
import kerastuner as kt
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import callbacks 
from dataloader import load_data

print(tf.__version__)

batch_size = 32
max_features = 5000
embedding_dim = 128
sequence_length = 500

def draw_graphs(history):
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def build_model(hp):
    activation = hp.Choice(
        'dense_activation',
        values=['relu', 'tanh', 'sigmoid', 'softmax'],
        default='relu')

    model = keras.Sequential([
      layers.Embedding(max_features + 1, embedding_dim),
      layers.Dropout(hp.Float('dropout', min_value = 0.1, max_value = 0.5, step = 0.1)),
      layers.GlobalAveragePooling1D(),
      layers.Dropout(hp.Float('dropout', min_value = 0.1, max_value = 0.5, step = 0.1)),
      layers.Dense(4, activation=activation),
    ])

    model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])

    return model

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_data()
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    tuner = kt.Hyperband(
            build_model,
            objective = 'val_accuracy', 
            max_epochs = 10,
            factor = 3,
            directory = 'my_dir',
            project_name = 'second')
    tuner.search(train_ds, epochs = 50, validation_data=val_ds)
    best_hps = tuner.get_best_hyperparameters(num_trials = 2)[0]

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        ])

    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    draw_graphs(history)
