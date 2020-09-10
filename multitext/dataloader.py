import re
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from stop_words import get_stop_words

batch_size = 32
max_features = 5000
embedding_dim = 128
sequence_length = 500
stop_words = get_stop_words('en')

# def custom_text_clean(text):
#     lowercase = tf.strings.lower(text)
#     stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
#     punctuation_regex = '[%s]' % re.escape(string.punctuation)
#     x = tf.strings.regex_replace(stripped_html, punctuation_regex, '')
#     return tf.strings.regex_replace(x, '[%s]' % '|'.join(stop_words), '')


def create_vectorize_text(ds):
    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    text_ds = ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    def vectorize_text(text, label):
      text = tf.expand_dims(text, -1)
      return vectorize_layer(text), label

    return vectorize_text

def load_data():
    raw_train_ds = keras.preprocessing.text_dataset_from_directory(
        './data/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=42)

    raw_val_ds = keras.preprocessing.text_dataset_from_directory(
        './data/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=42)

    raw_test_ds = keras.preprocessing.text_dataset_from_directory(
        './data/test', batch_size=batch_size)

    vectorize_text = create_vectorize_text(raw_train_ds)

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

