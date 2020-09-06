import re
import string
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

class DataLoader:
    def __init__(self):
        default_args = {
                "seed": 42,
                "batch_size": 32}
        self.raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            'train',
            validation_split=0.2,
            subset='training',
            **default_args)
        self.raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            'train',
            validation_split=0.2,
            subset='validation',
            **default_args)
        self.raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            'test', **default_args)

    def __call__(self):
        train_ds = self.preprocess(self.raw_train_ds)
        val_ds = self.preprocess(self.raw_val_ds)
        test_ds = self.preprocess(self.raw_test_ds)
        return train_ds, val_ds, test_ds

    def preprocess(self, raw_ds):
        max_features = 10000
        sequence_length = 250
        vectorize_layer = TextVectorization(
            standardize=DataLoader.custom_standardization,
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=sequence_length)
        raw_train_text = self.raw_train_ds.map(lambda text, l: text)
        vectorize_layer.adapt(raw_train_text)

        def vectorize_text(text, label):
            # why do we do this -> expand_dims??
            text = tf.expand_dims(text, -1)
            return vectorize_layer(text), label

        return raw_ds.map(vectorize_text)

    @staticmethod
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        punctuation_regex = '[%s]' % re.escape(string.punctuation)
        return tf.strings.regex_replace(stripped_html, punctuation_regex, '')




