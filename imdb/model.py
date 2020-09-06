import tensorflow as tf

embedding_dim = 16
max_features = 10000

class ImdbModel(tf.keras.Model):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(max_features + 1, embedding_dim)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.globalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout1(x)
        x = self.globalAveragePooling1D(x)
        x = self.dropout2(x)
        return self.dense(x)
