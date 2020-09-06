import tensorflow as tf

class ImdbTrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy = tf.metrics.BinaryAccuracy(threshold=0.0)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss_object(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

