import tensorflow as tf


class PerturbedBPRModel(tf.keras.Model):

    def __init__(self, perturbed_top_k_func, k=100):
        """k should match the k baked into the perturbed top_k func.
        we need k for when performing exact top k in evaluation step."""
        super(PerturbedBPRModel, self).__init__()
        self.perturbed_top_k_func = perturbed_top_k_func
        self.k = k
        self.hidden = tf.keras.layers.Dense(10, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        intermediate = self.hidden(inputs)
        outputs = self.output_layer(intermediate)
        # squeeze away feature dimension
        outputs = tf.squeeze(outputs, axis=-1)
        return outputs

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            top_100_indicators = self.perturbed_top_k_func(y_pred)
            true_top_100_val, true_top_100_idx = tf.math.top_k(y, k=self.k)

            denominator = tf.reduce_sum(true_top_100_val, axis=-1)
            numerator = tf.reduce_sum(top_100_indicators * y, axis=-1)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(numerator, denominator, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)  # Forward pass
        # use discrete topk to simulate making a decision
        _, pred_100_idx = tf.math.top_k(y_pred, k=self.k)
        true_top_100_val, true_top_100_idx = tf.math.top_k(y, k=self.k)

        denominator = tf.reduce_sum(true_top_100_val, axis=-1)
        numerator = tf.reduce_sum(tf.gather(y, pred_100_idx, batch_dims=-1), axis=-1)

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        self.compiled_loss(numerator, denominator, regularization_losses=self.losses)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}