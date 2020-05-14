import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error, mean_absolute_error


class DiversitySensitiveLoss:
    def __init__(self, alpha, beta, tau, discriminator_loss):
        self.alpha = tf.cast(alpha, tf.float32)
        self.beta = tf.cast(beta, tf.float32)
        self.tau = tau
        self.discriminator_loss = discriminator_loss
        self.diversity_loss = []
        self.distance_loss = []
        self.distribution_loss = []

    def dummy_loss(self, y_true, y_pred):
        return y_pred

    def loss_function(self, tensor):
        y_true, y_pred, noise1, noise2, pred1, pred2 = tensor[0], tensor[1], tensor[2], tensor[3], tensor[4], tensor[5]
        diversity_loss = K.minimum(K.mean(tf.math.divide_no_nan(tf.cast(K.abs(tf.math.subtract(pred2, pred1)), tf.float32),
                                                                tf.cast(K.abs(tf.math.subtract(noise2, noise1)), tf.float32))),
                                   self.tau)
        self.diversity_loss.append(K.mean(diversity_loss))
        distance_loss = mean_absolute_error(y_true, y_pred)
        self.distance_loss.append(K.mean(distance_loss))
        distribution_loss = self.discriminator_loss(y_true, y_pred)
        self.distribution_loss.append(K.mean(distribution_loss))
        output = tf.math.add(distribution_loss,
                             tf.math.subtract(tf.math.multiply(self.beta, distance_loss),
                                              tf.math.multiply(self.alpha, diversity_loss)))
        return output

