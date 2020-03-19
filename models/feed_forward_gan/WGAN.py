import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop
from keras import backend

from models.feed_forward_gan.GAN import GAN
from util.ClipConstraint import ClipConstraint


class WGAN(GAN):
    def __init__(self):
        GAN.__init__(self)
        self.plot_rate = 100
        self.plot_folder = 'feed_forward_WGAN'
        self.noise_vector_size = 10  # Try larger vector

        self.optimizer = RMSprop(lr=0.0005)
        self.loss_function = self.wasserstein_loss

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def build_discriminator(self):
        future_shape = (1,)
        future_inp = Input(shape=future_shape)

        const = ClipConstraint(0.1)

        x = Dense(64, kernel_constraint=const)(future_inp)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(64, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = Dropout(0.4)(x)
        validity = Dense(1, kernel_constraint=const)(x)

        model = Model(inputs=future_inp, outputs=validity)
        model.summary()

        return model

    def _get_labels(self, batch_size, real=True):
        if real:
            return np.ones((batch_size, 1))
        else:
            return -np.ones((batch_size, 1))


if __name__ == '__main__':
    gan = WGAN()
    gan.build_gan()
    gan.train(epochs=1500, batch_size=1024, discriminator_epochs=3)
    gan.monte_carlo_prediction(mc_forward_passes=5000)
