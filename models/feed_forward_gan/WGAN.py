import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop
# from keras_radam import RAdam
from keras import backend

from models.feed_forward_gan.GAN import GAN
from utility.ClipConstraint import ClipConstraint


class WGAN(GAN):
    def __init__(self, cfg):
        GAN.__init__(self, cfg)
        self.plot_rate = 100
        self.plot_folder = 'feed_forward_WGAN'
        self.noise_vector_size = 100  # Try larger vector

        self.optimizer = RMSprop(lr=cfg['learning_rate'])
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
        validity = Dense(1)(x)

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
    gan.train(epochs=5000, batch_size=64, discriminator_epochs=3)
    gan.monte_carlo_prediction(mc_forward_passes=5000)
