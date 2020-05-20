import numpy as np
from keras import backend
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop

from utility.ClipConstraint import ClipConstraint
from models.recurrent_gan.RecurrentGAN import RecurrentGAN
from models.feed_forward_gan.WGAN import WGAN


class RecurrentWGAN(RecurrentGAN, WGAN):
    def __init__(self, cfg):
        RecurrentGAN.__init__(self, cfg)
        self.plot_folder = 'RecurrentWGAN'
        self.optimizer = RMSprop(lr=cfg['learning_rate'])
        self.loss_function = self.wasserstein_loss

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def _get_labels(self, batch_size, real=True):
        if real:
            return np.ones((batch_size, 1))
        else:
            return -np.ones((batch_size, 1))

    def build_discriminator(self):

        historic_shape = (self.window_size, 1)
        future_shape = (self.output_size, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        x = Concatenate(axis=1)([historic_inp, future_inp])

        # define the constraint
        const = ClipConstraint(0.1)

        x = SimpleRNN(64, return_sequences=False, kernel_constraint=const)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(64, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
        validity = Dense(1)(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model


if __name__ == '__main__':
    gan = RecurrentWGAN()
    gan.build_gan()
    gan.train(epochs=500, batch_size=64, discriminator_epochs=3)
    gan.monte_carlo_forecast(steps=100, mc_forward_passes=500)
