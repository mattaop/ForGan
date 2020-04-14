import numpy as np
from keras import backend
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop

from utility.ClipConstraint import ClipConstraint
from models.recurrent_gan.RecurrentGAN import RecurrentGAN
from models.feed_forward_gan.WGAN import WGAN


class RecurrentWGAN(RecurrentGAN, WGAN):
    def __init__(self):
        RecurrentGAN.__init__(self)
        WGAN.__init__(self)
        self.plot_folder = 'RecurrentWGAN'

        self.optimizer = RMSprop(lr=0.001)
        self.loss_function = self.wasserstein_loss

    def build_discriminator(self):

        historic_shape = (self.window_size, 1)
        future_shape = (self.forecasting_horizon, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        x = Concatenate(axis=1)([historic_inp, future_inp])

        # define the constraint
        const = ClipConstraint(0.1)

        x = GRU(64, return_sequences=False, kernel_constraint=const)(x)
        x = BatchNormalization()(x)
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
