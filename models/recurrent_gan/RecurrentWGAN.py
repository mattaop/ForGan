import numpy as np
from keras import backend
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop

from util.ClipConstraint import ClipConstraint
from models.recurrent_gan.RecurrentGAN import RecurrentGAN


class RecurrentWGAN(RecurrentGAN):
    def __init__(self):
        RecurrentGAN.__init__(self)
        self.plot_rate = 100
        self.plot_folder = 'RecurrentWGAN'
        self.window_size = 24
        self.forecasting_horizon = 1
        self.noise_vector_size = 10  # Try larger vector

        self.optimizer = RMSprop(lr=0.001)
        self.loss_function = self.wasserstein_loss

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)


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
        # x = Dense(64)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        validity = Dense(1)(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model

    def _get_labels(self, batch_size, real=True):
        if real:
            return np.ones((batch_size, 1))
        else:
            return -np.ones((batch_size, 1))


if __name__ == '__main__':
    gan = RecurrentWGAN()
    gan.build_gan()
    gan.train(epochs=3000, batch_size=512, discriminator_epochs=1)
    gan.monte_carlo_forecast(steps=1, mc_forward_passes=5000)
