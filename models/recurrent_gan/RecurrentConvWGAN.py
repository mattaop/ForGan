import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras import backend

from models.recurrent_gan.RecurrentConvGAN import RecurrentConvGAN
from data.generate_noise import generate_noise
from utility.ClipConstraint import ClipConstraint


class RecurrentConvWGAN(RecurrentConvGAN):
    def __init__(self):
        RecurrentConvGAN.__init__(self)
        self.plot_rate = 25
        self.plot_folder = 'RecurrentConvWGAN'
        self.noise_vector_size = 100  # Try larger vector

        self.optimizer = RMSprop(lr=0.001)
        self.loss_function = self.wasserstein_loss

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)

        hist = SimpleRNN(32, return_sequences=False)(historic_inp)
        # hist = ReLU()(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        # x = Dropout(0.2)(x)
        # x = BatchNormalization()(x)
        x = Dense(32)(x)
        # x = Dropout(0.4)(x)
        x = ReLU()(x)
        prediction = Dense(self.forecasting_horizon)(x)

        model = Model(inputs=[historic_inp, noise_inp], outputs=prediction)
        model.summary()
        return model

    def build_discriminator(self):
        historic_shape = (self.window_size, 1)
        future_shape = (self.forecasting_horizon, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        x = Concatenate(axis=1)([historic_inp, future_inp])
        # x = future_inp

        # define the constraint
        const = ClipConstraint(0.1)

        x = Conv1D(32, kernel_size=4, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        # x = Dense(64, kernel_constraint=const)(x)
        # x = LeakyReLU(alpha=0.1)(x)
        x = Dense(64, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
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
    gan = RecurrentConvWGAN()
    gan.build_gan()
    gan.train(epochs=5000, batch_size=256, discriminator_epochs=3)
    gan.monte_carlo_forecast(data=generate_noise(gan.window_size+gan.forecasting_horizon),
                             steps=1, mc_forward_passes=5000, plot=True)
