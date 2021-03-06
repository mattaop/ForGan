import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras import backend

from models.conv_gan.ConvGAN import ConvGAN
from data.generate_noise import generate_noise
from utility.ClipConstraint import ClipConstraint


class ConvWGAN(ConvGAN):
    def __init__(self, cfg):
        ConvGAN.__init__(self, cfg)
        self.plot_folder = 'ConvWGAN'
        self.optimizer = RMSprop(lr=cfg['learning_rate'])
        self.loss_function = self.wasserstein_loss

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)

        hist = Conv1D(16, kernel_size=4, activation='relu')(historic_inp)
        # hist = Dropout(0.2)(hist)
        hist = MaxPooling1D(strides=4)(hist)
        hist = Flatten()(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        # x = noise_inp
        # x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        # x = Dense(16, activation='relu')(x)
        prediction = Dense(self.output_size)(x)

        model = Model(inputs=[historic_inp, noise_inp], outputs=prediction)
        model.summary()
        return model

    def build_discriminator(self):
        historic_shape = (self.window_size, 1)
        future_shape = (self.output_size, 1)

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
