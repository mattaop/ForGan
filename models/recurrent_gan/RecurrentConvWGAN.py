import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop, Adam
from keras import backend

from models.recurrent_gan.RecurrentWGAN import RecurrentWGAN
from data.generate_noise import generate_noise
from utility.ClipConstraint import ClipConstraint


class RecurrentConvWGAN(RecurrentWGAN):
    def __init__(self, cfg):
        RecurrentWGAN.__init__(self, cfg)
        self.plot_folder = 'RecurrentConvWGAN'

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)

        hist = SimpleRNN(16, return_sequences=False)(historic_inp)
        # hist = ReLU()(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        # x = Dropout(0.2)(x)
        # x = BatchNormalization()(x)
        x = Dense(64)(x)
        # x = Dropout(0.4)(x)
        x = ReLU()(x)
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

        x = Conv1D(self.discriminator_nodes, kernel_size=4, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(self.discriminator_nodes, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)

        validity = Dense(1)(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model
