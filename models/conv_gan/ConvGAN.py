import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
from keras_radam import RAdam
import tensorflow as tf
from tqdm import tqdm

from utility.split_data import split_sequence
from utility.adamod import AdaMod
from data.generate_noise import generate_noise
from data.generate_sine import generate_sine_data
from models.recurrent_gan.RecurrentGAN import RecurrentGAN


class ConvGAN(RecurrentGAN):
    def __init__(self, cfg):
        RecurrentGAN.__init__(self, cfg)
        self.plot_folder = 'ConvGAN'
        self.optimizer = RAdam(lr=cfg['learning_rate'])
        self.loss_function = 'binary_crossentropy'

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)

        hist = Conv1D(4, kernel_size=4, activation='relu')(historic_inp)
        # hist = Dropout(0.2)(hist)
        hist = MaxPooling1D(pool_size=2)(hist)
        hist = Flatten()(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        x = BatchNormalization()(x)
        x = Dense(self.window_size + self.forecasting_horizon)(x)
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

        # define the constraint

        x = Conv1D(32, kernel_size=4)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model
