import numpy as np
from keras import backend
from keras import Model
from keras.layers import *
#from keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam, RMSprop

from keras.layers.merge import _Merge
import tensorflow as tf
from functools import partial

from utility.ClipConstraint import ClipConstraint
from models.recurrent_gan.RecurrentGAN import RecurrentGAN
from models.feed_forward_gan.WGAN import WGAN


class RecurrentWGAN3(RecurrentGAN, WGAN):
    def __init__(self, cfg):
        RecurrentGAN.__init__(self, cfg)
        self.plot_folder = 'RecurrentWGAN'
        self.optimizer = Adam(lr=cfg['learning_rate'])
        self.loss_function = self.wasserstein_loss
        self.gradient_penalty_weight = 10
        self.batch_size = cfg['batch_size']

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def _get_labels(self, batch_size, real=True):
        if real:
            return np.ones((batch_size, 1))
        else:
            return -np.ones((batch_size, 1))

    def build_model(self):
        print('=== Config===', '\nModel name:', self.model_name, '\nNoise vector size:', self.noise_vector_size,
              '\nDiscriminator epochs:', self.discriminator_epochs, '\nGenerator nodes', self.generator_nodes,
              '\nDiscriminator nodes:', self.discriminator_nodes, '\nOptimizer:', self.optimizer,
              '\nLearning rate:', self.learning_rate)
        # Build and compile the discriminator
        self.expand_dims = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))

        critic = self.build_discriminator()
        # self.discriminator.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()

        self.generator.trainable = False

        # self.generator.compile(loss=self.loss_function, optimizer=self.optimizer)

        # The generator takes noise as input and generated forecasts
        z_disc = Input(shape=(self.noise_vector_size,))
        time_series_disc = Input(shape=(self.window_size, 1))
        real_forecast = Input(shape=(self.output_size, 1))

        fake_forecast = self.expand_dims(self.generator([time_series_disc, z_disc]))
        fake = critic([time_series_disc, fake_forecast])
        valid = critic([time_series_disc, real_forecast])

        interpolated_samples = RandomWeightedAverage(batch_size=self.batch_size)([real_forecast, fake_forecast])

        validity_interpolated = critic([time_series_disc, interpolated_samples])

        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_samples)
                                  #gradient_penalty_weight=self.gradient_penalty_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator = Model(inputs=[time_series_disc, real_forecast, z_disc],
                                   outputs=[valid, fake, validity_interpolated])

        self.discriminator.compile(loss=[self.loss_function, self.loss_function, partial_gp_loss],
                                   optimizer=self.optimizer,
                                   loss_weights=[1, 1, 10])

        # For the combined model we will only train the generator
        critic.trainable = False
        self.generator.trainable = True

        z_gen = Input(shape=(self.noise_vector_size,))
        time_series_gen = Input(shape=(self.window_size, 1))
        forecast = self.generator([time_series_gen, z_gen])
        valid = critic([time_series_gen, self.expand_dims(forecast)])

        self.combined = Model(inputs=[time_series_gen, z_gen], outputs=valid)
        self.combined.compile(loss=self.loss_function, optimizer=self.optimizer)

        self.generator.compile(loss=self.loss_function, optimizer=self.optimizer)

    def build_discriminator(self):

        historic_shape = (self.window_size, 1)
        future_shape = (self.output_size, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        x = Concatenate(axis=1)([historic_inp, future_inp])

        # define the constraint
        # const = ClipConstraint(0.01)

        # x = SimpleRNN(self.discriminator_nodes, return_sequences=False, kernel_constraint=const)(x)

        if self.layers == 'lstm':
            x = LSTM(self.discriminator_nodes, return_sequences=False, kernel_initializer=self.weight_init)(x)
        else:
            x = SimpleRNN(self.discriminator_nodes, return_sequences=False, kernel_initializer=self.weight_init)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
            # pass
        """

        x = Conv1D(self.discriminator_nodes, kernel_size=4)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(16, kernel_size=4)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Flatten()(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        """
        x = Dense(self.discriminator_nodes)(x)
        x = LeakyReLU(alpha=0.1)(x)
        validity = Dense(1)(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)
        hist = historic_inp
        if self.layers == 'lstm':
            for i in range(self.num_layers-1):
                hist = LSTM(self.generator_nodes, return_sequences=True)(hist)
            hist = LSTM(self.generator_nodes, return_sequences=False, kernel_initializer=self.weight_init)(hist)
        else:
            for i in range(self.num_layers-1):
                hist = SimpleRNN(self.generator_nodes, return_sequences=True)(hist)
            hist = SimpleRNN(self.generator_nodes, return_sequences=False, kernel_initializer=self.weight_init)(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        if self.dropout:
            x = Dropout(0.2)(x, training=True)
        if self.batch_norm:
            # x = BatchNormalization()(x)
            pass
        x = Dense(self.generator_nodes+self.noise_vector_size)(x)
        x = ReLU()(x)
        if self.dropout:
            x = Dropout(0.4)(x, training=True)
        # x = Dense(16)(x)
        # x = ReLU()(x)
        prediction = Dense(self.output_size)(x)

        model = Model(inputs=[historic_inp, noise_inp], outputs=prediction)
        # model.summary()
        return model

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):

        gradients = backend.gradients(y_pred, averaged_samples)[0]

        gradients_sqr = backend.square(gradients)

        gradients_sqr_sum = backend.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

        gradient_l2_norm = backend.sqrt(gradients_sqr_sum)

        gradient_penalty = backend.square(1 - gradient_l2_norm)

        return backend.mean(gradient_penalty)

    def train_generator_on_batch(self, x, batch_size):
        generator_noise = self._generate_noise(batch_size)
        idx = np.random.randint(0, x.shape[0], batch_size)
        historic_time_series = x[idx]

        valid = np.ones((batch_size, 1))

        # Train the generator
        g_loss = self.combined.train_on_batch([historic_time_series, generator_noise], valid)
        return g_loss

    def train_discriminator_on_batch(self, x, y, half_batch, mask, epoch):
        # Select a random half batch of images
        batch_size = half_batch*2
        valid = np.ones((batch_size, 1))
        fake = - np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))

        d_loss = [0, 0]
        if epoch < 100 or epoch % 100 == 0:
            n_range = 5*self.discriminator_epochs
        else:
            n_range = self.discriminator_epochs
        for i in range(n_range):
            idx = np.random.randint(0, x.shape[0], batch_size)
            historic_time_series = x[idx]
            future_time_series = y[idx]

            generator_noise = self._generate_noise(batch_size)

            # Generate a half batch of new images
            # gen_forecasts = self.generator.predict([historic_time_series, generator_noise])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch([historic_time_series, future_time_series, generator_noise],
                                                       [valid, fake, dummy])
        # print(d_loss)
        return [d_loss[1] - d_loss[0] + d_loss[2], d_loss[3]]


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""
    def __init__(self, batch_size):
        _Merge.__init__(self)
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        weights = backend.random_uniform((self.batch_size, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
