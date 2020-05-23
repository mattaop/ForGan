import numpy as np
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
from utility.compute_statistics import compute_coverage
from models.recurrent_gan.RecurrentGAN import RecurrentGAN
from utility.diversity_sensitive_loss import DiversitySensitiveLoss


class RecurrentDSGAN(RecurrentGAN):
    def __init__(self, cfg):
        RecurrentGAN.__init__(self, cfg)
        self.plot_folder = 'RecurrentConvDSGAN'

        self.optimizer = Adam(cfg['learning_rate'], 0.5)
        self.alpha = 0.01
        self.beta = 0.01
        self.tau = 5
        loss = DiversitySensitiveLoss(self.alpha, self.beta, self.tau, self.discriminator_loss)
        self.generator_loss = loss.dummy_loss
        self.combined_loss = loss.loss_function

    def build_model(self):
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.discriminator_loss, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.generator_loss, optimizer=self.optimizer)

        # The generator takes noise as input and generated forecasts
        z = Input(shape=(self.noise_vector_size,))
        z2 = Input(shape=(self.noise_vector_size,))
        time_series = Input(shape=(self.window_size, 1))

        forecast = self.generator([time_series, z])
        forecast2 = self.generator([time_series, z2])

        # For the combined model we will only train the generator
        frozen_discriminator = Model(inputs=self.discriminator.inputs, outputs=self.discriminator.outputs)
        frozen_discriminator.trainable = False
        # self.discriminator.trainable = False

        # Layer that add a dimension as the last axis
        self.expand_dims = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))

        # The valid takes generated images as input and determines validity
        valid = frozen_discriminator([time_series, self.expand_dims(forecast)])
        y_true = Input(shape=(1,))
        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        loss = Lambda(self.combined_loss, output_shape=(1,))([y_true, valid, z, z2, forecast, forecast2])

        self.combined = Model(inputs=[time_series, z, z2, y_true], outputs=loss)
        self.combined.summary()
        self.combined.compile(loss=self.generator_loss, optimizer=self.optimizer)

    def train_generator_on_batch(self, x, batch_size):
        generator_noise_1 = self._generate_noise(batch_size)
        generator_noise_2 = self._generate_noise(batch_size)

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid_y = np.array([1] * batch_size)

        idx = np.random.randint(0, x.shape[0], batch_size)
        historic_time_series = x[idx]
        # Train the generator
        g_loss = self.combined.train_on_batch([historic_time_series, generator_noise_1, generator_noise_2, valid_y],
                                              valid_y)
        return g_loss
