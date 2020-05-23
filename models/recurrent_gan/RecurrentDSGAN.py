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

    def fit(self, x, y,  epochs=1, batch_size=32, verbose=1):
        half_batch = int(batch_size / 2)
        forecast_mse = np.zeros(epochs)
        G_loss = np.zeros(epochs)
        D_loss = np.zeros(epochs)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for d_epochs in range(max(1, self.discriminator_epochs)):
                # Select a random half batch of images
                idx = np.random.randint(0, x.shape[0], half_batch)
                historic_time_series = x[idx]
                future_time_series = y[idx]

                generator_noise = self._generate_noise(half_batch)

                # Generate a half batch of new images
                gen_forecasts = self.generator.predict([historic_time_series, generator_noise])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([historic_time_series, future_time_series],
                                                                self._get_labels(batch_size=half_batch, real=True))
                d_loss_fake = self.discriminator.train_on_batch([historic_time_series,
                                                                tf.keras.backend.expand_dims(gen_forecasts, axis=-1)],
                                                                self._get_labels(batch_size=half_batch, real=False))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            generator_noise_1 = self._generate_noise(batch_size)
            generator_noise_2 = self._generate_noise(batch_size)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            idx = np.random.randint(0, x.shape[0], batch_size)
            historic_time_series = x[idx]
            # Train the generator
            g_loss = self.combined.train_on_batch([historic_time_series, generator_noise_1, generator_noise_2, valid_y], valid_y)

            # Measure forecast MSE of generator
            forecast_mse[epoch] = mean_squared_error(future_time_series[:, :, 0], gen_forecasts)
            # kl_divergence[epoch] = sum(self.kl_divergence(future_time_series[:, i, 0], gen_forecasts[:, i])
            #                           for i in range(self.forecasting_horizon))/self.forecasting_horizon
            G_loss[epoch] = g_loss
            D_loss[epoch] = d_loss[0]
            # Plot the progress
            if verbose == 1 and epoch % self.plot_rate == 0:
                if self.print_coverage:
                    idx = np.random.randint(0, x.shape[0], batch_size)
                    historic_time_series = x[idx]
                    future_time_series = y[idx]
                    forecasts = np.zeros([batch_size, self.output_size, 100])
                    for j in range(batch_size):
                        generator_noise = self._generate_noise(100)
                        x_input = np.vstack([np.expand_dims(historic_time_series[j], axis=0)] * 100)

                        forecasts[j] = self.generator.predict([x_input, generator_noise]).transpose()
                    coverage_80_pi = compute_coverage(actual_values=future_time_series[:, :, 0],
                                                      upper_limits=np.quantile(forecasts, q=0.9, axis=-1),
                                                      lower_limits=np.quantile(forecasts, q=0.1, axis=-1))
                    coverage_95_pi = compute_coverage(actual_values=future_time_series[:, :, 0],
                                                      upper_limits=np.quantile(forecasts, q=0.975, axis=-1),
                                                      lower_limits=np.quantile(forecasts, q=0.025, axis=-1))

                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f][Mean forecast mse: %f, "
                          "80%%-PI: %.2f%%, 95%%-PI: %.2f%%]" %
                          (epoch, d_loss[0], 100 * d_loss[1], g_loss, forecast_mse[epoch],
                           mean_squared_error(future_time_series[:, :, 0], np.mean(forecasts, axis=-1)),
                           100 * coverage_80_pi, 100 * coverage_95_pi))
                    # coverage_distance = (coverage_80_pi-0.8)*0.8 + (coverage_95_pi-0.95)*0.95
                    # self.alpha = self.alpha - coverage_distance*self.alpha
                    # print(self.alpha)
                else:
                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                          (epoch, d_loss[0], 100 * (d_loss[1]), g_loss, forecast_mse[epoch]))
        return {'mse': forecast_mse, 'G_loss': G_loss, 'D_loss': D_loss, 'Accuracy': 100 * d_loss[1]}
