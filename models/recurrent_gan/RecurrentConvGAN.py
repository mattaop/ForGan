import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm

from utility.ClipConstraint import ClipConstraint
from utility.split_data import split_sequence
from data.generate_noise import generate_noise
from models.feed_forward_gan.GAN import GAN


class RecurrentConvGAN(GAN):
    def __init__(self):
        GAN.__init__(self)
        self.plot_rate = 100
        self.plot_folder = 'RecurrentConvGAN'
        self.window_size = 24
        self.forecasting_horizon = 1
        self.noise_vector_size = 100  # Try larger vector

        self.optimizer = Adam(0.001, 0.5)
        self.loss_function = 'binary_crossentropy'

    def build_gan(self):
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.loss_function, optimizer=self.optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.noise_vector_size,))
        time_series = Input(shape=(self.window_size, 1))

        forecast = self.generator([time_series, z])
        # For the combined model we will only train the generator
        frozen_discriminator = Model(inputs=self.discriminator.inputs, outputs=self.discriminator.outputs)
        frozen_discriminator.trainable = False
        # self.discriminator.trainable = False

        # Layer that add a dimension as the last axis
        self.expand_dims = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))

        # The valid takes generated images as input and determines validity
        valid = frozen_discriminator([time_series, self.expand_dims(forecast)])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(inputs=[time_series, z], outputs=valid)
        self.combined.compile(loss=self.loss_function, optimizer=self.optimizer)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)

        hist = LSTM(32, return_sequences=False)(historic_inp)
        # hist = ReLU()(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        x = BatchNormalization()(x)
        x = Dense(32)(x)
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

        # define the constraint
        const = ClipConstraint(0.1)

        x = Conv1D(32, kernel_size=4, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(32, kernel_size=4, kernel_constraint=const)(x)
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

    def train(self, epochs, batch_size=128, data_samples=5000, discriminator_epochs=1):
        paths = ['ims',
                 'ims/' + self.plot_folder
                 ]
        for i in paths:
            if not os.path.exists(i):
                os.makedirs(i)

        # Load the data
        # time_series = generate_arp_data(5, 600, 500)
        time_series = generate_noise(data_samples)
        x_train, y_train = split_sequence(time_series, self.window_size, self.forecasting_horizon)

        half_batch = int(batch_size / 2)
        forecast_mse = np.zeros(epochs)
        kl_divergence = np.zeros(epochs)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for d_epochs in range(discriminator_epochs):
                # Select a random half batch of images
                idx = np.random.randint(0, x_train.shape[0], half_batch)
                historic_time_series = x_train[idx]
                future_time_series = y_train[idx]

                noise = self._generate_noise(half_batch)  # Normalisere til 1

                # Generate a half batch of new images
                gen_forecasts = self.generator.predict([historic_time_series, noise])

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

            noise = self._generate_noise(batch_size)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            historic_time_series = x_train[idx]
            # Train the generator
            g_loss = self.combined.train_on_batch([historic_time_series, noise], valid_y)

            # Measure forecast MSE of generator
            forecast_mse[epoch] = mean_squared_error(future_time_series[:, :, 0], gen_forecasts)
            # kl_divergence[epoch] = sum(self.kl_divergence(future_time_series[:, i, 0], gen_forecasts[:, i])
            #                           for i in range(self.forecasting_horizon))/self.forecasting_horizon

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss, forecast_mse[epoch]))
            # print("KL-divergence: ", kl_divergence[epoch])

            if epoch % self.plot_rate == 0:
                self.plot_distributions(future_time_series[:, :, 0], gen_forecasts,
                                        f'ims/' + self.plot_folder + f'/epoch{epoch:03d}.png')

        plt.figure()
        plt.plot(np.linspace(1, epochs, epochs), forecast_mse, label='Training loss generator')
        plt.legend()
        plt.show()

    def forecast(self, steps=1):
        # time_series = generate_arp_data(5, 600, self.window_size+self.forecasting_horizon+steps-1)
        time_series = generate_noise(self.window_size + self.forecasting_horizon + steps - 1)
        time_series = np.expand_dims(time_series, axis=0)
        forecast = np.zeros([steps, self.forecasting_horizon])
        for i in range(steps):
            noise = self._generate_noise(1)
            forecast[i] = self.generator.predict([time_series[:, i:self.window_size + i], noise])[0]
        plt.figure()
        plt.plot(np.linspace(1, len(time_series[0]), len(time_series[0])), time_series[0], label='real data')
        plt.plot(np.linspace(self.window_size, self.window_size + self.forecasting_horizon + steps - 1,
                             self.forecasting_horizon + steps - 1), forecast, label='forecasted data')
        plt.legend()
        plt.show()
        print('Forecast error:', mean_squared_error(time_series[0, -len(forecast):], forecast))

    def monte_carlo_forecast(self, data, steps=1, mc_forward_passes=500, plot=False):
        time_series = np.expand_dims(data, axis=0)
        forecast = np.zeros([steps, self.forecasting_horizon, mc_forward_passes])
        for i in tqdm(range(steps)):
            for j in range(mc_forward_passes):
                noise = self._generate_noise(1)
                forecast[i, :, j] = self.generator.predict([time_series[:, i:self.window_size + i], noise])[0]
        if plot:
            plt.figure()
            plt.plot(np.linspace(1, len(data[0]), len(data[0])), data[0], label='real data')
            plt.plot(np.linspace(self.window_size, self.window_size + steps, steps), forecast.mean(axis=2)[:, 0],
                     label='forecasted data')
            plt.legend()
            plt.show()
            print('Forecast error:',
                  mean_squared_error(time_series[0, -len(forecast):], forecast.mean(axis=2)[:, 0]))
            print('Forecast standard deviation', np.mean(forecast.std(axis=2)[:, 0], axis=0))
            self.plot_distributions(real_samples=generate_noise(mc_forward_passes), fake_samples=forecast[0, 0])
        return forecast


if __name__ == '__main__':
    gan = RecurrentConvGAN()
    gan.build_gan()
    gan.train(epochs=500, batch_size=512, discriminator_epochs=1)
    gan.monte_carlo_forecast(data=generate_noise(gan.window_size + gan.forecasting_horizon),
                             steps=1, mc_forward_passes=5000, plot=True)

