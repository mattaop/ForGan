import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop
from keras import backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults

import tensorflow as tf
from tqdm import tqdm

from models.feed_forward_gan.GAN import GAN
from utility.ClipConstraint import ClipConstraint
from utility.split_data import split_sequence


class ESRNNWGAN(GAN):
    def __init__(self, cfg):
        GAN.__init__(self, cfg)
        self.plot_rate = cfg['plot_rate']
        self.plot_folder = 'ESRNNWGAN'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']
        self.recurrent_forecasting = cfg['recurrent_forecasting']
        if self.recurrent_forecasting:
            self.output_size = 1
        else:
            self.output_size = self.forecasting_horizon

        self.noise_vector_size = cfg['noise_vector_size']  # Try larger vector
        self.discriminator_epochs = cfg['discriminator_epochs']
        self.mc_forward_passes = cfg['mc_forward_passes']

        self.optimizer = RMSprop(lr=cfg['learning_rate'])
        self.loss_function = self.wasserstein_loss
        self.exponential_smoothing = None

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def _get_labels(self, batch_size, real=True):
        if real:
            return np.ones((batch_size, 1))
        else:
            return -np.ones((batch_size, 1))

    def build_model(self):
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.loss_function, optimizer=self.optimizer)

        # The generator takes noise as input and generated forecasts
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

        hist = SimpleRNN(16, return_sequences=False)(historic_inp)
        # hist = Dropout(0.2)(hist)
        # hist = ReLU()(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        # x = BatchNormalization()(x)
        x = Dense(32)(x)
        x = ReLU()(x)
        # x = Dropout(0.4)(x)

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
        x = Dense(32, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
        validity = Dense(1)(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model

    def fit_es(self, train):
        # scaler = MinMaxScaler(feature_range=(10 ** (-10), 1))
        # train = np.array(x[:self.window_size], y)
        print(train.shape)
        trends = [None, 'add', 'add_damped', 'mul']
        seasons = [None, 'add', 'mul']
        best_model_parameters = [None, None, False]  # trend, season, damped
        best_aicc = np.inf
        for trend in trends:
            for season in seasons:
                if trend == 'add_damped':
                    trend = 'add'
                    damped = True
                else:
                    damped = False
                model_es = ExponentialSmoothing(train, seasonal_periods=12,
                                                trend=trend, seasonal=season,
                                                damped=damped)
                model_es = model_es.fit(optimized=True)
                if model_es.aicc < best_aicc:
                    best_model_parameters = [trend, season, damped]
                    best_aicc = model_es.aicc
        model_es = ExponentialSmoothing(train, seasonal_periods=12,
                                        trend=best_model_parameters[0], seasonal=best_model_parameters[1],
                                        damped=best_model_parameters[2])
        model_es = model_es.fit(optimized=True)

        print(model_es.params)
        print('ETS: T=', best_model_parameters[0], ', S=', best_model_parameters[1], ', damped=',
              best_model_parameters[2])
        print('AICc', model_es.aicc)
        self.exponential_smoothing = model_es
        return self.exponential_smoothing

    def fit_gan(self, x, y, epochs=1, batch_size=32, verbose=1):
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

            generator_noise = self._generate_noise(batch_size)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            idx = np.random.randint(0, x.shape[0], batch_size)
            historic_time_series = x[idx]
            # Train the generator
            g_loss = self.combined.train_on_batch([historic_time_series, generator_noise], valid_y)

            # Measure forecast MSE of generator
            forecast_mse[epoch] = mean_squared_error(future_time_series[:, :, 0], gen_forecasts)
            # kl_divergence[epoch] = sum(self.kl_divergence(future_time_series[:, i, 0], gen_forecasts[:, i])
            #                           for i in range(self.forecasting_horizon))/self.forecasting_horizon
            G_loss[epoch] = g_loss
            D_loss[epoch] = d_loss[0]
            # Plot the progress
            if verbose == 1:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss, forecast_mse[epoch]))
            # print("KL-divergence: ", kl_divergence[epoch])

            if epoch % self.plot_rate == 0 and verbose == 1:
                self.plot_distributions(future_time_series[:, :, 0], gen_forecasts,
                                        f'ims/' + self.plot_folder + f'/epoch{epoch:03d}.png')
        return {'mse': forecast_mse, 'G_loss': G_loss, 'D_loss': D_loss, 'Accuracy': 100 * d_loss[1]}

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1):
        # fit Exponential Smoothing
        train = np.concatenate([x[0, :, 0], y[:, 0, 0]])
        exponential_model = self.fit_es(train)
        # transform data
        # pred_es = exponential_model.predict(start=0, end=len(train)-1)
        pred_es = exponential_model.fittedvalues

        es_transform = train-pred_es
        es_transform = np.expand_dims(es_transform, axis=-1)
        x_train, y_train = split_sequence(es_transform, self.window_size, self.output_size)

        # Fit GAN
        self.fit_gan(x_train,  y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        pred_gan = self.forecast(x_train)

        # Combine predictions
        pred = np.expand_dims(pred_es[self.window_size:], axis=-1) + pred_gan
        # Print training mse
        print('ES MSE:', mean_squared_error(y[:, :, 0], pred_es[self.window_size:]))
        print('GAN MSE:', mean_squared_error(y_train[:, :, 0], pred_gan))
        print('Mean squared error:', mean_squared_error(y[:, :, 0], pred))

    def forecast(self, x):
        # forecast exponential smoothing
        es_forecasts = self.exponential_smoothing.forecast(steps=x.shape[0])

        forecast = np.zeros([x.shape[0], self.mc_forward_passes, self.output_size])
        for i in tqdm(range(x.shape[0])):
            generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
            x_input = np.vstack([np.expand_dims(x[i], axis=0)] * self.mc_forward_passes)
            forecast[i] = self.generator.predict([x_input, generator_noise])
        return forecast.mean(axis=1)

    def recurrent_forecast(self, time_series):
        time_series = np.vstack([time_series] * self.mc_forward_passes)
        x_input = np.zeros([self.mc_forward_passes, self.window_size + self.forecasting_horizon, 1])
        x_input[:, :self.window_size] = time_series
        for i in range(self.forecasting_horizon):
            generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
            x_input[:, self.window_size+i] = self.generator.predict([x_input[:, i:self.window_size+i], generator_noise])
        return x_input[:, -self.forecasting_horizon:].transpose()[0]

    def monte_carlo_forecast(self, data, steps=1, plot=False):
        # forecast ES
        es_series = np.expand_dims(self.exponential_smoothing.forecast(steps=steps+self.window_size+self.forecasting_horizon), axis=-1)
        es_series = np.expand_dims(es_series, axis=0)

        data = np.expand_dims(data, axis=0)

        time_series = data - es_series[:, :-self.forecasting_horizon]

        es_forecasts = np.zeros([steps, self.forecasting_horizon, 1])
        gan_forecast = np.zeros([steps, self.forecasting_horizon, self.mc_forward_passes])
        for i in tqdm(range(steps)):
            # forecast ES
            es_forecasts[i] = es_series[:, self.window_size + i:self.window_size + i+self.forecasting_horizon]
            if self.recurrent_forecasting:
                gan_forecast[i] = self.recurrent_forecast(time_series[:, i:self.window_size + i])
            else:
                generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
                x_input = np.vstack([time_series[:, i:self.window_size + i]]*self.mc_forward_passes)
                gan_forecast[i] = self.generator.predict([x_input, generator_noise]).transpose()
        forecast = gan_forecast + np.dstack([es_forecasts]*self.mc_forward_passes)
        return forecast
