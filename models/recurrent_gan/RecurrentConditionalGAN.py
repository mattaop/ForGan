import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import to_categorical

import keras.backend as K
from keras.models import load_model
import tensorflow as tf
from tqdm import tqdm, trange
import time

from config.load_config import load_config_file
from data.generate_noise import generate_noise
from utility.split_data import split_sequence
from utility.compute_statistics import compute_coverage, sliding_window_coverage
from models.feed_forward_gan.GAN import GAN


class RecurrentConditionalGAN(GAN):
    def __init__(self, cfg):
        GAN.__init__(self, cfg)
        self.plot_rate = cfg['plot_rate']
        self.print_coverage = cfg['print_coverage']
        self.plot_folder = 'RecurrentConditionalGAN'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']
        self.recurrent_forecasting = cfg['recurrent_forecasting']
        if self.recurrent_forecasting:
            self.output_size = 1
        else:
            self.output_size = self.forecasting_horizon
        self.new_training_loop = cfg['new_training_loop']
        self.noise_vector_size = cfg['noise_vector_size']
        self.discriminator_epochs = cfg['discriminator_epochs']
        self.mixed_batches = cfg['mixed_batches']
        self.mc_forward_passes = cfg['mc_forward_passes']

        self.layers = cfg['layers']
        self.optimizer = Adam(cfg['learning_rate'], 0.5)
        self.loss_function = 'binary_crossentropy'

    def build_model(self):
        print('=== Config===', '\nModel name:', self.model_name, '\nNoise vector size:', self.noise_vector_size,
              '\nDiscriminator epochs:', self.discriminator_epochs, '\nGenerator nodes', self.generator_nodes,
              '\nDiscriminator nodes:', self.discriminator_nodes, '\nOptimizer:', self.optimizer,
              '\nLearning rate:', self.learning_rate)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.loss_function, optimizer=self.optimizer)

        # The generator takes noise as input and generated forecasts
        z = Input(shape=(self.noise_vector_size,))
        time_series = Input(shape=(self.window_size, 1))
        conditional_input = Input(shape=(108,))

        forecast = self.generator([time_series, z, conditional_input])
        # For the combined model we will only train the generator
        frozen_discriminator = Model(inputs=self.discriminator.inputs, outputs=self.discriminator.outputs)
        frozen_discriminator.trainable = False
        # self.discriminator.trainable = False

        # Layer that add a dimension as the last axis
        self.expand_dims = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))

        # The valid takes generated images as input and determines validity
        valid = frozen_discriminator([time_series, self.expand_dims(forecast), conditional_input])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(inputs=[time_series, z, conditional_input], outputs=valid)
        self.combined.compile(loss=self.loss_function, optimizer=self.optimizer)

    def build_generator(self):

        noise_inp = Input(shape=(self.noise_vector_size,))
        historic_inp = Input(shape=(self.window_size, 1))
        conditional_inp = Input(shape=(108,))

        if self.layers == 'lstm':
            hist = LSTM(self.generator_nodes, return_sequences=False)(historic_inp)
        elif self.layers == 'gru':
            hist = GRU(self.generator_nodes, return_sequences=False)(historic_inp)
        else:
            hist = SimpleRNN(self.generator_nodes, return_sequences=False)(historic_inp)
        condition = Dense(108, activation='relu')(conditional_inp)
        # condition = Flatten()(condition)

        x = Concatenate(axis=1)([hist, noise_inp, condition])
        if self.dropout:
            x = Dropout(0.2)(x, training=True)

        # x = BatchNormalization()(x)
        x = Dense(self.generator_nodes+self.noise_vector_size)(x)
        x = ReLU()(x)
        if self.dropout:
            x = Dropout(0.4)(x, training=True)
        # x = Dense(16)(x)
        # x = ReLU()(x)
        prediction = Dense(self.output_size)(x)

        model = Model(inputs=[historic_inp, noise_inp, conditional_inp], outputs=prediction)
        model.summary()
        return model

    def build_discriminator(self):

        historic_inp = Input(shape=(self.window_size, 1))
        future_inp = Input(shape=(self.output_size, 1))
        conditional_inp = Input(shape=(108,))

        x = Concatenate(axis=1)([historic_inp, future_inp])

        if self.layers == 'lstm':
            x = LSTM(self.discriminator_nodes, return_sequences=False)(x)
        elif self.layers == 'gru':
            x = GRU(self.discriminator_nodes, return_sequences=False)(x)
        else:
            x = SimpleRNN(self.discriminator_nodes, return_sequences=False)(x)

        if self.batch_norm:
            x = BatchNormalization()(x)
        if self.dropout:
            x = Dropout(0.2)(x)
        condition = Dense(108, activation='relu')(conditional_inp)
        # condition = Flatten()(condition)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Concatenate(axis=1)([x, condition])
        x = Dense(self.discriminator_nodes)(x)
        x = LeakyReLU(alpha=0.1)(x)
        if self.dropout:
            x = Dropout(0.4)(x)
        # x = Dropout(0.2)(x)
        # x = Dense(32)(x)
        # x = LeakyReLU(alpha=0.1)(x)
        # x = Dropout(0.2)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[historic_inp, future_inp, conditional_inp], outputs=validity)
        model.summary()

        return model

    def find_training_mask(self, half_batch):
        mask = np.empty(self.discriminator_epochs * half_batch * 2, dtype=int)

        if self.discriminator_epochs > 1:
            """
            for i in range(0, self.discriminator_epochs, 2):
                print(i * half_batch * 2, i * half_batch * 2 + 2 * half_batch, i * half_batch, i * half_batch + 2 * half_batch)
                mask[i * half_batch * 2:i * half_batch * 2 + 2 * half_batch] = np.arange(i * half_batch,
                                                                                         i * half_batch + 2 * half_batch,
                                                                                         dtype=int)
                print((i + 1) * half_batch * 2, (i + 1) * half_batch * 2 + 2 * half_batch, self.discriminator_epochs * half_batch + i * half_batch,
                              self.discriminator_epochs * half_batch + i * half_batch + 2 * half_batch)
                mask[(i + 1) * half_batch * 2:(i + 1) * half_batch * 2 + 2 * half_batch] = \
                    np.arange(self.discriminator_epochs * half_batch + i * half_batch,
                              self.discriminator_epochs * half_batch + i * half_batch + 2 * half_batch, dtype=int)
            """
            for i in range(self.discriminator_epochs):
                mask[i * half_batch*2:i * half_batch*2 + half_batch] = \
                    np.arange(i * half_batch, i * half_batch + half_batch, dtype=int)
                mask[(2*i + 1) * half_batch:(2*i + 1) * half_batch + half_batch] = \
                    np.arange(self.discriminator_epochs * half_batch + i * half_batch,
                              self.discriminator_epochs * half_batch + i * half_batch + half_batch, dtype=int)
        else:
            mask = np.arange(0, 2 * half_batch, dtype=int)
        return mask

    def train_generator_on_batch(self, x, condition, batch_size):
        generator_noise = self._generate_noise(batch_size)
        idx = np.random.randint(0, x.shape[0], batch_size)
        historic_time_series = x[idx]
        time_series_index = condition[idx]

        valid_y = np.array([1] * batch_size)

        # Train the generator
        g_loss = self.combined.train_on_batch([historic_time_series, generator_noise, time_series_index], valid_y)
        return g_loss

    def train_discriminator_on_batch(self, x, y, condition, half_batch, mask):
        # Select a random half batch of images
        idx = np.random.randint(0, x.shape[0], self.discriminator_epochs*half_batch)
        historic_time_series = x[idx]
        future_time_series = y[idx]
        time_series_index = condition[idx]

        generator_noise = self._generate_noise(self.discriminator_epochs*half_batch)

        # Generate a half batch of new images
        gen_forecasts = self.generator.predict([historic_time_series, generator_noise, time_series_index])
        hist_input = np.concatenate([historic_time_series, historic_time_series], axis=0)
        future_input = np.concatenate([future_time_series,  np.expand_dims(gen_forecasts, axis=-1)], axis=0)
        time_series_index_input = np.concatenate([time_series_index, time_series_index], axis=0)
        y_input = np.concatenate([self._get_labels(batch_size=self.discriminator_epochs*half_batch, real=True),
                                  self._get_labels(batch_size=self.discriminator_epochs*half_batch, real=False)],
                                 axis=0)
        if self.mixed_batches:
            random_mask = np.arange(self.discriminator_epochs*half_batch*2)
            np.random.shuffle(random_mask)
            hist_input = hist_input[random_mask]
            future_input = future_input[random_mask]
            time_series_index_input = time_series_index_input[random_mask]
            y_input = y_input[random_mask]
        else:
            hist_input = hist_input[mask]
            future_input = future_input[mask]
            time_series_index_input = time_series_index_input[mask]
            y_input = y_input[mask]

        # Train the discriminator
        d_loss = self.discriminator.fit([hist_input, future_input, time_series_index_input], y_input, epochs=1, batch_size=half_batch,
                                        shuffle=False, verbose=0)
        return [np.mean(d_loss.history['loss']), np.mean(d_loss.history['acc'])]

    def fit(self, x, y, condition_train=None, x_val=None, y_val=None, condition_val=None, epochs=1, batch_size=32, verbose=1):
        half_batch = int(batch_size / 2)
        forecast_mse = np.zeros(epochs)
        condition_train = to_categorical(condition_train, num_classes=108)
        condition_val = to_categorical(condition_val, num_classes=108)
        G_loss = np.zeros(epochs)
        D_loss = np.zeros(epochs)
        training_mask = self.find_training_mask(half_batch)
        coverage_80_pi = []
        coverage_95_pi = []
        validation_mse = []

        best_validation_mse = 10 ** 10
        best_generator = self.generator
        best_epoch = 0
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            d_loss = self.train_discriminator_on_batch(x, y, condition_train, half_batch, training_mask)

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.train_generator_on_batch(x, condition_train, batch_size)

            # Measure forecast MSE of generator
            # forecast_mse[epoch] = mean_squared_error(future_time_series[:, :, 0], gen_forecasts)

            G_loss[epoch] = g_loss
            D_loss[epoch] = d_loss[0]

            # Print the progress
            if epoch % self.plot_rate == 0:
                if self.print_coverage and (x_val is None) and (y_val is None):
                    idx = np.random.randint(0, x.shape[0], batch_size)
                    historic_time_series = x[idx]
                    future_time_series = y[idx]
                    conditional_input = condition_train[idx]
                    forecasts = np.zeros([batch_size, self.output_size, 100])
                    for j in range(batch_size):
                        generator_noise = self._generate_noise(100)
                        x_input = np.vstack([np.expand_dims(historic_time_series[j], axis=0)] * 100)

                        forecasts[j] = self.generator.predict([x_input, generator_noise, conditional_input]).transpose()
                    if self.output_size > 1:
                        coverage_80_pi.append(sliding_window_coverage(actual_values=future_time_series,
                                                                      upper_limits=np.quantile(forecasts, q=0.9, axis=-1),
                                                                      lower_limits=np.quantile(forecasts, q=0.1, axis=-1)))
                        coverage_95_pi.append(sliding_window_coverage(actual_values=future_time_series[:, :, 0],
                                                                      upper_limits=np.quantile(forecasts, q=0.975, axis=-1),
                                                                      lower_limits=np.quantile(forecasts, q=0.025, axis=-1)))
                        validation_mse.append(
                            mean_squared_error(future_time_series.flatten(), np.mean(forecasts, axis=-1).flatten()))
                    else:
                        coverage_80_pi.append(compute_coverage(actual_values=future_time_series[:, :, 0],
                                                               upper_limits=np.quantile(forecasts, q=0.9, axis=-1),
                                                               lower_limits=np.quantile(forecasts, q=0.1, axis=-1)))
                        coverage_95_pi.append(compute_coverage(actual_values=future_time_series[:, :, 0],
                                                               upper_limits=np.quantile(forecasts, q=0.975, axis=-1),
                                                               lower_limits=np.quantile(forecasts, q=0.025, axis=-1)))
                        validation_mse.append(mean_squared_error(future_time_series[:, :, 0], np.mean(forecasts, axis=-1)))

                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f][Mean forecast mse: %f, "
                          "80%%-PI: %.2f%%, 95%%-PI: %.2f%%]" %
                          (epoch, d_loss[0], 100 * d_loss[1], g_loss, forecast_mse[epoch],
                           validation_mse[-1], 100*coverage_80_pi[-1], 100*coverage_95_pi[-1]))
                elif self.print_coverage and (x_val is not None) and (y_val is not None):
                    forecasts = np.zeros([len(y_val), self.output_size, 100])
                    for j in range(len(y_val)):
                        generator_noise = self._generate_noise(100)
                        x_input = np.vstack([np.expand_dims(x_val[j], axis=0)] * 100)
                        forecasts[j] = self.generator.predict([x_input, generator_noise, condition_val]).transpose()
                    if self.output_size > 1:
                        coverage_80_pi.append(sliding_window_coverage(actual_values=y_val,
                                                                      upper_limits=np.quantile(forecasts, q=0.9, axis=-1),
                                                                      lower_limits=np.quantile(forecasts, q=0.1, axis=-1),
                                                                      forecast_horizon=self.output_size).mean())
                        coverage_95_pi.append(sliding_window_coverage(actual_values=y_val,
                                                                      upper_limits=np.quantile(forecasts, q=0.975, axis=-1),
                                                                      lower_limits=np.quantile(forecasts, q=0.025, axis=-1),
                                                                      forecast_horizon=self.output_size).mean())
                        validation_mse.append(mean_squared_error(y_val.flatten(), np.mean(forecasts, axis=-1).flatten()))
                    else:
                        coverage_80_pi.append(compute_coverage(actual_values=y_val,
                                                               upper_limits=np.quantile(forecasts, q=0.9, axis=-1),
                                                               lower_limits=np.quantile(forecasts, q=0.1, axis=-1)))
                        coverage_95_pi.append(compute_coverage(actual_values=y_val,
                                                               upper_limits=np.quantile(forecasts, q=0.975, axis=-1),
                                                               lower_limits=np.quantile(forecasts, q=0.025, axis=-1)))
                        validation_mse.append(mean_squared_error(y_val[:, 0], np.mean(forecasts, axis=-1)))
                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f][Validation mse: %f, "
                          "80%%-PI: %.2f%%, 95%%-PI: %.2f%%]" %
                          (epoch, d_loss[0], 100 * d_loss[1], g_loss, forecast_mse[epoch],
                           validation_mse[-1], 100 * coverage_80_pi[-1], 100 * coverage_95_pi[-1]))
                else:
                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                          (epoch, d_loss[0], 100 * (d_loss[1]), g_loss, forecast_mse[epoch]))
            if (epoch+1) % self.save_model_interval == 0:
                self.discriminator.save(self.results_path + "/discriminator_%d.h5" % (epoch+1))
                self.generator.save(self.results_path + "/generator_%d.h5" % (epoch+1))
                if validation_mse[-1] < best_validation_mse:
                    best_validation_mse = validation_mse[-1]
                    best_generator = self.generator
                    best_epoch = epoch+1

        best_generator.save(self.results_path + "/generator_best_%d.h5" % best_epoch)
        # self.generator = best_generator
        print('Best model at epoch %d, validation mse: %f' % (best_epoch, best_validation_mse))

        if self.print_coverage:
            file_name = self.results_path + "/training_results.txt"
            with open(file_name, "a") as f:
                f.write("mse,coverage_80,coverage_95\n")
                for (validation_mse, coverage_80_pi, coverage_95_pi) in zip(validation_mse, coverage_80_pi, coverage_95_pi):
                    f.write("{0},{1},{2}\n".format(validation_mse, coverage_80_pi, coverage_95_pi))
        return {'mse': forecast_mse, 'G_loss': G_loss, 'D_loss': D_loss, 'Accuracy': 100 * d_loss[1]}

    def forecast(self, x, condition=None):
        forecast = np.zeros([x.shape[0], self.mc_forward_passes, self.output_size])
        condition = to_categorical(condition, num_classes=108)
        # condition = np.expand_dims(condition, axis=-1)
        print(condition.shape)
        for i in trange(x.shape[0]):
            generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
            x_input = np.vstack([np.expand_dims(x[i], axis=0)] * self.mc_forward_passes)
            condition_input = np.vstack([np.expand_dims(condition[i], axis=0)]*self.mc_forward_passes)
            forecast[i] = self.generator.predict([x_input, generator_noise, condition_input])
        """
        forecast = np.zeros([x.shape[0]*self.mc_forward_passes, self.forecasting_horizon])
        generator_noise = self._generate_noise(batch_size=self.mc_forward_passes*x.shape[0])
        x_input = np.vstack([np.expand_dims(x, axis=0)] * self.mc_forward_passes)
        forecast = self.generator.predict([x_input.reshape([x_input.shape[0]*x_input.shape[1], x_input.shape[2], x_input.shape[3]]), generator_noise])
        forecast = forecast.reshape([x.shape[0], self.mc_forward_passes, self.output_size])
        """
        return forecast.mean(axis=1)

    def recurrent_forecast(self, time_series, condition):
        time_series = np.vstack([time_series] * self.mc_forward_passes)
        condition = np.vstack([condition] * self.mc_forward_passes)
        x_input = np.zeros([self.mc_forward_passes, self.window_size + self.forecasting_horizon, 1])
        x_input[:, :self.window_size] = time_series
        for i in range(self.forecasting_horizon):
            generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
            x_input[:, self.window_size+i] = self.generator.predict([x_input[:, i:self.window_size+i], generator_noise, condition])
        return x_input[:, -self.forecasting_horizon:].transpose()[0]

    def monte_carlo_forecast(self, data, condition=None, steps=1, plot=False, disable_pbar=False):
        time_series = np.expand_dims(data, axis=0)
        condition = to_categorical(condition, num_classes=108)
        forecast = np.zeros([steps, self.forecasting_horizon, self.mc_forward_passes])
        for i in trange(steps, disable=disable_pbar):
            if self.recurrent_forecasting:
                forecast[i] = self.recurrent_forecast(time_series[:, i:self.window_size + i], condition[i])
            else:
                generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
                x_input = np.vstack([time_series[:, i:self.window_size + i]]*self.mc_forward_passes)
                forecast[i] = self.generator.predict([x_input, generator_noise, condition]).transpose()
        return forecast


