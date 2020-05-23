import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
from tqdm import tqdm

from config.load_config import load_config_file
from data.generate_noise import generate_noise
from utility.split_data import split_sequence
from utility.compute_statistics import compute_coverage
from models.feed_forward_gan.GAN import GAN


class RecurrentGAN(GAN):
    def __init__(self, cfg):
        GAN.__init__(self, cfg)
        self.plot_rate = cfg['plot_rate']
        self.print_coverage = cfg['print_coverage']
        self.plot_folder = 'RecurrentGAN'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']
        self.recurrent_forecasting = cfg['recurrent_forecasting']
        if self.recurrent_forecasting:
            self.output_size = 1
        else:
            self.output_size = self.forecasting_horizon

        self.noise_vector_size = cfg['noise_vector_size']
        self.discriminator_epochs = cfg['discriminator_epochs']
        self.mc_forward_passes = cfg['mc_forward_passes']

        self.optimizer = Adam(cfg['learning_rate'], 0.5)
        self.loss_function = 'binary_crossentropy'

    def build_model(self):
        print('=== Config===', '\nModel name:', self.model_name, '\nNoise vector size:', self.noise_vector_size,
              '\nDiscriminator epochs:', self.discriminator_epochs, '\n Generator nodes', self.generator_nodes,
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

        hist = SimpleRNN(self.generator_nodes, return_sequences=False)(historic_inp)
        # hist = LSTM(self.generator_nodes, return_sequences=False)(historic_inp)
        # hist = GRU(self.generator_nodes, return_sequences=False)(historic_inp)

        x = Concatenate(axis=1)([hist, noise_inp])
        # x = BatchNormalization()(x)
        x = Dense(self.generator_nodes+self.noise_vector_size)(x)
        x = ReLU()(x)
        # x = Dense(16)(x)
        # x = ReLU()(x)
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

        x = SimpleRNN(self.discriminator_nodes, return_sequences=False)(x)
        # x = GRU(self.discriminator_nodes, return_sequences=False)(x)
        x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(self.discriminator_nodes)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = Dropout(0.2)(x)
        # x = Dense(32)(x)
        # x = LeakyReLU(alpha=0.1)(x)
        # x = Dropout(0.2)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model

    def train_generator_on_batch(self, x, batch_size):
        generator_noise = self._generate_noise(batch_size)
        idx = np.random.randint(0, x.shape[0], batch_size)
        historic_time_series = x[idx]

        valid_y = np.array([1] * batch_size)

        # Train the generator
        g_loss = self.combined.train_on_batch([historic_time_series, generator_noise], valid_y)
        return g_loss

    def fit(self, x, y, x_val=None, y_val=None, epochs=1, batch_size=32, verbose=1):
        half_batch = int(batch_size / 2)
        forecast_mse = np.zeros(epochs)
        G_loss = np.zeros(epochs)
        D_loss = np.zeros(epochs)
        coverage_80_pi = []
        coverage_95_pi = []
        validation_mse = []

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
            g_loss = self.train_generator_on_batch(x, batch_size)

            # Measure forecast MSE of generator
            forecast_mse[epoch] = mean_squared_error(future_time_series[:, :, 0], gen_forecasts)

            G_loss[epoch] = g_loss
            D_loss[epoch] = d_loss[0]
            # Print the progress
            if epoch % self.plot_rate == 0:
                if self.print_coverage and (x_val is None) and (y_val is None):
                    idx = np.random.randint(0, x.shape[0], batch_size)
                    historic_time_series = x[idx]
                    future_time_series = y[idx]
                    forecasts = np.zeros([batch_size, self.output_size, 100])
                    for j in range(batch_size):
                        generator_noise = self._generate_noise(100)
                        x_input = np.vstack([np.expand_dims(historic_time_series[j], axis=0)] * 100)

                        forecasts[j] = self.generator.predict([x_input, generator_noise]).transpose()
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
                        forecasts[j] = self.generator.predict([x_input, generator_noise]).transpose()
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
                self.discriminator.save("results/" + self.data_source + "/" + self.model_name +
                                        "/Epochs_%d_D_epochs_%d_batch_size_%d_noise_vec_%d_lr_%f/discriminator_%d.h5" %
                                        (epochs, self.discriminator_epochs, batch_size, self.noise_vector_size,
                                         self.learning_rate,  epoch+1))
                self.generator.save("results/" + self.data_source + "/" + self.model_name +
                                    "/Epochs_%d_D_epochs_%d_batch_size_%d_noise_vec_%d_lr_%f/generator_%d.h5" %
                                    (epochs, self.discriminator_epochs, batch_size, self.noise_vector_size,
                                     self.learning_rate, epoch+1))
        if self.print_coverage:
            file_name = ("results/" + self.data_source + "/" + self.model_name +
                         "/Epochs_%d_D_epochs_%d_batch_size_%d_noise_vec_%d_lr_%f/training_results.txt" %
                         (epochs, self.discriminator_epochs, batch_size, self.noise_vector_size, self.learning_rate))
            with open(file_name, "a") as f:
                f.write("mse,coverage_80,coverage_95\n")
                for (validation_mse, coverage_80_pi, coverage_95_pi) in zip(validation_mse, coverage_80_pi, coverage_95_pi):
                    f.write("{0},{1},{2}\n".format(validation_mse, coverage_80_pi, coverage_95_pi))
        return {'mse': forecast_mse, 'G_loss': G_loss, 'D_loss': D_loss, 'Accuracy': 100 * d_loss[1]}

    def forecast(self, x):
        forecast = np.zeros([x.shape[0], self.mc_forward_passes, self.output_size])
        for i in tqdm(range(x.shape[0])):
            generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
            x_input = np.vstack([np.expand_dims(x[i], axis=0)] * self.mc_forward_passes)
            forecast[i] = self.generator.predict([x_input, generator_noise])
        """
        forecast = np.zeros([x.shape[0]*self.mc_forward_passes, self.forecasting_horizon])
        generator_noise = self._generate_noise(batch_size=self.mc_forward_passes*x.shape[0])
        x_input = np.vstack([np.expand_dims(x, axis=0)] * self.mc_forward_passes)
        forecast = self.generator.predict([x_input.reshape([x_input.shape[0]*x_input.shape[1], x_input.shape[2], x_input.shape[3]]), generator_noise])
        forecast = forecast.reshape([x.shape[0], self.mc_forward_passes, self.output_size])
        """
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
        time_series = np.expand_dims(data, axis=0)
        forecast = np.zeros([steps, self.forecasting_horizon, self.mc_forward_passes])
        for i in tqdm(range(steps)):
            if self.recurrent_forecasting:
                forecast[i] = self.recurrent_forecast(time_series[:, i:self.window_size + i])
            else:
                generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
                x_input = np.vstack([time_series[:, i:self.window_size + i]]*self.mc_forward_passes)
                forecast[i] = self.generator.predict([x_input, generator_noise]).transpose()
        if plot:
            plt.figure()
            plt.plot(np.linspace(1, self.window_size + self.forecasting_horizon,
                                 self.window_size + self.forecasting_horizon),
                     data[:self.window_size + self.forecasting_horizon, 0], label='Real data')
            plt.plot(np.linspace(self.window_size+1, self.window_size + self.forecasting_horizon,
                                 self.forecasting_horizon),
                     forecast.mean(axis=2)[0, :],
                     label='Forecast data')
            plt.fill_between(np.linspace(self.window_size+1, self.window_size + self.forecasting_horizon,
                                         self.forecasting_horizon),
                             forecast.mean(axis=-1)[0, :] - 1.28 * forecast.std(axis=-1)[0, :],
                             forecast.mean(axis=-1)[0, :] + 1.28 * forecast.std(axis=-1)[0, :],
                             alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
            plt.fill_between(np.linspace(self.window_size+1, self.window_size + self.forecasting_horizon,
                                         self.forecasting_horizon),
                             forecast.mean(axis=-1)[0, :] - 1.96 * forecast.std(axis=-1)[0, :],
                             forecast.mean(axis=-1)[0, :] + 1.96 * forecast.std(axis=-1)[0, :],
                             alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')

            plt.legend()
            plt.show()
        return forecast


if __name__ == '__main__':
    config = load_config_file('C:\\Users\\mathi\\PycharmProjects\\gan\\config\\config.yml')
    coverage_80_PI_1, coverage_95_PI_1 = [], []
    coverage_80_PI_2, coverage_95_PI_2 = [], []
    kl_div, js_div, uncertainty_list = [], [], []
    for _ in range(1):
        gan = RecurrentGAN(config['gan'])
        gan.build_model()
        gan.train(epochs=200, batch_size=32)
        predictions = gan.monte_carlo_forecast(generate_noise(5000))[0, 0]
        prediction_mean = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        kl_div.append(gan.compute_kl_divergence(predictions, generate_noise(5000)))
        js_div.append(gan.compute_js_divergence(predictions, generate_noise(5000)))
        uncertainty_list.append(uncertainty)
        coverage_80_PI_1.append(compute_coverage(upper_limits=np.vstack([np.quantile(predictions, q=0.9, axis=0)]*10000),
                                                 lower_limits=np.vstack([np.quantile(predictions, q=0.1, axis=0)]*10000),
                                                 actual_values=generate_noise(10000)))
        coverage_95_PI_1.append(compute_coverage(upper_limits=np.vstack([np.quantile(predictions, q=0.975, axis=0)]*10000),
                                                 lower_limits=np.vstack([np.quantile(predictions, q=0.025, axis=0)]*10000),
                                                 actual_values=generate_noise(10000)))
        coverage_80_PI_2.append(compute_coverage(upper_limits=np.vstack([prediction_mean+1.28*uncertainty]*10000),
                                                 lower_limits=np.vstack([prediction_mean-1.28*uncertainty]*10000),
                                                 actual_values=generate_noise(10000)))
        coverage_95_PI_2.append(compute_coverage(upper_limits=np.vstack([prediction_mean+1.96*uncertainty]*10000),
                                                 lower_limits=np.vstack([prediction_mean-1.96*uncertainty]*10000),
                                                 actual_values=generate_noise(10000)))
    print('80% PI Coverage:', np.mean(coverage_80_PI_1), ', std:', np.std(coverage_80_PI_1))
    print('95% PI Coverage:', np.mean(coverage_95_PI_1), ', std:', np.std(coverage_95_PI_1))

    print('80% PI Coverage:', np.mean(coverage_80_PI_2), ', std:', np.std(coverage_80_PI_2))
    print('95% PI Coverage:', np.mean(coverage_95_PI_2), ', std:', np.std(coverage_95_PI_2))
    print('KL-divergence mean:', np.mean(kl_div), ', std:', np.std(kl_div))
    print('JS-divergence mean:', np.mean(js_div), ', std:', np.std(js_div))
    print('Uncertainty mean:', np.mean(uncertainty_list), ', std:', np.std(uncertainty_list))
