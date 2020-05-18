import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import normalize
from tqdm import tqdm

from data.generate_sine import generate_sine_data
from utility.split_data import split_sequence


class RNN:
    def __init__(self, cfg):
        self.plot_rate = cfg['plot_rate']
        self.plot_folder = 'RNN'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']
        self.recurrent_forecasting = cfg['recurrent_forecasting']
        if self.recurrent_forecasting:
            self.output_size = 1
        else:
            self.output_size = self.forecasting_horizon

        self.mc_forward_passes = cfg['mc_forward_passes']
        self.optimizer = Adam(lr=cfg['learning_rate'])
        self.loss_function = 'mse'

        # Build and compile the discriminator
        self.model = None

    def build_model(self):
        input_shape = (self.window_size, 1)
        inp = Input(shape=input_shape)

        x = SimpleRNN(64, return_sequences=False)(inp)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)
        output = Dense(self.output_size)(x)

        model = Model(inputs=inp, outputs=output)
        model.compile(optimizer=self.optimizer, loss=self.loss_function)
        model.summary()

        self.model = model

    def train(self, epochs, batch_size=128, data_samples=5000):
        # Load the data
        data = generate_sine_data(data_samples)
        x_train, y_train = split_sequence(data, self.window_size, self.forecasting_horizon)

        history = self.model.fit(x_train, y_train[:, :, 0], epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

        plt.figure()
        plt.plot(np.linspace(1, epochs, epochs), history.history['loss'], label='Training loss')
        plt.legend()
        plt.show()

        return {'mse': history.history['loss'], 'G_loss': None, 'D_loss': None, 'Accuracy': None}

    def fit(self, x, y, epochs=1, batch_size=32, verbose=2):
        # Load the data
        history = self.model.fit(x, y[:, :, 0], epochs=epochs, batch_size=batch_size, validation_split=0.1,
                                 verbose=verbose)
        return history

    def forecast(self, x):
        return self.model.predict(x)

    def recurrent_forecast(self, func, time_series):
        time_series = np.vstack([time_series] * self.mc_forward_passes)
        x_input = np.zeros([self.mc_forward_passes, self.window_size + self.forecasting_horizon, 1])
        x_input[:, :self.window_size] = time_series
        for i in range(self.forecasting_horizon):
            x_input[:, self.window_size+i] = np.array(func([x_input[:, i:self.window_size+i], 0.4])[0])
        return x_input[:, -self.forecasting_horizon:].transpose()[0]

    def monte_carlo_forecast(self, data, steps=1, plot=False):
        """
        x = np.zeros([len(data) - self.window_size, self.window_size, 1])
        for i in range(len(data) - self.window_size):
            x[i] = data[i:self.window_size+i]
        time_series = np.expand_dims(x, axis=0)
        time_series = np.vstack([time_series] * self.mc_forward_passes)
        func = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        # forecast = self.recurrent_forecast(func, time_series)
        print(time_series.shape)
        x_input = np.zeros([self.mc_forward_passes*steps, self.window_size + self.forecasting_horizon, 1])
        x_input[:, :self.window_size] = time_series.reshape([time_series.shape[0]*time_series.shape[1], time_series.shape[2], time_series.shape[3]])
        for i in tqdm(range(self.forecasting_horizon)):
            print(x_input[:, i:self.window_size+i].shape)
            x_input[:, self.window_size + i] = np.array(func([x_input[:, i:self.window_size+i], 0.4])[0])
        forecast = x_input[:, -self.forecasting_horizon:].transpose()[0].reshape([steps, self.mc_forward_passes,
                                                                                  self.forecasting_horizon]).transpose(0, 2, 1)
        print(forecast.shape)
        """
        time_series = np.expand_dims(data, axis=0)
        forecast = np.zeros([steps, self.forecasting_horizon, self.mc_forward_passes])
        func = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        for i in tqdm(range(steps)):
            if self.recurrent_forecasting:
                forecast[i] = self.recurrent_forecast(func, time_series[:, i:self.window_size + i])
            else:
                x_input = np.vstack([time_series[:, i:self.window_size + i]]*self.mc_forward_passes)
                forecast[i] = np.array(func([x_input, 0.4])[0]).transpose()
        if plot:
            print(data.shape)
            print(forecast.shape)
            plt.figure()
            plt.plot(np.linspace(1, self.window_size + self.forecasting_horizon,
                                 self.window_size + self.forecasting_horizon),
                     data[:self.window_size+self.forecasting_horizon, 0],  label='Real data')
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
