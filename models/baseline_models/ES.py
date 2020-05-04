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


class ESWGAN(GAN):
    def __init__(self, cfg):
        GAN.__init__(self, cfg)
        self.plot_rate = cfg['plot_rate']
        self.plot_folder = 'ES'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']

        self.mc_forward_passes = cfg['mc_forward_passes']
        self.exponential_smoothing = None
        self.variance = []

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

        residual_variance = model_es.sse / len(train - 2)
        var = []
        alpha = model_es.params['smoothing_level']
        beta = model_es.params['smoothing_slope']
        gamma = model_es.params['smoothing_seasonal']
        for j in range(self.forecasting_horizon):
            s = 12
            h = j + 1
            k = int((h - 1) / s)

            if best_model_parameters[1] == 'add':
                if best_model_parameters[0] == 'add':
                    var.append(residual_variance * (
                                1 + (h - 1) * (alpha ** 2 + alpha * h * beta + h / 6 * (2 * h - 1) * beta ** 2)
                                + k * gamma * (2 * alpha + gamma + beta * s * (k + 1))))
                else:
                    var.append(
                        residual_variance * (1 + (h - 1) * alpha ** 2 + k * gamma * (2 * alpha + gamma)))
            elif best_model_parameters[1] == 'mul':
                var.append(residual_variance * h)
            else:
                if best_model_parameters[0] == 'add':
                    var.append(
                        residual_variance * (
                                    1 + (h - 1) * (alpha ** 2 + alpha * h * beta + h / 6 * (2 * h - 1) * beta ** 2)))
                else:
                    var.append(residual_variance * (1 + (h - 1) * alpha ** 2))
        self.variance = var
        return self.exponential_smoothing

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1):
        # fit Exponential Smoothing
        train = np.concatenate([x[0, :, 0], y[:, 0, 0]])
        exponential_model = self.fit_es(train)
        # transform data
        pred_es = exponential_model.fittedvalues

        # Print training mse
        print('ES MSE:', mean_squared_error(y[:, :, 0], pred_es[self.window_size:]))

    def forecast(self, x):
        # forecast exponential smoothing
        es_forecasts = self.exponential_smoothing.forecast(steps=x.shape[0])
        return es_forecasts

    def recurrent_forecast(self, time_series):
        time_series = np.vstack([time_series] * self.mc_forward_passes)
        x_input = np.zeros([self.mc_forward_passes, self.window_size + self.forecasting_horizon, 1])
        x_input[:, :self.window_size] = time_series
        for i in range(self.forecasting_horizon):
            generator_noise = self._generate_noise(batch_size=self.mc_forward_passes)
            x_input[:, self.window_size+i] = self.generator.predict([x_input[:, i:self.window_size+i, 0], generator_noise])
        return x_input[:, -self.forecasting_horizon:].transpose()[0]

    def monte_carlo_forecast(self, data, steps=1, plot=False):
        # forecast ES
        es_series = np.expand_dims(self.exponential_smoothing.forecast(steps=steps+self.window_size+self.forecasting_horizon), axis=-1)
        es_series = np.expand_dims(es_series, axis=0)

        data = np.expand_dims(data, axis=0)
        time_series = data - es_series[:, :-self.forecasting_horizon]

        es_forecasts = np.zeros([steps, self.forecasting_horizon, 1])
        for i in tqdm(range(steps)):
            # forecast ES
            es_forecasts[i] = es_series[:, self.window_size + i:self.window_size + i+self.forecasting_horizon]
        return es_forecasts
