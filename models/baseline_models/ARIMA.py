import numpy as np
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

from tqdm import tqdm


class ARIMA:
    def __init__(self, cfg):
        self.plot_rate = cfg['plot_rate']
        self.plot_folder = 'ES'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']

        self.mc_forward_passes = cfg['mc_forward_passes']
        self.arima_model = None
        self.variance = []

    def fit_arima(self, train):

        auto_model = auto_arima(train, start_p=1, start_q=1, max_p=11, max_q=11, max_d=3, max_P=5, max_Q=5, max_D=3,
                                m=12, start_P=1, start_Q=1, seasonal=True, d=None, D=None, suppress_warnings=True,
                                stepwise=True, information_criterion='aicc')

        print(auto_model.summary())
        return auto_model

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1):
        # fit Exponential Smoothing
        train = np.concatenate([x[0, :, 0], y[:, 0, 0]])
        arima = self.fit_arima(train)
        # transform data
        pred_arima = arima.fittedvalues

        # Print training mse
        print('ES MSE:', mean_squared_error(y[:, :, 0], pred_arima[self.window_size:]))

    def forecast(self, x):
        # forecast exponential smoothing
        es_forecasts = self.arima_model.forecast(steps=x.shape[0])
        return es_forecasts

    def monte_carlo_forecast(self, data, steps=1, plot=False):
        # forecast ES
        es_series = np.expand_dims(self.arima_model.forecast(steps=steps+self.window_size+self.forecasting_horizon), axis=-1)
        es_series = np.expand_dims(es_series, axis=0)

        data = np.expand_dims(data, axis=0)
        time_series = data - es_series[:, :-self.forecasting_horizon]

        forecasts = np.zeros([steps, self.forecasting_horizon, 1])
        for i in tqdm(range(steps)):
            # forecast ES
            forecasts[i] = es_series[:, self.window_size + i:self.window_size + i+self.forecasting_horizon]
        return forecasts
