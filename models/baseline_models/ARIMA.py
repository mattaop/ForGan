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
        self.pred_int_80 = []
        self.pred_int_95 = []

        self.recurrent_forecasting = cfg['recurrent_forecasting']
        if self.recurrent_forecasting:
            self.output_size = 1
        else:
            self.output_size = self.forecasting_horizon

    def build_model(self):
        pass

    def fit_arima(self, train):

        auto_model = auto_arima(train, start_p=1, start_q=1, max_p=11, max_q=11, max_d=3, max_P=5, max_Q=5, max_D=3,
                                m=12, start_P=1, start_Q=1, seasonal=True, d=None, D=None, suppress_warnings=True,
                                stepwise=True, information_criterion='aicc')

        print(auto_model.summary())
        self.arima_model = auto_model
        return auto_model

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1):
        # fit Exponential Smoothing
        train = np.concatenate([x[0, :, 0], y[:, 0, 0]])
        arima = self.fit_arima(train)
        # transform data
        # pred_arima = arima.fittedvalues

        # Print training mse
        # print('ES MSE:', mean_squared_error(y[:, :, 0], pred_arima[self.window_size:]))

    def forecast(self, x):
        # forecast exponential smoothing
        arima_forecasts = self.arima_model.forecast(steps=x.shape[0])
        return arima_forecasts

    def monte_carlo_forecast(self, data, steps=1, plot=False):
        data = data[-steps:]
        print(data.shape)
        forecasts = np.zeros([steps, self.forecasting_horizon])
        pred_int_80 = np.zeros([steps, self.forecasting_horizon, 2])
        pred_int_95 = np.zeros([steps, self.forecasting_horizon, 2])

        for i in tqdm(range(steps)):
            forecast_arima_95 = self.arima_model.predict(n_periods=self.forecasting_horizon,
                                                         return_conf_int=True, alpha=1 - 0.95)
            forecast_arima_80 = self.arima_model.predict(n_periods=self.forecasting_horizon,
                                                         return_conf_int=True, alpha=1 - 0.8)
            forecasts[i] = forecast_arima_95[0]
            pred_int_80[i] = forecast_arima_80[1]
            pred_int_95[i] = forecast_arima_95[1]
            self.arima_model.update(y=data[i])
        self.pred_int_80 = pred_int_80
        self.pred_int_95 = pred_int_95
        print(pred_int_80.shape)
        return forecasts
