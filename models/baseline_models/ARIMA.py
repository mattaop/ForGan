import numpy as np
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

from tqdm import trange


class ARIMA:
    def __init__(self, cfg):
        self.plot_rate = cfg['plot_rate']
        self.plot_folder = 'ES'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']
        self.seasonality = cfg['seasonality']

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

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1):
        # fit ARIMA
        train = np.concatenate([x[0, :, 0], y[:, 0, 0]])
        auto_model = auto_arima(train, start_p=1, start_q=1, max_p=5, max_q=5, max_d=3, max_P=2, max_Q=2, max_D=2,
                                m=self.seasonality, start_P=1, start_Q=1, seasonal=True, d=None, D=None,
                                suppress_warnings=True, stepwise=True, information_criterion='aicc',
                                error_action="ignore")

        print(auto_model.summary())
        self.arima_model = auto_model
        return auto_model

    def forecast(self, x):
        # forecast exponential smoothing
        arima_forecasts = self.arima_model.forecast(steps=x.shape[0])
        return arima_forecasts

    def monte_carlo_forecast(self, data, steps=1, plot=False, disable_pbar=False):
        data = data[-steps:]
        print(data.shape)
        forecasts = np.zeros([steps, self.forecasting_horizon])
        pred_int_80 = np.zeros([steps, self.forecasting_horizon, 2])
        pred_int_95 = np.zeros([steps, self.forecasting_horizon, 2])

        for i in trange(steps, disable=disable_pbar):
            forecast_arima_95 = self.arima_model.predict(n_periods=self.forecasting_horizon,
                                                         return_conf_int=True, alpha=1 - 0.95)
            forecast_arima_80 = self.arima_model.predict(n_periods=self.forecasting_horizon,
                                                         return_conf_int=True, alpha=1 - 0.8)
            forecasts[i] = forecast_arima_95[0]
            pred_int_80[i] = forecast_arima_80[1]
            pred_int_95[i] = forecast_arima_95[1]
            self.arima_model.update(y=data[i], maxiter=0)
        self.pred_int_80 = pred_int_80
        self.pred_int_95 = pred_int_95
        self.variance = (0.25 / 1.28 * (np.mean(pred_int_80[:, :, 1], axis=0) - np.mean(pred_int_80[:, :, 0], axis=0)) +\
                         0.25 / 1.96 * (np.mean(pred_int_95[:, :, 1], axis=0) - np.mean(pred_int_95[:, :, 0], axis=0)))**2
        print(pred_int_80.shape)
        return forecasts
