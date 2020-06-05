import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def compute_coverage(upper_limits, lower_limits, actual_values):
    coverage = 0
    for i in range(len(actual_values)):
        if lower_limits[i] < actual_values[i] < upper_limits[i]:
            coverage += 1
    return coverage / len(actual_values)


def sliding_window_coverage(upper_limits, lower_limits, actual_values, forecast_horizon):
    coverage = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        coverage[i] = compute_coverage(upper_limits[:-i or None, i], lower_limits[:-i or None, i],
                                       actual_values[i:, 0])
    return coverage


def sliding_window_mse(forecast, actual_values, forecast_horizon):
    mse = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        mse[i] = mean_squared_error(actual_values[i:].flatten(), forecast[:-i or None, i].flatten())
    return mse


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 2 * 100 * np.mean(diff)


def pinball_loss(upper_limits, lower_limits, actual_values):
    pass


def sliding_window_smape(forecast, actual_values, forecast_horizon):
    f_smape = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        f_smape[i] = symmetric_mean_absolute_percentage_error(actual_values[i:].flatten(), forecast[:-i or None, i].flatten())
    return f_smape


def sliding_window_mase(forecast, actual_values, forecast_horizon, naive_error):
    f_mase = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        f_mase[i] = mean_absolute_error(actual_values[i:].flatten(), forecast[:-i or None, i].flatten()) / naive_error[i]
    return f_mase


def compute_naive_error(training_data, seasonality=1, forecast_horizon=1):
    naive_error = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        temp_error = 0
        look_back = seasonality*(1+i//seasonality)
        # print('Forecast horizon:', i, ', Look back:', look_back)
        for j in range(len(training_data) - look_back):
            temp_error += np.abs(training_data[j + look_back] - training_data[j])
        naive_error[i] = temp_error/(len(training_data) - look_back - 1)
    return naive_error


def indicator_function(larger, lower):
    if larger > lower:
        return 1
    else:
        return 0


def mean_scaled_interval_score(y, u, l, alpha=0.05, h=1, naive_error=1):
    """

    :param y: Real values
    :param u: Upper bound of prediction interval
    :param l: Lower bound of prediction interval
    :param alpha: Significance level
    :param h: Forecast horizon of the h-step forecast
    :param naive_error: Seasonal error of last-season-prediciton
    :return: mean scaled interval score
    """
    if h > 1:
        mean_interval_score = np.zeros(h)
        for i in range(h):
            interval_score = np.zeros(len(y[i:, 0]))
            for j in range(len(y[i:, 0])):
                interval_score[j] = u[j, i]-l[j, i] + 2/alpha * (l[j, i]-y[i+j]) * indicator_function(larger=l[j, i],
                                                                                                    lower=y[i+j]) \
                                  + 2/alpha * (y[i+j] - u[j, i]) * indicator_function(larger=y[i+j], lower=u[j, i])
            mean_interval_score[i] = np.mean(interval_score)/naive_error[i]

        return mean_interval_score
    else:
        interval_score = u - l + 2 / alpha * (l - y) * indicator_function(larger=l, lower=y) \
                          + 2 / alpha * (y - u) * indicator_function(larger=y, lower=u)
        return interval_score/naive_error


def print_coverage(mean, uncertainty, actual_values):
    coverage_80pi = compute_coverage(upper_limits=mean + 1.28 * uncertainty,
                                     lower_limits=mean - 1.28 * uncertainty,
                                     actual_values=actual_values)
    coverage_95pi = compute_coverage(upper_limits=mean + 1.96 * uncertainty,
                                     lower_limits=mean - 1.96 * uncertainty,
                                     actual_values=actual_values)
    print('80%-prediction interval coverage: ', coverage_80pi)
    print('95%-prediction interval coverage: ', coverage_95pi)
