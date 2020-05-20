import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def compute_coverage(upper_limits, lower_limits, actual_values):
    coverage = 0
    for i in range(len(actual_values)):
        if lower_limits[i] < actual_values[i] < upper_limits[i]:
            coverage += 1
    return coverage/len(actual_values)


def sliding_window_coverage(upper_limits, lower_limits, actual_values, forecast_horizon):
    coverage = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        coverage[i] = compute_coverage(upper_limits[:len(actual_values)-i, i], lower_limits[:len(actual_values)-i, i],
                                       actual_values[i:, 0])
    return coverage


def sliding_window_mse(forecast_mean, actual_values, forecast_horizon):
    mse = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        mse[i] = mean_squared_error(actual_values[i:, 0], forecast_mean[:len(actual_values)-i, i])
    return mse


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 2*100 * np.mean(diff)


def sliding_window_smape(forecast_mean, actual_values, forecast_horizon):
    f_smape = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        f_smape[i] = smape(actual_values[i:, 0], forecast_mean[:len(actual_values)-i, i])
    return f_smape


def print_coverage(mean, uncertainty, actual_values):
    coverage_80pi = compute_coverage(upper_limits=mean + 1.28 * uncertainty,
                                     lower_limits=mean - 1.28 * uncertainty,
                                     actual_values=actual_values)
    coverage_95pi = compute_coverage(upper_limits=mean + 1.96 * uncertainty,
                                     lower_limits=mean - 1.96 * uncertainty,
                                     actual_values=actual_values)
    print('80%-prediction interval coverage: ', coverage_80pi)
    print('95%-prediction interval coverage: ', coverage_95pi)
