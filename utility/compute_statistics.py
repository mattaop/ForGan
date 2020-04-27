import numpy as np
from sklearn.metrics import mean_squared_error


def compute_coverage(upper_limits, lower_limits, actual_values):
    coverage = 0
    for i in range(len(actual_values)):
        if lower_limits[i] < actual_values[i] < upper_limits[i]:
            coverage += 1
    return coverage/len(actual_values)


def sliding_window_coverage(upper_limits, lower_limits, actual_values, forecast_horizon):
    coverage = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        coverage[i] = compute_coverage(upper_limits[i:len(actual_values), i], lower_limits[i:len(actual_values), i],
                                       actual_values[i:, 0])
    return coverage


def sliding_window_mse(forecast_mean, actual_values, forecast_horizon):
    mse = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        mse[i] = mean_squared_error(actual_values[i:, 0], forecast_mean[i:len(actual_values), i])
    return mse


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / (np.abs(y_true)+np.abs(y_pred))) * 100


def sliding_window_smape(forecast_mean, actual_values, forecast_horizon):
    smape = np.zeros(forecast_horizon)
    for i in range(forecast_horizon):
        smape[i] = 2/(i+1)*symmetric_mean_absolute_percentage_error(actual_values[i:, 0],
                                                                    forecast_mean[i:len(actual_values), i])
    return smape


def print_coverage(mean, uncertainty, actual_values):
    coverage_80pi = compute_coverage(upper_limits=mean + 1.28 * uncertainty,
                                     lower_limits=mean - 1.28 * uncertainty,
                                     actual_values=actual_values)
    coverage_95pi = compute_coverage(upper_limits=mean + 1.96 * uncertainty,
                                     lower_limits=mean - 1.96 * uncertainty,
                                     actual_values=actual_values)
    print('80%-prediction interval coverage: ', coverage_80pi)
    print('95%-prediction interval coverage: ', coverage_95pi)
