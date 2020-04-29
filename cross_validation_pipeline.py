import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import random as rn
import tensorflow as tf

print(tf.__version__)
seed = 1
rn.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from keras import backend as k

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count={'CPU': 1})
sess = tf.Session(graph=tf.get_default_graph(), config=config)
k.set_session(sess)

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.load_config import load_config_file
from models.get_model import get_gan
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from data.load_data import load_oslo_temperature
from utility.compute_statistics import *


def configure_model(cfg):
    gan = get_gan(cfg)
    gan.build_model()

    paths = ['ims',
             'ims/' + gan.plot_folder
             ]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    return gan


def load_data(cfg, window_size):
    if cfg['data_source'].lower() == 'sine':
        data = generate_sine_data(num_points=1000)
    elif cfg['data_source'].lower() == 'oslo':
        data = load_oslo_temperature()
    else:
        return None
    print('Data shape', data.shape)
    train = data[:-int(len(data) * cfg['test_split'])]
    test = data[-int(len(data) * cfg['test_split'] + window_size):]
    train, test = scale_data(train, test)
    return train, test


def scale_data(train, test):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test


def plot_results(y, label, y2=None, y2_label=None, title='', y_label='Value'):
    x = np.linspace(1, len(y), len(y))
    plt.figure()
    plt.title(title)
    plt.plot(x, y, label=label)
    if y2 is not None:
        plt.plot(x, y2, label=y2_label)
    plt.ylabel(y_label)
    plt.xlabel('Forecast horizon')
    plt.legend()
    plt.show()


def compute_validation_error(gan, data):
    # Split validation data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_val, y_val = split_sequence(data, gan.window_size, gan.output_size)

    # Compute inherent noise on validation set
    y_predicted = gan.forecast(x_val)

    validation_mse = np.zeros(gan.output_size)
    for i in range(gan.output_size):
        validation_mse[i] = mean_squared_error(y_val[:, i], y_predicted[:, i])
    return validation_mse


def train_cross_validation_gan(gan, data, cfg, epochs, batch_size=128):
    # Split data in training and validation set
    train, val = data[:-int(len(data) * 0.1)], data[-int(gan.window_size + len(data) * 0.1):]
    x_train, y_train = split_sequence(train, gan.window_size, gan.output_size)
    x_val, y_val = split_sequence(val, gan.window_size, gan.output_size)
    validation_forecasts = np.zeros([cfg['cross_validation_setup']['k-folds'], x_val.shape[0], gan.output_size])
    validation_mse = np.zeros([cfg['cross_validation_setup']['k-folds'], gan.output_size])
    for i in range(cfg['cross_validation_setup']['k-folds']):
        cross_val_gan = configure_model(cfg=cfg['gan'])
        idx = np.random.randint(0, x_train.shape[0], int(x_train.shape[0]))
        x_fold = x_train[idx]
        y_fold = y_train[idx]
        cross_val_gan.fit(x_fold, y_fold, epochs=epochs, batch_size=batch_size, verbose=0)
        # validation_mse = compute_validation_error(gan, val)
        validation_forecasts[i] = cross_val_gan.forecast(x_val)
        print(y_val.shape)
        for j in range(gan.output_size):
            validation_mse[i, j] = mean_squared_error(y_val[:, j], validation_forecasts[i, :, j])
    print(validation_forecasts.shape)
    gan.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    training_variance = validation_forecasts.var(axis=0).mean(axis=0)
    inherent_noise = validation_mse.mean(axis=0) - training_variance

    print('Validation_mse:', validation_mse.mean(axis=0))
    print('Training variance:', training_variance)
    print('Inherent noise estimate:', inherent_noise)

    return gan, inherent_noise


def test_model(gan, data, validation_mse):
    forecast = gan.monte_carlo_forecast(data,
                                        steps=int(len(data) - gan.window_size))  # steps x horizon x mc_forward_passes
    forecast_mean = forecast.mean(axis=-1)
    forecast_std = forecast.std(axis=-1)
    forecast_var = forecast.var(axis=-1)

    print('Mutual information:', normalized_mutual_info_score(forecast_mean[:, 0], data[gan.window_size:, 0]))
    total_uncertainty = np.sqrt(forecast_var + validation_mse)

    x_pred = np.linspace(gan.window_size + 1, len(data), len(data) - gan.window_size)
    plt.figure()
    plt.plot(np.linspace(1, len(data), len(data)), data, label='Data')
    plt.plot(x_pred, forecast_mean[:, 0], label='Predictions')
    plt.fill_between(x_pred, forecast_mean[:, 0] - 1.28 * forecast_std[:, 0],
                     forecast_mean[:, 0] + 1.28 * forecast_std[:, 0],
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(x_pred, forecast_mean[:, 0] - 1.96 * forecast_std[:, 0],
                     forecast_mean[:, 0] + 1.96 * forecast_std[:, 0],
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.legend()
    plt.show()
    print('Mean validaiton MSE:', validation_mse)
    print('Mean forecast MSE:',
          sliding_window_mse(forecast_mean, data[gan.window_size:], gan.forecasting_horizon).mean())
    print('Forecast MSE:', sliding_window_mse(forecast_mean, data[gan.window_size:], gan.forecasting_horizon))
    print('Mean forecast SMAPE:',
          sliding_window_smape(forecast_mean, data[gan.window_size:], gan.forecasting_horizon).mean())
    print('Forecast SMAPE:', sliding_window_smape(forecast_mean, data[gan.window_size:], gan.forecasting_horizon))

    # print('Mean validation MSE:', validation_mse.mean())
    # print('Validation MSE:', validation_mse)
    print('Mean forecast standard deviation:', forecast_std.mean(axis=0))
    print('Mean total forecast standard deviation:', total_uncertainty.mean(axis=0).mean())
    print('Forecast standard deviation:', total_uncertainty.mean(axis=0))
    print('80%-prediction interval coverage - Mean:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                  lower_limits=np.quantile(forecast, q=0.1, axis=-1),
                                  forecast_horizon=gan.forecasting_horizon).mean(),
          '\n Forecast horizon:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                  lower_limits=np.quantile(forecast, q=0.1, axis=-1),
                                  forecast_horizon=gan.forecasting_horizon))
    print('95%-prediction interval coverage - Mean:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                  lower_limits=np.quantile(forecast, q=0.025, axis=-1),
                                  forecast_horizon=gan.forecasting_horizon).mean(),
          '\n Forecast horizon:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                  lower_limits=np.quantile(forecast, q=0.025, axis=-1),
                                  forecast_horizon=gan.forecasting_horizon))

    print('80%-prediction interval coverage - Mean:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=forecast_mean + 1.28 * total_uncertainty,
                                  lower_limits=forecast_mean - 1.28 * total_uncertainty,
                                  forecast_horizon=gan.forecasting_horizon).mean(),
          '\n Forecast horizon:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=forecast_mean + 1.28 * total_uncertainty,
                                  lower_limits=forecast_mean - 1.28 * total_uncertainty,
                                  forecast_horizon=gan.forecasting_horizon))
    print('95%-prediction interval coverage - Mean:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=forecast_mean + 1.96 * total_uncertainty,
                                  lower_limits=forecast_mean - 1.96 * total_uncertainty,
                                  forecast_horizon=gan.forecasting_horizon).mean(),
          '\n Forecast horizon:',
          sliding_window_coverage(actual_values=data[gan.window_size:],
                                  upper_limits=forecast_mean + 1.96 * total_uncertainty,
                                  lower_limits=forecast_mean - 1.96 * total_uncertainty,
                                  forecast_horizon=gan.forecasting_horizon))

    plot_results(sliding_window_mse(forecast_mean, data[gan.window_size:], gan.forecasting_horizon),
                 label='Forecast MSE', title='Mean Squared Forecast Error', y_label='MSE')
    plot_results(sliding_window_coverage(actual_values=data[gan.window_size:],
                                         upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                         lower_limits=np.quantile(forecast, q=0.1, axis=-1),
                                         forecast_horizon=gan.forecasting_horizon),
                 label='80% PI coverage',
                 y2=sliding_window_coverage(actual_values=data[gan.window_size:],
                                            upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                            lower_limits=np.quantile(forecast, q=0.025, axis=-1),
                                            forecast_horizon=gan.forecasting_horizon),
                 y2_label='95% PI coverage',
                 title='Prediction Interval Coverage', y_label='Coverage')


def pipeline():
    cfg = load_config_file('config\\config.yml')
    gan = configure_model(cfg=cfg['gan'])
    train, test = load_data(cfg=cfg['data'], window_size=gan.window_size)
    trained_gan, validation_mse = train_cross_validation_gan(gan=gan, data=train, cfg=cfg, epochs=cfg['gan']['epochs'],
                                                             batch_size=cfg['gan']['batch_size'])
    test_model(gan=trained_gan, data=test, validation_mse=validation_mse)


if __name__ == '__main__':
    pipeline()
