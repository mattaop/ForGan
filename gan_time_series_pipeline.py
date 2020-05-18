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
seed = 4
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
from data.load_data import load_oslo_temperature, load_australia_temperature
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
        data = generate_sine_data(num_points=5000)
    elif cfg['data_source'].lower() == 'oslo':
        data = load_oslo_temperature()
    elif cfg['data_source'].lower() == 'australia':
        data = load_australia_temperature()

    else:
        return None
    train = data[:-int(len(data)*cfg['test_split'])]
    test = data[-int(len(data)*cfg['test_split']+window_size):]
    train, test = scale_data(train, test)
    return train, test


def scale_data(train, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
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


def train_gan(gan, data, epochs, batch_size=128, verbose=1):
    # Split data in training and validation set
    train, val = data[:-int(len(data)*0.1)], data[-int(gan.window_size+len(data)*0.1):]

    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(train, gan.window_size, gan.output_size)

    history = gan.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    validation_mse = compute_validation_error(gan, val)

    """
    plt.figure()
    plt.title('Training Mean Squared Error')
    plt.plot(np.linspace(1, epochs, epochs), history['mse'], label='Forecast MSE generator')
    plt.legend()
    # plt.show()
    
    plt.figure()
    plt.title('Training Generator and Discriminator Loss')
    plt.plot(np.linspace(1, epochs, epochs), history['G_loss'], label='Generator loss')
    plt.plot(np.linspace(1, epochs, epochs), history['D_loss'], label='Discriminator loss')
    plt.legend()
    # plt.show()
    """

    return gan, validation_mse


def test_model(gan, data, validation_mse, plot=True):
    forecast = gan.monte_carlo_forecast(data, steps=int(len(data)-gan.window_size))  # steps x horizon x mc_forward_passes
    forecast_mean = forecast.mean(axis=-1)
    forecast_std = forecast.std(axis=-1)
    forecast_var = forecast.var(axis=-1)

    total_uncertainty = np.sqrt(forecast_var + validation_mse)
    if plot:
        x_pred = np.linspace(gan.window_size+1, len(data), len(data)-gan.window_size)
        plt.figure()
        plt.plot(np.linspace(1, len(data), len(data)), data, label='Data')
        plt.plot(x_pred, forecast_mean[:, 0], label='Predictions')
        plt.fill_between(x_pred, forecast_mean[:, 0]-1.28*forecast_std[:, 0], forecast_mean[:, 0]+1.28*forecast_std[:, 0],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(x_pred, forecast_mean[:, 0]-1.96*forecast_std[:, 0], forecast_mean[:, 0]+1.96*forecast_std[:, 0],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.legend()
        plt.show()
    forecast_mse = sliding_window_mse(forecast_mean, data[gan.window_size:], gan.forecasting_horizon)

    forecast_smape = sliding_window_smape(forecast_mean, data[gan.window_size:], gan.forecasting_horizon)

    coverage_80_1 = sliding_window_coverage(actual_values=data[gan.window_size:],
                                            upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                            lower_limits=np.quantile(forecast, q=0.1, axis=-1),
                                            forecast_horizon=gan.forecasting_horizon)
    coverage_95_1 = sliding_window_coverage(actual_values=data[gan.window_size:],
                                            upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                            lower_limits=np.quantile(forecast, q=0.025, axis=-1),
                                            forecast_horizon=gan.forecasting_horizon)
    coverage_80_2 = sliding_window_coverage(actual_values=data[gan.window_size:],
                                            upper_limits=forecast_mean + 1.28*total_uncertainty,
                                            lower_limits=forecast_mean - 1.28*total_uncertainty,
                                            forecast_horizon=gan.forecasting_horizon)
    coverage_95_2 = sliding_window_coverage(actual_values=data[gan.window_size:],
                                            upper_limits=forecast_mean + 1.96 * total_uncertainty,
                                            lower_limits=forecast_mean - 1.96 * total_uncertainty,
                                            forecast_horizon=gan.forecasting_horizon)
    width_80_1 = np.quantile(forecast, q=0.9, axis=-1)-np.quantile(forecast, q=0.1, axis=-1)
    width_95_1 = np.quantile(forecast, q=0.975, axis=-1)-np.quantile(forecast, q=0.025, axis=-1)
    width_80_2 = 2*1.28*total_uncertainty
    width_95_2 = 2*1.96*total_uncertainty
    if plot:
        print('80%-prediction interval coverage - Mean:', coverage_80_1.mean(),
              '\n Forecast horizon:', coverage_80_1)
        print('95%-prediction interval coverage - Mean:', coverage_95_1.mean(),
              '\n Forecast horizon:', coverage_95_1)

        print('80%-prediction interval coverage - Mean:',coverage_80_2.mean(),
              '\n Forecast horizon:', coverage_80_2)
        print('95%-prediction interval coverage - Mean:',coverage_95_2.mean(),
              '\n Forecast horizon:', coverage_95_2)

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
    return forecast_mse, forecast_smape, coverage_80_1, coverage_95_1, coverage_80_2, coverage_95_2, width_80_1, width_95_1, width_80_2, width_95_2, forecast_std


def pipeline():
    cfg = load_config_file('config\\config.yml')
    forecast_mse_list, forecast_smape_list, validation_mse_list, forecast_std_list = [], [], [], []
    width_80_1_list, width_95_1_list, width_80_2_list, width_95_2_list = [], [], [], []
    coverage_80_1_list, coverage_95_1_list, coverage_80_2_list, coverage_95_2_list = [], [], [], []
    for i in range(1):
        gan = configure_model(cfg=cfg['gan'])
        train, test = load_data(cfg=cfg['data'], window_size=gan.window_size)
        trained_gan, validation_mse = train_gan(gan=gan, data=train, epochs=cfg['gan']['epochs'],
                                                batch_size=cfg['gan']['batch_size'], verbose=0)
        forecast_mse, forecast_smape, coverage_80_1, coverage_95_1, coverage_80_2, coverage_95_2, width_80_1, \
            width_95_1, width_80_2, width_95_2, forecast_std = test_model(gan=trained_gan, data=test, validation_mse=validation_mse, plot=False)
        forecast_mse_list.append(forecast_mse), forecast_smape_list.append(forecast_smape), validation_mse_list.append(validation_mse)
        forecast_std_list.append(forecast_std)
        coverage_80_1_list.append(coverage_80_1), coverage_95_1_list.append(coverage_95_1)
        coverage_80_2_list.append(coverage_80_2), coverage_95_2_list.append(coverage_95_2)
        width_80_1_list.append(width_80_1), width_95_1_list.append(width_95_1)
        width_80_2_list.append(width_80_2), width_95_2_list.append(width_95_2)

    print('Mean validation MSE:', np.mean(np.mean(validation_mse_list, axis=0)))
    print('Mean forecast MSE:', np.mean(np.mean(forecast_mse_list, axis=0)))
    print('Forecast MSE:', np.mean(np.array(forecast_mse_list), axis=0))
    print('Mean forecast SMAPE:', np.mean(np.mean(forecast_smape_list, axis=0)))
    print('Forecast SMAPE:', np.mean(np.array(forecast_smape_list), axis=0))
    print('Estimated Standard deviation:', np.mean(np.mean(forecast_std_list, axis=0)),
          np.std(np.mean(forecast_std_list, axis=-1), axis=0))
    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(coverage_80_1_list, axis=0)),
          ', width:', np.mean(width_80_1_list),
          '\n Forecast horizon:', np.mean(np.array(coverage_80_1_list), axis=0))
    print('95%-prediction interval coverage - Mean:',  np.mean(np.mean(coverage_95_1_list, axis=0)),
          ', width:', np.mean(width_95_1_list),
          '\n Forecast horizon:', np.mean(np.array(coverage_95_1_list), axis=0))

    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(coverage_80_2_list, axis=0)),
          ', width:', np.mean(width_80_2_list),
          '\n Forecast horizon:', np.mean(np.array(coverage_80_2_list), axis=0))
    print('95%-prediction interval coverage - Mean:', np.mean(np.mean(coverage_95_2_list, axis=0)),
          ', width:', np.mean(width_95_2_list),
          '\n Forecast horizon:', np.mean(np.array(coverage_95_2_list), axis=0))


if __name__ == '__main__':
    pipeline()
