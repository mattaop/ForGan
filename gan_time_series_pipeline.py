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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.load_config import load_config_file
from models.get_model import get_GAN
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from data.load_data import load_oslo_temperature
from utility.compute_coverage import print_coverage, compute_coverage, sliding_window_coverage, sliding_window_mse


def configure_model(model_name):
    gan = get_GAN(model_name)
    gan.build_gan()

    paths = ['ims',
             'ims/' + gan.plot_folder
             ]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    return gan


def load_data(cfg, window_size):
    if cfg['data_source'].lower() == 'sine':
        data = generate_sine_data(num_points=500)
    elif cfg['data_source'].lower() == 'oslo':
        data = load_oslo_temperature()
    else:
        return None
    print(data.shape)
    train = data[:-int(len(data)*cfg['test_split'])]
    test = data[-int(len(data)*cfg['test_split']+window_size):]
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
    x_val, y_val = split_sequence(data, gan.window_size, gan.forecasting_horizon)

    # Compute inherent noise on validation set
    y_predicted = gan.forecast(x_val, forward_passes=5000)

    validation_mse = np.zeros(gan.forecasting_horizon)
    for i in range(gan.forecasting_horizon):
        validation_mse[i] = mean_squared_error(y_val[:, i], y_predicted[:, i])
    return validation_mse


def train_gan(gan, data, epochs, batch_size=128, discriminator_epochs=1):
    # Split data in training and validation set
    train, val = data[:-int(len(data)*0.1)], data[-int(gan.window_size+len(data)*0.1):]

    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(train, gan.window_size, gan.forecasting_horizon)

    history = gan.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, discriminator_epochs=discriminator_epochs)

    validation_mse = compute_validation_error(gan, val)

    plt.figure()
    plt.plot(np.linspace(1, epochs, epochs), history['mse'], label='Forecast MSE generator')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(np.linspace(1, epochs, epochs), history['G_loss'], label='Generator loss')
    plt.plot(np.linspace(1, epochs, epochs), history['D_loss'], label='Discriminator loss')
    plt.legend()
    plt.show()

    return gan, validation_mse


def test_model(gan, data, validation_mse, mc_forward_passes=1000):
    forecast = gan.monte_carlo_forecast(data, int(len(data)-gan.window_size), mc_forward_passes)  # steps x horizon x mc_forward_passes
    forecast_mean = forecast.mean(axis=-1)
    forecast_std = forecast.std(axis=-1)
    forecast_var = forecast.var(axis=-1)

    total_uncertainty = np.sqrt(forecast_var + validation_mse)

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
    print('Mean forecast MSE:', sliding_window_mse(forecast_mean, data[gan.window_size:], gan.forecasting_horizon).mean())
    print('Forecast MSE:', sliding_window_mse(forecast_mean, data[gan.window_size:], gan.forecasting_horizon))
    print('Mean validation MSE:', validation_mse.mean())
    print('Validation MSE:', validation_mse)
    print('Mean forecast standard deviation:', forecast_std.mean(axis=0))
    print('Mean total forecast standard deviation:', total_uncertainty.mean(axis=0))
    print('80%-prediction interval coverage:', sliding_window_coverage(actual_values=data[gan.window_size:],
                                                                       upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                                                       lower_limits=np.quantile(forecast, q=0.1, axis=-1),
                                                                       forecast_horizon=gan.forecasting_horizon).mean())
    print('95%-prediction interval coverage:', sliding_window_coverage(actual_values=data[gan.window_size:],
                                                                       upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                                                       lower_limits=np.quantile(forecast, q=0.025, axis=-1),
                                                                       forecast_horizon=gan.forecasting_horizon).mean())
    print_coverage(mean=forecast_mean[:, 0], uncertainty=forecast_std[:, 0], actual_values=data[gan.window_size:])
    print_coverage(mean=forecast_mean[:, 0], uncertainty=total_uncertainty[:, 0], actual_values=data[gan.window_size:])

    plot_results(forecast_mean.mean(axis=0), label='Forecast MSE', title='Mean Squared Forecast Error', y_label='MSE')
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
    gan = configure_model(model_name=cfg['gan']['model_name'])
    train, test = load_data(cfg=cfg['data'], window_size=gan.window_size)
    trained_gan, validation_mse = train_gan(gan=gan, data=train, epochs=10, batch_size=256, discriminator_epochs=1)
    test_model(gan=trained_gan, data=test, validation_mse=validation_mse, mc_forward_passes=5000)


if __name__ == '__main__':
    pipeline()
