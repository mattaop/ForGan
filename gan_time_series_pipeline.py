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
from utility.compute_coverage import print_coverage, compute_coverage


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
    else:
        return None
    train = data[:-int(len(data)*cfg['test_split'])]
    test = data[-int(len(data)*cfg['test_split']+window_size):]
    train, test = scale_data(train, test)
    return train, test


def scale_data(train, test):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test


def compute_validation_error(model, data):
    # Split validation data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_val, y_val = split_sequence(data, model.window_size, model.forecasting_horizon)

    # Compute inherent noise on validation set
    y_predicted = model.forecast(x_val)
    print(y_predicted.shape)
    print(y_val.shape)
    inherent_noise = mean_squared_error(y_val[:, :, 0], y_predicted)
    return inherent_noise


def train_gan(gan, data, epochs, batch_size=128, discriminator_epochs=1):
    # Split data in training and validation set
    train, val = data[:-int(len(data)*0.1)], data[-int(gan.window_size+len(data)*0.1):]

    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(train, gan.window_size, gan.forecasting_horizon)

    history = gan.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, discriminator_epochs=discriminator_epochs)

    validation_error = compute_validation_error(gan, val)

    plt.figure()
    plt.plot(np.linspace(1, epochs, epochs), history['mse'], label='Forecast MSE generator')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(np.linspace(1, epochs, epochs), history['G_loss'], label='Generator loss')
    plt.plot(np.linspace(1, epochs, epochs), history['D_loss'], label='Discriminator loss')
    plt.legend()
    plt.show()

    return gan, validation_error


def test_model(gan, data, validation_error, mc_forward_passes=1000):
    forecast = gan.monte_carlo_forecast(data, int(len(data)-gan.window_size), mc_forward_passes)  # steps x horizon x mc_forward_passes
    forecast_mean = forecast.mean(axis=-1)
    forecast_std = forecast.std(axis=-1)
    forecast_var = forecast.var(axis=-1)

    total_uncertainty = np.sqrt(forecast_var + validation_error)

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
    print('Forecast MSE:', mean_squared_error(data[gan.window_size:], forecast_mean[:, 0]))
    print('Validation MSE:', validation_error)
    print('Mean forecast standard deviation:', forecast_std.mean(axis=0))
    print('Mean total forecast standard deviation:', total_uncertainty.mean(axis=0))
    print('80%-prediction interval coverage:', compute_coverage(actual_values=data[gan.window_size:],
                                                                upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                                                lower_limits=np.quantile(forecast, q=0.1, axis=-1)))
    print('95%-prediction interval coverage:', compute_coverage(actual_values=data[gan.window_size:],
                                                                upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                                                lower_limits=np.quantile(forecast, q=0.025, axis=-1)))
    print_coverage(mean=forecast_mean[:, 0], uncertainty=forecast_std[:, 0], actual_values=data[gan.window_size:])
    print_coverage(mean=forecast_mean[:, 0], uncertainty=total_uncertainty[:, 0], actual_values=data[gan.window_size:])


def pipeline():
    cfg = load_config_file('config\\config.yml')
    gan = configure_model(model_name=cfg['gan']['model_name'])
    train, test = load_data(cfg=cfg['data'], window_size=gan.window_size)
    trained_gan, validation_error = train_gan(gan=gan, data=train, epochs=1000, batch_size=256, discriminator_epochs=1)
    test_model(gan=trained_gan, data=test, validation_error=validation_error, mc_forward_passes=500)


if __name__ == '__main__':
    pipeline()
