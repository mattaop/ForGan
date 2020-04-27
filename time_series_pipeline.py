import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import random as rn
import tensorflow as tf

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
from models.get_model import get_model
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from utility.compute_statistics import print_coverage


def configure_model(model_name):
    model = get_model(model_name)
    model.build_model()

    return model


def load_data(cfg, window_size):
    if cfg['data_source'].lower() == 'sine':
        data = generate_sine_data(num_points=5000)
    else:
        return None
    train = data[:-int(len(data)*(cfg['test_split']))]
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
    print(x_val.shape)
    print(y_val.shape)
    y_predicted = model.forecast(x_val)
    print(y_predicted.shape)
    inherent_noise = mean_squared_error(y_val[:, :, 0], y_predicted)
    return inherent_noise


def train_model(model, data, epochs, batch_size=128):
    # Split data in training and validation set
    train, val = data[:-int(len(data)*0.1)], data[-int(model.window_size+len(data)*0.1):]

    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(train, model.window_size, model.forecasting_horizon)

    # Train model
    history = model.model.fit(x_train, y_train[:, :, 0], epochs=epochs, batch_size=batch_size, validation_split=0.1,
                              verbose=2)
    # Compute validation error
    validation_error = compute_validation_error(model, val)

    plt.figure()
    plt.plot(np.linspace(1, epochs, epochs), history.history['loss'], label='Training loss')
    plt.legend()
    plt.show()

    return model, validation_error


def test_model(model, data, validation_error, mc_forward_passes=500):
    forecast = model.monte_carlo_forecast(data, int(len(data)-model.window_size), mc_forward_passes)  # steps x horizon x mc_forward_passes
    forecast_mean = forecast.mean(axis=-1)
    forecast_variance = forecast.var(axis=-1)

    forecast_uncertainty = np.sqrt(validation_error + forecast_variance)

    x_pred = np.linspace(model.window_size+1, len(data), len(data)-model.window_size)
    plt.figure()
    plt.plot(np.linspace(1, len(data), len(data)), data, label='Data')
    plt.plot(x_pred, forecast_mean[:, 0], label='Predictions')
    plt.fill_between(x_pred, forecast_mean[:, 0]-1.28*forecast_uncertainty[:, 0], forecast_mean[:, 0]+1.28*forecast_uncertainty[:, 0],
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(x_pred, forecast_mean[:, 0]-1.96*forecast_uncertainty[:, 0], forecast_mean[:, 0]+1.96*forecast_uncertainty[:, 0],
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.legend()
    plt.show()
    print('Forecast MSE:', mean_squared_error(data[model.window_size:], forecast_mean[:, 0]))
    print('Mean forecast standard deviation:', forecast_uncertainty.mean(axis=0))
    print_coverage(mean=forecast_mean[:, 0], uncertainty=forecast_uncertainty[:, 0], actual_values=data[model.window_size:])


def pipeline():
    cfg = load_config_file('config\\config.yml')
    model = configure_model(model_name=cfg['gan']['model_name'])
    train, test = load_data(cfg=cfg['data'], window_size=model.window_size)
    trained_model, validation_error = train_model(model=model, data=train, epochs=500, batch_size=64)
    test_model(model=trained_model, data=test, validation_error=validation_error, mc_forward_passes=500)


if __name__ == '__main__':
    pipeline()
