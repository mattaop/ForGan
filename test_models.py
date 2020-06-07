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

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from keras.models import load_model

from config.load_config import load_config_file
from models.get_model import get_gan
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from data.load_data import load_oslo_temperature, load_australia_temperature
from utility.compute_statistics import *
from time_series_pipeline import configure_model, load_data, plot_results
from time_series_pipeline_with_validation import test_model, compute_validation_error


def train_model(model, data, cfg):
    # Split data in training and validation set
    train, val = data[:-int(len(data)*cfg['val_split'])], data[-int(model.window_size+len(data)*cfg['val_split']):]

    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(train, model.window_size, model.output_size)
    x_val, y_val = split_sequence(val, model.window_size, model.output_size)

    validation_mse = compute_validation_error(model, x_val, y_val)

    return model, validation_mse, val


def pipeline(model_path, model_name):
    cfg = load_config_file(model_path+'config.yml')
    cfg['mc_forward_passes'] = 5000
    forecast_mse_list, forecast_smape_list, forecast_mase_list = [], [], []
    validation_mse_list, forecast_std_list = [], []
    w_80_list, w_95_list = [], []
    c_80_list, c_95_list = [], []
    msis_80_list, msis_95_list = [], []

    for i in range(1):
        model = configure_model(cfg=cfg)
        model.generator = load_model(model_path+model_name)
        train, test, scaler = load_data(cfg=cfg, window_size=model.window_size)
        naive_error = compute_naive_error(scaler.inverse_transform(train), seasonality=12,
                                          forecast_horizon=model.forecasting_horizon)
        start_time = time.time()
        trained_model, validation_mse, val = train_model(model=model, data=train, cfg=cfg)
        training_time = time.time() - start_time
        test_model(model=trained_model, data=val, validation_mse=validation_mse, cfg=cfg, naive_error=naive_error,
                   scaler=scaler, plot=False, file_name="/validation_results_" + model_name + ".txt",   min_max=(np.max(
                                                                       scaler.inverse_transform(train)) - np.min(
                                                                       scaler.inverse_transform(train))) /
                                                                          (np.max(train) - np.min(train)))
        mse, smape, mase, std, c_80, c_95, w_80, w_95, msis_80, msis_95 \
            = test_model(model=trained_model, data=test, validation_mse=validation_mse, cfg=cfg,
                         naive_error=naive_error, scaler=scaler, plot=False,
                         file_name="/test_results_" + model_name + ".txt",
                         min_max=(np.max(scaler.inverse_transform(train)) - np.min(scaler.inverse_transform(train)))
                                 / (np.max(train) - np.min(train)))
        msis_80_list.append(msis_80), msis_95_list.append(msis_95)
        forecast_mse_list.append(mse), forecast_smape_list.append(smape), forecast_mase_list.append(mase)
        validation_mse_list.append(validation_mse)
        forecast_std_list.append(std)
        c_80_list.append(c_80), c_95_list.append(c_95)
        w_80_list.append(w_80), w_95_list.append(w_95)

    print('========================================================'
          '\n================ Point Forecast Metrics ================'
          '\n========================================================')
    print('Mean validation MSE:', np.mean(np.mean(validation_mse_list, axis=0)))
    print('Mean forecast MSE:', np.mean(np.mean(forecast_mse_list, axis=0)))
    # print('Forecast MSE:', np.mean(np.array(forecast_mse_list), axis=0))
    print('Mean forecast SMAPE:', np.mean(np.mean(forecast_smape_list, axis=0)))
    # print('Forecast SMAPE:', np.mean(np.array(forecast_smape_list), axis=0))
    print('Mean forecast MASE:', np.mean(np.mean(forecast_mase_list, axis=0)))
    print('Training time:', training_time)

    print('========================================================'
          '\n================== Model Uncertainty ==================='
          '\n========================================================')
    print('Estimated Standard deviation:', np.mean(np.mean(forecast_std_list, axis=0)))
    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(c_80_list, axis=0)),
          ', width:', np.mean(np.mean(np.array(w_80_list), axis=0)),
          # '\n Forecast horizon:', np.mean(np.array(c_80_list), axis=0)
          )
    print('80%-prediction interval MSIS:', np.mean(np.mean(msis_80_list, axis=0)))

    print('95%-prediction interval coverage - Mean:',  np.mean(np.mean(c_95_list, axis=0)),
          ', width:', np.mean(np.mean(np.array(w_95_list), axis=0)),
          # '\n Forecast horizon:', np.mean(np.array(c_95_list), axis=0)
          )
    print('95%-prediction interval MSIS:', np.mean(np.mean(msis_95_list, axis=0)))


def avocado_pipeline(model_path, model_name):
    cfg = load_config_file(model_path+'config.yml')
    df_train, df_test, scalers = load_data(cfg=cfg, window_size=cfg['window_size'])
    model = configure_model(cfg=cfg)
    model.generator = load_model(model_path + model_name)
    i = 0
    for columnName, columnData in tqdm(df_train.iteritems(), total=df_train.columns.shape[0]):
        train = df_train[columnName].values.reshape(-1, 1)
        test = df_test[columnName].values.reshape(-1, 1)
        scaler = scalers[i]
        naive_error = compute_naive_error(scaler.inverse_transform(train), seasonality=cfg['seasonality'],
                                          forecast_horizon=model.forecasting_horizon)
        trained_model, validation_mse, val = train_model(model=model, data=train, cfg=cfg)
        if cfg['data_source'] == 'avocado':
            test_name = "/" + model_name + "_" + columnName[1] + "_" + columnName[2] + "_test_results.txt"
        else:
            test_name = "/" + model_name + "_" + columnName + "_test_results.txt"
        test_model(model=trained_model, data=test, validation_mse=validation_mse, cfg=cfg,
                   naive_error=naive_error, scaler=scaler, plot=False, file_name=test_name,
                   min_max=(np.max(scaler.inverse_transform(train)) - np.min(scaler.inverse_transform(train))) /
                           (np.max(train) - np.min(train)), disable_pbar=True)
        i += 1


if __name__ == '__main__':
    model_path = 'results/avocado/recurrentgan/minmax/rnn_epochs_30000_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000100/'
    model_name = 'generator_30000.h5'
    avocado_pipeline(model_path, model_name)
