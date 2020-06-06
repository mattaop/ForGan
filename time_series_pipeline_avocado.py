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
from tqdm import tqdm
from config.load_config import load_config_file
from utility.split_data import split_sequence
from utility.compute_statistics import *
from time_series_pipeline import configure_model, load_data
from time_series_pipeline_with_validation import compute_validation_error, test_model


def train_model_on_multiple_series(model, data, epochs, cfg, batch_size=32, verbose=1):
    train_x = []
    train_y = []
    validation_x = []
    validation_y = []
    validation = []
    i = 1
    for name, values in data.iteritems():
        # Split data in training and validation set
        train = np.expand_dims(values[:-int(len(data) * cfg['val_split'])].values, axis=-1)
        val = np.expand_dims(values[-int(model.window_size + len(data) * cfg['val_split']):].values, axis=-1)
        validation.append(val)

        # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
        x_train, y_train = split_sequence(train, model.window_size, model.output_size)
        x_val, y_val = split_sequence(val, model.window_size, model.output_size)

        train_x.append(x_train)
        train_y.append(y_train)
        validation_x.append(x_val)
        validation_y.append(y_val)
        i += 1
    train_x = np.asarray(train_x)
    train_x = train_x.reshape([train_x.shape[0]*train_x.shape[1], train_x.shape[2], train_x.shape[3]])
    train_y = np.asarray(train_y)

    train_y = train_y.reshape([train_y.shape[0]*train_y.shape[1], train_y.shape[2], train_y.shape[3]])

    validation_x = np.asarray(validation_x)
    validation_x = validation_x.reshape([validation_x.shape[0] * validation_x.shape[1], validation_x.shape[2], validation_x.shape[3]])
    validation_y = np.asarray(validation_y)
    validation_y = validation_y.reshape([validation_y.shape[0] * validation_y.shape[1], validation_y.shape[2], validation_y.shape[3]])

    validation = np.asarray(validation)

    history = model.fit(train_x, train_y, x_val=validation_x, y_val=validation_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose)

    validation_mse = compute_validation_error(model, validation_x, validation_y)

    return model, validation_mse, validation


def time_series_avocado_pipeline(cfg):
    forecast_mse_list, forecast_smape_list, forecast_mase_list = [], [], []
    validation_mse_list, forecast_std_list = [], []
    w_80_list, w_95_list = [], []
    c_80_list, c_95_list = [], []
    msis_80_list, msis_95_list = [], []
    df_train, df_test, scalers = load_data(cfg=cfg, window_size=cfg['window_size'])
    model = configure_model(cfg=cfg)
    trained_model, validation_mse, val = train_model_on_multiple_series(model=model, data=df_train,
                                                                        epochs=cfg['epochs'], cfg=cfg,
                                                                        batch_size=cfg['batch_size'], verbose=0)
    i = 0
    for columnName, columnData in tqdm(df_train.iteritems(), total=df_train.columns.shape[0]):
        train = df_train[columnName].values.reshape(-1, 1)
        test = df_test[columnName].values.reshape(-1, 1)
        scaler = scalers[i]
        naive_error = compute_naive_error(scaler.inverse_transform(train), seasonality=cfg['seasonality'],
                                          forecast_horizon=model.forecasting_horizon)
        start_time = time.time()
        if cfg['data_source'] == 'avocado':
            validation_name = "/" + columnName[1] + "_" + columnName[2] + "_validation_results.txt"
            test_name = "/" + columnName[1] + "_" + columnName[2] + "_test_results.txt"
        else:
            validation_name = "/" + columnName + "_validation_results.txt"
            test_name = "/" + columnName + "_test_results.txt"

        training_time = time.time() - start_time
        test_model(model=trained_model, data=val[i], validation_mse=validation_mse, cfg=cfg, naive_error=naive_error,
                   scaler=scaler, plot=False, file_name=validation_name,
                   min_max=(np.max(scaler.inverse_transform(train))-np.min(scaler.inverse_transform(train)))/
                           (np.max(train)-np.min(train)),
                   disable_pbar=True)
        mse, smape, mase, std, c_80, c_95, w_80, w_95, msis_80, msis_95 = \
            test_model(model=trained_model, data=test, validation_mse=validation_mse, cfg=cfg,
                       naive_error=naive_error, scaler=scaler, plot=False, file_name=test_name,
                       min_max=(np.max(scaler.inverse_transform(train)) - np.min(scaler.inverse_transform(train))) /
                               (np.max(train) - np.min(train)), disable_pbar=True)
        msis_80_list.append(msis_80), msis_95_list.append(msis_95)
        forecast_mse_list.append(mse), forecast_smape_list.append(smape), forecast_mase_list.append(mase)
        validation_mse_list.append(validation_mse)
        forecast_std_list.append(std)
        c_80_list.append(c_80), c_95_list.append(c_95)
        w_80_list.append(w_80), w_95_list.append(w_95)
        i += 1

    print('========================================================'
          '\n================ Point Forecast Metrics ================'
          '\n========================================================')
    # print('Mean validation MSE:', np.mean(np.mean(validation_mse_list, axis=0)))
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

    file_path = cfg['results_path']+"/test_results_mean.txt"

    with open(file_path, "a") as f:
        f.write("mse,smape,mase,std,coverage_80,coverage_95,width_80,width_95,msis_80,msis_95\n")
        for (mse, smape, mase, std, c_80, c_95, w_80, w_95, m_80, m_95) in zip(np.mean(forecast_mse_list, axis=0),
                                                                               np.mean(forecast_smape_list, axis=0),
                                                                               np.mean(forecast_mase_list, axis=0),
                                                                               np.mean(forecast_std_list, axis=0),
                                                                               np.mean(c_80_list, axis=0),
                                                                               np.mean(c_95_list, axis=0),
                                                                               np.mean(w_80_list, axis=0),
                                                                               np.mean(w_95_list, axis=0),
                                                                               np.mean(msis_80_list, axis=0),
                                                                               np.mean(msis_95_list, axis=0)):
            f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n".format(mse, smape, mase, std, c_80, c_95, w_80, w_95,
                                                                       m_80, m_95))


if __name__ == '__main__':
    cfg = load_config_file('config\\config.yml')
    time_series_avocado_pipeline(cfg)