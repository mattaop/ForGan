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


from config.load_config import load_config_file
from models.get_model import get_gan
from time_series_pipeline import configure_model, load_data, plot_results, train_model
from utility.split_data import split_sequence
from utility.compute_statistics import *


def test_model(model, data, naive_error, cfg, plot=True, file_name="/test_results.txt"):
    forecast = model.monte_carlo_forecast(data, steps=int(len(data) - model.window_size), plot=plot)  # steps x horizon x mc_forward_passes
    forecast_mse = sliding_window_mse(forecast, data[model.window_size:], model.forecasting_horizon)
    forecast_std = np.sqrt(model.var)

    forecast_smape = sliding_window_smape(forecast, data[model.window_size:], model.forecasting_horizon)

    forecast_mase = sliding_window_mase(forecast, data[model.window_size:], model.forecasting_horizon, naive_error)

    coverage_80 = sliding_window_coverage(actual_values=data[model.window_size:],
                                          upper_limits=model.pred_int_80[:, :, 1],
                                          lower_limits=model.pred_int_80[:, :, 0],
                                          forecast_horizon=model.forecasting_horizon)
    coverage_95 = sliding_window_coverage(actual_values=data[model.window_size:],
                                          upper_limits=model.pred_int_95[:, :, 1],
                                          lower_limits=model.pred_int_95[:, :, 0],
                                          forecast_horizon=model.forecasting_horizon)

    width_80 = model.pred_int_80[:, :, 1] - model.pred_int_80[:, :, 0]
    width_95 = model.pred_int_95[:, :, 1] - model.pred_int_95[:, :, 0]

    file_path = cfg['results_path'] + file_name

    mse = forecast_mse
    smape = forecast_smape
    mase = forecast_mase
    std = np.mean(forecast_std, axis=0)
    c_80 = coverage_80
    c_95 = coverage_95
    w_80 = width_80
    w_95 = width_95

    with open(file_path, "a") as f:
        f.write("mse,smape,mase,std,coverage_80,coverage_95,width_80,width_95\n")
        for (mse, smape, mase, std, c_80, c_95, w_80, w_95) in zip(mse, smape, mase, std, c_80, c_95, w_80, w_95):
            f.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(mse, smape, mase, std, c_80, c_95, w_80, w_95))

    return mse, smape, mase, std, c_80, c_95, w_80, w_95


def pipeline():
    cfg = load_config_file('config\\config.yml')
    forecast_mse_list, forecast_smape_list, forecast_mase_list = [], [], []
    width_80_list, width_95_list = [], []
    coverage_80_list, coverage_95_list = [], []
    for i in range(1):
        model = configure_model(cfg=cfg)
        train, test = load_data(cfg=cfg, window_size=model.window_size)
        naive_error = compute_naive_error(train, seasonality=12, forecast_horizon=model.forecasting_horizon)
        start_time = time.time()
        trained_model = train_model(model=model, data=train, epochs=cfg['epochs'],
                                    batch_size=cfg['batch_size'], verbose=1)
        training_time = time.time() - start_time
        mse, smape, mase, std, c_80, c_95, w_80, w_95 = \
            test_model(model=trained_model, data=test, naive_error=naive_error, cfg=cfg, plot=False)
        forecast_mse_list.append(mse), forecast_smape_list.append(smape), forecast_mase_list.append(mase)
        coverage_80_list.append(c_80), coverage_95_list.append(c_95)
        width_80_list.append(w_80), width_95_list.append(w_95)

    print('========================================================'
          '\n================ Point Forecast Metrics ================'
          '\n========================================================')
    print('Mean forecast MSE:', np.mean(np.mean(forecast_mse_list, axis=0)))
    # print('Forecast MSE:', np.mean(np.array(forecast_mse_list), axis=0))
    print('Mean forecast SMAPE:', np.mean(np.mean(forecast_smape_list, axis=0)))
    # print('Forecast SMAPE:', np.mean(np.array(forecast_smape_list), axis=0))
    print('Mean forecast MASE:', np.mean(np.mean(forecast_mase_list, axis=0)))
    print('Training time:', training_time)

    print('========================================================'
          '\n================== Model Uncertainty ==================='
          '\n========================================================')
    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(coverage_80_list, axis=0)),
          ', width:', np.mean(width_80_list),
          # '\n Forecast horizon:', np.mean(np.array(coverage_80_list), axis=0)
          )
    print('95%-prediction interval coverage - Mean:',  np.mean(np.mean(coverage_95_list, axis=0)),
          ', width:', np.mean(width_95_list),
          # '\n Forecast horizon:', np.mean(np.array(coverage_95_list), axis=0)
    )


if __name__ == '__main__':
    pipeline()
