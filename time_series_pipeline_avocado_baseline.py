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
from utility.compute_statistics import *
from time_series_pipeline import configure_model, load_data, plot_results
from time_series_pipeline_baseline import test_model, train_model


def time_series_avocado_baseline_pipeline(cfg):
    forecast_mse_list, forecast_smape_list, forecast_mase_list = [], [], []
    forecast_std_list = []
    w_80_list, w_95_list = [], []
    c_80_list, c_95_list = [], []
    msis_80_list, msis_95_list = [], []
    df_train, df_test, scalers = load_data(cfg=cfg, window_size=cfg['window_size'])
    i = 0
    for columnName, columnData in tqdm(df_train.iteritems(), total=df_train.columns.shape[0]):
        model = configure_model(cfg=cfg)

        train = df_train[columnName].values.reshape(-1, 1)[int(0.5*len(df_train)):]  # dddddddddddddddd
        test = df_test[columnName].values.reshape(-1, 1)
        scaler = scalers[i]
        naive_error = compute_naive_error(scaler.inverse_transform(train), seasonality=cfg['seasonality'],
                                          forecast_horizon=model.forecasting_horizon)
        start_time = time.time()
        trained_model = train_model(model=model, data=train, epochs=cfg['epochs'],
                                    batch_size=cfg['batch_size'], verbose=1)
        training_time = time.time() - start_time
        if cfg['data_source'] == 'avocado':
            test_file_name = "/" + columnName[1] + "_" + columnName[2] + "_test_results.txt"
        else:
            test_file_name = "/" + columnName + "_test_results.txt"
        mse, smape, mase, std, c_80, c_95, w_80, w_95, msis_80, msis_95 = \
            test_model(model=trained_model, data=test, naive_error=naive_error, scaler=scaler, cfg=cfg, plot=True,
                       file_name=test_file_name,
                       min_max=(np.max(scaler.inverse_transform(train))-np.min(scaler.inverse_transform(train))) /
                               (np.max(train)-np.min(train)), disable_pbar=True)
        msis_80_list.append(msis_80), msis_95_list.append(msis_95)
        forecast_mse_list.append(mse), forecast_smape_list.append(smape), forecast_mase_list.append(mase)
        forecast_std_list.append(std)
        c_80_list.append(c_80), c_95_list.append(c_95)
        w_80_list.append(w_80), w_95_list.append(w_95)
        i += 1

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
    print('Estimated Standard deviation:', np.mean(np.mean(forecast_std_list, axis=0)))
    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(c_80_list, axis=0)),
          ', width:', np.mean(np.mean(np.array(w_80_list), axis=0)),
          # '\n Forecast horizon:', np.mean(np.array(c_80_list), axis=0)
          )
    print('95%-prediction interval coverage - Mean:',  np.mean(np.mean(c_95_list, axis=0)),
          ', width:', np.mean(np.mean(np.array(w_95_list), axis=0)),
          # '\n Forecast horizon:', np.mean(np.array(c_95_list), axis=0)
          )
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
    time_series_avocado_baseline_pipeline(cfg)
