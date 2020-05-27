import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from config.load_config import load_config_file


def read_files(file):
    cfg = load_config_file(file+'/config.yml')
    val = pd.read_csv(file + '/validation_results.txt', header=0)
    test = pd.read_csv(file + '/test_results.txt', header=0)
    return val, test, cfg


def print_results(folder):
    sub_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
    for sub_folder in sub_folders:
        try:
            val, test, cfg = read_files(sub_folder)
            print(val.dtype, test.dtype)
            length = cfg['forecast_horizon']
            mean_val = pd.DataFrame({'mse': np.zeros(length), 'smape': np.zeros(length), 'mase': np.zeros(length),
                                     'coverage_80': np.zeros(length), 'coverage_95': np.zeros(length),
                                     'width_80': np.zeros(length), 'width_95': np.zeros(length)})
            mean_test = pd.DataFrame({'mse': np.zeros(length), 'smape': np.zeros(length), 'mase': np.zeros(length),
                                      'coverage_80': np.zeros(length), 'coverage_95': np.zeros(length),
                                      'width_80': np.zeros(length), 'width_95': np.zeros(length)})
            print(mean_val.columns.values)
            print(len(val)//length)
            print(val.loc[0:length].to_numpy()/(len(val)//length))

            for i in range(len(val)//length):
                for value in mean_val.columns.values:
                    mean_val[value] += val[value].iloc[i:(i+1)*length]/(len(val)//length)
                    print(val)

            for i in range(len(test)//length):
                for value in mean_test.columns.values:
                    mean_test[value] += test[i:(i+1)*length][value]/(len(test)//length)

            print('Val:', mean_val)
            print('Test:', mean_test)

        except:
            pass


def main():
    folder = 'results/oslo/recurrentgan/'
    print_results(folder=folder)


if __name__ == '__main__':
    main()
