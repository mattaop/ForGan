import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml


from config.load_config import load_config_file


def read_files(file):
    with open(file+'/config.yml', 'r') as f:
        cfg = yaml.load(f)
    if cfg['data_source'].lower() in ['avocado', 'electricity']:
        df = pd.read_csv(file + '/test_results_mean.txt', header=0)
    else:
        df = pd.read_csv(file + '/validation_results.txt', header=0)
    return df, cfg


def print_results(folder):
    print("Layer | Epochs | Loss | D_epochs | Noise vector | Batch Size | G nodes | D nodes | MSE | MASE | 80c | 95c | ")
    sub_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
    for sub_folder in sub_folders:
        try:
            df, cfg = read_files(sub_folder)
            print(cfg['layers'] + "  |  %d  | " % (cfg['epochs']) + cfg['loss_function'] +  " |    %d    |     %d     |     %d     |    %d    |    %d   | %.4f |%.4f| %.2f| %.2f| "
                  %  (cfg['discriminator_epochs'], cfg['noise_vector_size'], cfg['batch_size'], cfg['generator_nodes'],
                     cfg['discriminator_nodes'], np.mean(df['mse'])*1, np.mean(df['mase'])*1,
                     np.mean(df['coverage_80'])*100, np.mean(df['coverage_95'])*100))
        except:
            pass


def main():
    folder = 'results\\electricity\\recurrentgan\\minmax'
    print_results(folder=folder)


if __name__ == '__main__':
    main()
