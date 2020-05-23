import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_training_files(file_name, training):
    if training:
        df = pd.read_csv('training_results/' + file_name, header=0)
    else:
        df = pd.read_csv('test_results/' + file_name, header=0)
    return df


def plot_results(save_file, file_names, file_labels, title, x_label, y_label, value='mse', training=True):
    plt.figure()
    for file_name, file_label in zip(file_names, file_labels):
        df = read_training_files(file_name, training)[value]
        t = np.linspace(1, len(df), len(df))
        plt.plot(t, df, label=file_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=4)
    plt.savefig('plots/'+save_file)
    plt.show()


def plot_training_results(file_names, file_labels, title, save_file='Train'):
    plot_results(save_file=save_file+'_train_mse', file_names=file_names, file_labels=file_labels, title=title[0],
                 x_label='Epochs', y_label='Mean Squared Error (MSE)', value='mse', training=True)
    plot_results(save_file=save_file+'_train_80', file_names=file_names, file_labels=file_labels, title=title[1],
                 x_label='Epochs', y_label='Coverage', value='coverage_80', training=True)
    plot_results(save_file=save_file+'_train_95', file_names=file_names, file_labels=file_labels, title=title[2],
                 x_label='Epochs', y_label='Coverage', value='coverage_95', training=True)


def plot_test_results(file_names, file_labels, title, save_file='Test'):
    plot_results(save_file=save_file+'_test_mse', file_names=file_names, file_labels=file_labels, title=title[0],
                 x_label='Forecast horizon', y_label='Mean Squared Error (MSE)', value='mse', training=False)
    plot_results(save_file=save_file+'_test_smape', file_names=file_names, file_labels=file_labels,  title=title[1],
                 x_label='Forecast horizon', y_label='Symmetric Mean Absolute Percentage Error (sMAPE)', value='smape',
                 training=False)
    plot_results(save_file=save_file+'_test_80', file_names=file_names, file_labels=file_labels,  title=title[2],
                 x_label='Forecast horizon', y_label='Coverage', value='coverage_80', training=False)
    plot_results(save_file=save_file+'_test_95', file_names=file_names, file_labels=file_labels,  title=title[3],
                 x_label='Forecast horizon', y_label='Coverage', value='coverage_95', training=False)


if __name__ == '__main__':
    plot_training_results(['Epochs_10_D_epochs_1_batch_size_64_noise_vec_50.txt',
                           'Epochs_10_D_epochs_2_batch_size_64_noise_vec_50.txt'],
                          ['D$_{epochs}$=1', 'D$_{epochs}$=2'],
                          save_file='sine',
                          title=['Training Mean Squared Error',
                                 'Training 80% Prediction Interval Coverage',
                                 'Training 95% Prediction Interval Coverage'])
    plot_test_results(['Epochs_10_D_epochs_1_batch_size_64_noise_vec_50_learning_rate_0.001000.txt'],
                      ['D$_{epochs}$=1'],
                      save_file='sine',
                      title=['Forecast Mean Squared Error',
                             'Forecast Symmetric Mean Absolute Percentage Error',
                             'Forecast 80% Prediction Interval Coverage',
                             'Forecast 95% Prediction Interval Coverage'])

