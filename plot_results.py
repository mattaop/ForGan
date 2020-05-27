import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler

show_plot = True
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200',
#                                                    '898989', 'A2C8EC', 'FFBC79', 'CFCFCF'])


def read_training_files(file_name, data):
    if data == 'training':
        df = pd.read_csv(file_name + '/training_results.txt', header=0)
    elif data == 'validation':
        df = pd.read_csv(file_name + '/validation_results.txt', header=0)
    else:
        df = pd.read_csv(file_name + '/test_results.txt', header=0)
    return df


def plot_results(save_file, model_paths, file_labels, title, x_label, y_label, value='mse', data='training', plot_rate=1):
    # color = plt.cm.viridis(np.linspace(0, 1, len(model_paths)))
    # color = ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79', 'CFCFCF']
    # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    plt.figure()
    #ax = plt.axes()
    # ax.set_prop_cycle('color', [plt.cm.jet(i) for i in np.linspace(0, 1, len(model_paths))])
    for model_paths, file_label in zip(model_paths, file_labels):
        df = read_training_files(model_paths, data)[value]
        t = np.linspace(1, len(df) * plot_rate, len(df))
        plt.plot(t, df, label=file_label)
    if value == 'mse' and data == 'training':
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig('plots/' + save_file)
    if show_plot:
        plt.show()


def plot_training_results(model_paths, file_labels, title, save_file='Train', plot_rate=1, data_set='sine'):
    plot_results(save_file=data_set + '/' + save_file + '_train_mse', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[0],
                 x_label='Training iterations', y_label='Mean Squared Error (MSE)', value='mse', data='training', plot_rate=plot_rate)
    plot_results(save_file=data_set + '/' + save_file + '_train_80', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[1],
                 x_label='Training iterations', y_label='Coverage', value='coverage_80',  data='training', plot_rate=plot_rate)
    plot_results(save_file=data_set + '/' + save_file + '_train_95', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[2],
                 x_label='Training iterations', y_label='Coverage', value='coverage_95', data='training', plot_rate=plot_rate)


def plot_validation_results(model_paths, file_labels, title, save_file='Validation', data_set='sine'):
    plot_results(save_file=data_set + '/' + save_file + '_validation_mse', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[0],
                 x_label='Forecast horizon', y_label='Mean Squared Error (MSE)', value='mse',  data='validation',
                 plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_validation_smape', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[1],
                 x_label='Forecast horizon', y_label='Symmetric Mean Absolute Percentage Error (sMAPE)', value='smape',
                 data='validation', plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_validation_mase', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[2],
                 x_label='Forecast horizon', y_label='Mean Absolute Scaled Error (MASE)', value='mase',
                 data='test', plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_validation_80', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[3],
                 x_label='Forecast horizon', y_label='Coverage', value='coverage_80', data='validation',
                 plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_validation_95', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels, title=title[4],
                 x_label='Forecast horizon', y_label='Coverage', value='coverage_95', data='validation',
                 plot_rate=1)


def plot_test_results(model_paths, file_labels, title, save_file='Test', data_set='sine'):
    plot_results(save_file=data_set + '/' + save_file + '_test_mse', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels,
                 title=title[0], x_label='Forecast horizon', y_label='Mean Squared Error (MSE)', value='mse',
                 data='test',   plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_test_smape', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels,
                 title=title[1], x_label='Forecast horizon', y_label='Symmetric Mean Absolute Percentage Error (sMAPE)',
                 value='smape',
                 data='test', plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_test_mase', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels,
                 title=title[2], x_label='Forecast horizon', y_label='Mean Absolute Scaled Error (MASE)', value='mase',
                 data='test', plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_test_80', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels,
                 title=title[3], x_label='Forecast horizon', y_label='Coverage', value='coverage_80', data='test',
                 plot_rate=1)
    plot_results(save_file=data_set + '/' + save_file + '_test_95', model_paths=['results/' + data_set + '/' + s for s in model_paths], file_labels=file_labels,
                 title=title[4], x_label='Forecast horizon', y_label='Coverage', value='coverage_95', data='test',
                 plot_rate=1)


def main():
    noise_paths = ['recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_1_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_5_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_10_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_25_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_50_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000'
                   ]
    noise_label = ['Noise vector = 1', 'Noise vector = 5', 'Noise vector = 10', 'Noise vector = 25',
                   'Noise vector = 50', 'Noise vector = 100']

    d_epochs_paths = ['recurrentgan/rnn_epochs_3000_D_epochs_1_batch_size_16_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      # 'results/sine/recurrentgan/Epochs_1500_D_epochs_2_batch_size_64_noise_vec_50_lr_0.001000',
                      'recurrentgan/rnn_epochs_3000_D_epochs_3_batch_size_16_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      'recurrentgan/rnn_epochs_3000_D_epochs_5_batch_size_16_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      'recurrentgan/rnn_epochs_3000_D_epochs_10_batch_size_16_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      # 'results/sine/recurrentgan/Epochs_1500_D_epochs_15_batch_size_64_noise_vec_50_lr_0.001000',
                      'recurrentgan/rnn_epochs_3000_D_epochs_20_batch_size_16_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      # 'results/sine/recurrentgan/Epochs_1500_D_epochs_50_batch_size_64_noise_vec_50_lr_0.001000'
                      ]
    d_epochs_label = ['D$_{epochs}=1$',
                      # 'D$_{epochs}=2$',
                      'D$_{epochs}=3$',
                      'D$_{epochs}=5$',
                      'D$_{epochs}=10$',
                      # 'D$_{epochs}=15$',
                      'D$_{epochs}=20$',
                      # 'D$_{epochs}=50$'
                      ]
    batch_size_paths = ['recurrentgan/rnn_epochs_1500_D_epochs_5_batch_size_16_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                        'recurrentgan/rnn_epochs_1500_D_epochs_5_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                        'recurrentgan/rnn_epochs_1500_D_epochs_5_batch_size_64_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                        'recurrentgan/rnn_epochs_1500_D_epochs_5_batch_size_128_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                        'recurrentgan/rnn_epochs_1500_D_epochs_5_batch_size_256_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',]
    batch_size_labels = ['Batch size = 16', 'Batch size = 32', 'Batch size = 64', 'Batch size = 128', 'Batch size = 256']

    data_set = 'sine'
    save_file_as = 'noise_vec'
    model_paths = noise_paths
    labels = noise_label
    plot_training_results(model_paths=model_paths,
                          file_labels=labels,
                          save_file=save_file_as,
                          title=['Training Mean Squared Error',
                                 'Training 80% Prediction Interval Coverage',
                                 'Training 95% Prediction Interval Coverage'],
                          plot_rate=25, data_set=data_set)

    plot_validation_results(model_paths=model_paths,
                            file_labels=labels,
                            save_file=save_file_as,
                            title=['Forecast Validation Mean Squared Error',
                                   'Forecast Validation Symmetric Mean Absolute Percentage Error',
                                   'Forecast Mean Absolute Scaled Error',
                                   'Forecast Validation 80% Prediction Interval Coverage',
                                   'Forecast Validation 95% Prediction Interval Coverage'],
                            data_set=data_set)

    # model_paths = ['arima',
    #                'es',
    #                'rnn/rnn_epochs_1500_D_epochs_1_batch_size_64_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
    #                'recurrentgan/rnn_epochs_3000_D_epochs_20_batch_size_32_noise_vec_100_gnodes_64_dnodes_64_loss_kl_lr_0.001000']
    # labels = ['ARIMA',
    #           'Exponential Smoothing',
    #           'MC Dropout',
    #           'ForGAN']
    plot_test_results(model_paths=model_paths,
                      file_labels=labels,
                      save_file=save_file_as,
                      title=['Forecast Mean Squared Error',
                             'Forecast Symmetric Mean Absolute Percentage Error',
                             'Forecast Mean Absolute Scaled Error',
                             'Forecast 80% Prediction Interval Coverage',
                             'Forecast 95% Prediction Interval Coverage'],
                      data_set=data_set)


if __name__ == '__main__':
    main()
