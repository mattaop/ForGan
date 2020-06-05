import yaml


def load_config_file(file_name, print_config=False):
    with open(file_name, 'r') as f:
        cfg = yaml.load(f)
    if print_config:
        print(cfg)
    if cfg['model_name'].lower() in ['es', 'arima']:
        cfg['results_path'] = "results/" + cfg['data_source'].lower() + "/" + cfg['model_name'].lower()

    else:
        cfg['results_path'] = "results/" + cfg['data_source'].lower() + "/" + cfg['model_name'].lower() + "/" \
                            + cfg['scaler'].lower() + "/" + cfg['layers'].lower() + \
                            "_epochs_%d_D_epochs_%d_batch_size_%d_noise_vec_%d_gnodes_%d_dnodes_%d_loss_%s_lr_%f" % \
                            (cfg['epochs'], cfg['discriminator_epochs'], cfg['batch_size'], cfg['noise_vector_size'],
                             cfg['generator_nodes'], cfg['discriminator_nodes'], cfg['loss_function'].lower(),
                             cfg['learning_rate'])
    if cfg['data_source'].lower() in ['sine', 'oslo']:
        cfg['seasonality'] = 12
    elif cfg['data_source'].lower() in ['avocado']:
        cfg['seasonality'] = 52
    elif cfg['data_source'].lower() in ['electricity']:
        cfg['seasonality'] = 365
    return cfg


def write_config_file(file_name, cfg):
    with open(file_name, 'w') as f:
        yaml.dump(cfg, f)


def get_path():
    with open('config\\paths.yml', 'r') as ymlfile:
        paths = yaml.load(ymlfile)
    return paths
