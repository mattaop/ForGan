import yaml


def load_config_file(file, print_config=False):
    with open(file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    if print_config:
        print(cfg)
    return cfg
