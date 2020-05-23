import yaml


def load_config_file(file_name, print_config=False):
    with open(file_name, 'r') as f:
        cfg = yaml.load(f)
    if print_config:
        print(cfg)
    return cfg


def write_config_file(file_name, cfg):
    with open(file_name, 'w') as f:
        yaml.dump(cfg, f)


def get_path():
    with open('config\\paths.yml', 'r') as ymlfile:
        paths = yaml.load(ymlfile)
    return paths
