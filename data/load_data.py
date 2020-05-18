import numpy as np
import pandas as pd
from config.load_config import get_path


def load_raw_data(file_name=''):
    path = get_path()
    return pd.read_csv(path['project path'] + 'data//' + file_name, header=0, index_col=0)


def load_oslo_temperature():
    path = get_path()
    df = pd.read_csv(path['project path'] + 'data//data_files//OsloTemperature.csv',
                     header=0, sep=';', index_col=0)
    df.set_index('time', inplace=True)
    df.drop(columns=['station', 'id', 'max(air_temperature P1M)', 'min(air_temperature P1M)'], inplace=True)
    df.dropna(how='any', inplace=True)
    df.rename(columns={"mean(air_temperature P1M)": "y"}, inplace=True)
    idx = pd.date_range('1937-01-31', freq='M', periods=994)
    for index, row in df.iterrows():
        df.loc[index, 'y'] = df.loc[index, 'y'].replace(',', '.')
    # df = pd.DataFrame(data=df['mean_temperature'].index[1:], index=idx, columns=['mean_temperature'])
    df.set_index(idx, inplace=True)
    df.drop(df.index[0], inplace=True)
    df["y"] = pd.to_numeric(df["y"])
    return df["y"].values.reshape(-1, 1)


def load_australia_temperature():
    path = get_path()
    df = pd.read_csv(path['project path'] + 'data//data_files//daily-min-temperatures.csv')
    df.set_index('Date', inplace=True)
    df.dropna(how='any', inplace=True)
    idx = pd.date_range('1981-01-01', freq='D', periods=3650)
    df.set_index(idx, inplace=True)
    df["y"] = pd.to_numeric(df["Temp"])
    return df["y"].values.reshape(-1, 1)


def load_electricity():
    data_array = np.load('data_files/electricity.npy')
    print(data_array.shape)
    return data_array


if __name__ == "__main__":
    print(load_australia_temperature().shape)
    load_electricity()

