import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.load_config import get_path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def load_raw_data(file_name=''):
    path = get_path()
    return pd.read_csv(path['project path'] + 'data//' + file_name, header=0, index_col=0)


def load_oslo_temperature(project_path=None):
    if not project_path:
        path = get_path()
        project_path = path['project path']
    df = pd.read_csv(project_path + 'data//data_files//OsloTemperature.csv',
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
    idx = pd.DatetimeIndex(freq="h", start="2018-01-01", periods=26136)
    # df = pd.DataFrame(data_array)
    df = pd.DataFrame(data=data_array.transpose(), index=idx, columns=np.arange(0, 370))

    print(df.shape)
    return df


def load_traffic():
    data_array = np.load('data_files/traffic.npy')
    # idx = pd.DatetimeIndex(freq="h", start="2018-01-01", periods=26136)
    # df = pd.DataFrame(data_array)
    # df = pd.DataFrame(data=data_array.transpose(), index=idx, columns=np.arange(0, 370))

    print(data_array.shape)
    return data_array


def load_avocado(project_path=None):
    if not project_path:
        path = get_path()
        project_path = path['project path']
    df = pd.read_csv(project_path+'data/data_files/avocado.csv', header=0, index_col=0)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.loc[:, ('AveragePrice', 'region', 'type')]
    df = df.pivot_table(index='Date', columns=['region', 'type'], aggfunc='mean')
    df = df.fillna(method='backfill').dropna()
    df.sort_index(inplace=True)
    # df = df.drop(columns=[('AveragePrice', 'TotalUS', 'organic')])
    return df


if __name__ == "__main__":
    data = load_oslo_temperature(project_path="C:\\Users\\mathi\\PycharmProjects\\gan\\")
    data = data[:int(len(data)*0.8)]
    print(data.shape)
    x = np.linspace(1, len(data), len(data)).reshape(-1, 1)
    y = data
    print(x.shape, y.shape)
    reg = LinearRegression().fit(x, y)
    print(reg.coef_)
    print(reg.intercept_)
    plt.plot(x, y)
    plt.plot(x, reg.coef_*x + reg.intercept_)
    plt.show()
    print(reg.coef_*x[-1]-reg.coef_*x[0])
    print(reg.score)
    #scaler = MinMaxScaler((0, 1))
    #print(data[('AveragePrice', 'TotalUS', 'organic')].values.reshape(-1, 1))
    #plt.plot(data[('AveragePrice', 'TotalUS', 'organic')])
    #plt.show()