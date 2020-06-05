import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


num_points = 1000000
noise = np.random.normal(0, 0.1, num_points)
noise_2 = np.random.normal(0, 0.1, num_points)
print(np.mean(np.abs(noise - noise_2)))
print(np.mean(np.abs(noise)))
x = np.linspace(1, num_points, num_points)
y = np.sin(x*np.pi/6) + noise
predictions = np.sin(x*np.pi/6)
print(y.shape, predictions.shape)

scaler = MinMaxScaler(feature_range=(10 ** (-10), 1))
y = scaler.fit_transform(y.reshape(-1, 1))
predictions = scaler.transform(predictions.reshape(-1, 1))


print(mean_squared_error(y, predictions))

es = pd.read_csv('results/oslo/es' + '/test_results.txt', header=0)
arima = pd.read_csv('results/oslo/arima' + '/test_results.txt', header=0)

print('ES: ', np.mean(es['mse'][:12]) / 5.75, np.mean(es['coverage_80'][:12]),  np.mean(es['coverage_95'][:12]))
print('ARIMA: ', np.mean(arima['mse'][:12]) / 5.75, np.mean(arima['coverage_80'][:12]),  np.mean(arima['coverage_95'][:12]))
