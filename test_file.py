import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


num_points = 10000000
noise = np.random.normal(0, 0.1, num_points)
x = np.linspace(1, num_points, num_points)
y = np.sin(x*np.pi/6) + noise
predictions = np.sin(x*np.pi/6)
print(y.shape, predictions.shape)

scaler = MinMaxScaler(feature_range=(10 ** (-10), 1))
y = scaler.fit_transform(y.reshape(-1, 1))
predictions = scaler.transform(predictions.reshape(-1, 1))


print(mean_squared_error(y, predictions))
