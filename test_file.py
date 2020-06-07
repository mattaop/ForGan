import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


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


def relu_fucntion(x):
    return np.max(0, x)

a = 5
x = np.linspace(-a, a, 2*a*100)

relu = np.zeros(len(x))
for i in range(len(x)):
    relu[i] = np.exp(x[i])/(1 + np.exp(x[i]))

fig, ax = plt.subplots()
ax.plot(x, relu)
# ax.set_aspect('equal')
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.show()


real_samples = np.random.normal(0, 0.1, 5000)
sns.kdeplot(real_samples.flatten(), color='blue', alpha=0.6, shade=True)
plt.xlabel('Sample value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Gaussian Distributed Samples', fontsize=14)
plt.legend()
plt.show()

np.random.seed(0)
x = np.linspace(1, num_points, num_points)
mean = 0
std = 0.1
noise = np.random.normal(mean, std, num_points)

y = np.sin(x * np.pi / 6) + noise