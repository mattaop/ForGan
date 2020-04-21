import numpy as np

n = 100000
mse = 0
variables = np.zeros(n)
for i in range(n):
    variables[i] = np.random.normal(0, 0.1)
    mse += 1/n*variables[i]**2
print('MSE', mse)
print('Std', np.std(variables))
print('Var', np.var(variables))