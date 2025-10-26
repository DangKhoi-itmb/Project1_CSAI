import numpy as np

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

x = np.array([1.0, 2.0, 3.0])
print("Rastrigin(1,2,3) =", rastrigin(x))
