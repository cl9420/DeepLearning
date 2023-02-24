import numpy as np
from matplotlib import pyplot as plt
import torch


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1


def get_function(x):
    return (x**2)-3


def get_tangent(function, x, point):
    h = 0.0001
    grad = (function(point + h) - function(point)) / h
    return grad * (x - point) + function(point)


x = np.arange(0, 3.0, 0.1)
y = get_function(x)
y_tangent = get_tangent(get_function, x=x, point=2)
plt.plot(x, y)
plt.plot(x, y_tangent)
plt.show()


