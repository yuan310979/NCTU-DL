import numpy as np
from scipy.special import xlogy

def linear(w, x):
    return np.dot(w, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def mse_loss(y_pred, y):
    return (y_pred - y) ** 2

def derivative_mse_loss(y_pred, y):
    return 1/2 * (y_pred - y)

def cross_entropy_loss(y_pred, y):
    return -xlogy(y, y_pred) - xlogy(1-y, 1-y_pred) 

def derivative_cross_entropy_loss(y_pred, y):
    if y_pred == 0:
        return (1-y) / (1-y_pred)
    elif y_pred == 1:
        return (-y) / y_pred
    else:
        return -y / y_pred + (1-y) / (1-y_pred)
