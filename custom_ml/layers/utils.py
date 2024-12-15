import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def pad_X(X, padding, kernel_size, stride):
    if padding == 'same':
        pad_h = ((X.shape[1] - 1) * stride + kernel_size[0] - X.shape[1]) // 2
        pad_w = ((X.shape[2] - 1) * stride + kernel_size[1] - X.shape[2]) // 2
        return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    return X