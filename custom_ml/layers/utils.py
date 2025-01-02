import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_weights(shape, activation):
    if activation == 'relu':
        return np.random.randn(*shape) * np.sqrt(2. / np.prod(shape[:-1]))
    elif activation == 'sigmoid':
        return np.random.randn(*shape) * np.sqrt(1. / np.prod(shape[:-1]))
    else:
        return np.random.randn(*shape) * 0.01
# def pad_X(X, padding, kernel_size, stride):
#     if padding == 'same':
#         pad_h = ((X.shape[1] - 1) * stride + kernel_size[0] - X.shape[1]) // 2
#         pad_w = ((X.shape[2] - 1) * stride + kernel_size[1] - X.shape[2]) // 2
#         return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
#     return X


def pad_X(X, padding, kernel_size, stride):
    if padding == 'same':
        input_height, input_width = X.shape[1], X.shape[2]
        kernel_height, kernel_width = kernel_size

        # Calculate required padding
        out_height = np.ceil(input_height / stride).astype(int)
        out_width = np.ceil(input_width / stride).astype(int)

        pad_height = max((out_height - 1) * stride + kernel_height - input_height, 0)
        pad_width = max((out_width - 1) * stride + kernel_width - input_width, 0)

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding
        return np.pad(X, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    elif padding == 'valid':
        return X
    else:
        raise ValueError("Unsupported padding type")



class OptimizerAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
