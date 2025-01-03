import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.weights, x) + self.bias
        return self.output

    def backward(self, grad_output, learning_rate):
        grad_weights = np.dot(grad_output, self.input.T)
        grad_bias = np.sum(grad_output, axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_output)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input
