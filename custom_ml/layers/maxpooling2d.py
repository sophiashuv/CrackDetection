import numpy as np


class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, x):
        self.input = x
        self.output = np.zeros((
            x.shape[0],
            x.shape[1],
            x.shape[2] // self.pool_size,
            x.shape[3] // self.pool_size
        ))
        self.max_indices = {}
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                for i in range(self.output.shape[2]):
                    for j in range(self.output.shape[3]):
                        patch = x[b, c, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                        self.output[b, c, i, j] = np.max(patch)
                        self.max_indices[(b, c, i, j)] = np.unravel_index(np.argmax(patch), patch.shape)
        return self.output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        for b in range(self.output.shape[0]):
            for c in range(self.output.shape[1]):
                for i in range(self.output.shape[2]):
                    for j in range(self.output.shape[3]):
                        max_index = self.max_indices[(b, c, i, j)]
                        grad_input[b, c, i*self.pool_size + max_index[0], j*self.pool_size + max_index[1]] = \
                            grad_output[b, c, i, j]
        return grad_input
