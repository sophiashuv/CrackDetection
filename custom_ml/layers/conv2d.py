from layers.utils import *


class Conv2D:
    def __init__(self, filters, kernel_size, input_shape, stride=1, padding='valid', activation=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        fan_in = kernel_size[0] * kernel_size[1] * input_shape[2]
        self.weights = np.random.randn(kernel_size[0], kernel_size[1], input_shape[2], filters) * np.sqrt(2. / fan_in)
        self.biases = np.zeros(filters)
        self.biases = np.zeros(filters)

    def forward(self, X):
        self.X = pad_X(X, self.padding, self.kernel_size, self.stride)
        self.input_shape = X.shape

        out_height = (self.X.shape[1] - self.kernel_size[0]) // self.stride + 1
        out_width = (self.X.shape[2] - self.kernel_size[1]) // self.stride + 1

        self.output = np.zeros((X.shape[0], out_height, out_width, self.filters))

        for batch in range(X.shape[0]):
            for i in range(out_height):
                for j in range(out_width):
                    for f in range(self.filters):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size[1]

                        region = self.X[batch, h_start:h_end, w_start:w_end, :]
                        self.output[batch, i, j, f] = np.sum(region * self.weights[:, :, :, f]) + self.biases[f]

        if self.activation == 'relu':
            self.output = relu(self.output)
        elif self.activation == 'sigmoid':
            self.output = sigmoid(self.output)

        return self.output

    def backward(self, d_output, learning_rate):
        if self.activation == 'relu':
            d_output *= relu_derivative(self.output)
        elif self.activation == 'sigmoid':
            d_output *= sigmoid_derivative(self.output)

        d_X = np.zeros_like(self.X)
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)

        for batch in range(self.input_shape[0]):
            for i in range(d_output.shape[1]):
                for j in range(d_output.shape[2]):
                    for f in range(self.filters):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size[1]

                        region = self.X[batch, h_start:h_end, w_start:w_end, :]

                        d_weights[:, :, :, f] += region * d_output[batch, i, j, f]
                        d_biases[f] += d_output[batch, i, j, f]
                        d_X[batch, h_start:h_end, w_start:w_end, :] += self.weights[:, :, :, f] * d_output[
                            batch, i, j, f]

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_X