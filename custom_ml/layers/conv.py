import numpy as np

class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding='same', use_batch_norm=False):
        self.kernel = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * np.sqrt(
            2.0 / (input_channels * kernel_size * kernel_size))
        self.bias = np.zeros((output_channels, 1))
        self.stride = stride
        self.padding = padding
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.gamma = np.ones((output_channels, 1))
            self.beta = np.zeros((output_channels, 1))
            self.running_mean = np.zeros((output_channels, 1))
            self.running_variance = np.ones((output_channels, 1))

    def forward(self, x):
        pad_h = (self.kernel.shape[2] - 1) // 2 if self.padding == 'same' else 0
        pad_w = (self.kernel.shape[3] - 1) // 2 if self.padding == 'same' else 0
        self.input = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        output_height = (self.input.shape[2] - self.kernel.shape[2]) // self.stride + 1
        output_width = (self.input.shape[3] - self.kernel.shape[3]) // self.stride + 1
        self.output = np.zeros((self.input.shape[0], self.kernel.shape[0], output_height, output_width))

        for b in range(self.input.shape[0]):
            for c in range(self.kernel.shape[0]):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel.shape[2]
                        end_j = start_j + self.kernel.shape[3]
                        self.output[b, c, i, j] = np.sum(
                            self.input[b, :, start_i:end_i, start_j:end_j] * self.kernel[c]
                        ) + self.bias[c]

        if self.use_batch_norm:
            self.mean = np.mean(self.output, axis=(0, 2, 3), keepdims=True)
            self.variance = np.var(self.output, axis=(0, 2, 3), keepdims=True)
            self.normalized_output = (self.output - self.mean) / np.sqrt(self.variance + 1e-7)
            self.output = self.gamma * self.normalized_output + self.beta

        return self.output

    def backward(self, grad_output, learning_rate):
        pad_h = (self.kernel.shape[2] - 1) // 2 if self.padding == 'same' else 0
        pad_w = (self.kernel.shape[3] - 1) // 2 if self.padding == 'same' else 0

        grad_input = np.zeros_like(self.input)
        grad_kernel = np.zeros_like(self.kernel)
        grad_bias = np.zeros_like(self.bias)

        if self.use_batch_norm:
            grad_gamma = np.sum(grad_output * self.normalized_output, axis=(0, 2, 3), keepdims=True)
            grad_beta = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)
            grad_output = grad_output * self.gamma / np.sqrt(self.variance + 1e-7)
            grad_output -= (np.sum(grad_output, axis=(0, 2, 3), keepdims=True) +
                            self.normalized_output * np.sum(grad_output * self.normalized_output, axis=(0, 2, 3),
                                                            keepdims=True)) / (self.output.shape[2] * self.output.shape[3])

        for b in range(self.input.shape[0]):
            for c in range(self.kernel.shape[0]):
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel.shape[2]
                        end_j = start_j + self.kernel.shape[3]
                        grad_input[b, :, start_i:end_i, start_j:end_j] += self.kernel[c] * grad_output[b, c, i, j]
                        grad_kernel[c] += self.input[b, :, start_i:end_i, start_j:end_j] * grad_output[b, c, i, j]
                grad_bias[c] += np.sum(grad_output[b, c])

        grad_input = grad_input[:, :, pad_h:-pad_h or None, pad_w:-pad_w or None]

        self.kernel -= learning_rate * grad_kernel
        self.bias -= learning_rate * grad_bias
        if self.use_batch_norm:
            self.gamma -= learning_rate * grad_gamma
            self.beta -= learning_rate * grad_beta

        return grad_input
