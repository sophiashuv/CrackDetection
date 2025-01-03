import numpy as np

class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size):
        self.kernel = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * np.sqrt(
            2.0 / (input_channels * kernel_size * kernel_size))
        self.bias = np.zeros((output_channels, 1))

    def forward(self, x):
        pad_h = (self.kernel.shape[2] - 1) // 2
        pad_w = (self.kernel.shape[3] - 1) // 2
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        self.input = x_padded
        self.output = np.zeros((
            x.shape[0],
            self.kernel.shape[0],
            x.shape[2],
            x.shape[3]
        ))

        for b in range(x.shape[0]):
            for c in range(self.kernel.shape[0]):
                for i in range(self.output.shape[2]):
                    for j in range(self.output.shape[3]):
                        self.output[b, c, i, j] = np.sum(
                            x_padded[b, :, i:i + self.kernel.shape[2], j:j + self.kernel.shape[3]] *
                            self.kernel[c]
                        ).item() + self.bias[c].item()
        return self.output

    def backward(self, grad_output, learning_rate):
        pad_h = (self.kernel.shape[2] - 1) // 2
        pad_w = (self.kernel.shape[3] - 1) // 2

        grad_input = np.zeros_like(self.input)
        grad_kernel = np.zeros_like(self.kernel)
        grad_bias = np.zeros_like(self.bias)

        for b in range(self.input.shape[0]):
            for c in range(self.kernel.shape[0]):
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        grad_input[b, :, i:i + self.kernel.shape[2], j:j + self.kernel.shape[3]] += self.kernel[c] * \
                                                                                                    grad_output[
                                                                                                        b, c, i, j]
                        grad_kernel[c] += self.input[b, :, i:i + self.kernel.shape[2], j:j + self.kernel.shape[3]] * \
                                          grad_output[b, c, i, j]
                grad_bias[c] += np.sum(grad_output[b, c])

        grad_input = grad_input[:, :, pad_h:-pad_h or None, pad_w:-pad_w or None]

        self.kernel -= learning_rate * grad_kernel
        self.bias -= learning_rate * grad_bias
        return grad_input
