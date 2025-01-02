import numpy as np
#
# def relu(x):
#     return np.maximum(0, x)
#
# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def sigmoid_derivative(x):
#     sig = sigmoid(x)
#     return sig * (1 - sig)
#
# class Conv2D:
#     def __init__(self, filters, kernel_size, input_shape, stride=1, padding='same', activation=None):
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.input_shape = input_shape
#         self.stride = stride
#         self.padding = padding
#         self.activation = activation
#
#         self.kernels = np.random.randn(filters, kernel_size[0], kernel_size[1], input_shape[2]) * 0.1
#         self.biases = np.zeros((filters, 1))
#
#         if activation == 'relu':
#             self.activation_function = relu
#             self.activation_derivative = relu_derivative
#         elif activation == 'sigmoid':
#             self.activation_function = sigmoid
#             self.activation_derivative = sigmoid_derivative
#         else:
#             self.activation_function = lambda x: x
#             self.activation_derivative = lambda x: np.ones_like(x)
#
#     def _pad_input(self, X):
#         if self.padding == 'same':
#             pad_h = (self.kernel_size[0] - 1) // 2
#             pad_w = (self.kernel_size[1] - 1) // 2
#             pad_h_extra = (self.kernel_size[0] - 1) % 2
#             pad_w_extra = (self.kernel_size[1] - 1) % 2
#             return np.pad(X, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra), (0, 0)),
#                           mode='constant')
#         elif self.padding == 'valid':
#             return X
#         else:
#             raise ValueError("Unsupported padding type. Use 'same' or 'valid'.")
#
#     def forward(self, X):
#         self.input = X
#         X = self._pad_input(X)
#
#         batch_size, input_height, input_width, _ = X.shape
#         if self.padding == 'same':
#             output_height = input_height
#             output_width = input_width
#         else:
#             output_height = (input_height - self.kernel_size[0]) // self.stride + 1
#             output_width = (input_width - self.kernel_size[1]) // self.stride + 1
#
#         self.output = np.zeros((batch_size, output_height, output_width, self.filters))
#
#         for b in range(batch_size):
#             for i in range(output_height):
#                 for j in range(output_width):
#                     for f in range(self.filters):
#                         h_start = i * self.stride
#                         h_end = h_start + self.kernel_size[0]
#                         w_start = j * self.stride
#                         w_end = w_start + self.kernel_size[1]
#
#                         self.output[b, i, j, f] = (
#                                 np.sum(X[b, h_start:h_end, w_start:w_end, :] * self.kernels[f])
#                                 + self.biases[f]
#                         )
#
#         self.output = self.activation_function(self.output)
#         print("Forward Pass: Input Shape:", X.shape)
#         print("Forward Pass: Output Shape:", self.output.shape)
#
#
#         return self.output
#
#     def backward(self, d_output, learning_rate):
#         print("Backward Pass: d_output Shape:", d_output.shape)
#         d_output = d_output * self.activation_derivative(self.output)
#         X = self._pad_input(self.input)
#         batch_size, input_height, input_width, input_depth = X.shape
#         _, output_height, output_width, _ = d_output.shape
#
#         d_kernels = np.zeros_like(self.kernels)
#         d_biases = np.zeros_like(self.biases)
#         d_input = np.zeros_like(X)
#
#         for b in range(batch_size):
#             for i in range(output_height):
#                 for j in range(output_width):
#                     for f in range(self.filters):
#                         h_start = i * self.stride
#                         h_end = h_start + self.kernel_size[0]
#                         w_start = j * self.stride
#                         w_end = w_start + self.kernel_size[1]
#
#                         d_kernels[f] += d_output[b, i, j, f] * X[b, h_start:h_end, w_start:w_end, :]
#                         d_biases[f] += d_output[b, i, j, f]
#                         d_input[b, h_start:h_end, w_start:w_end, :] += d_output[b, i, j, f] * self.kernels[f]
#
#         if self.padding == 'same':
#             pad_h = (self.kernel_size[0] - 1) // 2
#             pad_w = (self.kernel_size[1] - 1) // 2
#             d_input = d_input[:, pad_h:-pad_h, pad_w:-pad_w, :]
#
#         self.kernels -= learning_rate * d_kernels
#         self.biases -= learning_rate * d_biases
#
#         return d_input


class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size):
        # Replace in ConvLayer
        self.kernel = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * np.sqrt(
            2.0 / (input_channels * kernel_size * kernel_size))

        self.bias = np.zeros((output_channels, 1))

    def forward(self, x):
        self.input = x
        self.output = np.zeros((
            x.shape[0],
            self.kernel.shape[0],
            x.shape[2] - self.kernel.shape[2] + 1,
            x.shape[3] - self.kernel.shape[3] + 1
        ))

        for b in range(x.shape[0]):  # batch
            for c in range(self.kernel.shape[0]):  # filters
                for i in range(self.output.shape[2]):  # height
                    for j in range(self.output.shape[3]):  # width
                        self.output[b, c, i, j] = np.sum(
                            x[b, :, i:i+self.kernel.shape[2], j:j+self.kernel.shape[3]] *
                            self.kernel[c]
                        ).item() + self.bias[c].item()
        return self.output


    def backward(self, grad_output, learning_rate):
        grad_input = np.zeros_like(self.input)
        grad_kernel = np.zeros_like(self.kernel)
        grad_bias = np.zeros_like(self.bias)

        for b in range(self.input.shape[0]):
            for c in range(self.kernel.shape[0]):
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        grad_input[b, :, i:i+self.kernel.shape[2], j:j+self.kernel.shape[3]] += self.kernel[c] * grad_output[b, c, i, j]
                        grad_kernel[c] += self.input[b, :, i:i+self.kernel.shape[2], j:j+self.kernel.shape[3]] * grad_output[b, c, i, j]
                grad_bias[c] += np.sum(grad_output[b, c])

        self.kernel -= learning_rate * grad_kernel
        self.bias -= learning_rate * grad_bias
        return grad_input