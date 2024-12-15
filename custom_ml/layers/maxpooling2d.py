from layers.utils import *


class MaxPooling2D:
    def __init__(self, pool_size=(2, 2), stride=2, padding='valid'):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def _pad_input(self, X):
        if self.padding == 'same':
            pad_h = ((self.pool_size[0] - 1) // 2, (self.pool_size[0] - 1) // 2)
            pad_w = ((self.pool_size[1] - 1) // 2, (self.pool_size[1] - 1) // 2)
            return np.pad(X, ((0, 0), pad_h, pad_w, (0, 0)), mode='constant')
        return X

    def forward(self, X):
        self.X = pad_X(X, self.padding, self.pool_size, self.stride)
        batch_size, in_height, in_width, in_channels = self.X.shape

        out_height = (in_height - self.pool_size[0]) // self.stride + 1
        out_width = (in_width - self.pool_size[1]) // self.stride + 1

        self.output = np.zeros((batch_size, out_height, out_width, in_channels))
        self.max_indices = np.zeros_like(self.X, dtype=bool)

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size[1]

                        region = self.X[b, h_start:h_end, w_start:w_end, c]
                        max_val = np.max(region)
                        self.output[b, i, j, c] = max_val
                        self.max_indices[b, h_start:h_end, w_start:w_end, c] = (region == max_val)

        return self.output

    def backward(self, d_out, learning_rate):
        dX = np.zeros_like(self.X)

        batch_size, out_height, out_width, in_channels = d_out.shape

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size[1]

                        dX[b, h_start:h_end, w_start:w_end, c] += (
                            d_out[b, i, j, c] * self.max_indices[b, h_start:h_end, w_start:w_end, c] * learning_rate
                        )

        return dX