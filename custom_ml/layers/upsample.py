import numpy as np
import cv2


class Upsample:
    def __init__(self, scale_factor=2, mode='bilinear'):
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        self.input = x
        batch_size, channels, height, width = x.shape
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor

        if self.mode == 'bilinear':
            self.output = np.zeros((batch_size, channels, new_height, new_width))
            for b in range(batch_size):
                for c in range(channels):
                    self.output[b, c] = cv2.resize(x[b, c], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        elif self.mode == 'nearest':
            self.output = np.zeros((batch_size, channels, new_height, new_width))
            for b in range(batch_size):
                for c in range(channels):
                    self.output[b, c] = cv2.resize(x[b, c], (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return self.output

    def backward(self, grad_output):
        batch_size, channels, height, width = self.input.shape
        grad_input = np.zeros_like(self.input)

        if self.mode == 'bilinear':
            for b in range(batch_size):
                for c in range(channels):
                    grad_input[b, c] = cv2.resize(grad_output[b, c], (width, height), interpolation=cv2.INTER_LINEAR)
        elif self.mode == 'nearest':
            for b in range(batch_size):
                for c in range(channels):
                    grad_input[b, c] = cv2.resize(grad_output[b, c], (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return grad_input
