# import tensorflow as tf
# from tensorflow.keras.layers import Conv2DTranspose
#
#
# class Conv2DTransposeWrapper:
#     def __init__(self, filters, kernel_size, strides, padding, activation):
#         self.layer = Conv2DTranspose(filters=filters,
#                                      kernel_size=kernel_size,
#                                      strides=strides,
#                                      padding=padding,
#                                      activation=activation,
#                                      kernel_initializer=tf.keras.initializers.HeNormal())
#
#     def forward(self, input):
#         self.input = tf.convert_to_tensor(input)
#         self.output = self.layer(self.input).numpy()
#         return self.output
#
#     def backward(self, d_output, learning_rate):
#         d_output_tensor = tf.convert_to_tensor(d_output)
#         with tf.GradientTape() as tape:
#             tape.watch(self.input)
#             output = self.layer(self.input)
#         grads = tape.gradient(output, [self.input, self.layer.weights])
#
#         weight_grad, bias_grad = grads[1]
#         self.layer.weights[0].assign_sub(learning_rate * weight_grad)
#         self.layer.weights[1].assign_sub(learning_rate * bias_grad)
#
#         return grads[0].numpy()
from layers.utils import *
import cv2
class Upsample:
    def __init__(self, scale_factor=2, mode='bilinear'):
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor
        self.output = np.zeros((batch_size, new_height, new_width, channels))

        for b in range(batch_size):
            for c in range(channels):
                self.output[b, :, :, c] = cv2.resize(X[b, :, :, c], (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return self.output

    def backward(self, d_out, learning_rate=None):
        batch_size, height, width, channels = d_out.shape
        original_height = height // self.scale_factor
        original_width = width // self.scale_factor
        dX = np.zeros((batch_size, original_height, original_width, channels))

        for b in range(batch_size):
            for c in range(channels):
                dX[b, :, :, c] = cv2.resize(d_out[b, :, :, c], (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        return dX
