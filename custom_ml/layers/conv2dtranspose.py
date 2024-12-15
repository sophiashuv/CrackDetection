import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose


class Conv2DTransposeWrapper:
    def __init__(self, filters, kernel_size, strides, padding, activation):
        self.layer = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)

    def forward(self, input):
        self.input = tf.convert_to_tensor(input)
        self.output = self.layer(self.input).numpy()
        return self.output

    def backward(self, d_output, learning_rate):
        d_output_tensor = tf.convert_to_tensor(d_output)
        with tf.GradientTape() as tape:
            tape.watch(self.input)
            output = self.layer(self.input)
        grads = tape.gradient(output, [self.input, self.layer.weights])

        weight_grad, bias_grad = grads[1]
        self.layer.weights[0].assign_sub(learning_rate * weight_grad)
        self.layer.weights[1].assign_sub(learning_rate * bias_grad)

        return grads[0].numpy()
