from tensorflow.keras import layers, models

class CNNModel:
    def __init__(self, input_shape, depth=3, initial_filters=16, use_residual=False, use_batch_norm=False, use_dropout=False):
        self.input_shape = input_shape
        self.depth = depth
        self.initial_filters = initial_filters
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

    def residual_block(self, x, filters):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.add([shortcut, x])
        x = layers.Activation("relu")(x)
        return x

    def build(self):
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        filters = self.initial_filters

        # Encoder
        for _ in range(self.depth):
            x = layers.Conv2D(filters, (3, 3), padding="same")(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            if self.use_residual:
                x = self.residual_block(x, filters)
            if self.use_dropout:
                x = layers.Dropout(0.3)(x)
            x = layers.MaxPooling2D((2, 2))(x)
            filters *= 2

        # Bottleneck
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)

        # Decoder
        for _ in range(self.depth):
            filters //= 2
            x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            if self.use_dropout:
                x = layers.Dropout(0.3)(x)

        # Output layer
        outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

        model = models.Model(inputs, outputs)
        return model

# Example usage
if __name__ == "__main__":
    cnn = CNNModel(input_shape=(128, 128, 3), depth=4, initial_filters=16, use_residual=True, use_batch_norm=True, use_dropout=True)
    model = cnn.build()
    model.summary()
