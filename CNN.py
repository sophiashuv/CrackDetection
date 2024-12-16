from tensorflow.keras import layers, models

class CNNModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Bottleneck
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

        # Decoder
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)

        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
        model = models.Model(inputs, outputs)
        return model
