from tensorflow.keras import layers, models

class CNNModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

        # Decoder
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), activation="relu", padding="same")(x)

        outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
        model = models.Model(inputs, outputs)
        return model
