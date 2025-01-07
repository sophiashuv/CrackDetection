from tensorflow.keras import layers, models

class UNETModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        inputs = layers.Input(shape=self.input_shape)

        # Encoder with additional convolutional layers
        c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
        c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
        c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
        c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(c5)

        # Decoder with dropout layers
        u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
        u4 = layers.concatenate([u4, c4])
        c6 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(u4)
        c6 = layers.Dropout(0.3)(c6)
        c6 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c6)

        u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
        u3 = layers.concatenate([u3, c3])
        c7 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u3)
        c7 = layers.Dropout(0.3)(c7)
        c7 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c7)

        u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
        u2 = layers.concatenate([u2, c2])
        c8 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2)
        c8 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c8)

        u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
        u1 = layers.concatenate([u1, c1])
        c9 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u1)
        c9 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c9)

        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

        model = models.Model(inputs, outputs)
        return model
