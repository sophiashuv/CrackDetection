from tensorflow.keras import layers, models

class AdvancedUNETModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def squeeze_excite_block(self, input_tensor, ratio=16):
        filters = input_tensor.shape[-1]
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Dense(filters // ratio, activation="relu")(se)
        se = layers.Dense(filters, activation="sigmoid")(se)
        return layers.multiply([input_tensor, se])

    def multi_scale_fusion(self, x, y, filters):
        x1 = layers.Conv2D(filters, (1, 1), padding="same")(x)
        y1 = layers.Conv2D(filters, (3, 3), padding="same")(y)
        fused = layers.add([x1, y1])
        return layers.Activation("relu")(fused)

    def build(self):
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        c1 = layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        c1 = self.squeeze_excite_block(c1)
        c1 = layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same")(c1)
        c1 = layers.Dropout(0.3)(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(p1)
        c2 = self.squeeze_excite_block(c2)
        c2 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(c2)
        c2 = layers.Dropout(0.3)(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(p2)
        c3 = self.squeeze_excite_block(c3)
        c3 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(c3)
        c3 = layers.Dropout(0.3)(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        # Bottleneck
        c4 = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(p3)
        c4 = self.squeeze_excite_block(c4)
        c4 = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(c4)

        # Decoder
        u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c4)
        fused_u3 = self.multi_scale_fusion(u3, c3, 128)
        u3 = layers.concatenate([fused_u3, c3])
        c5 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(u3)
        c5 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(c5)

        u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
        fused_u2 = self.multi_scale_fusion(u2, c2, 64)
        u2 = layers.concatenate([fused_u2, c2])
        c6 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(u2)
        c6 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(c6)

        u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
        fused_u1 = self.multi_scale_fusion(u1, c1, 32)
        u1 = layers.concatenate([fused_u1, c1])
        c7 = layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same")(u1)
        c7 = layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same")(c7)

        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)

        model = models.Model(inputs, outputs)
        return model

# Example usage
if __name__ == "__main__":
    unet = AdvancedUNETModel(input_shape=(128, 128, 3))
    model = unet.build()
    model.summary()
