import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from layers.conv2d import Conv2D
from layers.maxpooling2d import MaxPooling2D
from layers.conv2dtranspose import Conv2DTransposeWrapper


def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.float32)
    return np.mean(y_true == y_pred)


def load_data(base_folder, image_size):
    inputs = []
    masks = []

    crack_folder = os.path.join(base_folder, "crack")
    nocrack_folder = os.path.join(base_folder, "nocrack")

    for category_folder in [crack_folder, nocrack_folder]:
        image_folder = os.path.join(category_folder, "images")
        mask_folder = os.path.join(category_folder, "masks")

        image_files = sorted(os.listdir(image_folder))

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            mask_file = base_name + "_mask" + ".png"

            image_path = os.path.join(image_folder, image_file)
            mask_path = os.path.join(mask_folder, mask_file)

            input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if input_image is None or mask_image is None:
                print(f"Error loading image or mask: {image_file}, {mask_file}")
                continue

            input_image = cv2.resize(input_image, image_size) / 255.0
            mask_image = cv2.resize(mask_image, image_size) / 255.0

            mask_image = np.expand_dims(mask_image, axis=-1)

            inputs.append(input_image)
            masks.append(mask_image)

    return np.array(inputs)[:30], np.array(masks)[:30]


def split_data(inputs, masks):
    validation_split = 0.15
    test_split = 0.15

    np.random.seed(seed=42)
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)

    inputs = inputs[indices]
    masks = masks[indices]

    test_index = int(test_split * inputs.shape[0])
    val_index = int((1 - validation_split - test_split) * inputs.shape[0])

    X_test, y_test = inputs[:test_index], masks[:test_index]
    X_val, y_val = inputs[test_index:test_index + val_index], masks[test_index:test_index + val_index]
    X_train, y_train = inputs[test_index + val_index:], masks[test_index + val_index:]

    return X_train, X_val, X_test, y_train, y_val, y_test


class SimpleModel:
    def __init__(self, input_shape):
        self.layers = []

        self.layers.append(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, stride=1, padding='same',
                                  activation='relu'))
        self.layers.append(MaxPooling2D(pool_size=(2, 2), stride=2, padding='valid'))

        self.layers.append(
            Conv2D(filters=64, kernel_size=(3, 3), input_shape=(input_shape[0] // 2, input_shape[1] // 2, 32), stride=1,
                   padding='same', activation='relu'))
        self.layers.append(MaxPooling2D(pool_size=(2, 2), stride=2, padding='valid'))

        # Bottleneck
        self.layers.append(
            Conv2D(filters=128, kernel_size=(3, 3), input_shape=(input_shape[0] // 4, input_shape[1] // 4, 64),
                   stride=1, padding='same', activation='relu'))
        self.layers.append(Conv2DTransposeWrapper(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
        self.layers.append(Conv2DTransposeWrapper(filters=32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
        self.layers.append(
            Conv2D(filters=1, kernel_size=(1, 1), input_shape=(input_shape[0], input_shape[1], 32), stride=1,
                   padding='same', activation='sigmoid'))

    def forward(self, X):
        for layer in self.layers:
            if callable(layer):
                X = layer(X)
            else:
                X = layer.forward(X)
            print(f"Layer {layer}: output shape {X.shape}")
        return X

    def backward(self, d_out, learning_rate=0.01):
        for layer in reversed(self.layers):
            print(layer, d_out.shape)
            if hasattr(layer, 'backward'):

                d_out = layer.backward(d_out, learning_rate)


class BinarySegmentationTrainer:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, output_folder):
        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []

        for epoch in range(epochs):
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]

            epoch_loss = 0
            epoch_acc = 0

            for batch_start in range(0, len(X_train), batch_size):

                batch_end = min(batch_start + batch_size, len(X_train))
                print(f"Training Images: {batch_start} - {batch_end}/{len(X_train)}")
                X_batch, Y_batch = X_train[batch_start:batch_end], y_train[batch_start:batch_end]

                # Forward pass
                predictions = self.model.forward(X_batch)

                # Compute loss
                loss = binary_crossentropy(Y_batch, predictions)
                acc = accuracy(Y_batch, predictions)
                epoch_loss += loss
                epoch_acc += acc

                # Backward pass
                d_loss = (predictions - Y_batch) / (predictions * (1 - predictions) + 1e-7)
                d_output = d_loss / batch_size
                self.model.backward(d_output, learning_rate)

            training_losses.append(epoch_loss / (len(X_train)/batch_size))
            training_accuracies.append(epoch_acc / (len(X_train)/batch_size))

            val_predictions = self.model.forward(X_val)
            val_loss = binary_crossentropy(y_val, val_predictions)
            val_acc = accuracy(y_val, val_predictions)

            validation_losses.append(val_loss)
            validation_accuracies.append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {training_losses[-1]:.4f}, Accuracy: {training_accuracies[-1]:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), training_losses, label="Training Loss")
        plt.plot(range(1, epochs + 1), validation_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig(os.path.join(output_folder, "loss_plot.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), training_accuracies, label="Training Accuracy")
        plt.plot(range(1, epochs + 1), validation_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation Accuracy")
        plt.savefig(os.path.join(output_folder, "accuracy_plot.png"))
        plt.close()

        return training_losses, validation_losses, training_accuracies, validation_accuracies

    def evaluate(self, X_test, y_test, output_folder, seed=42):
        print("Evaluating model...")

        predictions = self.model.forward(X_test)
        test_loss = binary_crossentropy(y_test, predictions)
        test_accuracy = accuracy(y_test, predictions)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        visualization_folder = os.path.join(output_folder, "visualizations")
        os.makedirs(visualization_folder, exist_ok=True)

        np.random.seed(seed)
        random_indices = np.random.choice(len(X_test), size=min(20, len(X_test)), replace=False)

        print("Saving predictions to visualization folder...")
        for i, idx in enumerate(random_indices):
            plt.figure(figsize=(10, 3))

            plt.subplot(1, 3, 1)
            plt.imshow(X_test[idx])
            plt.title("Input")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(y_test[idx].squeeze(), cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(predictions[idx].squeeze(), cmap="viridis")
            plt.title("Prediction Heatmap")
            plt.axis("off")

            plt.tight_layout()
            output_dir = os.path.join(visualization_folder, f"sample_{i + 1}.png")
            plt.savefig(output_dir)
            plt.close()

        print(f"Visualizations saved in '{visualization_folder}' folder.")

        hyperparams = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

        hyperparams_path = os.path.join(output_folder, "hyperparameters.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)

        print(f"Hyperparameters saved at '{hyperparams_path}'")


if __name__ == "__main__":
    # Load data
    base_folder = "/data"
    output_folder = "/workspace/evaluation_output"
    os.makedirs(output_folder, exist_ok=True)
    image_size = (64, 64)
    X, Y = load_data(base_folder, image_size)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y)

    # Define model
    model = SimpleModel(input_shape=(64, 64, 3))

    # Train model
    trainer = BinarySegmentationTrainer(model, input_shape=(64, 64, 3))
    trainer.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=32, learning_rate=0.001, output_folder=output_folder)

    # Evaluate model
    trainer.evaluate(X_test, y_test, output_folder)
