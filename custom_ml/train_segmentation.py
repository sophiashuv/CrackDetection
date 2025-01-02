import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from layers.conv2d import Conv2D
from layers.maxpooling2d import MaxPooling2D
from layers.conv2dtranspose import Upsample
from layers.utils import *


def binary_crossentropy(y_true, y_pred, epsilon=1e-7):
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

    return np.array(inputs)[:10], np.array(masks)[:10]


def split_data(inputs, masks):
    validation_split = 0.15
    test_split = 0.15

    np.random.seed(seed=42)
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)

    inputs = inputs[indices]
    masks = masks[indices]

    test_index = int(test_split * inputs.shape[0])
    train_index = int((1 - validation_split - test_split) * inputs.shape[0])
    X_test, y_test = inputs[:test_index], masks[:test_index]
    X_train, y_train = inputs[test_index:test_index + train_index], masks[test_index:test_index + train_index]
    X_val, y_val = inputs[test_index + train_index:], masks[test_index + train_index:]
    return X_train, X_val, X_test, y_train, y_val, y_test


# class SimpleModel:
#     def __init__(self, input_shape):
#         self.layers = []
#
#         self.layers.append(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, stride=1, padding='same',
#                                   activation='relu'))
#         self.layers.append(MaxPooling2D(pool_size=(2, 2), stride=2, padding='valid'))
#
#         self.layers.append(
#             Conv2D(filters=64, kernel_size=(3, 3), input_shape=(input_shape[0] // 2, input_shape[1] // 2, 32), stride=1,
#                    padding='same', activation='relu'))
#         self.layers.append(MaxPooling2D(pool_size=(2, 2), stride=2, padding='valid'))
#
#         # Bottleneck
#         self.layers.append(
#             Conv2D(filters=128, kernel_size=(3, 3), input_shape=(input_shape[0] // 4, input_shape[1] // 4, 64),
#                    stride=1, padding='same', activation='relu'))
#         self.layers.append(Conv2DTransposeCustom(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
#         self.layers.append(Conv2DTransposeCustom(filters=32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
#         self.layers.append(
#             Conv2D(filters=1, kernel_size=(1, 1), input_shape=(input_shape[0], input_shape[1], 32), stride=1,
#                    padding='same', activation='sigmoid'))
#
#     def forward(self, X):
#         for layer in self.layers:
#             if callable(layer):
#                 X = layer(X)
#             else:
#                 X = layer.forward(X)
#             # print(f"Layer {layer}: output shape {X.shape}")
#         return X
#
#     def backward(self, d_out, learning_rate=0.01):
#         for layer in reversed(self.layers):
#             print(layer, d_out.shape)
#             if hasattr(layer, 'backward'):
#                 d_out = layer.backward(d_out, learning_rate)

class SimpleModelCustom:
    def __init__(self, input_shape):
        self.layers = []

        # Encoder
        self.layers.append(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, stride=1, padding='same',
                                  activation='relu'))

        self.layers.append(MaxPooling2D(pool_size=(2, 2), stride=2, padding='valid'))

        # Bottleneck
        self.layers.append(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(input_shape[0] // 2, input_shape[1] // 2, 16),
                                  stride=1, padding='same', activation='relu'))

        # Decoder with Bilinear Upsampling
        self.layers.append(Upsample(scale_factor=2, mode='bilinear'))
        self.layers.append(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(input_shape[0], input_shape[1], 32),
                                  stride=1, padding='same', activation='relu'))
        self.layers.append(
            Conv2D(filters=1, kernel_size=(1, 1), input_shape=(input_shape[0], input_shape[1], 16), stride=1,
                   padding='same', activation='sigmoid'))

    def forward(self, X):
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                X = layer.forward(X)
            else:
                X = layer(X)
        return X

    def backward(self, d_out, learning_rate=0.01):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                d_out = layer.backward(d_out, learning_rate)
            print(f"Backward pass through layer {layer}: d_out shape {d_out.shape}")

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
            total_samples = 0

            for i, batch_start in enumerate(range(0, len(X_train), batch_size)):
                batch_end = min(batch_start + batch_size, len(X_train))
                print(f"Training Images: {batch_start} - {batch_end}/{len(X_train)}")
                X_batch, Y_batch = X_train[batch_start:batch_end], y_train[batch_start:batch_end]

                # Forward pass
                predictions = self.model.forward(X_batch)
                print("max pred, min pred: ", np.max(predictions), np.min(predictions))
                print(Y_batch.shape, predictions.shape)
                loss = binary_crossentropy(Y_batch, predictions)

                print(loss)
                acc = accuracy(Y_batch, predictions)
                batch_size = Y_batch.shape[0]

                epoch_loss += loss * batch_size
                epoch_acc += acc * batch_size
                total_samples += batch_size

                # Backward pass
                d_loss = (predictions - Y_batch) / (predictions * (1 - predictions) + 1e-7)
                d_output = d_loss / batch_size
                self.model.backward(d_output, learning_rate)

            # Average the epoch loss and accuracy
            epoch_loss /= total_samples
            epoch_acc /= total_samples

            training_losses.append(epoch_loss)
            training_accuracies.append(epoch_acc)

            # Validation
            val_predictions = self.model.forward(X_val)
            val_loss = binary_crossentropy(y_val, val_predictions)
            val_acc = accuracy(y_val, val_predictions)

            validation_losses.append(val_loss)
            validation_accuracies.append(val_acc)

            # Print metrics for the epoch
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

            # Evaluate and save visualizations for validation
            self.evaluate(X_val, y_val, output_folder, seed=42, epoch=epoch + 1)

        # Plot losses and accuracies
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

    def evaluate(self, X_test, y_test, output_folder, seed=42, epoch=None):
        print("Evaluating model...")

        predictions = self.model.forward(X_test)
        test_loss = binary_crossentropy(y_test, predictions)
        test_accuracy = accuracy(y_test, predictions)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        if epoch is not None:
            visualization_folder = os.path.join(output_folder, f"visualizations{epoch}")
        else:
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
        if epoch is not None:
            hyperparams_path = os.path.join(output_folder, f"hyperparameters{epoch}.json")
        else:
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
    model = SimpleModelCustom(input_shape=(64, 64, 3))

    # Train model
    trainer = BinarySegmentationTrainer(model, input_shape=(64, 64, 3))
    print(X_train.shape, X_train.shape)
    trainer.train(X_train, y_train, X_val, y_val, epochs=15, batch_size=32, learning_rate=0.0001, output_folder=output_folder)

    # Evaluate model
    trainer.evaluate(X_test, y_test, output_folder)
