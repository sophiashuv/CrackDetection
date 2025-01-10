import argparse
import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from layers.conv import ConvLayer
from layers.upsample import Upsample
from layers.maxpooling import MaxPoolLayer

from activations.relu import ReLU
from activations.sigmoid import Sigmoid

from losses.bce import BCELoss


def accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.float32)
    return np.mean(y_true == y_pred)


def load_data(base_folder, image_size, samples_amount=200):
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

    return np.array(inputs)[:samples_amount], np.array(masks)[:samples_amount]


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


class CrackSegmentationModel:
    def __init__(self):
        self.conv1 = ConvLayer(input_channels=3, output_channels=16, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(pool_size=2)

        self.conv2 = ConvLayer(input_channels=16, output_channels=32, kernel_size=3)
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(pool_size=2)

        self.conv3 = ConvLayer(input_channels=32, output_channels=64, kernel_size=3)
        self.relu3 = ReLU()

        self.upsample1 = Upsample(scale_factor=2, mode='bilinear')
        self.deconv1 = ConvLayer(input_channels=64, output_channels=32, kernel_size=3)
        self.relu4 = ReLU()

        self.upsample2 = Upsample(scale_factor=2, mode='bilinear')
        self.deconv2 = ConvLayer(input_channels=32, output_channels=16, kernel_size=3)
        self.relu5 = ReLU()

        self.final_conv = ConvLayer(input_channels=16, output_channels=1, kernel_size=3)
        self.sigmoid = Sigmoid()

    def forward(self, X):
        X = self.conv1.forward(X)
        X = self.relu1.forward(X)
        X = self.pool1.forward(X)

        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.pool2.forward(X)

        X = self.conv3.forward(X)
        X = self.relu3.forward(X)

        X = self.upsample1.forward(X)
        X = self.deconv1.forward(X)
        X = self.relu4.forward(X)

        X = self.upsample2.forward(X)
        X = self.deconv2.forward(X)
        X = self.relu5.forward(X)

        X = self.final_conv.forward(X)
        X = self.sigmoid.forward(X)
        return X

    def backward(self, grad_output, learning_rate=0.01):
        grad_output = self.sigmoid.backward(grad_output)
        grad_output = self.final_conv.backward(grad_output, learning_rate)

        grad_output = self.relu5.backward(grad_output)
        grad_output = self.deconv2.backward(grad_output, learning_rate)
        grad_output = self.upsample2.backward(grad_output)

        grad_output = self.relu4.backward(grad_output)
        grad_output = self.deconv1.backward(grad_output, learning_rate)
        grad_output = self.upsample1.backward(grad_output)

        grad_output = self.relu3.backward(grad_output)
        grad_output = self.conv3.backward(grad_output, learning_rate)

        grad_output = self.pool2.backward(grad_output)
        grad_output = self.relu2.backward(grad_output)
        grad_output = self.conv2.backward(grad_output, learning_rate)

        grad_output = self.pool1.backward(grad_output)
        grad_output = self.relu1.backward(grad_output)
        grad_output = self.conv1.backward(grad_output, learning_rate)

        return grad_output


class BinarySegmentationTrainer:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, output_folder):
        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []

        loss_fn = BCELoss()

        for epoch in range(epochs):
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]

            epoch_loss = 0
            epoch_acc = 0
            total_samples = 0

            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                predictions = self.model.forward(X_batch)
                loss = loss_fn.forward(predictions, y_batch)
                print(f"Training Images: ({batch_start} - {batch_end})/{len(X_train)}, loss: {loss}.")
                epoch_loss += loss * len(X_batch)

                accuracy_val = accuracy(y_batch, predictions)
                epoch_acc += accuracy_val * len(X_batch)

                grad_output = loss_fn.backward()
                self.model.backward(grad_output, learning_rate)
                total_samples += len(X_batch)

            training_losses.append(epoch_loss / total_samples)
            training_accuracies.append(epoch_acc / total_samples)

            val_predictions = self.model.forward(X_val)
            val_loss = loss_fn.forward(val_predictions, y_val)
            val_acc = accuracy(y_val, val_predictions)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"\tTraining Loss: {training_losses[-1]:.4f}, Training Accuracy: {training_accuracies[-1]:.4f}")
            print(f"\tValidation Loss: {validation_losses[-1]:.4f}, Validation Accuracy: {validation_accuracies[-1]:.4f}")
            self.evaluate(X_val, y_val, output_folder, seed=42, epoch=epoch + 1)

        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, training_losses, label="Training Loss")
        plt.plot(epochs_range, validation_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, training_accuracies, label="Training Accuracy")
        plt.plot(epochs_range, validation_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()

        plot_path = os.path.join(output_folder, "training_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Training plot saved at '{plot_path}'")
        self.evaluate(X_val, y_val, output_folder, seed=42)

        return training_losses, validation_losses, training_accuracies, validation_accuracies

    def evaluate(self, X_test, y_test, output_folder, seed=42, epoch=None):
        print("Evaluating model...")
        loss_fn = BCELoss()
        predictions = self.model.forward(X_test)
        test_loss = loss_fn.forward(predictions, y_test)
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
            plt.imshow(X_test[idx].transpose(1, 2, 0))
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


def train_handler(args):
    os.makedirs(args.output_folder, exist_ok=True)
    X, Y = load_data(args.input_folder, tuple(args.image_size))
    X = np.transpose(X, (0, 3, 1, 2))
    Y = np.transpose(Y, (0, 3, 1, 2))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y)

    model = CrackSegmentationModel()
    trainer = BinarySegmentationTrainer(model, input_shape=tuple(args.image_size) + (3,))

    trainer.train(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, output_folder=args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a segmentation model.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder with subfolders: crack, "
                                                                        "nocrack.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[64, 64], help="Size of input images (height, "
                                                                                  "width).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save outputs, including models and "
                                                                         "visualizations.")

    parser.set_defaults(func=train_handler)

    args = parser.parse_args()
    args.func(args)
