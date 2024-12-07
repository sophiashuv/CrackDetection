import os
import argparse
import datetime
import json
import cv2
import shutil
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.metrics import MeanIoU
from tqdm import tqdm

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

    return np.array(inputs), np.array(masks)


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


def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)

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
    return models.Model(inputs, outputs)



def train_model_from_scratch(model, X_train, y_train, X_val, y_val, args):
    # Initialize weights and biases in the model
    def initialize_weights(layer):
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.assign(np.random.randn(*layer.kernel.shape) * 0.01)
        if hasattr(layer, 'bias_initializer') and layer.use_bias:
            layer.bias.assign(np.zeros(layer.bias.shape))

    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.Conv2DTranspose)):
            initialize_weights(layer)

    # Define loss function (e.g., Binary Crossentropy)
    def binary_crossentropy(y_true, y_pred):
        epsilon = 1e-7  # Small value to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # Compute gradients
    def compute_gradients(X, y, model, loss_fn):
        # Convert NumPy arrays to TensorFlow tensors
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = model(X, training=True)
            # Calculate the loss using TensorFlow-compatible functions
            loss = loss_fn(y, y_pred)
        # Calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        return gradients, loss

    # Update parameters using gradients
    def update_parameters(model, gradients, learning_rate):
        for param, grad in zip(model.trainable_variables, gradients):
            param.assign_sub(learning_rate * grad)  # Gradient descent step

    # Training loop
    training_loss = []
    validation_loss = []

    for epoch in range(args.epochs):
        epoch_loss = 0
        val_loss = 0

        # Training phase
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train[i:i + args.batch_size]
            y_batch = y_train[i:i + args.batch_size]

            # Compute gradients and update weights
            gradients, loss = compute_gradients(X_batch, y_batch, model, binary_crossentropy)
            update_parameters(model, gradients, 0.01)
            epoch_loss += loss

        # Validation phase
        for i in range(0, len(X_val), args.batch_size):
            X_batch = X_val[i:i + args.batch_size]
            y_batch = y_val[i:i + args.batch_size]

            y_pred = model(X_batch, training=False)

            loss = binary_crossentropy(y_batch, y_pred)
            val_loss += loss

        # Append losses for the epoch
        training_loss.append(epoch_loss / len(X_train))
        validation_loss.append(val_loss / len(X_val))

        print(f"Epoch {epoch + 1}/{args.epochs}, "
              f"Train Loss: {training_loss[-1]:.4f}, "
              f"Val Loss: {validation_loss[-1]:.4f}")

        # Early stopping logic
        patience = 15
        if epoch > patience and validation_loss[-1] > np.min(validation_loss[-patience:]):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save training history
    history = {
        'train_loss': training_loss,
        'val_loss': validation_loss
    }

    return history



def evaluate_model(model, X_test, y_test, args, zip_path, hyperparams, seed=42):
    print("Evaluating model...")

    results = model.evaluate(X_test, y_test, batch_size=args.batch_size)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    iou_metric = MeanIoU(num_classes=2)
    y_pred = model.predict(X_test, batch_size=args.batch_size)
    iou_metric.update_state(y_test, y_pred)
    iou_score = iou_metric.result().numpy()
    print(f"Mean IoU: {iou_score}")

    visualization_folder = os.path.join(zip_path, "visualization")
    os.makedirs(visualization_folder, exist_ok=True)

    np.random.seed(seed)
    random_indices = np.random.choice(len(X_test), size=20, replace=False)

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
        plt.imshow(y_pred[idx].squeeze(), cmap="viridis")
        plt.title("Prediction Heatmap")
        plt.axis("off")

        plt.tight_layout()
        output_dir = os.path.join(visualization_folder, f"sample_{i + 1}.png")
        plt.savefig(output_dir)
        plt.close()

    print(f"Visualizations saved in '{visualization_folder}' folder.")

    hyperparams["test"] = {
        "batch_size": args.batch_size,
        "loss": results[0],
        "accuracy": results[1],
        "mean iou": float(iou_score)
    }

    return visualization_folder


def create_zip_and_cleanup(source_folder, destination_folder, zip_name):
    os.makedirs(destination_folder, exist_ok=True)
    zip_path = os.path.join(destination_folder, f"{zip_name}.zip")
    with ZipFile(zip_path, "w") as zip_file:
        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_folder)  # Preserve folder structure
                zip_file.write(file_path, arcname)

    print(f"Zip archive created at: {zip_path}")
    shutil.rmtree(source_folder)
    print(f"Deleted folder: {source_folder}")
    return zip_path


def main(args):
    # os.makedirs(args.output_dir, exist_ok=True)
    zip_path = os.path.join(args.output_dir, "model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # os.makedirs(zip_path, exist_ok=True)

    hyperparams = {}
    print("Loading data...")
    inputs, masks = load_data(args.input_folder, tuple(args.image_size))

    print(f"Data loaded. Input shape: {inputs.shape}, Mask shape: {masks.shape}")
    hyperparams["inputs_shape"] = inputs.shape
    hyperparams["masks_shape"] = masks.shape

    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(inputs, masks)

    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    hyperparams["train_size"] = X_train.shape[0]
    hyperparams["validation_size"] = X_val.shape[0]
    hyperparams["test_size"] = X_test.shape[0]

    print("Creating model...")
    model = create_model((args.image_size[0], args.image_size[1], 3))
    print(model.summary())

    print("Training model...")
    history = train_model_from_scratch(model, X_train, y_train, X_val, y_val, args)

    print("Training completed.")
    print(history)

    # print("Evaluating model on test data...")
    # evaluate_model(model, X_test, y_test, args, zip_path, hyperparams)
    #
    # file_path = os.path.join(zip_path, "hyperparams.json")
    # with open(file_path, "w") as json_file:
    #     json.dump(hyperparams, json_file, indent=4)
    # print(f"Hyperparameters saved to {file_path}")

    # create_zip_and_cleanup(zip_path, args.output_dir, "model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a segmentation model.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the folder with supfolders: crack, nocrack.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[64, 64],
                        help="Size of input images (height, width).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training.")
    parser.add_argument("--loss", type=str, default="binary_crossentropy", help="Loss function for training.")
    parser.add_argument("--metrics", type=str, default="accuracy", help="Metrics to evaluate during training.")
    parser.add_argument("--output_dir", type=str, help="Where to save models.")

    args = parser.parse_args()
    main(args)
