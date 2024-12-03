import os
import argparse
import datetime
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.metrics import MeanIoU


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
    X_train, X_temp, y_train, y_temp = train_test_split(inputs, masks, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
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


def train_model(model, X_train, y_train, X_val, y_val, args):
    model.compile(optimizer=args.optimizer, loss=args.loss, metrics=[args.metrics])

    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint = ModelCheckpoint(
        filepath="model_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[checkpoint, tensorboard_callback, early_stopping]
    )
    print(f"TensorBoard logs are stored in: {log_dir}")
    return history


def evaluate_model(model, X_test, y_test, batch_size):
    print("Evaluating model...")

    results = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    iou_metric = MeanIoU(num_classes=2)
    y_pred = model.predict(X_test, batch_size=batch_size)
    iou_metric.update_state(y_test, y_pred)
    iou_score = iou_metric.result().numpy()
    print(f"Mean IoU: {iou_score}")


def main(args):
    print("Loading data...")
    inputs, masks = load_data(args.input_folder, tuple(args.image_size))
    print(f"Data loaded. Input shape: {inputs.shape}, Mask shape: {masks.shape}")

    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(inputs, masks)
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    print("Creating model...")
    model = create_model((args.image_size[0], args.image_size[1], 3))
    print(model.summary())

    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val, args)

    print("Training completed.")

    print("Evaluating model on test data...")
    evaluate_model(model, X_test, y_test, args.batch_size)


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

    args = parser.parse_args()
    main(args)
