import argparse
import datetime
import json
import os
import shutil
import glob
from zipfile import ZipFile

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model

from CNN import CNNModel
from UNET import UNETModel


def augment_data(X_train, y_train, augment_params, save_vis=False):
    augmented_images = []
    augmented_masks = []

    transform = A.Compose([
        A.HorizontalFlip(p=augment_params.get("horizontal_flip", 0.5)),
        A.ShiftScaleRotate(
            shift_limit=augment_params.get("shift_limit", 0.05),
            scale_limit=augment_params.get("scale_limit", 0.1),
            rotate_limit=augment_params.get("rotate_limit", 50),
            p=augment_params.get("shift_scale_rotate_p", 0.5)
        )
    ])

    num_augmentations = len(X_train) // 2
    for _ in range(num_augmentations):
        for img, mask in zip(X_train, y_train):
            augmented = transform(image=img, mask=mask)
            augmented_images.append(augmented["image"])
            augmented_masks.append(augmented["mask"])
            if len(augmented_images) >= num_augmentations:
                break
        if len(augmented_images) >= num_augmentations:
            break
    if save_vis:
        augmented_images_ = (np.array(augmented_images) * 255).astype(np.uint8)
        augmented_masks_ = (np.array(augmented_masks) * 255).astype(np.uint8)
        original_images_ = (X_train * 255).astype(np.uint8)
        original_masks_ = (y_train * 255).astype(np.uint8)

        for i in range(min(10, len(augmented_images_))):
            plt.figure(figsize=(15, 7))

            plt.subplot(2, 2, 1)
            plt.imshow(original_images_[i])
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(original_masks_[i].squeeze(), cmap="gray")
            plt.title("Original Mask")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(augmented_images_[i])
            plt.title("Augmented Image")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(augmented_masks_[i].squeeze(), cmap="gray")
            plt.title("Augmented Mask")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"/workspace/augmentation_comparison_{i + 1}.png")
            plt.close()

    return np.array(augmented_images), np.array(augmented_masks)


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
            mask_image = (mask_image > 0.5).astype(np.uint8)
            inputs.append(input_image)
            masks.append(mask_image)

    return np.array(inputs), np.array(masks)


def split_data(inputs, masks):
    X_train, X_temp, y_train, y_temp = train_test_split(inputs, masks, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(input_shape, model_name):
    if model_name == "CNN":
        model_class = CNNModel(input_shape)
    elif model_name == "UNET":
        model_class = UNETModel(input_shape)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model_class.build()


def train_model(model, X_train, y_train, X_val, y_val, args, zip_path, hyperparams, model_name):
    model.compile(optimizer=args.optimizer, loss=args.loss, metrics=[args.metrics])

    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_save_path = os.path.join(
        zip_path, f"{model_name}_epoch-{{epoch:02d}}_val_loss-{{val_loss:.4f}}.h5"
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=False,
        monitor="val_loss",
        mode="min"
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[checkpoint, tensorboard_callback]
    )

    print(f"TensorBoard logs are stored in: {log_dir}")
    min_val_loss = float("inf")
    best_model_path = None
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_model_path = os.path.join(
            zip_path, f"{model_name}_epoch-{epoch + 1:02d}_val_loss-*.h5"
        )
        for file in sorted(glob.glob(epoch_model_path)):
            val_loss = float(file.split("val_loss-")[-1].split(".h5")[0])
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model_path = file
                best_epoch = epoch + 1
    for file in glob.glob(os.path.join(zip_path, f"{model_name}_epoch-*_val_loss-*.h5")):
        if file != best_model_path:
            os.remove(file)
    print(f"Retained only the best model: {best_model_path}")

    print(f"Loading best model from {best_model_path} with validation loss {min_val_loss}")
    best_model = load_model(best_model_path)

    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_train_accuracy = history.history.get("accuracy", [None])[-1]
    final_val_accuracy = history.history.get("val_accuracy", [None])[-1]

    hyperparams["training"] = {
        "optimizer": args.optimizer,
        "loss": args.loss,
        "metrics": [args.metrics],
        "batch_size": args.batch_size,
        "total_epochs": args.epochs,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_accuracy": final_train_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "best_model_path": best_model_path
    }
    hyperparams["time"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return best_model, history


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)


def evaluate_model(model, X_test, y_test, args, zip_path, hyperparams, seed=42):
    print("Evaluating model...")

    results = model.evaluate(X_test, y_test, batch_size=args.batch_size)
    iou_metric = MeanIoU(num_classes=2)
    y_pred = model.predict(X_test, batch_size=args.batch_size)
    iou_metric.update_state(y_test, y_pred > 0.5)
    iou_score = iou_metric.result().numpy()

    thresholds = np.arange(0.1, 1.0, 0.1)
    best_threshold = 0.5
    best_dice = 0
    threshold_results = {}

    for threshold in thresholds:
        y_pred_thresh = (y_pred > threshold).astype(np.uint8)
        current_dice = dice_coefficient(y_test.flatten(), y_pred_thresh.flatten())
        threshold_results[threshold] = {
            "Dice Coefficient": current_dice
        }
        if current_dice > best_dice:
            best_dice = current_dice
            best_threshold = threshold
    y_pred_binary = (y_pred > best_threshold).astype(np.uint8)
    precision = precision_score(y_test.flatten(), y_pred_binary.flatten())
    recall = recall_score(y_test.flatten(), y_pred_binary.flatten())
    f1 = f1_score(y_test.flatten(), y_pred_binary.flatten())
    dice = dice_coefficient(y_test.flatten(), y_pred_binary.flatten())

    visualization_folder = os.path.join(zip_path, "visualization")
    os.makedirs(visualization_folder, exist_ok=True)

    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    fpr, tpr, _ = roc_curve(y_test_flat, y_pred_flat)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    roc_plot_path = os.path.join(visualization_folder, "roc_curve.png")
    plt.savefig(roc_plot_path)
    plt.close()

    precision_vals, recall_vals, _ = precision_recall_curve(y_test_flat, y_pred_flat)
    pr_auc = auc(recall_vals, precision_vals)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f"PR curve (area = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_plot_path = os.path.join(visualization_folder, "precision_recall_curve.png")
    plt.savefig(pr_plot_path)
    plt.close()

    np.random.seed(seed)
    random_indices = np.random.choice(len(X_test), size=20, replace=False)

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

    hyperparams["test"] = {
        "batch_size": args.batch_size,
        "loss": results[0],
        "accuracy": results[1],
        "mean_iou": float(iou_score),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "dice_coefficient": float(dice),
        "best_threshold": float(best_threshold),
        "threshold_investigation": threshold_results,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


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
    os.makedirs(args.output_dir, exist_ok=True)
    zip_path = os.path.join(args.output_dir, "model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(zip_path, exist_ok=True)

    hyperparams = {}
    print("Loading data...")
    inputs, masks = load_data(args.input_folder, tuple(args.image_size))

    print(f"Data loaded. Input shape: {inputs.shape}, Mask shape: {masks.shape}")
    hyperparams["inputs_shape"] = inputs.shape
    hyperparams["masks_shape"] = masks.shape

    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(inputs, masks)
    print(np.unique(y_test))

    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    hyperparams["train_size"] = X_train.shape[0]
    hyperparams["validation_size"] = X_val.shape[0]
    hyperparams["test_size"] = X_test.shape[0]

    print("Applying data augmentation...")
    augment_params = {
        "horizontal_flip": 0.5,
        "shift_limit": 0.05,
        "scale_limit": 0.1,
        "rotate_limit": 50,
        "shift_scale_rotate_p": 0.5
    }

    X_aug, y_aug = augment_data(X_train, y_train, augment_params)
    X_train = np.concatenate((X_train, X_aug))
    y_train = np.concatenate((y_train, y_aug))

    print(f"Data after augmentation: Train shape: {X_train.shape}, Mask shape: {y_train.shape}")
    hyperparams["augmentation"] = augment_params
    hyperparams["train_size_after_augmentation"] = X_train.shape[0]

    print("Creating model...")
    model = create_model((args.image_size[0], args.image_size[1], 3), model_name=args.model_name)
    print(model.summary())

    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val, args, zip_path, hyperparams, model_name=args.model_name)

    print("Training completed.")

    print("Evaluating model on test data...")
    evaluate_model(model, X_test, y_test, args, zip_path, hyperparams)

    file_path = os.path.join(zip_path, "hyperparams.json")
    with open(file_path, "w") as json_file:
        json.dump(hyperparams, json_file, indent=4)
    print(f"Hyperparameters saved to {file_path}")

    create_zip_and_cleanup(zip_path, args.output_dir, "model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


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
    parser.add_argument("--model_name", type=str, default="CNN", help="Name of the model to use.")

    args = parser.parse_args()
    main(args)
