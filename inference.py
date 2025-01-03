import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import argparse
from scipy.ndimage import label
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score


def tiles_generator(big_img, tile_size, overlap_size, resize_size):
    height, width, _ = big_img.shape
    tile_width, tile_height = tile_size
    overlap_width, overlap_height = overlap_size
    resize_width, resize_height = resize_size

    effective_stride_width = tile_width - overlap_width
    effective_stride_height = tile_height - overlap_height

    x_starts = []
    y_starts = []

    x = 0
    while x + tile_width <= width:
        x_starts.append(x)
        x += effective_stride_width
    if x_starts[-1] + tile_width < width:
        x_starts.append(width - tile_width)

    y = 0
    while y + tile_height <= height:
        y_starts.append(y)
        y += effective_stride_height
    if y_starts[-1] + tile_height < height:
        y_starts.append(height - tile_height)

    for x in x_starts:
        for y in y_starts:
            tile = big_img[y:y + tile_height, x:x + tile_width]
            if tile.shape[1] != resize_width or tile.shape[0] != resize_height:
                tile = cv2.resize(tile, (resize_width, resize_height))

            yield tile, x, y


def apply_model_to_tile(model, tile):
    tile = np.array([tile]) / 255.0
    prediction = model.predict(tile)
    return prediction[0]


def remove_small_blobs(mask, min_size):
    labeled_array, num_features = label(mask)
    for i in range(1, num_features + 1):
        blob = labeled_array == i
        if np.sum(blob) < min_size:
            mask[blob] = 0
    return mask


def normalize_mask(mask):
    return (mask > 127).astype(np.uint8)


def calculate_metrics(gt_mask, pred_mask):
    gt_mask = normalize_mask(gt_mask)
    pred_mask = normalize_mask(pred_mask)
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    iou = jaccard_score(gt_flat, pred_flat, average='binary')
    dice = f1_score(gt_flat, pred_flat, average='binary')
    return iou, dice


def save_visualization(big_img, segmentation_mask, postprocessed_mask, gt_mask, visualization_folder, filename):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[1].imshow(segmentation_mask, cmap="gray")
    axs[1].set_title("Prediction")
    axs[2].imshow(postprocessed_mask, cmap="gray")
    axs[2].set_title("Postprocessed")
    axs[3].imshow(gt_mask, cmap="gray")
    axs[3].set_title("Ground Truth")
    for ax in axs:
        ax.axis("off")
    vis_path = os.path.join(visualization_folder, f"{os.path.splitext(filename)[0]}_visualization.png")
    plt.savefig(vis_path)
    plt.close()
    print(f"Saved visualization for {filename} to {vis_path}")


def process_images(model, input_folder, output_folder, tile_size, overlap_size, resize_size, threshold, min_blob_size, gt_folder=None, visualization_folder=None):
    os.makedirs(output_folder, exist_ok=True)
    if visualization_folder is not None:
        os.makedirs(visualization_folder, exist_ok=True)

    metrics_before = []
    metrics_after = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            big_img = cv2.imread(img_path)

            if big_img is None:
                print(f"Could not read image {filename}. Skipping.")
                continue

            height, width, _ = big_img.shape
            segmentation_mask = np.zeros((height, width), dtype=np.uint8)

            for tile, x, y in tiles_generator(big_img, tile_size, overlap_size, resize_size):
                prediction = apply_model_to_tile(model, tile)
                prediction = cv2.resize(prediction, (tile_size[0], tile_size[1]), interpolation=cv2.INTER_NEAREST)
                prediction = (prediction > threshold).astype(np.uint8) * 255

                segmentation_mask[y:y + tile_size[1], x:x + tile_size[0]] = np.maximum(
                    segmentation_mask[y:y + tile_size[1], x:x + tile_size[0]],
                    prediction
                )

            gt_mask = None
            if gt_folder is not None:
                gt_path = os.path.join(gt_folder, filename)
                if os.path.exists(gt_path):
                    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if gt_mask is not None:
                iou_before, dice_before = calculate_metrics(gt_mask, segmentation_mask)
                metrics_before.append((iou_before, dice_before))

            postprocessed_mask = remove_small_blobs(segmentation_mask, min_blob_size)

            if gt_mask is not None:
                iou_after, dice_after = calculate_metrics(gt_mask, postprocessed_mask)
                metrics_after.append((iou_after, dice_after))

            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask.png")
            cv2.imwrite(output_path, postprocessed_mask)
            print(f"Saved segmentation mask for {filename} to {output_path}")

            if visualization_folder is not None and gt_mask is not None:
                save_visualization(big_img, segmentation_mask, postprocessed_mask, gt_mask, visualization_folder, filename)

    if gt_folder is not None:
        print("Metrics before postprocessing:")
        for i, (iou, dice) in enumerate(metrics_before):
            print(f"Image {i + 1}: IoU={iou:.4f}, Dice={dice:.4f}")

        avg_iou_before = np.mean([m[0] for m in metrics_before])
        avg_dice_before = np.mean([m[1] for m in metrics_before])
        print(f"Average IoU before postprocessing: {avg_iou_before:.4f}")
        print(f"Average Dice before postprocessing: {avg_dice_before:.4f}")

        print("\nMetrics after postprocessing:")
        for i, (iou, dice) in enumerate(metrics_after):
            print(f"Image {i + 1}: IoU={iou:.4f}, Dice={dice:.4f}")

        avg_iou_after = np.mean([m[0] for m in metrics_after])
        avg_dice_after = np.mean([m[1] for m in metrics_after])
        print(f"Average IoU after postprocessing: {avg_iou_after:.4f}")
        print(f"Average Dice after postprocessing: {avg_dice_after:.4f}")


def inference_handler(args):
    model = load_model(args.h5_file)
    process_images(
        model=model,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        tile_size=args.tile_size,
        overlap_size=args.overlap_size,
        resize_size=args.resize_size,
        threshold=args.threshold,
        min_blob_size=args.min_blob_size,
        gt_folder=args.gt_folder,
        visualization_folder=args.visualization_folder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for segmentation models.")

    parser.add_argument("--h5_file", help="Path to the .h5 file with model weights.", required=True)
    parser.add_argument("--input_folder", help="Path to the folder with input images.", required=True)
    parser.add_argument("--output_folder", help="Path to save the output segmentation masks.", required=True)
    parser.add_argument("--tile_size", nargs=2, type=int, default=[64, 64], help="Tile size as (width, height).")
    parser.add_argument("--resize_size", nargs=2, type=int, default=[64, 64], help="Resize size as (width, height).")
    parser.add_argument("--overlap_size", nargs=2, type=int, default=[16, 16], help="Overlap size as (width, height).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for segmentation prediction.")
    parser.add_argument("--min_blob_size", type=int, default=100, help="Minimum size of blobs to keep.")
    parser.add_argument("--gt_folder", help="Path to the folder with ground truth segmentation masks.", default=None)
    parser.add_argument("--visualization_folder", help="Path to save visualizations.", default=None)

    parser.set_defaults(func=inference_handler)

    args = parser.parse_args()
    args.func(args)




