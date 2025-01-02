import os
import numpy as np
import cv2
import argparse


def tiles_generator(big_img, big_mask, tile_size, overlap_size, resize_size):

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

            mask = big_mask[y:y + tile_height, x:x + tile_width]
            if mask.shape[1] != resize_size[1] or mask.shape[0] != resize_size[0]:
                mask = cv2.resize(mask, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)

            yield tile, mask, x, y


def process_images_and_masks(input_folder, mask_folder, output_base_folder, tile_size, overlap_size, resize_size):
    crack_folder = os.path.join(output_base_folder, 'crack')
    nocrack_folder = os.path.join(output_base_folder, 'nocrack')
    crack_images_folder = os.path.join(crack_folder, 'images')
    crack_masks_folder = os.path.join(crack_folder, 'masks')
    nocrack_images_folder = os.path.join(nocrack_folder, 'images')
    nocrack_masks_folder = os.path.join(nocrack_folder, 'masks')

    os.makedirs(crack_images_folder, exist_ok=True)
    os.makedirs(crack_masks_folder, exist_ok=True)
    os.makedirs(nocrack_images_folder, exist_ok=True)
    os.makedirs(nocrack_masks_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        print(f"Processing {filename}.")
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            big_img = cv2.imread(img_path)

            if big_img is None:
                continue

            base_filename = os.path.splitext(filename)[0]
            mask_filename = f"{base_filename}_label.PNG"
            mask_path = os.path.join(mask_folder, mask_filename)

            big_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if big_mask is None:
                print(f"Mask {mask_filename} not found, skipping.")
                continue

            for i, (tile, mask, x, y) in enumerate(tiles_generator(big_img, big_mask, tile_size, overlap_size, resize_size)):
                if np.all(mask == 0):
                    images_folder = nocrack_images_folder
                    masks_folder = nocrack_masks_folder
                else:
                    images_folder = crack_images_folder
                    masks_folder = crack_masks_folder

                tile_filename = f"{base_filename}_tile_{i}.png"
                tile_path = os.path.join(images_folder, tile_filename)
                cv2.imwrite(tile_path, tile)

                mask_tile_filename = f"{base_filename}_tile_{i}_mask.png"
                mask_tile_path = os.path.join(masks_folder, mask_tile_filename)
                cv2.imwrite(mask_tile_path, mask)


def tiler_handler(args):
    process_images_and_masks(args.input_folder,
                             args.mask_folder,
                             args.output_base_folder,
                             args.tile_size,
                             args.overlap_size,
                             args.resize_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_tiler = subparsers.add_parser("tiler")
    parser_tiler.add_argument("--input_folder", help="Directory containing original images.", required=True)
    parser_tiler.add_argument("--mask_folder", help="Directory containing original masks.", required=True)
    parser_tiler.add_argument("--output_base_folder", help="Where to save cropped data", required=True)

    parser_tiler.add_argument("--tile_size", help="Size of the tiles in pixels as a tuple (height, width).",
                              nargs=2, type=int, default=[128, 128])
    parser_tiler.add_argument("--resize_size",
                              help="Size to which tiles are resized in pixels as a tuple (height, width).",
                              nargs=2, type=int, default=[128, 128])
    parser_tiler.add_argument("--overlap_size",
                              help="Size of overlap between tiles in pixels as a tuple (height, width).",
                              nargs=2, type=int, default=[32, 32])

    parser_tiler.set_defaults(func=tiler_handler)

    args = parser.parse_args()
    args.func(args)
