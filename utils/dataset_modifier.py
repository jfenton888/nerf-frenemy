


import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from rgb_transforms import RGBTransform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--all', help='Whether to use all the files', default=False, action='store_true')
args = parser.parse_args()


# source_dataset_dir: Path = Path("../../datasets/PROPS-NeRF")
# out_dataset_dir: Path = Path("../../datasets/PROPS-NeRF-modified")
source_dataset_dir: Path = Path("../../datasets/PROPS-Pose-Dataset")
out_dataset_dir: Path = Path("../../datasets/PROPS-Pose-Dataset-modified")


def color_noise(image: Image) -> Image:
    image_blue = RGBTransform().mix_with((0, 0, 0),factor=0.6).applied_to(image)

    return image_blue

def gaussian_noise(image: Image) -> Image:
    image_np = np.array(image)

    # Genearte noise with same shape as that of the image
    noise = np.random.normal(0, 50, image_np.shape) 

    # Add the noise to the image
    img_noised = image_np + noise

    # Clip the pixel values to be between 0 and 255.
    img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)

    return Image.fromarray(img_noised)

def darker_noise(image: Image):
    rgb_image = image.convert(mode="RGB")
    image.convert(mode="HSV")

def switch_colors(image: Image):
    image_np = np.array(image)

    image_np = image_np[:, :, np.array([1, 2, 0])]

    return Image.fromarray(image_np)


rgb_dir = os.path.join(source_dataset_dir, "val/rgb")
rgb_list = os.listdir(rgb_dir)
rgb_list.sort()

rgb_list_used = rgb_list if args.all else rgb_list[0:1]

for rgb_fname in rgb_list_used:
    rgb_path = os.path.join(rgb_dir, rgb_fname)
    
    image = Image.open(rgb_path)
    image = image.convert('RGB')

    # modified_image = color_noise(image)
    modified_image = switch_colors(image)
    # modified_image = gaussian_noise(image)

    out_path = os.path.join(out_dataset_dir, "val/rgb", rgb_fname)
    modified_image.save(out_path)
