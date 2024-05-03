


import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from rgb_transforms import RGBTransform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--all', help='Whether to use all the files', default=False, action='store_true')
parser.add_argument('--all-mask', help='Whether to use all the masks', default=False, action='store_true')
parser.add_argument('--source', help="Source dataset folder", default="../../datasets/PROPS-NeRF", action='store')
parser.add_argument('--dest', help="Destination dataset folder", default="../../datasets/PROPS-NeRF-6", action='store')
args = parser.parse_args()


source_dataset_dir: Path = Path(args.source)
out_dataset_dir: Path = Path(args.dest)
# source_dataset_dir: Path = Path("../../datasets/PROPS-Pose-Dataset")
# out_dataset_dir: Path = Path("../../datasets/PROPS-Pose-Dataset-modified")


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
    img_noised = np.clip(img_noised, 20, 255).astype(np.uint8)

    return Image.fromarray(img_noised)

def darker_noise(image: Image):
    rgb_image = image.convert(mode="RGB")
    image.convert(mode="HSV")

def switch_colors(image: Image):
    image_np = np.array(image)

    image_np = image_np[:, :, np.array([1, 2, 0])]

    return Image.fromarray(image_np)

def switch_colors_masked(image: Image, mask: Image):
    image_np = np.array(image)
    mask_np = np.array(mask, dtype=bool)

    # Create a copy of the image to avoid modifying the original
    modified_image_np = image_np.copy()

    # Apply the mask to the image and switch the color channels
    modified_image_np[mask_np] = image_np[mask_np][:, np.array([2, 0, 1])]

    return Image.fromarray(modified_image_np)

# For HSV
def change_hsv(image: Image, mask: Image, hue):
    hsv_image = image.convert(mode="HSV")
    mask_np = np.array(mask, dtype=bool)

    hsv_np = np.array(hsv_image)
    modified_image_np = hsv_np.copy()

    modified_image_np[mask_np, 0] = (hsv_np[mask_np, 0] + hue) % 255
    
    return Image.fromarray(modified_image_np, mode="HSV").convert(mode="RGB")


rgb_dir = os.path.join(source_dataset_dir, "images")
rgb_list = os.listdir(rgb_dir)
rgb_list.sort()

mask_dir = os.path.join(source_dataset_dir, "mask_visib")
mask_list = os.listdir(rgb_dir)
mask_list.sort()

rgb_list_used = rgb_list if args.all else rgb_list[0:1]
mask_list_used = range(10) if args.all_mask else [1]
mask_hue = [(obj_idx, np.random.randint(0, 255)) for obj_idx in mask_list_used]

for rgb_fname in rgb_list_used:
    rgb_path = os.path.join(rgb_dir, rgb_fname)
    
    image = Image.open(rgb_path)
    image = image.convert('RGB')
    
    idx = int(rgb_fname.split(".")[0])

    modified_image = image.copy()
    for (obj_idx, hue_shift) in mask_hue:    

        mask_path = os.path.join(mask_dir, f"{idx:006d}_{obj_idx:06d}.png")
        mask = Image.open(mask_path)

        # modified_image = color_noise(image)
        # modified_image = switch_colors(image)
        # modified_image = switch_colors_masked(image, mask)
        # modified_image = gaussian_noise(image)
        modified_image = change_hsv(modified_image, mask, hue_shift)

    out_path = os.path.join(out_dataset_dir, "images", rgb_fname)
    modified_image.save(out_path)
