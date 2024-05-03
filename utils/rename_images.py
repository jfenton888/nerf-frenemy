


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


source_dataset_dir: Path = Path("../datasets/PROPS-Table2")
out_dataset_dir: Path = Path("../datasets/PROPS-Table2")


rgb_dir = os.path.join(source_dataset_dir, "images")
rgb_list = os.listdir(rgb_dir)
rgb_list.sort()

# mask_dir = os.path.join(source_dataset_dir, "mask_visib")
# mask_list = os.listdir(rgb_dir)
# mask_list.sort()

rgb_list_used = rgb_list if args.all else rgb_list[0:1]

for rgb_fname in rgb_list_used:
    rgb_path = os.path.join(rgb_dir, rgb_fname)
    
    image = Image.open(rgb_path)
    image = image.convert('RGB')
    
    idx = int(rgb_fname.split(".")[0].split("_")[1])
    # obj_idx = 4

    # mask_path = os.path.join(mask_dir, f"{idx:006d}_{obj_idx:06d}.png")
    # mask = Image.open(mask_path)

    long_name = os.path.join(out_dataset_dir, "images", f"{idx:006d}.png")

    image.save(long_name)
