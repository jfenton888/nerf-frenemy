

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
parser.add_argument('--source', help="Source dataset folder", default="../../datasets/PROPS-Table1/valid", action='store')
parser.add_argument('--dest', help="Destination dataset folder", default="../../datasets/PROPS-Table1", action='store')
args = parser.parse_args()


source_dataset_dir: Path = Path(args.source)
out_dataset_dir: Path = Path(args.dest)


input_dir = os.path.join(source_dataset_dir, "")
images_list = os.listdir(input_dir)

rgb_list = [f for f in images_list if f.endswith(".jpg")]
rgb_list.sort()

# mask_list = [f for f in images_list if f.endswith(".png")]
# mask_list.sort()

rgb_list_used = rgb_list if args.all else rgb_list[0:1]


obj_id_list = [
    1, # master chef
    2, # cracker box
    3, # sugar box
    4, # soup can
    5, # mustard bottle
    6, # tuna can
    8, # jello box
    9, # meat can
    14,# mug
    18 # marker
]
id2label = {}
for idx, id in enumerate(obj_id_list):
    id2label[id] = idx + 1

class_mapping = { # Match the class name used for file names to those used in the dataset. NOTE: Only the 3 classes are actually used.
    0: 10, # master chef
    1: 11, # cracker box
    2: 3, # sugar box
    3: 2, # soup can
    4: 12, # mustard bottle
    5: 13, # tuna can
    6: 14, # jello box
    7: 1, # meat can
    8: 15, # mug
    9: 16, # marker
}

for rgb_fname in rgb_list_used:
    
    image_num = int(rgb_fname.split("_")[0])
    image_stem = ".".join(rgb_fname.split('.')[:-1])
    mask_fname = f"{image_stem}_mask.png"
    rgb_path = os.path.join(input_dir, rgb_fname)
    mask_path = os.path.join(input_dir, mask_fname)

    image_rgb = Image.open(rgb_path)
    image_rgb = image_rgb.convert('RGB')
    # image_rgb = image_rgb.crop((0, 420, 1440, 1500)) # Crop the image from (1920, 1440) to (1080, 1440) by removing the top and bottom 420 pixels 
    # image_rgb = image_rgb.resize((640, 480), Image.NEAREST)

    image_mask = Image.open(mask_path)
    mask_np = np.array(image_mask)

    for obj_name, obj_idx in class_mapping.items():
        mask = mask_np == obj_idx
        mask = mask.astype(np.uint8) * 255
        mask = Image.fromarray(mask, mode="L")
        # mask = mask.crop((0, 420, 1440, 1500))
        # mask = mask.resize((640, 480), Image.NEAREST)

        out_mask_fname = f"{image_num:06d}_{obj_name:06d}.png"
        out_mask_path = os.path.join(out_dataset_dir, "mask_visib", out_mask_fname)

        mask.save(out_mask_path)

        
    out_rgb_fname = f"{image_num:06d}.png"
    out_rgb_path = os.path.join(out_dataset_dir, "images", out_rgb_fname)
    image_rgb.save(out_rgb_path)

    # modified_image = image.copy()
    # for obj_idx in mask_list_used:
    #     mask_path = os.path.join(input_dir, f"{obj_idx:06d}.png")
    #     mask = Image.open(mask_path)
    #     modified_image = switch_colors_masked(image, mask)
    # out_path = os.path.join(out_dataset_dir, "images", rgb_fname)
    # modified_image.save(out_path)


# for rgb_fname in rgb_list_used:
#     rgb_path = os.path.join(rgb_dir, rgb_fname)
    
#     image = Image.open(rgb_path)
#     image = image.convert('RGB')
    
#     idx = int(rgb_fname.split(".")[0])

#     modified_image = image.copy()
#     for (obj_idx, hue_shift) in mask_hue:    

#         mask_path = os.path.join(mask_dir, f"{idx:006d}_{obj_idx:06d}.png")
#         mask = Image.open(mask_path)

#         # modified_image = color_noise(image)
#         # modified_image = switch_colors(image)
#         # modified_image = switch_colors_masked(image, mask)
#         # modified_image = gaussian_noise(image)
#         modified_image = change_hsv(modified_image, mask, hue_shift)

#     out_path = os.path.join(out_dataset_dir, "images", rgb_fname)
#     modified_image.save(out_path)
