import os
from pathlib import Path
import cv2
from PIL import Image
import imageio
import numpy as np
from rgb_transforms import RGBTransform


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--all', help='Whether to use all the files', default=False, action='store_true')
parser.add_argument('--all-mask', help='Whether to use all the masks', default=False, action='store_true')
parser.add_argument('--source', help="Source dataset folder", default="../datasets/Table2", action='store')
parser.add_argument('--dest', help="Destination dataset folder", default="../datasets/Table2-small", action='store')
args = parser.parse_args()


source_dataset_dir: Path = Path(args.source)
out_dataset_dir: Path = Path(args.dest)


input_dir = os.path.join(source_dataset_dir, "rgb")
# depth_dir = os.path.join(source_dataset_dir, "depth")
rgb_list = os.listdir(input_dir)
# rgb_list = [f for f in images_list if f.endswith(".jpg")]
rgb_list.sort()

rgb_list_used = rgb_list if args.all else rgb_list[0:1]

for rgb_fname in rgb_list_used:
    
    image_num = int(rgb_fname.split(".")[0])
    rgb_path = os.path.join(input_dir, rgb_fname)

    image_rgb = Image.open(rgb_path)
    image_rgb = image_rgb.convert('RGB')
    # Crop the image from (1920, 1440) to (1080, 1440) by removing the top and bottom 420 pixels 
    image_rgb = image_rgb.crop((0, 420, 1440, 1500))
    image_rgb = image_rgb.resize((640, 480), Image.NEAREST)
        
    # out_rgb_fname = f"{image_num:06d}.png"
    out_rgb_path = os.path.join(out_dataset_dir, "rgb", rgb_fname)
    image_rgb.save(out_rgb_path)

    # depth_fname = f"{image_num}.exr"
    # depth_path = os.path.join(depth_dir, depth_fname)

    # im = imageio.imread(depth_path)

    # img = cv2.resize(img, (128, 128))

    # # im_gamma_correct = np.clip(np.power(im, 0.45), 0, 1)
    # # im_fixed = Image.fromarray(np.uint8(im_gamma_correct*255))

    # # im_fixed = im_fixed.crop((0, 420, 1440, 1500))
    # # im_fixed = im_fixed.resize((640, 480), Image.NEAREST)

    # out_depth_path = os.path.join(out_dataset_dir, "depth", depth_fname)
    # im_fixed.save(out_depth_path)