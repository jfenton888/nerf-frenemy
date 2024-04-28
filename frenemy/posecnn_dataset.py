# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PoseCNN dataset.
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE


class PoseCNNDataset(InputDataset):
    """PoseCNNDataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth_image", "objs_id", "label", "bbx", "RTs", "centermaps", "centers"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        # if there are no depth images than we want to generate them all with zoe depth

        ## parameter
        self.max_instance_num = 10
        self.H = 480
        self.W = 640
        self.rgb_aug_prob = 0.4
        self.cam_intrinsic = np.array([
                            [902.19, 0.0, 342.35],
                            [0.0, 902.39, 252.23],
                            [0.0, 0.0, 1.0]])
        self.resolution = [640, 480]

        self.obj_id_list = [
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
        self.id2label = {}
        for idx, id in enumerate(self.obj_id_list):
            self.id2label[id] = idx + 1

        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
        self.objs_dict_list = self.metadata["objs_dict"]


    def get_metadata(self, data: Dict) -> Dict:
        metadata = {}

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self.cameras.height[data["image_idx"]])
        width = int(self.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )
        metadata["depth_image"] = depth_image

        detections = self.get_detections(data["image_idx"])
        metadata.update(**detections)

        camera = self.cameras[data["image_idx"]]
        metadata["cameras"] = camera

        # # Turn detections into a mask so that only pixels of objects are used
        # mask = ~torch.tensor(detections["label"][0]).unsqueeze(-1)
        # metadata["mask"] = mask

        return metadata

    def get_detections(self, idx):
        """
        objs_dict = {
            0: {
                cam_R_m2c:
                cam_t_m2c:
                obj_id:
                visible:
                bbox_visib:
                visiable_mask_path:
            }
            ...
        }

        data_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs',
            'centermaps', []
        }
        """
        # print("{} has {}".format(type(self.objs_dict_list), type(idx)))
        depth_path = self.depth_filenames[idx]
        objs_dict = self.objs_dict_list[idx]

        data_dict = {}
        # with Image.open(rgb_path) as im:
        #     rgb = np.array(im)

        # if self.split == 'train' and np.random.rand(1) > 1 - self.rgb_aug_prob:
        #     rgb = chromatic_transform(rgb)
        #     rgb = add_noise(rgb)
        # rgb = rgb.astype(np.float32)/255
        # data_dict['rgb'] = rgb.transpose((2,0,1))

        with Image.open(depth_path) as im:
            data_dict['depth'] = np.array(im, dtype=np.int16)[np.newaxis, :]
        # ## TODO data-augmentation of depth 
        assert(len(objs_dict) <= self.max_instance_num)
        objs_id = np.zeros(self.max_instance_num, dtype=np.uint8)
        label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)
        bbx = np.zeros((self.max_instance_num, 4))
        RTs = np.zeros((self.max_instance_num, 3, 4))
        centers = np.zeros((self.max_instance_num, 2))
        centermaps = np.zeros((self.max_instance_num, 3, self.resolution[1], self.resolution[0]))

        for idx in objs_dict.keys():
            if len(objs_dict[idx]['bbox_visib']) > 0:
                ## have visible mask 
                objs_id[idx] = self.id2label[objs_dict[idx]['obj_id']]
                assert(objs_id[idx] > 0)
                with Image.open(objs_dict[idx]['visible_mask_path']) as im:
                    label[objs_id[idx]] = np.array(im, dtype=bool)
                ## [x_min, y_min, width, height]
                bbx[idx] = objs_dict[idx]['bbox_visib']
                
                RT = np.zeros((4, 4))
                RT[3, 3] = 1
                RT[:3, :3] = objs_dict[idx]['R']
                RT[:3, [3]] = objs_dict[idx]['T']
                RT = np.linalg.inv(RT)                
                RTs[idx] = RT[:3]
                
                center_homo = self.cam_intrinsic @ RT[:3, [3]]
                center = center_homo[:2]/center_homo[2]
                
                x = np.linspace(0, self.resolution[0] - 1, self.resolution[0])
                # print("data x {}".format(x.shape))
                y = np.linspace(0, self.resolution[1] - 1, self.resolution[1])
                xv, yv = np.meshgrid(x, y)
                dx, dy = center[0] - xv, center[1] - yv
                distance = np.sqrt(dx ** 2 + dy ** 2)
                nx, ny = dx / distance, dy / distance
                Tz = np.ones((self.resolution[1], self.resolution[0])) * RT[2, 3]

                # centermaps[idx] = np.array([nx, ny, Tz]) # Centermaps is too large to be able to store for every image

                centers[idx] = np.array([float(center[0]), float(center[1])])
        label[0] = 1 - label[1:].sum(axis=0)

        data_dict['objs_id'] = objs_id
        data_dict['label'] = label
        data_dict['bbx'] = bbx
        data_dict['RTs'] = RTs
        data_dict['centers'] = centers
        return data_dict



# class SemanticDataset(InputDataset):
#     """Dataset that returns images and semantics and masks.

#     Args:
#         dataparser_outputs: description of where and how to read input images.
#     """

#     exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "semantics"]

#     def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
#         super().__init__(dataparser_outputs, scale_factor)
#         assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
#         self.semantics = self.metadata["semantics"]
#         self.mask_indices = torch.tensor(
#             [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
#         ).view(1, 1, -1)

#     def get_metadata(self, data: Dict) -> Dict:
#         # handle mask
#         filepath = self.semantics.filenames[data["image_idx"]]
#         semantic_label, mask = get_semantics_and_mask_tensors_from_path(
#             filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
#         )
#         if "mask" in data.keys():
#             mask = mask & data["mask"]
#         return {"mask": mask, "semantics": semantic_label}