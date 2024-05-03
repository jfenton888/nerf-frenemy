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


class PoseCNNSegmentationDataset(InputDataset):
    """PoseCNNDataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["objs_id", "label"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        # if there are no depth images than we want to generate them all with zoe depth

        ## parameter
        self.max_instance_num = 10
        self.H = 480
        self.W = 640

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

        height = int(self.cameras.height[data["image_idx"]])
        width = int(self.cameras.width[data["image_idx"]])


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
        objs_dict = self.objs_dict_list[idx]

        data_dict = {}

        assert(len(objs_dict) <= self.max_instance_num)
        objs_id = np.zeros(self.max_instance_num, dtype=np.uint8)
        label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)

        for idx in objs_dict.keys():
            objs_id[idx] = idx+1
            with Image.open(objs_dict[idx]['visible_mask_path']) as im:
                label[objs_id[idx]] = np.array(im, dtype=bool)
        label[0] = 1 - label[1:].sum(axis=0)

        data_dict['objs_id'] = objs_id
        data_dict['label'] = label

        return data_dict