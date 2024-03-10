"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from collections import defaultdict
import os

import torch
from torch.nn import Parameter

from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.field_heads import FieldHeadNames

# from nerfstudio.nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.models.base_model import ModelConfig

from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel  # for custom Model


@dataclass
class FrenemyModelConfig(ModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: FrenemyModel)
    """target class to instantiate"""
    # enable_collider: bool = None
    """Whether to create a scene collider to filter rays."""
    # collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    # """parameters to instantiate scene collider with"""
    # loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    # """parameters to instantiate density field with"""
    num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    nerf_config: ModelConfig = field(default_factory=ModelConfig)
    """specifies the nerf type of nerf to use"""
    nerf_path: Optional[Path] = None
    """path to the pretrained nerf model to load"""


class FrenemyModel(Model):
    """Template Model."""

    config: FrenemyModelConfig

    def populate_modules(self):
        # super().populate_modules()

        self.nerf = self.config.nerf_config.setup(scene_box=self.scene_box, 
                                                  num_train_data=self.num_train_data, 
                                                  **self.kwargs)

        
        # # TODO: Load state dict for nerf model
        # if self.config.nerf_path is None:
        #     raise ValueError("Nerf path is required to load the model.")
        
        # load_dir = self.config.nerf_path
        # print("Loading latest Nerfstudio checkpoint from load_dir...")
        # # NOTE: this is specific to the checkpoint name format
        # load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
        
        # load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
        # assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        # loaded_state = torch.load(load_path, map_location="cpu")
        
        # self.load_nerf_dict(loaded_state)

        # TODO: Add perturbation modules here

        self.nerf.get_field_outputs = self.get_field_outputs


    def load_nerf_dict(self, loaded_state: Mapping[str, Any], strict: Optional[bool] = None):

        state_dict = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        model_state = {
            (key[len("_model.") :] if key.startswith("_model.") else key): value for key, value in state_dict.items()
        }

        try:
            self.nerf.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.nerf.load_state_dict(model_state, strict=False)
            else:
                raise



    def get_param_groups(self) -> Dict[str, List[Parameter]]:

        # TODO: Once pretrained models are being loaded in, these parameters should only be those from the perturbation model
        # These determine the parameters updated by the optimizer
        
        return self.nerf.get_param_groups()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        return self.nerf.get_training_callbacks(training_callback_attributes)
    
    def get_field_outputs(self, ray_samples: RaySamples, **kwargs) -> Dict[FieldHeadNames, torch.Tensor]:

        # TODO: Actually add in the perturbation model here

        nerf_outputs = NerfactoModel.get_field_outputs(self.nerf, ray_samples, **kwargs)

        outputs = {}
        for output_name, outputs_tensor in nerf_outputs.items():
            if output_name == FieldHeadNames.RGB:
                outputs[output_name] = outputs_tensor # + 0.4
            else:
                outputs[output_name] = outputs_tensor

        return outputs

    def get_outputs(self, ray_bundle: RayBundle):

        return self.nerf(ray_bundle)

    
    # def forward(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Union[torch.Tensor, List]]:
    #     """Run forward starting with a ray bundle. This outputs different things depending on the configuration
    #     of the model and whether or not the batch is provided (whether or not we are training basically)

    #     Args:
    #         ray_bundle: containing all the information needed to render that ray latents included
    #     """
    #     if self.collider is not None:
    #         ray_bundle = self.collider(ray_bundle)

    #     return self.get_outputs(ray_bundle)
    
        # if isinstance(ray_bundle, RayBundle):
        #     if self.collider is not None:
        #         ray_bundle = self.collider(ray_bundle)

        #     return self.get_outputs(ray_bundle)
            
        
        # elif isinstance(ray_bundle, Cameras): 
        #     return self._get_outputs_for_camera(ray_bundle)
        
        # return {}
         

    def get_metrics_dict(self, outputs, batch):

        return self.nerf.get_metrics_dict(outputs, batch)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):

        # TODO: Add loss from task specific network here. This will require loading the network in during setup

        return self.nerf.get_loss_dict(outputs, batch, metrics_dict)

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        return self.nerf.get_image_metrics_and_images(outputs, batch)


    # def _get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
    #     """Takes in a camera, generates the raybundle, and computes the output of the model.
    #     Assumes a ray-based model.

    #     Args:
    #         camera: generates raybundle
    #     """
    #     print(camera.shape)
    #     return self._get_outputs_for_camera_ray_bundle(
    #         camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
    #     )

    # def _get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
    #     """Takes in camera parameters and computes the output of the model.

    #     Args:
    #         camera_ray_bundle: ray bundle to calculate outputs over
    #     """
    #     num_rays_per_chunk = self.config.num_rays_per_chunk
    #     image_height, image_width = camera_ray_bundle.origins.shape[:2]
    #     num_rays = len(camera_ray_bundle)
    #     # print(f"Total number of rays: {num_rays}")
    #     outputs_lists = defaultdict(list)
    #     # print(f"Processing n chunks: {num_rays // num_rays_per_chunk}")
 
    #     import random
    #     for i in random.sample(range(0, num_rays, num_rays_per_chunk), 10):
    #         print(f"Processing rays {i} to {i + num_rays_per_chunk}")
    #         start_idx = i
    #         end_idx = i + num_rays_per_chunk
    #         ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
    #         # move the chunk inputs to the model device
    #         ray_bundle = ray_bundle.to(self.device)
    #         outputs = self.forward(ray_bundle=ray_bundle)
    #         for output_name, output in outputs.items():  # type: ignore
    #             if not isinstance(output, torch.Tensor):
    #                 # TODO: handle lists of tensors as well
    #                 continue
    #             # move the chunk outputs from the model device back to the device of the inputs.
    #             outputs_lists[output_name].append(output)
    #     outputs = {}
    #     for output_name, outputs_list in outputs_lists.items():
    #         outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

    #     return outputs

    


