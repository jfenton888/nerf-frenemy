"""
FrenemyModel for learning a perturbation field to improve object performance on tasks.

This is a subclass of the BaseModel. It will train a NerfactoField whose RGB outputs will 
be added to the outputs from the pretrained (and fixed) nerf that is loaded in. Other
fields output by the NerfactoField are currently not used.
"""
from collections import defaultdict
import copy
import os

import torch
from torch.nn import Parameter

from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union

import torchvision.transforms.functional
import torchvision.transforms.v2
from frenemy.task_models.base_task import Task, TaskConfig
from frenemy.task_models.fasterrcnn_task import FasterRCNNTaskConfig
from frenemy.task_models.posecnn_task import PoseCNNTaskConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.renderers import RGBRenderer

# from nerfstudio.nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.models.base_model import ModelConfig

from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils import colormaps  # for custom Model


@dataclass
class FrenemyModelConfig(ModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: FrenemyModel)
    """target class to instantiate"""
    num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    nerf: ModelConfig = field(default_factory=ModelConfig)
    """specifies the nerf type of nerf to use"""
    nerf_path: Optional[Path] = None
    """path to the pretrained nerf model to load"""
    task_model: TaskConfig = field(default_factory=PoseCNNTaskConfig)
    """Task model to use for evaluation"""
    
    rgb_loss_mult: float = 1.0
    """RGB loss multiplier."""
    regulatization_loss_mult: float = 1e-4
    """Output regulatization loss multiplier."""
    perturbation_weight: float = 1.0
    """Perturbation weight."""
    
    ## Perturbation field config components
    hidden_dim: int = 64 # 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64 # 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64 # 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16 # 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    
    # use_proposal_weight_anneal: bool = True
    # """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    # proposal_weights_anneal_slope: float = 10.0
    # """Slope of the annealing function for the proposal weights."""
    # proposal_weights_anneal_max_num_iters: int = 1000
    # """Max num iterations for the annealing function."""
    # use_single_jitter: bool = True
    # """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32 # 32
    """Dimension of the appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """



class FrenemyModel(Model):
    """Template Model."""

    config: FrenemyModelConfig
    perturbation_field: NerfactoField
    task_model: Task

    def populate_modules(self):

        # Loss Functions
        self.rgb_loss = MSELoss()
        
        # Set up the perturbation field
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        self.perturbation_field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        # Task model
        self.task_model = self.config.task_model.setup()

        # Set up the nerf model
        self.nerf = self.config.nerf.setup(scene_box=self.scene_box, 
                                                  num_train_data=self.num_train_data, 
                                                  **self.kwargs)
        # And load the model into the 
        self.load_nerf_model(self.config.nerf_path)

        for (name_pert, state_pert), (name_field, state_field) in zip(self.perturbation_field.named_children(), self.nerf.field.named_children()):
            if name_field != "mlp_head":
                state_pert.load_state_dict(state_field.state_dict())

        # self.perturbation_field = copy.deepcopy(self.nerf.field) # Previously copied the entire nerf field, but better performance when keeping mlp_head randomly initialized

        # Overwrite the get_field_outputs method from the nerf model to add in the outputs from the perturbation module
        self.use_perturbation = True
        self.nerf.get_field_outputs = self.get_field_outputs


    def load_nerf_model(self, path: Mapping[Path, Any], strict: Optional[bool] = None):

        if path is None:
            raise ValueError("Nerf path is required to load the model.")
        
        load_dir = path
        
        # NOTE: this is specific to the checkpoint name format
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
        print(f"Loading latest Nerfstudio checkpoint from load_dir... at step {load_step}")

        load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")

        state_dict = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state["pipeline"].items()
        }

        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        self.nerf.update_to_step(load_step)
        try:
            self.nerf.load_state_dict(model_state, strict=True)
            print("Successfully Loaded NeRF Strict State")
        except RuntimeError:
            print("Failed to load Model")
            if not strict:
                self.nerf.load_state_dict(model_state, strict=False)
            else:
                raise


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # Once pretrained models are being loaded in, these parameters should only be those from the perturbation model
        # These determine the parameters updated by the optimizer
        
        param_groups = {}
        param_groups["perturbation_field"] = list(self.perturbation_field.mlp_head.parameters())
        return param_groups
        

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        
        callbacks = []
        return callbacks

    def get_field_outputs(self, ray_samples: RaySamples, **kwargs) -> Dict[FieldHeadNames, torch.Tensor]:

        # Call the unaltered method from the original model
        nerf_outputs = self.config.nerf._target.get_field_outputs(self.nerf, ray_samples, **kwargs)

        # Find perturbation field outputs
        perturbation_outputs = self.perturbation_field(ray_samples, **kwargs)

        # Combine the outputs without altering torch graph
        outputs = {}
        for output_name, outputs_tensor in nerf_outputs.items():
            if output_name == FieldHeadNames.RGB:
                perturbation_rgb = perturbation_outputs[FieldHeadNames.RGB]
                outputs[FieldHeadNames.PERTURBATION] = perturbation_rgb
                if self.use_perturbation:
                    outputs[FieldHeadNames.RGB] = outputs_tensor + 2*(self.config.perturbation_weight*perturbation_rgb - 0.5)
                else: 
                    outputs[FieldHeadNames.RGB] = outputs_tensor
            else:
                outputs[output_name] = outputs_tensor

        return outputs

    def get_outputs(self, ray_bundle: RayBundle):
        # The outputs for this model come from those produced by the nerf, using the modified get_field_outputs method

        return self.nerf(ray_bundle)

    def get_metrics_dict(self, outputs, batch):

        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.nerf.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.nerf.psnr(predicted_rgb, gt_rgb)

        self.task_model.get_metrics_dict(metrics_dict, outputs, batch)

        patch_images = torch.unique(batch["indices"][:,0])
        assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"

        return metrics_dict
    

        # metrics_dict = {}
        # gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        # gt_rgb = self.nerf.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        # predicted_rgb = outputs["rgb"]
        # metrics_dict["psnr"] = self.nerf.psnr(predicted_rgb, gt_rgb)

        # # gt_rgb_shaped = gt_rgb.permute(1, 0).view(3, 64, 64)
        # # predicted_rgb_shaped = predicted_rgb.permute(1, 0).view(3, 64, 64)

        # # self.task_model.eval()
        # # task_outputs = self.task_model(gt_rgb_shaped.unsqueeze(0))

        # # if self.training:
        # #     metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        # # # self.nerf.camera_optimizer.get_metrics_dict(metrics_dict)

        # # TODO: Add metrics from task specific network here. This will require loading the network in during setup

        # return metrics_dict


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # TODO: Add loss from task specific network here. This will require loading the network in during setup
        
        loss_dict = {}

        # L2 norm regularization loss
        loss_dict["regularization_loss"] = self.config.regulatization_loss_mult * torch.mean(
            torch.norm(outputs["field_outputs"][FieldHeadNames.PERTURBATION] - 0.5)
        )

        patch_images = torch.unique(batch["indices"][:,0])
        assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"
        patch_image = patch_images.item()

        batch["rendered_image"] = self.get_outputs_for_camera(batch["cameras"][patch_image])["rgb"]


        # RGB loss if it's need
        image = batch["image"].to(self.device)

        # Mask of indices that are not assigned to any class (background)
        mask = batch["label"][patch_image].to(self.device)
        # Get indices shaped so they can be used to slice the image rays        
        background_mask = mask[0, batch["indices"][:,1], batch["indices"][:,2]]

        pred_rgb, gt_rgb = self.nerf.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"][background_mask, :],
            pred_accumulation=outputs["accumulation"],
            gt_image=image[background_mask, :],
        )
        # Can result in empty mask, no rgb loss if full patch is inside an object
        if torch.numel(pred_rgb) > 0 and torch.numel(gt_rgb):
            loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(gt_rgb, pred_rgb)
        else:
            loss_dict["rgb_loss"] = torch.zeros(1, device=self.device)

        # Get the loss components from the task model
        self.task_model.get_loss_dict(loss_dict, outputs, batch, metrics_dict)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        # This will be called during evaluation, and will be called with full images as outputs shaped [H, W, C]

        # print("training: {}".format(self.nerf.training))
        
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.nerf.renderer_rgb.blend_background(gt_rgb)
        # acc = colormaps.apply_colormap(outputs["accumulation"])
        # depth = colormaps.apply_depth_colormap(
        #     outputs["depth"],
        #     accumulation=outputs["accumulation"],
        # )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        # combined_acc = torch.cat([acc], dim=1)
        # combined_depth = torch.cat([depth], dim=1)

        # Switch images from[H, W, C] to [1, C, H, W]  for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.nerf.psnr(gt_rgb, predicted_rgb)
        ssim = self.nerf.ssim(gt_rgb, predicted_rgb)
        lpips = self.nerf.lpips(gt_rgb, torch.clamp(predicted_rgb, min=0, max=1))

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb} #, "accumulation": combined_acc, "depth": combined_depth}

        # for i in range(self.config.nerf.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

        # Visualize the regions that have been changed by the model (probably better done with some form of greyscale)
        diff_image = torch.abs(gt_rgb - predicted_rgb)
        images_dict["difference"] = diff_image[0].permute(1, 2, 0)

        # Add all the metrics and images specific to the task model
        self.task_model.get_image_metrics_and_images(metrics_dict, images_dict, outputs, batch)

        return metrics_dict, images_dict


