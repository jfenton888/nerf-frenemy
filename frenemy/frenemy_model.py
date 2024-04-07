"""
FrenemyModel for learning a perturbation field to improve object performance on tasks.

This is a subclass of the BaseModel. It will train a NerfactoField whose RGB outputs will 
be added to the outputs from the pretrained (and fixed) nerf that is loaded in. Other
fields output by the NerfactoField are currently not used.
"""
from collections import defaultdict
import os

import torch
from torch.nn import Parameter

from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union
from frenemy.task_models.base_task import Task, TaskConfig
from frenemy.task_models.fasterrcnn_task import FasterRCNNTaskConfig
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
    # enable_collider: bool = None
    """Whether to create a scene collider to filter rays."""
    # collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    # """parameters to instantiate scene collider with"""
    # loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    # """parameters to instantiate density field with"""
    num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    nerf: ModelConfig = field(default_factory=ModelConfig)
    """specifies the nerf type of nerf to use"""
    nerf_path: Optional[Path] = None
    """path to the pretrained nerf model to load"""
    task_model: TaskConfig = field(default_factory=FasterRCNNTaskConfig)
    """Task model to use for evaluation"""
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
    
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
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





    # """Whether to randomize the background color."""
    # hidden_dim: int = 64
    # """Dimension of hidden layers"""
    # hidden_dim_color: int = 64
    # """Dimension of hidden layers for color network"""
    # hidden_dim_transient: int = 64
    # """Dimension of hidden layers for transient network"""
    # num_levels: int = 16
    # """Number of levels of the hashmap for the base mlp."""
    # base_res: int = 16
    # """Resolution of the base grid for the hashgrid."""
    # max_res: int = 2048
    # """Maximum resolution of the hashmap for the base mlp."""
    # log2_hashmap_size: int = 19
    # """Size of the hashmap for the base mlp"""
    # features_per_level: int = 2
    # """How many hashgrid features per level"""
    # num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    # """Number of samples per ray for each proposal network."""
    # num_nerf_samples_per_ray: int = 48
    # """Number of samples per ray for the nerf network."""
    # proposal_update_every: int = 5
    # """Sample every n steps after the warmup"""
    # proposal_warmup: int = 5000
    # """Scales n from 1 to proposal_update_every over this many steps"""
    # num_proposal_iterations: int = 2
    # """Number of proposal network iterations."""
    # use_same_proposal_network: bool = False
    # """Use the same proposal network. Otherwise use different ones."""
    # proposal_net_args_list: List[Dict] = field(
    #     default_factory=lambda: [
    #         {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
    #         {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
    #     ]
    # )
    # """Arguments for the proposal density fields."""
    # proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    # """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    # interlevel_loss_mult: float = 1.0
    # """Proposal loss multiplier."""
    # distortion_loss_mult: float = 0.002
    # """Distortion loss multiplier."""
    # orientation_loss_mult: float = 0.0001
    # """Orientation loss multiplier on computed normals."""
    # pred_normal_loss_mult: float = 0.001
    # """Predicted normal loss multiplier."""
    # use_proposal_weight_anneal: bool = True
    # """Whether to use proposal weight annealing."""
    # use_appearance_embedding: bool = True
    # """Whether to use an appearance embedding."""
    # use_average_appearance_embedding: bool = True
    # """Whether to use average appearance embedding or zeros for inference."""
    # proposal_weights_anneal_slope: float = 10.0
    # """Slope of the annealing function for the proposal weights."""
    # proposal_weights_anneal_max_num_iters: int = 1000
    # """Max num iterations for the annealing function."""
    # use_single_jitter: bool = True
    # """Whether use single jitter or not for the proposal networks."""
    # predict_normals: bool = False
    # """Whether to predict normals or not."""
    # disable_scene_contraction: bool = False
    # """Whether to disable scene contraction or not."""
    # use_gradient_scaling: bool = False
    # """Use gradient scaler where the gradients are lower for points closer to the camera."""
    # implementation: Literal["tcnn", "torch"] = "tcnn"
    # """Which implementation to use for the model."""
    # appearance_embed_dim: int = 32
    # """Dimension of the appearance embedding."""
    # average_init_density: float = 1.0
    # """Average initial density output from MLP. """
    # camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    # """Config of the camera optimizer to use"""


class FrenemyModel(Model):
    """Template Model."""

    config: FrenemyModelConfig
    perturbation_field: NerfactoField
    # task_model: Task

    def populate_modules(self):
        # super().populate_modules()
        
        # Renderer (use the one from the nerf)
        # self.renderer_rgb = RGBRenderer(background_color=self.config.nerf.background_color)

        # Loss Functions
        self.rgb_loss = MSELoss()
        
        from torchmetrics.image import PeakSignalNoiseRatio
        # self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        # self.ssim = structural_similarity_index_measure
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.task_model = self.config.task_model.setup()
        # from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights # maskrcnn_resnet50_fpn
        
        # self.task_weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        # self.task_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.task_weights) # maskrcnn_resnet50_fpn(pretrained=True)

        # # Step 1: Initialize model with the best available weights
        # self.task_model.eval()

        # # Step 2: Initialize the inference transforms
        # self.task_preprocess = self.task_weights.transforms()

        self.nerf = self.config.nerf.setup(scene_box=self.scene_box, 
                                                  num_train_data=self.num_train_data, 
                                                  **self.kwargs)


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
        
        # TODO: Load state dict for nerf model
        self.load_nerf_model(self.config.nerf_path)

        # Overwrite the get_field_outputs method from the nerf model to add in the outputs from the perturbation module
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

        # model_state = {
        #     (key[len("_model.") :] if key.startswith("_model.") else key): value for key, value in state_dict.items()
        # }

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
        # TODO: Once pretrained models are being loaded in, these parameters should only be those from the perturbation model
        # These determine the parameters updated by the optimizer
        
        # return self.nerf.get_param_groups()

        param_groups = {}
        # param_groups = self.nerf.get_param_groups()
        param_groups["perturbation_field"] = list(self.perturbation_field.parameters())
        return param_groups
        

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        # return self.nerf.get_training_callbacks(training_callback_attributes)
        
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
            # if output_name in perturbation_outputs.keys(): 
            if output_name == FieldHeadNames.RGB:
                outputs[output_name] = outputs_tensor + (1.0*perturbation_outputs[output_name])
                # outputs[FieldHeadNames.NERF] = outputs_tensor
                # outputs[FieldHeadNames.PERTURBATION] = perturbation_outputs[output_name]
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
    
    #     if isinstance(ray_bundle, RayBundle):
    #         if self.collider is not None:
    #             ray_bundle = self.collider(ray_bundle)

    #         return self.get_outputs(ray_bundle)
            
        
    #     elif isinstance(ray_bundle, Cameras): 
    #         return self._get_outputs_for_camera(ray_bundle)
        
    #     return {}
         

    def get_metrics_dict(self, outputs, batch):
        # Need to use the full image, currently only gives the subset of ray indices

        # return self.nerf.get_metrics_dict(outputs, batch)


        # metrics_dict = {}

        # # patch_images = torch.unique(batch["indices"][:,0])
        # # assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"

        # image = batch["full_image"].to(self.device)#[patch_images[0]].to(self.device)
        # modified_image = image.clone()

        # modified_image[batch["indices"][:,1], batch["indices"][:,2], :] = outputs["rgb"]

        # metrics_dict["psnr"] = self.nerf.psnr(image, modified_image)

        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.nerf.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.nerf.psnr(predicted_rgb, gt_rgb)

        self.task_model.get_metrics_dict(metrics_dict, outputs, batch)

        # # Step 3: Apply inference preprocessing transforms
        # target_batch = [self.task_preprocess(image.permute(2,0,1))]


        # # Step 4: Use the model and visualize the prediction
        # self.task_model.eval()
        # target = self.task_model(target_batch)[0]

        # # Predict for the image
        # self.task_model.train()
        # img = self.task_preprocess(modified_image.permute(2,0,1))

        # detection_outputs = self.task_model(img.unsqueeze(0), [target])
        # # print(detection_outputs)

        # task_loss = sum(loss for loss in detection_outputs.values())
        # # print(task_loss)

        # metrics_dict["task"] = task_loss

        # # labels = [self.task_weights.meta["categories"][i] for i in prediction["labels"]]
        # # print(labels)
        # # # box = draw_bounding_boxes(img, boxes=prediction["boxes"],
        # # #                         labels=labels,
        # # #                         colors="red",
        # # #                         width=4, font_size=30)
        
        # metrics_dict["psnr"] = self.nerf.psnr(image, modified_image)

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
        
        # return self.nerf.get_loss_dict(outputs, batch, metrics_dict)


        # loss_dict = {}

        # # patch_images = torch.unique(batch["indices"][:,0])
        # # assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"

        # image = batch["full_image"].to(self.device)#[patch_images[0]].to(self.device)
        # modified_image = image.clone()

        # modified_image[batch["indices"][:,1], batch["indices"][:,2], :] = outputs["rgb"]

        # loss_dict["rgb_loss"] = self.rgb_loss(image, modified_image)

        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.nerf.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)

        self.task_model.get_loss_dict(loss_dict, outputs, batch, metrics_dict)


        # loss_dict = {}
        # image = batch["image"].to(self.device)
        # pred_rgb, gt_rgb = self.nerf.renderer_rgb.blend_background_for_loss_computation(
        #     pred_image=outputs["rgb"],
        #     pred_accumulation=outputs["accumulation"],
        #     gt_image=image,
        # )
        # loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)

        return loss_dict
        
        # if self.training:
        #     loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
        #         outputs["weights_list"], outputs["ray_samples_list"]
        #     )
        #     assert metrics_dict is not None and "distortion" in metrics_dict
        #     loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
        #     if self.config.predict_normals:
        #         # orientation loss for computed normals
        #         loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
        #             outputs["rendered_orientation_loss"]
        #         )

        #         # ground truth supervision for normals
        #         loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
        #             outputs["rendered_pred_normal_loss"]
        #         )
        # #     # Add loss from camera optimizer
        # #     self.camera_optimizer.get_loss_dict(loss_dict)
        # return loss_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        # return self.nerf.get_image_metrics_and_images(outputs, batch)
        
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.nerf.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.nerf.psnr(gt_rgb, predicted_rgb)
        ssim = self.nerf.ssim(gt_rgb, predicted_rgb)
        lpips = self.nerf.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.nerf.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
        

















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

    


