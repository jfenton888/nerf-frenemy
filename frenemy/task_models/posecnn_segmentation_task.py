"""
Preamble
"""

# Task model related configs
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Type

import numpy as np
import torch
from torchvision.models import vgg16, VGG16_Weights

from frenemy.task_models.base_task import Task, TaskConfig
from frenemy.PoseCNN.pose_cnn import PoseCNN
from frenemy.PoseCNN.rob599.utils import Visualize, format_gt_RTs

@dataclass
class PoseCNNSegmentationTaskConfig(TaskConfig):
    """Configuration for task instantiation"""

    _target: Type = field(default_factory=lambda: PoseCNNSegmentationTask)
    """target class to instantiate"""
    model_path: Path = Path("../nerfrenemy/frenemy/PoseCNN/posecnn_model.pth")
    """Path to the PoseCNN model weights"""
    dataset_dir: Path = Path("../datasets/PROPS-NeRF")
    """Path to the PoseCNN model weights"""
    segmentation_loss_mult: float = 1.0
    """Segmentation loss multiplier."""
    translation_loss_mult: float = 1.0
    """Translation loss multiplier."""
    rotation_loss_mult: float = 1.0
    """Rotation loss multiplier."""




class PoseCNNSegmentationTask(Task):
    """Task class

    Args:
        config: configuration for instantiating model
    """
    config: PoseCNNSegmentationTaskConfig

    def __init__(self, config: TaskConfig, **kwargs):
        super().__init__(config, **kwargs)


    def populate_modules(self):
        """Populate the modules of the model."""

        self.max_instance_num = 10
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

        self.cam_intrinsic = np.array([
                            [902.19, 0.0, 342.35],
                            [0.0, 902.39, 252.23],
                            [0.0, 0.0, 1.0]])
        
        self.models_pcd = self.parse_model()

        vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        self.posecnn_model = PoseCNN(pretrained_backbone = vgg16_model, \
                                     models_pcd = torch.tensor(self.models_pcd).to(self.device, dtype=torch.float32), 
                                     cam_intrinsic = self.cam_intrinsic,
                                     only_segmentation=True
                                    ).to(self.device)
        
        self.posecnn_model.load_state_dict(torch.load(self.config.model_path))


    def parse_model(self):
        model_path = os.path.join(self.config.dataset_dir, "model")
        objpathdict = {
            1: ["master_chef_can", os.path.join(model_path, "1_master_chef_can", "textured_simple.obj")],
            2: ["cracker_box", os.path.join(model_path, "2_cracker_box", "textured_simple.obj")],
            3: ["sugar_box", os.path.join(model_path, "3_sugar_box", "textured_simple.obj")],
            4: ["tomato_soup_can", os.path.join(model_path, "4_tomato_soup_can", "textured_simple.obj")],
            5: ["mustard_bottle", os.path.join(model_path, "5_mustard_bottle", "textured_simple.obj")],
            6: ["tuna_fish_can", os.path.join(model_path, "6_tuna_fish_can", "textured_simple.obj")],
            7: ["gelatin_box", os.path.join(model_path, "8_gelatin_box", "textured_simple.obj")],
            8: ["potted_meat_can", os.path.join(model_path, "9_potted_meat_can", "textured_simple.obj")],
            9: ["mug", os.path.join(model_path, "14_mug", "textured_simple.obj")],
            10: ["large_marker", os.path.join(model_path, "18_large_marker", "textured_simple.obj")],
        }
        self.visualizer = Visualize(objpathdict, self.cam_intrinsic, self.resolution)
        models_pcd_dict = {index:np.array(self.visualizer.objnode[index]['mesh'].vertices) for index in self.visualizer.objnode}
        models_pcd = np.zeros((len(models_pcd_dict), 1024, 3))
        for m in models_pcd_dict:
            model = models_pcd_dict[m]
            models_pcd[m - 1] = model[np.random.randint(0, model.shape[0], 1024)]
        return models_pcd

    def posecnn_dict(self, outputs, batch, image_idx):
        """
        input_dict = {
            'rgb', # [batch_size, 3, H, W]
            'depth', # [batch_size, 1, H, W]
            'objs_id', # [batch_size, N]
            'mask', # [batch_size, N, H, W]
            'bbx', # [batch_size, N, 4]
            'RTs' # [batch_size, N, 3, 4]
        }
        """
        posecnn_dict = {}

        # There is only one image in the batch
        if image_idx == None:
            posecnn_dict["label"] = batch["label"].to(self.device) # [B, 11, H, W]
            # posecnn_dict["depth"] = batch["depth"].to(self.device) # [B, 1, H, W]
            # posecnn_dict["objs_id"] = batch["objs_id"].to(self.device) # [B, N]
            # posecnn_dict["bbx"] = batch["bbx"].to(self.device) # [B, N, 4]
            # posecnn_dict["RTs"] = batch["RTs"].to(self.device) # [B, N, 3, 4]
            # posecnn_dict["centers"] = batch["centers"].to(self.device) # [B, N, 3]
        else:
            posecnn_dict["label"] = batch["label"][image_idx].to(self.device) # [B, 11, H, W]
            # posecnn_dict["depth"] = batch["depth"][image_idx].to(self.device) # [B, 1, H, W]
            # posecnn_dict["objs_id"] = batch["objs_id"][image_idx].to(self.device) # [B, N]
            # posecnn_dict["bbx"] = batch["bbx"][image_idx].to(self.device) # [B, N, 4]
            # posecnn_dict["RTs"] = batch["RTs"][image_idx].to(self.device) # [B, N, 3, 4]
            # posecnn_dict["centers"] = batch["centers"][image_idx].to(self.device) # [B, N, 3]

        # centermaps = torch.zeros((self.max_instance_num, 3, self.resolution[1], self.resolution[0]), device=self.device)

        # for idx in range(posecnn_dict["objs_id"].shape[0]):
        #     if torch.any(posecnn_dict["bbx"][idx] > 0):

        #         RT = posecnn_dict["RTs"][idx]
        #         center = posecnn_dict["centers"][idx]
                
        #         x = torch.linspace(0, self.resolution[0] - 1, self.resolution[0], device=self.device)
        #         y = torch.linspace(0, self.resolution[1] - 1, self.resolution[1], device=self.device)
        #         xv, yv = torch.meshgrid(x, y, indexing="xy")
        #         dx, dy = float(center[0]) - xv, float(center[1]) - yv
        #         distance = torch.sqrt(dx ** 2 + dy ** 2)
        #         nx, ny = dx / distance, dy / distance
        #         Tz = torch.ones((self.resolution[1], self.resolution[0]), device=self.device) * RT[2, 3]
        #         centermaps[idx] = torch.stack([nx, ny, Tz])

        # posecnn_dict["centermaps"] = centermaps.reshape(-1, self.resolution[1], self.resolution[0])

        posecnn_dict["label"] = posecnn_dict["label"].unsqueeze(0)
        # posecnn_dict["depth"] = posecnn_dict["depth"].unsqueeze(0)
        # posecnn_dict["objs_id"] = posecnn_dict["objs_id"].unsqueeze(0)
        # posecnn_dict["bbx"] = posecnn_dict["bbx"].unsqueeze(0)
        # posecnn_dict["RTs"] = posecnn_dict["RTs"].unsqueeze(0)
        # posecnn_dict["centers"] = posecnn_dict["centers"].unsqueeze(0).to(int)
        # posecnn_dict["centermaps"] = posecnn_dict["centermaps"].unsqueeze(0)

        return posecnn_dict
    

    def get_metrics_dict(self, metrics_dict, outputs, batch):
        """Compute the metrics of the model."""
        
        # # patch_images = torch.unique(batch["indices"][:,0])
        # # assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"

        # image = batch["full_image"].to(self.device)#[patch_images[0]].to(self.device)
        # modified_image = image.clone()

        # modified_image[batch["indices"][:,1], batch["indices"][:,2], :] = outputs["rgb"]


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



    def get_loss_dict(self, loss_dict, outputs, batch, metrics_dict):
        """Compute the loss of the model."""

        # print("posecnn_model {}".format(self.posecnn_model.training))
        
        patch_images = torch.unique(batch["indices"][:,0])
        assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"
        patch_image = patch_images.item()

        original_image = batch["full_image"][patch_image].unsqueeze(0).to(self.device)
        modified_image = batch["rendered_image"].unsqueeze(0).to(self.device)

        """
        input_dict = {
            'rgb', # [batch_size, 3, H, W]
            'depth', # [batch_size, 1, H, W]
            'objs_id', # [batch_size, N]
            'mask', # [batch_size, N, H, W]
            'bbx', # [batch_size, N, 4]
            'RTs' # [batch_size, N, 3, 4]
        }
        """
        posecnn_dict = self.posecnn_dict(outputs, batch, patch_image)
        
        # Get the loss of the original image to compare against the modified image
        posecnn_dict["rgb"] = original_image.permute(0, 3, 1, 2)
        with torch.no_grad():
            original_task_loss = self.posecnn_model(posecnn_dict)

        # Get the loss of the modified image
        modified_image[0, batch["indices"][:,1], batch["indices"][:,2], :] = outputs["rgb"]
        posecnn_dict["rgb"] = modified_image.permute(0, 3, 1, 2)

        task_loss = self.posecnn_model(posecnn_dict)

        loss_dict["segmentation_loss"] = self.config.segmentation_loss_mult * task_loss["loss_segmentation"]
        loss_dict["original_segmentation_loss"] = self.config.segmentation_loss_mult * original_task_loss["loss_segmentation"]


    def get_image_metrics_and_images(
        self, 
        metrics_dict: Dict[str, float], 
        images_dict: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ):
        # This will be called during evaluation, and will be called with full images as outputs shaped [H, W, C]

        rgb, predicted_rgb = torch.tensor_split(images_dict["img"].clone(), 2, dim=1)

        gt_rgb, original_rgb = rgb.clone(), rgb.clone() # This will be used for detections on the original image
        pred_rgb = predicted_rgb.clone()
        
        # Get segmentation mask for the original image, the predicted image, and the ground truth mask
        # Labels are in [1, 11, H, W]
        gt_segmentation = torch.tensor(batch["label"]).to(self.device)

        # Transform inputs from [H, W, C] to [1, C, H, W], adding batch dimension
        # Output segmentation will be in [1, H, W] with value of each pixel being the class label
        original_output_dict, original_segmentation = self.posecnn_model({"rgb": rgb.permute(2, 0, 1).unsqueeze(0)}, eval=True)
        pred_output_dict, pred_segmentation = self.posecnn_model({"rgb": pred_rgb.permute(2, 0, 1).unsqueeze(0)}, eval=True)
        
        cmp = color_map(10, normalized=True).to(self.device)
        alpha = 0.5

        
        for i in range(1, 11): # Don't use the 0 class
            # Start with ground truth masks
            mask = gt_segmentation[i] 
            color = cmp[i-1]
            gt_rgb[mask, :] = alpha * gt_rgb[mask, :] + (1 - alpha) * color[:]

            # Then segmentations on the original image
            mask = original_segmentation.squeeze() == i
            color = cmp[i-1]
            original_rgb[mask, :] = alpha * original_rgb[mask, :] + (1 - alpha) * color[:]

            # Then segmentations on the modified image
            mask = pred_segmentation.squeeze() == i
            color = cmp[i-1]
            pred_rgb[mask, :] = alpha * pred_rgb[mask, :] + (1 - alpha) * color[:]

        combined_segmentation_rgb = torch.cat([gt_rgb, original_rgb, pred_rgb], dim=1)
        images_dict["predictions"] = combined_segmentation_rgb

        # Get pose visualizations
        rgb_np =  (rgb.cpu().numpy() * 255).astype(np.uint8)
        rgb_predicted_np =  (predicted_rgb.cpu().numpy() * 255).astype(np.uint8)
        

        original_pose_rgb = torch.tensor(self.visualizer.vis_oneview(
            ipt_im = rgb_np, 
            obj_pose_dict = original_output_dict[0],
            alpha = alpha
            ))
        
        pred_pose_rgb = torch.tensor(self.visualizer.vis_oneview(
            ipt_im = rgb_predicted_np, 
            obj_pose_dict = pred_output_dict[0],
            alpha = alpha
            ))
        
        combined_pose_rgb = torch.cat([original_pose_rgb, pred_pose_rgb], dim=1).to(torch.float) / 255.
        images_dict["pose"] = combined_pose_rgb




def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = torch.float32 if normalized else torch.uint8
    cmap = torch.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = torch.tensor([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap
