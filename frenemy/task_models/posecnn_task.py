"""
Preamble
"""

# Task model related configs
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16, VGG16_Weights

from frenemy.task_models.base_task import Task, TaskConfig
from frenemy.PoseCNN.p4_helper import HoughVoting, IOUselection, loss_Rotation, loss_cross_entropy
from frenemy.PoseCNN.pose_cnn import FeatureExtraction, RotationBranch, SegmentationBranch, TranslationBranch
from frenemy.PoseCNN.rob599.utils import Visualize, format_gt_RTs, quaternion_to_matrix
from nerfstudio.utils.io import load_from_json

@dataclass
class PoseCNNTaskConfig(TaskConfig):
    """Configuration for task instantiation"""

    _target: Type = field(default_factory=lambda: PoseCNNTask)
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




class PoseCNNTask(Task):
    """Task class

    Args:
        config: configuration for instantiating model
    """
    config: PoseCNNTaskConfig

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

        self.posecnn_model = PoseCNN(pretrained_backbone = vgg16_model, 
                models_pcd = torch.tensor(self.models_pcd).to(self.device, dtype=torch.float32),
                cam_intrinsic = self.cam_intrinsic).to(self.device)
        
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

        # image = batch["full_image"][patch_image].unsqueeze(0).to(self.device)
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
        posecnn_dict = {}

        # There is only one image in the batch
        # print(batch["cameras"][patch_image])
        posecnn_dict["label"] = batch["label"][patch_image].to(self.device) # [B, 11, H, W]
        posecnn_dict["depth"] = batch["depth"][patch_image].to(self.device) # [B, 1, H, W]
        posecnn_dict["objs_id"] = batch["objs_id"][patch_image].to(self.device) # [B, N]
        posecnn_dict["bbx"] = batch["bbx"][patch_image].to(self.device) # [B, N, 4]
        posecnn_dict["RTs"] = batch["RTs"][patch_image].to(self.device) # [B, N, 3, 4]
        posecnn_dict["centers"] = batch["centers"][patch_image].to(self.device) # [B, N, 3]

        centermaps = torch.zeros((self.max_instance_num, 3, self.resolution[1], self.resolution[0]), device=self.device)

        for idx in range(posecnn_dict["objs_id"].shape[0]):
            if torch.any(posecnn_dict["bbx"][idx] > 0):

                RT = posecnn_dict["RTs"][idx]
                center = posecnn_dict["centers"][idx]
                
                x = torch.linspace(0, self.resolution[0] - 1, self.resolution[0], device=self.device)
                y = torch.linspace(0, self.resolution[1] - 1, self.resolution[1], device=self.device)
                xv, yv = torch.meshgrid(x, y, indexing="xy")
                dx, dy = float(center[0]) - xv, float(center[1]) - yv
                distance = torch.sqrt(dx ** 2 + dy ** 2)
                nx, ny = dx / distance, dy / distance
                Tz = torch.ones((self.resolution[1], self.resolution[0]), device=self.device) * RT[2, 3]
                centermaps[idx] = torch.stack([nx, ny, Tz])

        posecnn_dict["centermaps"] = centermaps.reshape(-1, self.resolution[1], self.resolution[0])

        posecnn_dict["label"] = posecnn_dict["label"].unsqueeze(0)
        posecnn_dict["depth"] = posecnn_dict["depth"].unsqueeze(0)
        posecnn_dict["objs_id"] = posecnn_dict["objs_id"].unsqueeze(0)
        posecnn_dict["bbx"] = posecnn_dict["bbx"].unsqueeze(0)
        posecnn_dict["RTs"] = posecnn_dict["RTs"].unsqueeze(0)
        posecnn_dict["centers"] = posecnn_dict["centers"].unsqueeze(0).to(int)
        posecnn_dict["centermaps"] = posecnn_dict["centermaps"].unsqueeze(0)

        modified_image[0, batch["indices"][:,1], batch["indices"][:,2], :] = outputs["rgb"]
        posecnn_dict["rgb"] = modified_image.permute(0, 3, 1, 2)

        # print("posecnn_dict[\"rgb\"] {} {}".format(posecnn_dict["rgb"].shape, posecnn_dict["rgb"].dtype))
        # print("posecnn_dict[\"label\"] {} {}".format(posecnn_dict["label"].shape, posecnn_dict["label"].dtype))
        # print("posecnn_dict[\"depth\"] {} {}".format(posecnn_dict["depth"].shape, posecnn_dict["depth"].dtype))
        # print("posecnn_dict[\"objs_id\"] {} {}".format(posecnn_dict["objs_id"].shape, posecnn_dict["objs_id"].dtype))
        # print("posecnn_dict[\"bbx\"] {} {}".format(posecnn_dict["bbx"].shape, posecnn_dict["bbx"].dtype))
        # print("posecnn_dict[\"RTs\"] {} {}".format(posecnn_dict["RTs"].shape, posecnn_dict["RTs"].dtype))
        # print("posecnn_dict[\"centermaps\"] {} {}".format(posecnn_dict["centermaps"].shape, posecnn_dict["centermaps"].dtype))
        # print("posecnn_dict[\"centers\"] {} {}".format(posecnn_dict["centers"].shape, posecnn_dict["centers"].dtype))
        

        task_loss = self.posecnn_model(posecnn_dict)


        loss_dict["segmentation_loss"] = self.config.segmentation_loss_mult * task_loss["loss_segmentation"]
        loss_dict["translation_loss"] = self.config.translation_loss_mult * task_loss["loss_centermap"]
        loss_dict["rotation_loss"] = self.config.rotation_loss_mult * task_loss["loss_R"]


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
        
        pose_dict = format_gt_RTs(batch["RTs"])
        render = torch.tensor(self.visualizer.vis_oneview(
            ipt_im = rgb_np, 
            obj_pose_dict = pose_dict,
            alpha = alpha
            ))

        original_pose_rgb = torch.tensor(self.visualizer.vis_oneview(
            ipt_im = rgb_np, 
            obj_pose_dict = original_output_dict[0],
            alpha = alpha
            ))
        
        # pred_pose_rgb = (predicted_rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pred_pose_rgb = torch.tensor(self.visualizer.vis_oneview(
            ipt_im = rgb_predicted_np, 
            obj_pose_dict = pred_output_dict[0],
            alpha = alpha
            ))
        
        combined_pose_rgb = torch.cat([render, original_pose_rgb, pred_pose_rgb], dim=1).to(torch.float) / 255.
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


# def display_detections(rgb, obj_pose_dict, mask, alpha=0.5):
#     img = rgb.copy()
#     cmp = color_map(10)

#     for obj_label in obj_pose_dict:

#         # node = self.objnode[obj_label]['node']
#         # node.mesh.is_visible = True
#         # depth = self.render.render(self.scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
#         # node.mesh.is_visible = False
#         # mask = np.logical_and(
#         #     (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0.2
#         # )
#         # if np.sum(mask) > 0:

#         _, prediction = posecnn_model({'rgb': rgb})
#         prediction = prediction.cpu().numpy().astype(np.float64)
#         prediction /= prediction.max()
#         prediction = (np.tile(prediction, (3, 1, 1)) * 255).astype(np.uint8)

#         color = self.cmp[obj_label - 1]
#         img[mask, :] = alpha * img[mask, :] + (1 - alpha) * color[:]
#         obj_pose = obj_pose_dict[obj_label]
#         obj_center = self.project2d(self.intrinsic, obj_pose[:3, -1])
#         rgb_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
#         for j in range(3):
#             obj_xyz_offset_2d = self.project2d(self.intrinsic, obj_pose[:3, -1] + obj_pose[:3, j] * 0.001)
#             obj_axis_endpoint = obj_center + (obj_xyz_offset_2d - obj_center) / np.linalg.norm(obj_xyz_offset_2d - obj_center) * axis_len
#             cv2.arrowedLine(img, (int(obj_center[0]), int(obj_center[1])), (int(obj_axis_endpoint[0]), int(obj_axis_endpoint[1])), rgb_colors[j], thickness=2, tipLength=0.15)  
#         return img




class PoseCNN(nn.Module):
    """
    PoseCNN
    """
    def __init__(self, pretrained_backbone, models_pcd, cam_intrinsic):
        super(PoseCNN, self).__init__()

        self.iou_threshold = 0.7
        self.models_pcd = models_pcd
        self.cam_intrinsic = cam_intrinsic

        ######################################################################
        # TODO: Initialize layers and components of PoseCNN.                 #
        #                                                                    #
        # Create an instance of FeatureExtraction, SegmentationBranch,       #
        # TranslationBranch, and RotationBranch for use in PoseCNN           #
        ######################################################################
        
        device = next(pretrained_backbone.parameters()).device

        # 1. Feature Extraction
        self.feature_extractor = FeatureExtraction(pretrained_backbone).to(device)

        # 2. Segmentation
        self.segmentation_branch = SegmentationBranch().to(device)

        # 3. Translation
        self.translation_branch = TranslationBranch().to(device)
        # self.translation_branch = self.translation_branch.to(torch.float32)

        # 4. Rotation
        self.rotation_branch = RotationBranch().to(device)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, input_dict, eval=False):
        """
        input_dict = {
            'rgb', # [batch_size, 3, H, W]
            'depth', # [batch_size, 1, H, W]
            'objs_id', # [batch_size, N]
            # 'mask', # [batch_size, N, H, W]
            'bbx', # [batch_size, N, 4]
            'RTs' # [batch_size, N, 3, 4]
        }
        """


        if not eval:
            loss_dict = {
                "loss_segmentation": 0,
                "loss_centermap": 0,
                "loss_R": 0
            }


            gt_bbx = self.getGTbbx(input_dict)
            gt_label = input_dict['label']

            ######################################################################
            # TODO: Implement PoseCNN's forward pass for training.               #
            #                                                                    #
            # Model should extract features, segment the objects, identify roi   #
            # object bounding boxes, and predict rotation and translations for   #
            # each roi box.                                                      #
            #                                                                    #
            # The training loss for semantic segmentation should be stored in    #
            # loss_dict["loss_segmentation"] and calculated using the            #
            # loss_cross_entropy(.) function.                                    #
            #                                                                    #
            # The training loss for translation should be stored in              #
            # loss_dict["loss_centermap"] using the L1loss function.             #
            #                                                                    #
            # The training loss for rotation should be stored in                 #
            # loss_dict["loss_R"] using the given loss_Rotation function.        #
            ######################################################################
            # Important: the rotation loss should be calculated only for regions
            # of interest that match with a ground truth object instance.
            # Note that the helper function, IOUselection, may be used for 
            # identifying the predicted regions of interest with acceptable IOU 
            # with the ground truth bounding boxes.
            # If no ROIs result from the selection, don't compute the loss_R
            
            # 1. Feature Extraction
            
            # print("input_dict[\"rgb\"] {} {}".format(input_dict["rgb"].shape, input_dict["rgb"].dtype))
            # print(self.feature_extractor)
            # for param in self.feature_extractor.embedding2.children():
            #     try:
            #         print(param.weight.dtype)
            #         print(param.bias.dtype)
            #     except:
            #         pass
            
            feature1, feature2 = self.feature_extractor(input_dict)
            # print("feature1 {} {}".format(feature1.shape, feature1.dtype))
            # print("feature2 {} {}".format(feature2.shape, feature2.dtype))

            # self.feature_extractor = self.feature_extractor.to(torch.float32)
            # feature1, feature2 = self.feature_extractor(input_dict)
            # # feature1 = feature1.to(torch.float32)
            # # feature2 = feature2.to(torch.float32)
            # print("feature1 {} {}".format(feature1.shape, feature1.dtype))
            # print("feature2 {} {}".format(feature2.shape, feature2.dtype))

            # 2. Segmentation
            pred_prob, pred_seg, pred_bbx = self.segmentation_branch(feature1, feature2)
            # print("pred_prob {} pred_seg {} pred_bbx {}".format(pred_prob.shape, pred_seg.shape, pred_bbx.shape))
            # Seg Loss
            loss_segmentation = loss_cross_entropy(pred_prob, gt_label)

            # # Filter Predicted Boxes
            filter_bbx = IOUselection(pred_bbx, gt_bbx, self.iou_threshold)

            # 3. Translation
            translation = self.translation_branch(feature1, feature2)

            # Trans Loss
            loss_centermap = F.l1_loss(translation, input_dict['centermaps'].to(torch.float16))
            
            if filter_bbx.shape[0] > 0:
                # 4. Rotation
                rotation = self.rotation_branch(feature1, feature2, filter_bbx[:,:-1])
                pred_rot, pred_label = self.estimateRotation(rotation, filter_bbx)
                gt_rot = self.gtRotation(filter_bbx, input_dict)
                # Rot Loss
                loss_R = loss_Rotation(pred_rot, gt_rot, pred_label, self.models_pcd)
            else:
                loss_R = 0

            loss_dict = {
                "loss_segmentation": loss_segmentation,
                "loss_centermap": loss_centermap,
                "loss_R": loss_R,
            }

            ######################################################################
            #                            END OF YOUR CODE                        #
            ######################################################################
            
            return loss_dict
        else:
            output_dict = None
            segmentation = None

            with torch.no_grad():
                ######################################################################
                # TODO: Implement PoseCNN's forward pass for inference.              #
                ######################################################################
                # gt_bbx = self.getGTbbx(input_dict)

                # 1. Features
                feature1, feature2 = self.feature_extractor(input_dict)

                # 2. Segmentation
                pred_prob, pred_seg, pred_bbx = self.segmentation_branch(feature1, feature2)

                # Filter Predicted Boxes
                # filter_bbx = IOUselection(pred_bbx, gt_bbx, self.iou_threshold)

                # 3. Translation
                translation = self.translation_branch(feature1, feature2)
                # pred_trans = self.estimateTrans(translation, filter_bbx, pred_seg)
                pred_centers, pred_depths = HoughVoting(pred_seg, translation)
                
                if pred_bbx.shape[0] > 0:
                    # 4. Rotation
                    rotation = self.rotation_branch(feature1, feature2, pred_bbx.float()[:,:-1])
                    pred_rot, label = self.estimateRotation(rotation, pred_bbx.float())

                    # Output
                    segmentation = pred_seg
                    output_dict = self.generate_pose(pred_rot, pred_centers, pred_depths, pred_bbx)
                else:
                    print("No bounding boxes")
                    segmentation = pred_seg
                    output_dict = {0: {}}

                
                # feature1, feature2 = self.feat_extract(input_dict)
                # probability, segmentation, bbx = self.seg_branch(feature1, feature2)

                # centermaps = self.trans_branch(feature1, feature2)

                # quaternion = self.rot_branch(feature1, feature2, bbx[:, :5].float())
                # pred_Rs, label = self.estimateRotation(quaternion, bbx)
                
                # pred_centers, pred_depths = self.HoughVoting(segmentation, centermaps)

                # output_dict = self.generate_pose(pred_Rs, pred_centers, pred_depths, bbx.to(pred_Rs.device))

                ######################################################################
                #                            END OF YOUR CODE                        #
                ######################################################################

            return output_dict, segmentation
    
    def estimateTrans(self, translation_map, filter_bbx, pred_label):
        """
        translation_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        pred_label: a tensor [batch_size, num_classes, height, width]
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            trans_map = translation_map[batch_id, (cls-1) * 3 : cls * 3, :]
            label = (pred_label[batch_id] == cls).detach()
            pred_T = trans_map[:, label].mean(dim=1)
            pred_Ts[idx] = pred_T
        return pred_Ts

    def gtTrans(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Ts[idx] = input_dict['RTs'][batch_id][cls - 1][:3, [3]].T
        return gt_Ts 

    def getGTbbx(self, input_dict):
        """
            bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
        """
        gt_bbx = []
        objs_id = input_dict['objs_id']
        device = objs_id.device
        ## [x_min, y_min, width, height]
        bbxes = input_dict['bbx']
        for batch_id in range(bbxes.shape[0]):
            for idx, obj_id in enumerate(objs_id[batch_id]):
                if obj_id.item() != 0:
                    # the obj appears in this image
                    bbx = bbxes[batch_id][idx]
                    gt_bbx.append([batch_id, bbx[0].item(), bbx[1].item(),
                                  bbx[0].item() + bbx[2].item(), bbx[1].item() + bbx[3].item(), obj_id.item()])
        return torch.tensor(gt_bbx).to(device=device, dtype=torch.int16)
        
    def estimateRotation(self, quaternion_map, filter_bbx):
        """
        quaternion_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Rs = torch.zeros(N_filter_bbx, 3, 3)
        label = []
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            quaternion = quaternion_map[idx, (cls-1) * 4 : cls * 4]
            quaternion = nn.functional.normalize(quaternion, dim=0)
            pred_Rs[idx] = quaternion_to_matrix(quaternion)
            label.append(cls)
        label = torch.tensor(label)
        return pred_Rs, label

    def gtRotation(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Rs = torch.zeros(N_filter_bbx, 3, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Rs[idx] = input_dict['RTs'][batch_id][cls - 1][:3, :3]
        return gt_Rs 

    def generate_pose(self, pred_Rs, pred_centers, pred_depths, bbxs):
        """
        pred_Rs: a tensor [pred_bbx_size, 3, 3]
        pred_centers: [batch_size, num_classes, 2]
        pred_depths: a tensor [batch_size, num_classes]
        bbx: a tensor [pred_bbx_size, 6]
        """        
        output_dict = {}
        for idx, bbx in enumerate(bbxs):
            bs, _, _, _, _, obj_id = bbx
            R = pred_Rs[idx].numpy()
            center = pred_centers[bs, obj_id - 1].detach().cpu().numpy()
            depth = pred_depths[bs, obj_id - 1].detach().cpu().numpy()
            if (center**2).sum().item() != 0:
                T = np.linalg.inv(self.cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
                T = T[:, np.newaxis]
                if bs.item() not in output_dict:
                    output_dict[bs.item()] = {}
                output_dict[bs.item()][obj_id.item()] = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))
        return output_dict









# class Visualize:
#     def __init__(self, object_dict, cam_intrinsic, resolution):
#         '''
#         object_dict is a dict store object labels, object names and object model path, 
#         example:
#         object_dict = {
#                     1: ["beaker_1", path]
#                     2: ["dropper_1", path]
#                     3: ["dropper_2", path]
#                 }
#         '''
#         self.objnode = {}
#         self.render = pyrender.OffscreenRenderer(resolution[0], resolution[1])
#         self.scene = pyrender.Scene()
#         cam = pyrender.camera.IntrinsicsCamera(cam_intrinsic[0, 0],
#                                                cam_intrinsic[1, 1], 
#                                                cam_intrinsic[0, 2], 
#                                                cam_intrinsic[1, 2], 
#                                                znear=0.05, zfar=100.0, name=None)
#         self.intrinsic = cam_intrinsic
#         Axis_align = np.array([[1, 0, 0, 0],
#                                 [0, -1, 0, 0],
#                                 [0, 0, -1, 0],
#                                 [0, 0, 0, 1]])
#         self.nc = pyrender.Node(camera=cam, matrix=Axis_align)
#         self.scene.add_node(self.nc)

#         for obj_label in object_dict:
#             objname = object_dict[obj_label][0]
#             objpath = object_dict[obj_label][1]
#             tm = trimesh.load(objpath)
#             mesh = pyrender.Mesh.from_trimesh(tm, smooth = False)
#             node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
#             node.mesh.is_visible = False
#             self.objnode[obj_label] = {"name":objname, "node":node, "mesh":tm}
#             self.scene.add_node(node)
#         self.cmp = self.color_map(N=len(object_dict))
#         self.object_dict = object_dict

#     def vis_oneview(self, ipt_im, obj_pose_dict, alpha = 0.5, axis_len=30):
#         '''
#         Input:
#             ipt_im: numpy [H, W, 3]
#                 input image
#             obj_pose_dict:
#                 is a dict store object poses within input image
#                 example:
#                 poselist = {
#                     15: numpy_pose 4X4,
#                     37: numpy_pose 4X4,
#                     39: numpy_pose 4X4,
#                 }
#             alpha: float [0,1]
#                 alpha for labels' colormap on image 
#             axis_len: int
#                 pixel lengths for draw axis
#         '''
#         img = ipt_im.copy()
#         for obj_label in obj_pose_dict:
#             if obj_label in self.object_dict:
#                 pose = obj_pose_dict[obj_label]
#                 node = self.objnode[obj_label]['node']
#                 node.mesh.is_visible = True
#                 self.scene.set_pose(node, pose=pose)
#         full_depth = self.render.render(self.scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
#         for obj_label in obj_pose_dict:
#             if obj_label in self.object_dict:
#                 node = self.objnode[obj_label]['node']
#                 node.mesh.is_visible = False
#         for obj_label in self.object_dict:
#             node = self.objnode[obj_label]['node']
#             node.mesh.is_visible = False
#         for obj_label in obj_pose_dict:
#             if obj_label in self.object_dict:
#                 node = self.objnode[obj_label]['node']
#                 node.mesh.is_visible = True
#                 depth = self.render.render(self.scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
#                 node.mesh.is_visible = False
#                 mask = np.logical_and(
#                     (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0.2
#                 )
#                 if np.sum(mask) > 0:
#                     color = self.cmp[obj_label - 1]
#                     img[mask, :] = alpha * img[mask, :] + (1 - alpha) * color[:]
#                     obj_pose = obj_pose_dict[obj_label]
#                     obj_center = self.project2d(self.intrinsic, obj_pose[:3, -1])
#                     rgb_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
#                     for j in range(3):
#                         obj_xyz_offset_2d = self.project2d(self.intrinsic, obj_pose[:3, -1] + obj_pose[:3, j] * 0.001)
#                         obj_axis_endpoint = obj_center + (obj_xyz_offset_2d - obj_center) / np.linalg.norm(obj_xyz_offset_2d - obj_center) * axis_len
#                         cv2.arrowedLine(img, (int(obj_center[0]), int(obj_center[1])), (int(obj_axis_endpoint[0]), int(obj_axis_endpoint[1])), rgb_colors[j], thickness=2, tipLength=0.15)  
#         return img

#     def color_map(self, N=256, normalized=False):
#         def bitget(byteval, idx):
#             return ((byteval & (1 << idx)) != 0)
#         dtype = 'float32' if normalized else 'uint8'
#         cmap = np.zeros((N, 3), dtype=dtype)
#         for i in range(N):
#             r = g = b = 0
#             c = i
#             for j in range(8):
#                 r = r | (bitget(c, 0) << 7-j)
#                 g = g | (bitget(c, 1) << 7-j)
#                 b = b | (bitget(c, 2) << 7-j)
#                 c = c >> 3
#             cmap[i] = np.array([r, g, b])
#         cmap = cmap/255 if normalized else cmap
#         return cmap
    
#     def project2d(self, intrinsic, point3d):
#         return (intrinsic @ (point3d / point3d[2]))[:2]
