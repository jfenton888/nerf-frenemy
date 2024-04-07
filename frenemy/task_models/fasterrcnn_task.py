




# Task model related configs
from dataclasses import dataclass, field
from typing import Type

from frenemy.task_models.base_task import Task, TaskConfig

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights # maskrcnn_resnet50_fpn

@dataclass
class FasterRCNNTaskConfig(TaskConfig):
    """Configuration for task instantiation"""

    _target: Type = field(default_factory=lambda: FasterRCNNTask)
    """target class to instantiate"""



class FasterRCNNTask(Task):
    """Task class

    Args:
        config: configuration for instantiating model
    """

    config: FasterRCNNTaskConfig
    
    def populate_modules(self):
        """Populate the modules of the model."""
        
        self.task_weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        self.task_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.task_weights) # maskrcnn_resnet50_fpn(pretrained=True)


        # Step 1: Initialize model with the best available weights
        self.task_model.eval()

        # Step 2: Initialize the inference transforms
        self.task_preprocess = self.task_weights.transforms()
    
    
    def get_metrics_dict(self, metrics_dict, outputs, batch):
        """Compute the metrics of the model."""
        
        # patch_images = torch.unique(batch["indices"][:,0])
        # assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"

        image = batch["full_image"].to(self.device)#[patch_images[0]].to(self.device)
        modified_image = image.clone()

        modified_image[batch["indices"][:,1], batch["indices"][:,2], :] = outputs["rgb"]


        # Step 3: Apply inference preprocessing transforms
        target_batch = [self.task_preprocess(image.permute(2,0,1))]


        # Step 4: Use the model and visualize the prediction
        self.task_model.eval()
        target = self.task_model(target_batch)[0]

        # Predict for the image
        self.task_model.train()
        img = self.task_preprocess(modified_image.permute(2,0,1))

        detection_outputs = self.task_model(img.unsqueeze(0), [target])
        # print(detection_outputs)

        task_loss = sum(loss for loss in detection_outputs.values())
        # print(task_loss)

        metrics_dict["task"] = task_loss

    
        
    def get_loss_dict(self, loss_dict, outputs, batch, metrics_dict=None):
        """Compute the loss of the model."""
        
        # patch_images = torch.unique(batch["indices"][:,0])
        # assert patch_images.size(0) == 1, f"All indices in batch must be from same image, but found {patch_images}"

        image = batch["full_image"].to(self.device)#[patch_images[0]].to(self.device)
        modified_image = image.clone()

        modified_image[batch["indices"][:,1], batch["indices"][:,2], :] = outputs["rgb"]

        # loss_dict["rgb_loss"] = self.rgb_loss(image, modified_image)

        # Step 3: Apply inference preprocessing transforms
        target_batch = [self.task_preprocess(image.permute(2,0,1))]

        # Step 4: Use the model and visualize the prediction
        self.task_model.eval()
        target = self.task_model(target_batch)[0]

        # Predict for the image
        self.task_model.train()
        img = self.task_preprocess(modified_image.permute(2,0,1))

        detection_outputs = self.task_model(img.unsqueeze(0), [target])
        # print(detection_outputs)

        task_loss = sum(loss for loss in detection_outputs.values())
        # print(task_loss)

        loss_dict["task"] = task_loss

    
