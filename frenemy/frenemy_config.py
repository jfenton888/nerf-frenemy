"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations
from pathlib import Path

from frenemy.frenemy_model import FrenemyModelConfig
from frenemy.posecnn_dataset import PoseCNNDataset
from frenemy.propspose_dataparser import PROPSPoseDataParserConfig
from frenemy.propspose_segmentation_dataparser import PROPSPoseSegmentationDataParserConfig
from frenemy.posecnn_segmentation_dataset import PoseCNNSegmentationDataset
from frenemy.task_models.posecnn_segmentation_task import PoseCNNSegmentationTaskConfig
from frenemy.task_models.posecnn_task import PoseCNNTaskConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

frenemy_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfrenemy",
        steps_per_eval_batch=100,
        steps_per_eval_image=100,
        steps_per_save=500,
        max_num_iterations=5000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=ParallelDataManager[PoseCNNDataset],
                dataparser=PROPSPoseDataParserConfig(
                    train_split_fraction=0.8
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=64,
                # This needs to be set to True in order to comapare task performance to the original image 
                # if that were removed this could be set to False to speed up runtime
                keep_full_image=True,
            ),
            model=FrenemyModelConfig(
                nerf=NerfactoModelConfig(
                    eval_num_rays_per_chunk=1 << 15,
                    average_init_density=0.01,
                    camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                    near_plane=0.5,
                ),
                nerf_path=Path("outputs/PROPS-NeRF/nerfacto/2024-04-16_122316/nerfstudio_models/"), # Set in command line by "--pipeline.model.nerf-path"
                task_model=PoseCNNTaskConfig(
                    model_path=Path("../nerfrenemy/frenemy/PoseCNN/posecnn_model.pth"), # Set in command line by "--pipeline.model.task-model.model-path"
                    dataset_dir=Path("../datasets/PROPS-NeRF"), # Set in command line by "--pipeline.model.task-model.dataset-dir"
                ),
            ),
        ),
        optimizers={
            "perturbation_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15), # LR of 1e-2 works well for fast training (100-300 iterations)
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRF-Frenemy template that uses NeRFacto.",
)


frenemy_segmentation_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfrenemy-segmentation",
        steps_per_eval_batch=100,
        steps_per_eval_image=100,
        steps_per_save=500,
        max_num_iterations=5000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=ParallelDataManager[PoseCNNSegmentationDataset],
                dataparser=PROPSPoseSegmentationDataParserConfig(
                    train_split_fraction=0.8
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=64,
                # This needs to be set to True in order to comapare task performance to the original image 
                # if that were removed this could be set to False to speed up runtime
                keep_full_image=True, 
            ),
            model=FrenemyModelConfig(
                nerf=NerfactoModelConfig(
                    eval_num_rays_per_chunk=1 << 15,
                    average_init_density=0.01,
                    camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                    near_plane=0.5,
                ),
                nerf_path=Path("outputs/poster/nerfacto/2024-02-15_133222/nerfstudio_models/"),
                task_model=PoseCNNSegmentationTaskConfig(),
            ),
        ),
        optimizers={
            "perturbation_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15), # LR of 1e-2 works well for fast training (100-300 iterations)
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRF-Frenemy template that uses NeRFacto.",
)


nerfacto_props = MethodSpecification(
    TrainerConfig(
        method_name="nerfacto-props",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=ParallelDataManager[PoseCNNDataset],
                dataparser=PROPSPoseDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRFacto2, which uses The PROPS-Pose datset to test that it works.",
)