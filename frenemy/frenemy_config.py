"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations
from pathlib import Path

# from frenemy.template_datamanager import (
#     TemplateDataManagerConfig,
# )
# from frenemy.template_model import FrenemyModelConfig, TemplateModelConfig
# from method_template.template_pipeline import (
#     TemplatePipelineConfig,
# )
from frenemy.frenemy_model import FrenemyModelConfig
from frenemy.posecnn_dataset import PoseCNNDataset
from frenemy.propspose_dataparser import PROPSPoseDataParserConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.plugins.types import MethodSpecification

frenemy_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfrenemy",
        steps_per_eval_batch=100,
        steps_per_eval_image=100,
        steps_per_save=500,
        max_num_iterations=2000,
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
            ),
        ),
        optimizers={
            "perturbation_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRF-Frenemy template that uses NeRFacto.",
)





nerfacto2 = MethodSpecification(
    TrainerConfig(
        method_name="nerfacto2",
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
    description="NeRFacto2, which uses FullImageDatamanager.",
)