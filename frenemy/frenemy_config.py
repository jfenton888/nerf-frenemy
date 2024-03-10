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
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

frenemy_method = MethodSpecification(
    config=TrainerConfig(
        method_name="frenemy-model",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=64,
            ),
            # datamanager=FullImageDatamanagerConfig(
            #     dataparser=NerfstudioDataParserConfig(),
            #     cache_images_type="uint8",
            # ),
            model=FrenemyModelConfig(
                nerf_config=NerfactoModelConfig(
                    eval_num_rays_per_chunk=1 << 15,
                    average_init_density=0.01,
                    camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                ),
                nerf_path=Path("outputs/poster/nerfacto/2024-02-15_133222/nerfstudio_models/"),
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
    description="NeRF-Frenemy template that uses NeRFacto.",
)


# nerfacto2 = MethodSpecification(
#     TrainerConfig(
#         method_name="nerfacto2",
#         steps_per_eval_batch=500,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=True,
#         pipeline=VanillaPipelineConfig(
#             datamanager=FullImageDatamanagerConfig(
#                 dataparser=NerfstudioDataParserConfig(),
#                 cache_images_type="uint8",
#             ),
#             # datamanager=ParallelDataManagerConfig(
#             #     dataparser=NerfstudioDataParserConfig(),
#             #     train_num_rays_per_batch=4096,
#             #     eval_num_rays_per_batch=4096,
#             # ),
#             model=NerfactoModelConfig(
#                 eval_num_rays_per_chunk=1 << 15,
#                 average_init_density=0.01,
#                 camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
#             ),
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#             "fields": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#             "camera_opt": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="NeRFacto2, which uses FullImageDatamanager.",
# )