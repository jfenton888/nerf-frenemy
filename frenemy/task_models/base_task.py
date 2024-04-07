
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type
from nerfstudio.configs.base_config import InstantiateConfig

import torch
from torch import nn


# Task model related configs
@dataclass
class TaskConfig(InstantiateConfig):
    """Configuration for task instantiation"""

    _target: Type = field(default_factory=lambda: Task)
    """target class to instantiate"""


class Task(nn.Module):
    """Task class

    Args:
        config: configuration for instantiating task model
    """

    config: TaskConfig

    def __init__(
        self,
        config: TaskConfig,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.kwargs = kwargs

        self.populate_modules()  # populate the modules of the task model
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device
    
    @abstractmethod
    def populate_modules(self):
        """Populate the modules of the model."""
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        raise NotImplementedError
    
    @abstractmethod
    def get_metrics_dict(self, metrics_dict, outputs, batch):
        """Compute the metrics of the model."""
        raise NotImplementedError
        
    @abstractmethod
    def get_loss_dict(self, loss_dict, outputs, batch, metrics_dict=None):
        """Compute the loss of the model."""
        raise NotImplementedError
    

