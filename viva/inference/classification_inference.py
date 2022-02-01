import os
import sys

import torch
from torchvision import models

from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from viva.udfs.inference import imagenet_model_udf
from viva.inference.abstract_inference import AbstractInference

class ClassificationInference(AbstractInference):
    """
    Class for classification inference
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        resnet50 = models.resnet50(pretrained=True)
        bc_resnet50_state = self.sc.broadcast(resnet50.state_dict())

        def resnet50_fn():
            """Gets the broadcasted model."""
            model = models.resnet50(pretrained=False)
            model.load_state_dict(bc_resnet50_state.value)
            return model

        return imagenet_model_udf(resnet50_fn)

    def _get_model_udf(self):
        return self.model_udf
