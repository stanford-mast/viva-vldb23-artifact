import os
import sys

import torch
from torchvision import models
from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from viva.udfs.inference import imagenet_model_udf
from viva.inference.abstract_inference import AbstractInference

class ProxyClassificationInference(AbstractInference):
    """
    Class for proxy classification inference (cheap classification)
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        squeezenet1_1 = models.squeezenet1_1(pretrained=True)
        bc_squeezenet1_1_state = self.sc.broadcast(squeezenet1_1.state_dict())

        def squeezenet1_1_fn():
            """Gets the broadcasted model."""
            model = models.squeezenet1_1(pretrained=False)
            model.load_state_dict(bc_squeezenet1_1_state.value)
            model.eval()
            return model

        return imagenet_model_udf(squeezenet1_1_fn)

    def _get_model_udf(self):
        return self.model_udf
