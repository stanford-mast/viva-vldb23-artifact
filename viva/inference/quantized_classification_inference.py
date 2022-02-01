import os
import sys

import torch
from torchvision.models.quantization import resnet18, ResNet18_QuantizedWeights

from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from viva.udfs.inference import qclassification_model_udf
from viva.inference.abstract_inference import AbstractInference

class QuantizedClassificationInference(AbstractInference):
    """
    Class for classification inference
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        weights = ResNet18_QuantizedWeights.DEFAULT
        model = resnet18(weights=weights, quantize=True)
        bc_resnet18_state = self.sc.broadcast(model.state_dict())

        def resnet18_fn():
            """Gets the broadcasted model."""
            weights = ResNet18_QuantizedWeights.DEFAULT
            model = resnet18(weights=weights, quantize=True)
            model.load_state_dict(bc_resnet18_state.value)
            preprocess = weights.transforms()
            return model, preprocess, weights

        return qclassification_model_udf(resnet18_fn)

    def _get_model_udf(self):
        return self.model_udf
