import os
import sys

import torch
from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from viva.udfs.inference import action_model_udf
from viva.inference.abstract_inference import AbstractInference

class ActionDetectInference(AbstractInference):
    """
    Class for action detection inference
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        action = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True, verbose=False)
        bc_action_state = self.sc.broadcast(action.state_dict())
        use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
        device = torch.device('cuda' if use_cuda else 'cpu')

        def action_fn():
            """Gets the broadcasted model."""
            model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True, verbose=False)
            model.load_state_dict(bc_action_state.value)
            model.to(device)
            model.eval()
            return model

        return action_model_udf(action_fn)

    def _get_model_udf(self):
        return self.model_udf
