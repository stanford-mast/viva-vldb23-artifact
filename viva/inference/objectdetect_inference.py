import os
import sys
import torch

from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from viva.udfs.inference import yolo_model_udf
from viva.inference.abstract_inference import AbstractInference

class ObjectDetectInference(AbstractInference):
    """
    Class for object detection inference
    """

    def __init__(self, size = 'yolov5s'):
        super().__init__()
        self.model_udf = self._prepare_model_udf(size)

    def _prepare_model_udf(self, size = 'yolov5s'):
        yolov5 = torch.hub.load('ultralytics/yolov5', size, pretrained=True, verbose=False)
        bc_yolov5_state = self.sc.broadcast(yolov5.state_dict())
        use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
        device = torch.device('cuda' if use_cuda else 'cpu')

        def yolo_fn():
            """Gets the broadcasted model."""
            model = torch.hub.load('ultralytics/yolov5', size, pretrained=True, verbose=False)
            model.load_state_dict(bc_yolov5_state.value)
            model.eval()
            model.to(device)
            return model

        return yolo_model_udf(yolo_fn)

    def _get_model_udf(self):
        return self.model_udf
