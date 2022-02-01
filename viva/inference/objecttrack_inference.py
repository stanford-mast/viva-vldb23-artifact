import os
import sys

import torch

from viva.inference.abstract_inference import AbstractInference
from viva.udfs.inference import tracking_model_udf
from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

sys.path.append(os.path.join(os.path.dirname(__file__), '../udfs/deep_sort/deep_sort/deep/reid'))
from viva.udfs.deep_sort.deep_sort.utils.parser import get_config
from viva.udfs.deep_sort.deep_sort import DeepSort

class ObjectTrackInference(AbstractInference):
    """
    Class for object detection inference
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
        device = torch.device('cuda' if use_cuda else 'cpu')

        def deepsort_fn():
            cfg = get_config()
            config = 'viva/udfs/deep_sort/deep_sort/configs/deep_sort.yaml'

            cfg.merge_from_file(config)
            model = DeepSort(
                'osnet_x0_25',
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
            return model

        return tracking_model_udf(deepsort_fn)

    def _get_model_udf(self):
        return self.model_udf
