import os
import sys

from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

import torch
from img2vec_pytorch import Img2Vec

from viva.udfs.inference import img2vec_model_udf
from viva.inference.abstract_inference import AbstractInference

class Img2VecInference(AbstractInference):
    """
    Class for image to vector using img2vec (ResNet-18)
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        def img2vec_fn():
            use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
            model = Img2Vec(cuda=use_cuda)
            return model

        return img2vec_model_udf(img2vec_fn)

    def _get_model_udf(self):
        return self.model_udf
