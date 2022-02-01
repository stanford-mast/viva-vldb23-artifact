import os
import sys

from urllib import request

from viva.udfs.inference import animal_model_udf
from viva.inference.abstract_inference import AbstractInference

class AnimalDetectInference(AbstractInference):
    """
    Class for animal detection inference
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        def animal_fn():
            remote_path = 'https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb'
            local_path = '/tmp/megadetector_v4_1_0.pb'
            if not os.path.exists(local_path):
                request.urlretrieve(remote_path, local_path)

            return local_path

        return animal_model_udf(animal_fn)

    def _get_model_udf(self):
        return self.model_udf
