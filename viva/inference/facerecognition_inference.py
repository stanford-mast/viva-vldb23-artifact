import os
import sys
import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN

from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from viva.udfs.inference import facenet_model_udf
from viva.inference.abstract_inference import AbstractInference

class FaceRecognitionInference(AbstractInference):
    """
    Class for face recognition inference
    """

    def __init__(self):
        super().__init__()
        self.model_udf = self._prepare_model_udf()

    def _prepare_model_udf(self):
        facenet = InceptionResnetV1(pretrained='vggface2')
        bc_facenet_state = self.sc.broadcast(facenet.state_dict())
        use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
        device = torch.device('cuda' if use_cuda else 'cpu')

        def facenet_fn():
            """Gets the broadcasted model."""
            model = InceptionResnetV1(pretrained='vggface2')
            model.load_state_dict(bc_facenet_state.value)
            model.eval()
            model.classify = True
            model.to(device)

            mtcnn = MTCNN(keep_all=True, device=device)

            return model, mtcnn

        return facenet_model_udf(facenet_fn)

    def _get_model_udf(self):
        return self.model_udf
