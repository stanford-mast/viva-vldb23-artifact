from deepface import DeepFace
from viva.udfs.inference import deepface_model_udf, deepface_prefix_model_udf, deepface_suffix_model_udf
from viva.inference.abstract_inference import AbstractInference

from viva.utils.tf_helpers import split_tf_model

class DeepFaceInference(AbstractInference):
    def __init__(self, model_type='Age'):
        super().__init__()
        self.model_udf = self._prepare_model_udf(model_type)

    def _prepare_model_udf(self, model_type='Age'):
        deepface_model = DeepFace.build_model(model_type)
        return deepface_model_udf(deepface_model, model_type)

    def _get_model_udf(self):
        return self.model_udf

# No need to pass a model_type here since the prefix model is shared among all types
class DeepFacePrefixInference(AbstractInference):
    def __init__(self, layer_id = 31):
        super().__init__()
        self.model_udf = self._prepare_model_udf(layer_id)

    def _prepare_model_udf(self, layer_id=31):
        deepface_model = DeepFace.build_model('Age')
        model_pre, model_post = split_tf_model(deepface_model, layer_id)
        return deepface_prefix_model_udf(model_pre)

    def _get_model_udf(self):
        return self.model_udf

class DeepFaceSuffixInference(AbstractInference):
    def __init__(self, model_type='Age', layer_id = 32):
        super().__init__()
        self.model_udf = self._prepare_model_udf(model_type, layer_id)

    # We split the model in UDF instead of here because PySpark throws error when serialising the suffix model.
    def _prepare_model_udf(self, model_type='Age', layer_id = 32):
        deepface_model = DeepFace.build_model(model_type)
        return deepface_suffix_model_udf(deepface_model, model_type, layer_id)

    def _get_model_udf(self):
        return self.model_udf
