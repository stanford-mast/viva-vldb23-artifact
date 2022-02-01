from typing import List

from viva.nodes.node import Node
from viva.inference.actiondetect_inference import ActionDetectInference
from viva.inference.classification_inference import ClassificationInference
from viva.inference.quantized_classification_inference import QuantizedClassificationInference
from viva.inference.proxyclassification_inference import ProxyClassificationInference
from viva.inference.objectdetect_inference import ObjectDetectInference
from viva.inference.emotiondetect_inference import EmotionDetectInference
from viva.inference.facerecognition_inference import FaceRecognitionInference
from viva.inference.animaldetect_inference import AnimalDetectInference
from viva.inference.objecttrack_inference import ObjectTrackInference
from viva.inference.img2vec_inference import Img2VecInference
from viva.inference.kmeans_inference import KMeansInference
from viva.inference.deepface_inference import (
    DeepFaceInference, DeepFacePrefixInference, DeepFaceSuffixInference
)

class ObjectDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'objectdetect'
        size = 'yolov5s'
        operator = ObjectDetectInference(size).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ObjectDetectNodeNano(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'objectdetect_nano'
        size = 'yolov5n'
        operator = ObjectDetectInference(size).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ObjectDetectNodeMedium(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'objectdetect_medium'
        size = 'yolov5m'
        operator = ObjectDetectInference(size).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ObjectDetectNodeLarge(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'objectdetect_large'
        size = 'yolov5l'
        operator = ObjectDetectInference(size).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ObjectDetectNodeXLarge(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'objectdetect_xlarge'
        size = 'yolov5x'
        operator = ObjectDetectInference(size).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class FaceDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'facedetect'
        operator = FaceRecognitionInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class EmotionDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'emotiondetect'
        operator = EmotionDetectInference(use_mtcnn=True).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class EmotionDetectNodeCascades(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'emotiondetect_cascades'
        operator = EmotionDetectInference(use_mtcnn=False).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ClassificationNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'classification'
        operator = ClassificationInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class QuantizedClassificationNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'qclassification'
        operator = QuantizedClassificationInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ProxyClassificationNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'proxyclassification'
        operator = ProxyClassificationInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class AnimalDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'animaldetect'
        operator = AnimalDetectInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ObjectTrackNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'objecttrack'
        operator = ObjectTrackInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class ActionDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'actiondetect'
        operator = ActionDetectInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class Img2VecNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'img2vec'
        operator = Img2VecInference().model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class TASTIObjectDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'objectdetect_tasti'
        operator = KMeansInference('objectdetect').model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class TASTIEmotionDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'emotiondetect_tasti'
        operator = KMeansInference('emotiondetect').model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class TASTIFaceDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'facedetect_tasti'
        operator = KMeansInference('facedetect').model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class TASTIActionDetectNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'actiondetect_tasti'
        operator = KMeansInference('actiondetect').model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class DeepFaceNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = [], model_type = 'Age'):
        out_column = 'deepface'+model_type
        operator = DeepFaceInference(model_type).model_udf
        #print('Initialized DeepFaceNode!! with model_type %s, out_column %s'%(model_type, out_column ))
        super().__init__(in_columns, out_column, operator, in_literals)

class DeepFacePrefixNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = [], layer_id = 31):
        out_column = 'dfprefixembed'
        operator = DeepFacePrefixInference(layer_id).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)

class DeepFaceSuffixNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = [], model_type = 'Age', layer_id = 32):
        out_column = 'deepfaceSuffix'+model_type
        operator = DeepFaceSuffixInference(model_type, layer_id).model_udf
        super().__init__(in_columns, out_column, operator, in_literals)
