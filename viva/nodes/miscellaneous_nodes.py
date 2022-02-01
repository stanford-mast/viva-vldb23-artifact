from typing import List

from viva.nodes.node import Node
from viva.udfs.compvision import (
    image_similarity, image_brightness, svm_classification, motion_detect,
    executor_overhead, simple_transfer, complex_transfer, data_generator
)

class TranscriptSearchNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'transcriptsearch'
        operator = None
        super().__init__(in_columns, out_column, operator, [])

class SimilarityNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'similarity'
        operator = image_similarity
        super().__init__(in_columns, out_column, operator, [])

class BrightnessNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'brightness'
        operator = image_brightness
        super().__init__(in_columns, out_column, operator, [])

class SVMNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'svm'
        operator = svm_classification
        super().__init__(in_columns, out_column, operator, [])

class MotionNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'motion'
        operator = motion_detect
        super().__init__(in_columns, out_column, operator, [])

class SimpleTransferNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'simple_transfer'
        operator = simple_transfer
        super().__init__(in_columns, out_column, operator, [])

class OverheadNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'overhead'
        operator = executor_overhead
        super().__init__(in_columns, out_column, operator, [])

class ComplexTransferNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'transfer'
        operator = complex_transfer
        super().__init__(in_columns, out_column, operator, [])

class DataGeneratorNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'data_generator'
        operator = data_generator
        super().__init__(in_columns, out_column, operator, [])
