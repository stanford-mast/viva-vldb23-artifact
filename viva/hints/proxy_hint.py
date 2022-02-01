import os
import sys
import itertools

from viva.hints.abstract_hint import AbstractHint

from typing import List, Any, Type, Tuple

class ProxyHint(AbstractHint):
    """
    Class for proxy hint

    left_column PROXY right_column
    """

    def __init__(self, left_col_arg: str, right_col_arg: str, confidence: float):
        super().__init__('PX', left_col_arg, right_col_arg)
        self.confidence = confidence

    def get_confidence(self):
        return self.confidence

    def __eq__(self, other):
        return isinstance(other, ProxyHint) and \
                (self.left_col_arg == other.left_col_arg and \
                self.right_col_arg == other.right_col_arg)
