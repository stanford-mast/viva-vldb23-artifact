import os
import sys
import itertools

from viva.hints.abstract_hint import AbstractHint

from typing import List, Any, Type, Tuple

class SupersetHint(AbstractHint):
    """
    Class for superset hint

    left_column SUPERSET right_column
    """

    def __init__(self, left_col_arg: str, right_col_arg: str):
        super().__init__('SS', left_col_arg, right_col_arg)

    def __eq__(self, other):
        return isinstance(other, SupersetHint) and \
                (self.left_col_arg == other.left_col_arg and \
                self.right_col_arg == other.right_col_arg)
