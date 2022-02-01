import os
import sys
import itertools

import pandas as pd

from viva.hints.abstract_hint import AbstractHint

from typing import List, Any, Type

class EqualsHint(AbstractHint):
    """
    Class for equals hint

    left_column EQUALS right_column
    """

    def __init__(self, left_col_arg: str, right_col_arg: str):
        super().__init__('EQ', left_col_arg, right_col_arg)

    def __eq__(self, other):
        return isinstance(other, EqualsHint) and \
                (self.left_col_arg == other.left_col_arg and \
                self.right_col_arg == other.right_col_arg) or \
                (self.left_col_arg == other.right_col_arg and \
                self.right_col_arg == other.left_col_arg)
