import os
import sys
import pandas as pd
from abc import ABC, abstractmethod

from typing import List, Any

class AbstractHint(ABC):
    """
    An abstract class for hints
    """

    def __init__(self, name: str, left_col_arg: str, right_col_arg: str):
        self.left_col_arg = left_col_arg
        self.right_col_arg = right_col_arg
        self.name = name

    def __repr__(self):
        if self.name == 'PX':
            return f'{self.left_col_arg} {self.name} {self.right_col_arg} {self.confidence}'
        return f'{self.left_col_arg} {self.name} {self.right_col_arg}'

    def get_args(self):
        return self.left_col_arg, self.right_col_arg
