from abc import ABC
from typing import List, Callable

from pyspark.sql.functions import col, lit

def filt_to_str(func, args=None):
    if args is None:
        return func.__name__
    else:
        if isinstance(args, list):
            args = list(map(str, args))
        return func.__name__ + ':' + '.'.join(args)

class Node(ABC):
    def __init__(self, in_columns: List[str], out_column: str,
                 operator: Callable[[], None] = None, in_literals: List[str] = []):
        self._in_columns = in_columns
        self._out_column = out_column
        self._operator = operator
        self._in_literals = in_literals
        self._filters = []

    def __repr__(self):
        if self._filters:
            return f'{self._out_column}->{self.str_filters()}'
        return f'{self._out_column}'

    def apply_op(self, df):
        if self.operator is None:
            return df

        incols = list(map(lambda x: col(x), self.in_columns))
        inlits = list(map(lambda x: lit(x), self.in_literals))

        return df.select('*', self.operator(*incols, *inlits).alias(self.out_column)).cache()

    def apply_filters(self, df):
        for func, args in self._filters:
            df = func(df) if args is None else func(df, *args)
        return df

    def add_filter(self, filter_func, args=None):
        self._filters.append((filter_func, args))

    def pop_front_filter(self):
        if len(self._filters) > 0:
            self._filters = self._filters[1:]

    def str_filters(self):
        allf = []
        for func, args in self._filters:
            if func.__name__ == 'explode_preds':
                continue
            allf.append(filt_to_str(func, args))
        strf = '_&_'.join(allf)

        return strf

    @property
    def filters(self):
        return self._filters

    @property
    def in_columns(self):
        return self._in_columns

    @property
    def out_column(self):
        return self._out_column

    @property
    def operator(self):
        return self._operator

    @property
    def in_literals(self):
        return self._in_literals
