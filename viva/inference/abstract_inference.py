import os
import sys
import pandas as pd
from abc import ABC, abstractmethod

from pyspark.sql import SparkSession
from pyspark.sql.functions import col 

class AbstractInference(ABC):
    """
    An abstract class for inference
    """

    def __init__(self):
        self.sp =  SparkSession.builder.getOrCreate()
        self.sc = self.sp.sparkContext

    @abstractmethod
    def _get_model_udf(self):
        """
        Abstract method to run the prediction
        """

    def _explode_dataframe(self, df: pd.DataFrame, frames_col: str) -> pd.DataFrame:
        """
        Private method for exploding the DataFrame for querying
        """
        df_expand = df.select(frames_col, 'predictions.xmin', 'predictions.ymin',
                                          'predictions.xmax', 'predictions.ymax',
                                          'predictions.label', 'predictions.score')
        df_second_expand = df_expand.selectExpr('%s as %s' % (frames_col, frames_col),
                                                'inline(arrays_zip(xmin,ymin,xmax,ymax,label,score))')
        return df_second_expand

    def predict(self, df: pd.DataFrame, frames_col: str) -> pd.DataFrame:
        model_udf = self._get_model_udf()
        df_result = df.select(frames_col, model_udf(col(frames_col)).alias('predictions'))
        df_explode = self._explode_dataframe(df_result, frames_col)
        return df_explode
