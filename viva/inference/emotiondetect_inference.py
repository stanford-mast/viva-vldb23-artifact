# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

from fer import FER

from viva.udfs.inference import emotion_model_udf
from viva.inference.abstract_inference import AbstractInference

class EmotionDetectInference(AbstractInference):
    """
    Class for emotion inference
    """

    def __init__(self, use_mtcnn = True):
        super().__init__()
        self.model_udf = self._prepare_model_udf(use_mtcnn)

    def _prepare_model_udf(self, use_mtcnn = True):
        # Not broadcasted for now since it's more of a library
        def emotion_fn():
            """Gets the broadcasted model."""
            model = FER(mtcnn=use_mtcnn)
            return model

        return emotion_model_udf(emotion_fn)

    def _get_model_udf(self):
        return self.model_udf
