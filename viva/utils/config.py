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

import yaml
import os
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import logging

def read_value_config(cfg, category, key):
    return cfg.get(category, {}).get(key)

def update_value_config(cfg, category, key, value):
    category_data = cfg.get(category, None)
    if category_data:
        category_data[key] = value

def viva_setup():
    config = ConfigManager()
    spark_conf = SparkConf()
    pyspark_config = config.get_value('spark', 'property')
    for key, value in pyspark_config.items():
        spark_conf.set(key, value)

    logging.basicConfig(level=logging.WARN, format='%(message)s')
    os.environ['YOLOv5_VERBOSE'] = 'False'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Disable GPUs based on config
    use_gpu = config.get_value('execution', 'gpu')
    if use_gpu:
        # TODO(JAH): this changes config for GPU to a single executor. the
        # proper way we should implement is to set the right spark configs to
        # do this.
        spark_conf.set('spark.master', 'local[1]')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # don't allocate the whole GPU to TF, grow memory as needed
    # needs to be set before importing TF so it is by design not in the GPU
    # case statement
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    import torch
    import tensorflow as tf

    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    sparkloglevel = config.get_value('logging', 'sparkloglevel')
    spark.sparkContext.setLogLevel(sparkloglevel)

    return spark

class ConfigManager(object):
    _instance = None
    _cfg = None

    # JAH: no idea what any of this class fluff does
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)

            ymlpath = 'conf.yml'
            with open(ymlpath, 'r') as ymlfile:
                cls._cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        return cls._instance

    def get_value(self, category, key):
        return read_value_config(self._cfg, category, key)

    def update_value(self, category, key, value):
        update_value_config(self._cfg, category, key, value)
