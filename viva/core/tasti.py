import os
import sys
import json

from viva.utils.config import viva_setup
spark = viva_setup()

from timeit import default_timer as now
from typing import Callable, NamedTuple, Tuple, List, Any, Dict, Type

from pyspark.sql import Window
import pyspark.sql.dataframe as ppd
from pyspark.sql.functions import row_number, col

from viva.sparkmodels import IngestVideo
from viva.nodes.data_nodes import WalkRows
from viva.plans.tasti_plan import Img2VecPlan
from viva.plans.ingest_opt_plan import Plan as IngestPlan

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

#def gen_input_df(frame_limit: int) -> ppd.DataFrame:
def gen_input_df(fraction_to_sample: float) -> ppd.DataFrame:
    videos = ['data/']
    data = WalkRows(videos, ['mp4']).custom_op(None)
    df_i = spark.createDataFrame(data, IngestVideo)

    for node in IngestPlan:
        df_i = node.apply_op(df_i)
        df_i = node.apply_filters(df_i)

    w = Window.partitionBy().orderBy(col("id"))
    df_i = df_i.withColumn("rn",row_number().over(w)).filter(col("rn") % int(1/fraction_to_sample) == 0)    .drop(*["rn"])

    #return df_i.limit(frame_limit)
    return df_i

def gen_indexes(df: ppd.DataFrame, vector_size: int, k_value: int) -> Dict:
    # Any plan will do
    plan = Img2VecPlan.all_plans[0]

    for i,node in enumerate(plan):
        df = node.apply_op(df)

        if i >= 10:
            break

    df = df.drop('uri', 'id', 'width', 'height', 'framebytes')
    col_to_select = ['img2vec'] + [col(f'{n}.label').alias(n) for n in df.schema.names if n != 'img2vec']
    df = df.select(*col_to_select)
    df_vec = df.collect()

    vec_np = np.array([np.frombuffer(bytearray(b.img2vec), dtype=np.float32).reshape(vector_size,) for b in df_vec])

    kmeans = KMeans(init='k-means++', n_clusters=k_value, n_init=10)
    kmeans.fit(vec_np)
    preds = kmeans.predict(vec_np)
    centers = kmeans.cluster_centers_

    # For each cluster, find the furthest point (FPF) and save its index
    # {Key: cluster_index, Value: FPF_index}
    fpf_map = {}
    for i,p in enumerate(preds):
        next_center = centers[p]
        next_vec = vec_np[i,:]
        sim = cosine_similarity(next_center.reshape((1, -1)), next_vec.reshape((1, -1)))[0][0]

        if p not in fpf_map:
            fpf_map[p] = i
        else:
            if fpf_map[p] > sim:
                fpf_map[p] = i

    # For each prediction (center), determine the furthest point from it, and assign that the label (i.e., FPF)
    rep_map = {} # {Key: model, Value: [(cluster_center, label(s)), ...]}
    for n in df.schema.names:
        if n != 'img2vec':
            for k,v in fpf_map.items():
                cluster_center = centers[k]
                label = df_vec[v][n]
                if n not in rep_map:
                    rep_map[n] = []
                rep_map[n].append((cluster_center, label))

    for k,v in rep_map.items():
        print(k)
        for vv in v:
            print('\t', vv[1])

    return rep_map

def run_tasti(output_name: str, fraction_to_sample: float, vector_size: int, k_value: int) -> None:
    df = gen_input_df(fraction_to_sample)
    generated_indexes = gen_indexes(df, vector_size, k_value)

    pandas_df = pd.DataFrame(generated_indexes)
    pd.set_option('display.max_colwidth', None)
    pandas_df.to_pickle(output_name)

if __name__ == '__main__':
    fname = 'long_tasti_index.bin'
    fraction_to_sample = 0.9
    vector_size = 512 # ResNet-18
    k_value = 50
    default_output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/', fname)
    run_tasti(output_name=default_output_name, fraction_to_sample=fraction_to_sample, vector_size=vector_size, k_value=k_value)

