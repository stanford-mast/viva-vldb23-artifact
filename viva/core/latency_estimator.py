import os
import sys
import json
from os import path
basepath = path.dirname(__file__)
sys.path.append(path.abspath(path.join(basepath, '../../')))

from viva.utils.config import viva_setup
spark = viva_setup()
from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from timeit import default_timer as now
from typing import Callable, NamedTuple, Tuple, List, Any, Dict, Type

import pyspark.sql.dataframe as ppd

from viva.sparkmodels import IngestVideo
from viva.nodes.data_nodes import WalkRows
from viva.plans.profile_plan import ProfilePlan
from viva.plans.ingest_opt_plan import Plan as IngestPlan

# Produce batch_scale * batch_size inputs so that we get an accurate estimate
# of time per batch without initial startup overhead. We then divide the final
# end to end time by this value.
batch_scale = 8
batch_size = 16
overwrite_ops = False

GPU_NODES = [
    'actiondetect',
    'classification',
    'qclassification',
    'facedetect',
    'objectdetect',
    'objectdetect_large',
    'objectdetect_medium',
    'objectdetect_nano',
    'objectdetect_xlarge',
    'objecttrack',
    'proxyclassification',
    'img2vec'
]

def gen_input_df(batch_size: int) -> ppd.DataFrame:
    videos = ['data/']
    data = WalkRows(videos, ['mp4']).custom_op(None)
    df_i = spark.createDataFrame(data, IngestVideo)

    for node in IngestPlan:
        df_i = node.apply_op(df_i)
        df_i = node.apply_filters(df_i)

    return df_i.limit(batch_size * batch_scale)

def profile_node_latencies(df: ppd.DataFrame, prof_map: Dict[str, int], warmup: bool=False) ->  Dict[str, int]:
    # Any plan will do
    plan = ProfilePlan.all_plans[0]
    use_gpu = config.get_value('execution', 'gpu')
    if warmup:
        print('Warm up run')

    # Add the first node twice since there is a startup overhead that will
    # produce misleading profiling results for it
    mod_plan = [plan[0]] + plan
    for i,node in enumerate(mod_plan):
        model = node.out_column

        # Skip; don't re-profile
        if model in prof_map:
            print(f'Skipping {model}.')
            continue

        if use_gpu and model not in GPU_NODES:
            prof_map[model] = 'NOT_GPU_OP' # hack
            continue
        print(f'Profiling {model}')

        # Only profile the op, not the filter
        df = node.apply_op(df)
        start = now()
        df.count()
        end = now()
        e2e = end - start

        # Ignore first node
        if i != 0 and not warmup:
            prof_map[model] = e2e / batch_scale
            print(f'Latency: {round(prof_map[model], 2)}')

        df = node.apply_filters(df)

    return prof_map

def profile_ops(output_name: str, batch_size: int, overwrite_ops: bool = False) -> None:
    prof_map = {}

    if not overwrite_ops:
        # Read in profiled ops if they already exist so we don't re-profile
        if os.path.exists(output_name):
            fd = open(output_name, 'r')
            prof_map = json.load(fd)
            fd.close()

    # warm up run
    _ = profile_node_latencies(gen_input_df(batch_size), prof_map, True)

    df = gen_input_df(batch_size)
    prof_map = profile_node_latencies(df, prof_map)

    fd = open(output_name, 'w')
    json.dump(prof_map, fd, indent=4, sort_keys=True)
    fd.close()

if __name__ == '__main__':
    use_gpu = config.get_value('execution', 'gpu')
    if use_gpu:
        fname = 'op_latency_bmarks_gpu.json'
    else:
        fname = 'op_latency_bmarks.json'

    default_output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/', fname)
    profile_ops(output_name=default_output_name, batch_size=batch_size, overwrite_ops=overwrite_ops)
