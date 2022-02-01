import os
import sys
import json
from time import perf_counter
import torch
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import pyspark.sql.dataframe as ppd
from os import path
basepath = path.dirname(__file__)
sys.path.append(path.abspath(path.join(basepath, '../../')))

from viva.utils.config import viva_setup, ConfigManager
spark = viva_setup()
config = ConfigManager()

use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
device = torch.device('cuda' if use_cuda else 'cpu')

from viva.sparkmodels import IngestVideo
from viva.nodes.data_nodes import WalkRows
from viva.plans.ingest_opt_plan import Plan as IngestPlan
from viva.plans.transfer_plan import TransferPlan

# frames = [0, 1, 500, 1000, 2000, 3000]
frames = [5000]
iterations = 5
t_sleep = 0
h = config.get_value('ingest', 'height')
w = config.get_value('ingest', 'width')
fname = 'data/gpu_data_transfer.json'

def gen_input_df() -> ppd.DataFrame:
    videos = ['data/']
    data = WalkRows(videos, ['mp4']).custom_op(None)
    df_i = spark.createDataFrame(data, IngestVideo)

    for node in IngestPlan:
        df_i = node.apply_op(df_i)
        df_i = node.apply_filters(df_i)

    return df_i

def warmup(plan):
    df_i = gen_input_df()
    for node in plan:
        df = node.apply_op(df_i)
        f = df.count()
    print('num frames', f)
    size = (1, 240, 360, 3)
    tensor = torch.rand(size, dtype=torch.float32)
    tensor = tensor.to(device)
    print('warm up done')

def execute_transfer(it = 2, latencies = {}) -> None:
    plan = TransferPlan.all_plans[0]
    warmup(plan)

    for f in frames:
        if str(f) in latencies:
            print(f'skipping {f}')
            continue

        latencies[f] = []

        for i in range(0, it):
            df_i = gen_input_df()
            df = df_i.limit(f)
            for node in plan:
                name = node.out_column
                s = perf_counter()
                df = node.apply_op(df)
                _ = df.count()
                t = perf_counter()
                d = t-s
                if name == 'transfer':
                    latencies[f].append(d)

        avg_lat = sum(latencies[f])/len(latencies[f])
        latencies[f] = avg_lat
        if avg_lat < 1:
            print('{0}: {1:.2f}ms'.format(f, avg_lat*1e3))
        else:
            print('{0}: {1:.2f}s'.format(f, avg_lat))

    return latencies

def fit_and_save(data):
    x_train = np.array([k for k in data]).reshape((-1, 1))
    y_train = np.array([data[k] for k in data])
    model = LinearRegression().fit(x_train, y_train)
    fname = 'data/gpu_data_transfer_model.pkl'
    pickle.dump(model, open(fname, 'wb'))

    return model

if __name__ == '__main__':
    with open(fname, 'r') as fd:
        loaded_latencies = json.load(fd)

    latencies = execute_transfer(iterations, loaded_latencies)
    with open(fname, 'w') as fd:
        json.dump(latencies, fd, indent=4)

    model = fit_and_save(latencies)

    x_pred = np.array([500, 5000]).reshape((-1, 1))
    y_pred = model.predict(x_pred)
    print(f'Predicted latencies for {x_pred}:', y_pred)
