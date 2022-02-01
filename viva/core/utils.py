import os
import csv
import json
import pathlib
import hashlib
import pickle
import numpy as np
from typing import  List, Type, Dict, Tuple
import logging
from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

import pandas as pd
import pyspark.sql.dataframe as ppd
from pyspark.sql import Window, Row
from pyspark.sql.functions import row_number, col

from viva.nodes.node import Node
from viva.nodes.data_nodes import WalkRows
from viva.sparkmodels import IngestVideo
from viva.plans.ingest_opt_plan import Plan as IngestPlan

selectivity_db_name = 'data/sel.db.csv'
f1_db_name = 'data/f1.db.csv'
profiled_ops_path = 'data/op_latency_bmarks.json'
gpu_profiled_ops_path = 'data/op_latency_bmarks_gpu.json'
gpu_tx_model_path = 'data/gpu_data_transfer_model.pkl'

def ingest(custom_path = None):
    spark = viva_setup()
    if isinstance(custom_path, Row):
        data = [custom_path]
    else:
        videos = [config.get_value('storage', 'input')]
        data = WalkRows(videos, ['mp4']).custom_op(None)

    logging.warn(f'Ingest->{data}')
    df_i = spark.createDataFrame(data, IngestVideo)
    for node in IngestPlan:
        df_i = node.apply_op(df_i)
        df_i = node.apply_filters(df_i)

    return df_i

def build_row(name):
    return Row(str(os.path.abspath(name)), 0)

def gen_canary_results(df_c: ppd.DataFrame, canary_name: str, plans: List[List[Type[Node]]]):
    spark = viva_setup()

    canary_path = config.get_value('storage', 'canary')
    canary_cache = os.path.join(canary_path, 'canary_results_cache', canary_name)
    pathlib.Path(canary_cache).mkdir(parents=True, exist_ok=True)

    # If we run img2vec, save for anything with TASTI
    # Similarly, if we run dfprefix, save for anything with deepface.
    embed_cached_df = {'img2vec': None, 'dfprefixembed': None}

    # Run the node if it has not previously been run
    # Parquet filename format is: <modelname>_<canary>.parquet
    for p in plans:
        for node in p:
            model = node.out_column
            next_filename = '%s_%s.parquet' % (model, canary_name)
            next_path = os.path.join(canary_cache, next_filename)

            # Don't profile a node more than once; load if needed
            if os.path.exists(next_path):
                if model != 'img2vec' and model != 'dfprefixembed':
                    continue
                elif model == 'img2vec':
                    if embed_cached_df['img2vec'] is None:
                        embed_cached_df['img2vec'] = spark.read.parquet(next_path)
                    continue
                elif model == 'dfprefixembed':
                    if embed_cached_df['dfprefixembed'] is None:
                        embed_cached_df['dfprefixembed'] = spark.read.parquet(next_path)
                    continue

            if 'tasti' in model:
                if embed_cached_df['img2vec'] is None:
                    # This plan should be invalid; skip
                    print('WARNING: attempted to get accuracy for', model, 'before running img2vec. Skipping...')
                    break
                else:
                    df_ss = embed_cached_df['img2vec']
            elif 'deepfaceSuffix' in model:
                if embed_cached_df['dfprefixembed'] is None:
                    # This plan should be invalid; skip
                    print('WARNING: attempted to get accuracy for', model, 'before running dfprefixembed. Skipping...')
                    break
                else:
                    df_ss = embed_cached_df['dfprefixembed']
            elif model != 'objecttrack':
                # Reset back to original if it's not objecttrack
                df_ss = df_c

            df_ss = df_ss.sort('id').repartitionByRange(2, col('id'))
            df_ss = node.apply_op(df_ss)

            # Write out to parquet before applying filter
            df_ss.write.parquet(next_path)

            df_ss = node.apply_filters(df_ss)

            if model == 'img2vec':
                embed_cached_df['img2vec'] = df_ss
            elif model == 'dfprefixembed':
                embed_cached_df['dfprefixembed'] = df_ss

    return

def profile_node_selectivity(df: ppd.DataFrame, plans: List[List[Type[Node]]],
                             fraction_to_sample: float, do_random: bool = False, sel_map: Dict = {}):
    # Sample
    if do_random:
        df_s = df.sample(withReplacement=False, fraction=fraction_to_sample, seed=None)
    else:
        w = Window.partitionBy().orderBy(col("id"))
        df_s = df.withColumn("rn",row_number().over(w)).filter(col("rn") % int(1/fraction_to_sample) == 0).drop(*["rn"])

    # Compute how many frames we will explore
    num_start_frames = df_s.select('id').distinct().count()

    # If we run img2vec, save for anything with TASTI
    # Similarly, if we run dfprefix, save for anything with deepface.
    embed_cached_df = {'img2vec': None, 'dfprefixembed': None}

    # Run the node and the filters, estimating along the way
    sel_map = sel_map
    for p in plans:
        for node in p:
            model = node.out_column
            if str(node) in sel_map:
                # Don't profile a node more than once UNLESS it's img2vec and
                # img2vec_df is NONE or UNLESS its dfprefixembed and dfprefixembed_df is NONE
                if model != 'img2vec' and model != 'dfprefixembed':
                    continue
                elif model == 'img2vec' and embed_cached_df['img2vec'] is not None:
                    continue
                elif model == 'dfprefixembed' and embed_cached_df['dfprefixembed'] is not None:
                    continue

            if 'tasti' in model:
                if embed_cached_df['img2vec'] is None:
                    # This plan should be invalid; skip
                    print('WARNING: attempted to get selectivity for', model, 'before running img2vec. Skipping...')
                    break
                else:
                    df_ss = embed_cached_df['img2vec']
            elif 'deepfaceSuffix' in model:
                if embed_cached_df['dfprefixembed'] is None:
                    # This plan should be invalid; skip
                    print('WARNING: attempted to get selectivity for', model, 'before running dfprefixembed. Skipping...')
                    break
                else:
                    df_ss = embed_cached_df['dfprefixembed']
            elif model != 'objecttrack':
                # Reset back to original if it's not objecttrack
                df_ss = df_s

            df_ss = df_ss.sort('id').repartitionByRange(2, col('id'))
            df_ss = node.apply_op(df_ss)
            df_ss = node.apply_filters(df_ss)

            if model == 'img2vec':
                embed_cached_df['img2vec'] = df_ss
            elif model == 'dfprefixembed':
                embed_cached_df['dfprefixembed'] = df_ss

            # Account for proxy case
            if isinstance(df_ss, tuple):
                # For estimating selectivity, want what the next node would process (first arg)
                df_ss = df_ss[0]

            num_filt_frames = df_ss.select('id').distinct().count()
            comp_select = num_filt_frames / num_start_frames

            sel_map[str(node)] = round(comp_select, 6)
            logging.warn(f'Optimizer->selectivity estimation: {node}={comp_select}')

    return sel_map

def make_unique_ids(fids: List) -> List:
    """
    append ids with an identifier for easier search
    """
    output = []
    counts = {}
    for val in fids:
        if val not in counts:
            counts[val] = 0
        counts[val] += 1
        uid = '_'.join([str(val), str(counts[val])])
        output.append(uid)
    return output

def load_sel_db(keys) -> Dict[str, float]:
    key = keys.get('key', None)
    sels = {}
    if key is None:
        return sels
    if os.path.exists(selectivity_db_name):
        sdb = pd.read_csv(selectivity_db_name)
        sdb = sdb[sdb.key == key]
        if not sdb.empty:
            logging.warn('Optimizer->loading selectivities from db.')
        for _, row in sdb.iterrows():
            sels[row['op']] = row['selectivity']

    return sels

def load_f1_db(keys) -> Dict[str, float]:
    key = keys.get('key', None)
    f1 = {}
    if key is None:
        return f1

    if os.path.exists(f1_db_name):
        f1db = pd.read_csv(f1_db_name)
        f1db = f1db[f1db.key == key]
        if not f1db.empty:
            logging.warn('Optimizer->loading f1 scores from db.')
        for _, row in f1db.iterrows():
            f1[row['plan']] = {}
            f1[row['plan']]['f1'] = row['f1']
            f1[row['plan']]['precision'] = row['precision']
            f1[row['plan']]['recall'] = row['recall']

    return f1

def save_to_db(pdf, fname):
    if not os.path.exists(fname):
        pdf_csv = pdf.to_csv(index=False)
        with open(fname, 'w') as fd:
            fd.write(pdf_csv)
    else:
        data = pd.read_csv(fname)
        merged = pd.concat([data, pdf]).drop_duplicates()
        merged.to_csv(fname, mode='w', index=False)

def save_selectivities(keys: Dict, sellogs: Dict):
    if keys == None:
        logging.warn('Optimizer dataset keys not set, not saving selectivities.')
        return

    key = keys['key']
    params = keys['params']

    data = []
    cols = ['key', 'params', 'op', 'selectivity']
    for op in sellogs:
        data.append((key, params, op, sellogs[op]))

    pdf = pd.DataFrame(data, columns=cols)
    save_to_db(pdf, selectivity_db_name)

def save_f1_scores(keys: Dict, plans: List[Type[Tuple]]):
    if keys == None:
        logging.warn('Optimizer dataset keys not set, not saving plan f1 scores.')
        return

    key = keys['key']
    params = keys['params']

    data = []
    cols = ['key', 'params', 'plan', 'f1', 'precision', 'recall']
    for plan, f1, precision, recall in plans:
        data.append((key, params, str(plan), f1, precision, recall))

    pdf = pd.DataFrame(data, columns=cols)
    save_to_db(pdf, f1_db_name)

def load_op_latency() -> Dict[str, int]:
    lat_profiles = {}
    if os.path.exists(profiled_ops_path):
        with open(profiled_ops_path, 'r') as fd:
            lat_profiles['cpu'] = json.load(fd)

    if os.path.exists(gpu_profiled_ops_path):
        with open(gpu_profiled_ops_path, 'r') as fd:
            lat_profiles['gpu'] = json.load(fd)

    return lat_profiles

def keygenerator(params: Dict) -> Dict:
    skeys = sorted(params.keys())
    key = '_'.join([f'{k}:{params[k]}' for k in skeys])
    ekey = str.encode(key)

    return {'params': key, 'key': hashlib.sha224(ekey).hexdigest()}

def hash_input_dataset(df_i):
    uris = '&'.join(sorted([r.uri for r in df_i.select(df_i.uri).collect()]))
    uris = hashlib.sha224(str.encode(uris)).hexdigest()
    return uris

def create_log_dict(args, config, in_uris):
    argkeys = ['selectivityfraction', 'selectivityrandom', 'query']
    logs = {}
    logs['inputs'] = in_uris
    for ak in argkeys:
        logs[ak] = args[ak]

    cfg = config._cfg['ingest']
    logs.update(cfg)

    return logs

def load_gpu_tx_model():
    model = pickle.load(open(gpu_tx_model_path, 'rb'))
    return model

def get_gpu_tx_cost(model, frames):
    x_pred = np.array([frames]).reshape(-1, 1)
    cost = model.predict(x_pred)[0]
    return cost
