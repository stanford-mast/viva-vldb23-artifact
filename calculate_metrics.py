import os
import sys
import argparse
import json
import pathlib
import hashlib
from viva.utils.config import viva_setup, ConfigManager
spark = viva_setup()
config = ConfigManager()

import pandas as pd
from pprint import pprint

from timeit import default_timer as now

from pyspark.sql.functions import col
from viva.core.session import VIVA
from viva.core.optimizer import Optimizer
from viva.core.utils import (
    build_row, ingest, keygenerator, hash_input_dataset, create_log_dict,
    gen_canary_results
)

def canary_frame_ids(plan, session, df, logs, canary_name):
    plan_c = plan.all_plans[0]
    df_o = session.run(df, plan_c, {}, canary_name)
    return [r.id for r in df_o.select(df_o.id).collect()]

def calculate_f1_scores(opt, hints, df_c, log_times, canary_name):
    # calculate selectivity on input dataset using all plans with hints and get optimal plan
    log_times['selectivities'] = opt.sel_profiles
    # update the input to the optimizer but don't reset.
    # Need to estimate plans using calculated selectivities
    opt.set_df(df_c)
    opt.set_hints(hints)
    opt.set_canary_name(canary_name)

    # ignore returned best plan just need computation
    opt.get_optimal_plan()
    log_times.update(opt.log_times)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selectivityfraction', '-s', type=float,
                        required=False, default=0.1,
                        dest='selectivityfraction',
                        help='Fraction of frames to estimate selectivity over (Default: 0.1). 0 to disable estimating.')
    parser.add_argument('--selectivityrandom', '-r', action='store_true',
                        dest='selectivityrandom',
                        help='Estimate selectivity by randomly choosing the fraction of frames. Not setting will do fixed rate.')
    parser.add_argument('--pruneplans', '-p', action='store_true',
                        dest='pruneplans',
                        help='Enable plan pruning for certain models')
    parser.add_argument('--query', '-q', type=str,
                        required=False, default='angrybernie',
                        choices=['angrybernie', 'dunk', 'amsterdamdock', 'deepface'],
                        dest='query',
                        help='Query to run (Default: angrybernie)')
    parser.add_argument('--canary', '-c', type=str, required=True, dest='canary', help='canary input video')

    return parser.parse_args()

def main(args):
    query = args.query
    canary = args.canary
    sel_fraction = args.selectivityfraction
    sel_random = args.selectivityrandom
    prune_plans = args.pruneplans
    do_cache = True

    # load canary and dataset
    df_c = ingest(build_row(canary))
    df_i = ingest()

    # hash input dataset
    input_dataset_hash = hash_input_dataset(df_i)
    keys = {}
    log_times = create_log_dict(vars(args), config, input_dataset_hash)
    keys['selectivity'] = keygenerator(log_times)
    log_times['canary'] = os.path.basename(canary)
    keys['f1'] = keygenerator(log_times)

    viva = VIVA(caching=do_cache)
    cp = None
    p = None
    f1_threshold = 0.9
    if query == 'angrybernie':
        from viva.plans.angry_bernie_plan import AngryBernieCanaryPlan as cp
        from viva.plans.angry_bernie_plan import AngryBerniePlan as p
    elif query == 'amsterdamdock':
        from viva.plans.amsterdam_dock_plan import AmsterdamCanaryPlan as cp
        from viva.plans.amsterdam_dock_plan import AmsterdamDockPlan as p
    elif query == 'dunk':
        from viva.plans.dunk_plan import DunkCanaryPlan as cp
        from viva.plans.dunk_plan import DunkPlan as p
    elif query == 'deepface':
        from viva.plans.deepface_plan import DeepFaceCanaryPlan as cp
        from viva.plans.deepface_plan import DeepFacePlan as p
    elif cp is None and p is None:
        print('%s is not an implemented query' % query)
        return

    canary_name = pathlib.Path(canary).stem

    # Generate canary results
    gen_canary_results(df_c, canary_name, p.all_plans)

    # calculate accuracy on canary using canary plan
    fids = canary_frame_ids(cp, viva, df_c, log_times, canary_name)

    opt = Optimizer(
        p.all_plans, df_i, fids, viva, sel_fraction, sel_random,
        f1_threshold=f1_threshold, keys=keys, prune_plans=prune_plans
    )
    calculate_f1_scores(opt, p.hints, df_c, log_times, canary_name)

    pprint(log_times)

    print(f'Done profiling: {query}.')

if __name__ == '__main__':
    main(get_args())
