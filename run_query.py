import logging
import os
import sys
import argparse
from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()
viva_setup()

import torch
from uuid import uuid1
from timeit import default_timer as now

from pyspark.sql.functions import col
from viva.utils.video import write_video
from viva.core.session import VIVA
from viva.core.optimizer import Optimizer
from viva.core.utils import (
    ingest, keygenerator, hash_input_dataset, create_log_dict
)
device = 'gpu' if config.get_value('execution', 'gpu') else 'cpu'

def run_plan(viva_session, df, plan, sel_fraction, sel_random, keys, costminmax, f1thresh, opt_target):
    # create optimizer and find optimal plan
    opt = Optimizer(
        plan.all_plans, df, session=viva_session, sel_fraction=sel_fraction,
        sel_random=sel_random, keys=keys, costminmax=costminmax,
        f1_threshold=f1thresh, opt_target=opt_target
    )

    # Set hints in case accuracy needs to be computed
    opt.set_hints(plan.hints)

    # Get optimal plan
    best_plan = opt.get_optimal_plan()
    best_execution_plan = best_plan['plan']

    device = best_plan['platform']
    for k in best_plan:
        # don't add the plan, doesn't serialize for pandas
        if k not in opt.log_times and k != 'plan':
            viva_session.update_log(k, best_plan[k])

    # Convert best_plan into a string for logging
    best_plan_str = ','.join([p.out_column for p in best_execution_plan])
    num_trees = len(plan.all_trees)
    num_plans = len(plan.all_plans)

    use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
    if device == 'cpu' and use_cuda:
        print(f'Opt target: {opt_target} plan requires {device}. Exit and rerun with {device}.')
        return df, opt, (best_plan_str, num_trees, num_plans)

    if device == 'gpu':
        logging.warn('Query->GPU warmup.')
        df2 = ingest()
        _ = viva_session.run(df2, best_execution_plan, plan.hints)
        viva_session.reset_cache()

    # run the plan on df
    df = viva_session.run(df, best_execution_plan, plan.hints)
    df = df.sort(df.id.asc())
    df_s = df.select(df.uri, df.id, df.label, df.score)
    df_s.show(truncate=False)

    return df, opt, (best_plan_str, num_trees, num_plans)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', '-l', type=str, nargs='?',
                        dest='logging', const='', default=None,
                        help='Do logging (optionally supply suffix for logfile name)')
    parser.add_argument('--cache', '-C', action='store_true',
                        dest='cache',
                        help='Enable caching and potential reuse of results')
    parser.add_argument('--ingestwarmup', '-w', action='store_true',
                        dest='ingestwarmup',
                        help='Perform ingest (transcoding) warmup')
    parser.add_argument('--selectivityfraction', '-s', type=float,
                        required=False, default=0.1,
                        dest='selectivityfraction',
                        help='Fraction of frames to estimate selectivity over (Default: 0). 0 to disable estimating.')
    parser.add_argument('--selectivityrandom', '-r', action='store_true',
                        dest='selectivityrandom',
                        help='Estimate selectivity by randomly choosing the fraction of frames. Not setting will do fixed rate.')
    parser.add_argument('--costminmax', '-e', type=str,
                        required=False, default='min',
                        choices=['min', 'max'],
                        dest='costminmax',
                        help='Select plan based on min/max cost (Default: min)')
    parser.add_argument('--f1thresh', '-f', type=float,
                        required=False, default=0.8,
                        dest='f1thresh',
                        help='F1 threshold (Default: 0.8)')
    parser.add_argument('--opttarget', '-o', type=str,
                        required=False, default='performance',
                        dest='opttarget',
                        choices=['performance', 'cost', 'dollar'],
                        help='Plan optimization target (Default: performance)')
    parser.add_argument('--query', '-q', type=str,
                        required=False, default='angrybernie',
                        choices=['angrybernie', 'dunk', 'amsterdamdock', 'deepface'],
                        dest='query',
                        help='Query to run (Default: angrybernie)')
    parser.add_argument('--canary', type=str, required=True, dest='canary', help='Canary input video to find database key.')
    parser.add_argument('--logname', type=str, required=False, dest='logname')

    return parser.parse_args()

def main(args):
    do_logging = args.logging is not None
    do_cache = args.cache
    do_ingestwarmup = args.ingestwarmup
    costminmax = args.costminmax
    f1thresh = args.f1thresh
    query = args.query
    canary = args.canary
    sel_fraction = args.selectivityfraction
    sel_random = args.selectivityrandom
    opt_target = args.opttarget

    start_ingest_v = now()
    df_i = ingest()
    if do_logging:
        df_i.count() # any action op will do
    end_ingest_v = now()

    if do_ingestwarmup:
        print('Ingest warmup complete')
        sys.exit(0)

    input_dataset_hash = hash_input_dataset(df_i)
    log_times = create_log_dict(vars(args), config, input_dataset_hash)
    keys = {}
    keys['selectivity'] = keygenerator(log_times)
    log_times['canary'] = os.path.basename(canary)
    keys['f1'] = keygenerator(log_times)

    if do_logging:
        log_times['ingest_video'] = (end_ingest_v - start_ingest_v)

    viva_session = VIVA(log_times, do_cache)
    if query == 'angrybernie':
        from viva.plans.angry_bernie_plan import AngryBerniePlan as p
    elif query == 'dunk':
        from viva.plans.dunk_plan import DunkPlan as p
    elif query == 'amsterdamdock':
        from viva.plans.amsterdam_dock_plan import AmsterdamDockPlan as p
    elif query == 'deepface':
        from viva.plans.deepface_plan import DeepFacePlan as p
    else:
        print('%s is not an implemented query' % query)
        return

    df_res, opt, logging_data = run_plan(
        viva_session, df_i, p, sel_fraction, sel_random, keys, costminmax,
        f1thresh, opt_target
    )

    if do_logging:
        final_frame_count = df_res.select('id').distinct().count()
        best_plan, num_trees, num_plans = logging_data
        viva_session.update_log('num_trees', num_trees)
        viva_session.update_log('num_plans', num_plans)
        viva_session.update_log('final_frame_count', final_frame_count)
        viva_session.print_logs(query, best_plan, args.logging, args.logname)
        opt.save_plans(query, args.logging)

    data = df_res.collect()
    if len(data) == 0 or len(data) == df_i.count():
        print('Query->no results.')
        return

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outname = f'{output_dir}/out-{str(uuid1())[0:7]}.mp4'
    write_video(data, outname)
    print(f'Done running: {query}. Results to: {outname}')

if __name__ == '__main__':
    main(get_args())
