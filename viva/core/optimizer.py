import os
import json
from copy import deepcopy
import logging
from typing import Dict, List, Type, Tuple
from timeit import default_timer as now
import numpy as np
import pyspark.sql.dataframe as ppd
import torch

from viva.utils.config import ConfigManager

from viva.nodes.node import Node
from viva.nodes.node_mappings import (
    objdet_hierarchy, emodet_hierarchy, facedet_hierarchy
)
from viva.core.utils import (
    profile_node_selectivity, make_unique_ids, load_op_latency, load_sel_db,
    save_selectivities, save_f1_scores, load_f1_db, load_gpu_tx_model,
    get_gpu_tx_cost
)
from viva.core.session import VIVA

MAX_COST = 1e10
PRICES = {
    'cpu':      0.00018,  # CPU  $/s -> 0.66 $/hr
    'gpu': {
        'T4':   0.00025,  # T4   $/s -> 0.91 $/hr
        'V100': 0.00067   # V100 $/s -> 2.40 $/hr
    }
}

class Optimizer:
    def __init__(self,
                 plans: List[List[Type[Node]]],
                 df_i: ppd.DataFrame,
                 reference_frames: List = [],
                 session: Type[VIVA] = None,
                 sel_fraction: float = 0,
                 sel_random: bool = False,
                 f1_threshold: float = 0.8,
                 keys: Dict = {},
                 costminmax: str = 'min',
                 opt_target: str = 'performance',
                 prune_plans: bool = False
                ):
        self.config = ConfigManager()
        self.plans = plans
        self.df_i = df_i
        self.frames_to_process = self.df_i.count()
        self.reference_frames = reference_frames
        self.viva = session
        self.sel_fraction = sel_fraction
        self.sel_random = sel_random
        self.f1_threshold = f1_threshold
        self._costminmax = costminmax
        self.opt_target = opt_target
        # to access accuracy and selectivity database
        self.keys = keys
        self.lat_profiles = load_op_latency()
        self._sel_profiles = load_sel_db(keys.get('selectivity', None))
        self._f1_scores = load_f1_db(keys.get('f1', None))
        self._log_times = {}
        self._df_cache = {}
        self._hints = {}
        self._canary_name = None
        self._use_gpu = self.config.get_value('execution', 'gpu')
        self._gpu_tx_model = load_gpu_tx_model() if self._use_gpu else None
        self._tx_cache = {}
        self.cost_plans = {}
        self.price_plans = {}
        self._acc_cache = None if not prune_plans else {}

    @property
    def sel_profiles(self):
        if self._sel_profiles or self.sel_fraction == 0:
            return self._sel_profiles
        logging.warn(f'Optimizer->selectivity estimation: {self.sel_fraction*100}%.')
        start_sel = now()
        self._sel_profiles = profile_node_selectivity(
            self.df_i, self.plans, self.sel_fraction, self.sel_random
        )
        end_sel = now()
        e2e_sel = end_sel - start_sel
        self._log_times['est_sel'] = e2e_sel
        # save to file
        save_selectivities(self.keys['selectivity'], self._sel_profiles)
        return self._sel_profiles

    @property
    def log_times(self):
        return self._log_times

    def save_plans(self, query_name: str, file_suffix: str):
        if self.opt_target == 'performance':
            logging.warn(f'Optimizer->save_plans: Warning: costs using performance opt target are not accurate when early_exit=True.')

        base_output_dir = self.config.get_value('logging', 'output')
        output_dir = os.path.join(base_output_dir, query_name)

        # Create directory for query if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output = os.path.join(output_dir, f'plan_costs_s_{file_suffix}.json')
        with open(output, 'w') as fd:
            data = deepcopy(self.cost_plans)
            for c in data:
                data[c]['plan'] = str(data[c]['plan'])
            json.dump(data, fd, indent=4, sort_keys=True)

    def set_df_cache(self, df_cache: Dict):
        self._df_cache = df_cache

    def set_df(self, df_i: ppd.DataFrame):
        self.df_i = df_i

    def set_hints(self, hints: Dict):
        self._hints = hints

    def set_canary_name(self, canary_name: str):
        self._canary_name = canary_name

    def _estimate_plan_metrics(self, plan: List[Type[Node]]) -> float:
        plan_str_list = [str(p) for p in plan]
        strplan = ','.join(plan_str_list)
        if strplan in self._f1_scores:
            f1 = self._f1_scores[strplan]['f1']
            precision = self._f1_scores[strplan]['precision']
            recall = self._f1_scores[strplan]['recall']
            # logging.warn(f'Loaded {strplan} f1={f1}, precision={precision}, recall={recall} from f1.db')
            return f1, precision, recall

        ref = make_unique_ids(self.reference_frames)
        def intersect(test: List) -> List:
            out = ['TP' if val in ref else 'FP' for val in test]
            out.extend(['FN' for r in ref if r not in test])
            return out

        # Prune plans if possible (skipping its accuracy computation)
        if self._acc_cache is not None:
            for k,v in self._acc_cache.items():
                comp_str_list = k.split(',')
                if len(plan_str_list) != len(comp_str_list):
                    continue

                num_diff = 0
                candidate_to_skip = None
                for curr,comp in zip(plan_str_list, comp_str_list):
                    curr_clean = ''.join(curr[:curr.find('->')]) if '->' in curr else curr
                    comp_clean = ''.join(comp[:comp.find('->')]) if '->' in comp else comp
                    if curr_clean != comp_clean:
                        num_diff += 1
                    if num_diff > 1:
                        candidate_to_skip = None
                        break

                    # Two cases for not checking this plan: (1) this plan uses
                    # a higher accuracy model and we've already met the threshold with a lower
                    # accuracy one, (2) this plan uses a lower accuracy model and we did not meet
                    # the threshold with a higher accuracy one
                    if 'objectdetect' in curr_clean and 'objectdetect' in comp_clean:
                        if (objdet_hierarchy[curr_clean] < objdet_hierarchy[comp_clean]) and \
                            self._acc_cache[k][0] >= self.f1_threshold:
                            candidate_to_skip = self._acc_cache[k]
                        elif (objdet_hierarchy[curr_clean] > objdet_hierarchy[comp_clean]) and \
                            self._acc_cache[k][0] < self.f1_threshold:
                            candidate_to_skip = self._acc_cache[k]
                    elif 'emotiondetect' in curr_clean and 'emotiondetect' in comp_clean:
                        if (emodet_hierarchy[curr_clean] < emodet_hierarchy[comp_clean]) and \
                            self._acc_cache[k][0] >= self.f1_threshold:
                            candidate_to_skip = self._acc_cache[k]
                        elif (emodet_hierarchy[curr_clean] > emodet_hierarchy[comp_clean]) and \
                            self._acc_cache[k][0] < self.f1_threshold:
                            candidate_to_skip = self._acc_cache[k]
                    elif 'facedetect' in curr_clean and 'facedetect' in comp_clean:
                        if (facedet_hierarchy[curr_clean] < facedet_hierarchy[comp_clean]) and \
                            self._acc_cache[k][0] >= self.f1_threshold:
                            candidate_to_skip = self._acc_cache[k]
                        elif (facedet_hierarchy[curr_clean] > facedet_hierarchy[comp_clean]) and \
                            self._acc_cache[k][0] < self.f1_threshold:
                            candidate_to_skip = self._acc_cache[k]
                if candidate_to_skip is not None:
                    #logging.warn(f'{plan} was skipped!!!')
                    return self._acc_cache[k]

        df_o = self.viva.run(self.df_i, plan, self._hints, self._canary_name)
        fids = make_unique_ids([r.id for r in df_o.select(df_o.id).collect()])
        res = intersect(fids)
        tp = res.count('TP')
        fp = res.count('FP')
        fn = res.count('FN')

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision*recall)/(precision+recall) if (precision+recall) > 0 else 0

        if self._acc_cache is not None:
            self._acc_cache[strplan] = (f1, precision, recall)

        # rnd = 2
        # logging.warn(f'{plan} f1={round(f1,rnd)}, precision={round(precision,rnd)}, recall={round(recall,rnd)}')
        return f1, precision, recall

    def _estimate_plan_cost(self, plan: List[Type[Node]], curr_best, early_exit = True) -> int:
        # Determine how many frames need to be processed
        batch_size = 16

        costs = {}
        platforms = ['cpu', 'gpu'] if self._use_gpu else ['cpu']
        for platform in platforms:
            # reset for each platform
            running_frames_to_process = self.frames_to_process
            if platform not in costs:
                costs[platform] = 0

            for p in plan:
                # If model is cached, set the latency to be 0 since we only need to run the filter
                if p.out_column in self._df_cache:
                    prof_lat = 0
                else:
                    # Get latency from op profile
                    prof_lat = self.lat_profiles[platform].get(p.out_column, None)

                if platform == 'gpu':
                    if prof_lat == 'NOT_GPU_OP':
                        prof_lat = self.lat_profiles['cpu'].get(p.out_column, None)
                    else:
                        # data transfer cost incurred for each op that runs on
                        # the GPU for the new number of frames to transfer.
                        # cost to transfer data back is not considered, negligible?
                        if running_frames_to_process in self._tx_cache:
                            costs[platform] += self._tx_cache[running_frames_to_process]
                        else:
                            frame_tx_cost = get_gpu_tx_cost(self._gpu_tx_model, running_frames_to_process)
                            costs[platform] += frame_tx_cost
                            self._tx_cache[running_frames_to_process] = frame_tx_cost

                cost_of_node = prof_lat * (running_frames_to_process / batch_size)
                costs[platform] += cost_of_node

                # early exit
                if costs[platform] > curr_best and early_exit:
                    break

                # Get selectivity from op profile
                # there could be a case where we loaded selectivities from the
                # given a dataset and query but an actual op wasn't saved. This can
                # happen if a hint wasn't included. Cost will be wrong but it'll
                # still run. Running it again will fix because it'll get written out.
                prof_sel = self.sel_profiles.get(str(p), None)
                if prof_sel is None and self.sel_fraction > 0:
                    logging.warn(f'Optimizer->{p} not found in db, profiling its selectivity. See inline comment.')
                    self._sel_profiles = profile_node_selectivity(
                        self.df_i, self.plans, self.sel_fraction, self.sel_random, self._sel_profiles
                    )
                    # Retry
                    prof_sel = self.sel_profiles.get(str(p), None)

                # Drop number of frames to process for the next op
                running_frames_to_process *= prof_sel

        return costs

    def _find_optimal_plan(self, early_exit:bool = False) -> Tuple[List[Type[Node]], Dict]:
        start_p_o = now()
        filtered_plans = []
        if self.reference_frames or self._f1_scores:
            logging.warn('Optimizer->estimating f1 scores.')
            to_save = []
            for i,pp in enumerate(self.plans):
                plan_f1, plan_precision, plan_recall = self._estimate_plan_metrics(pp)
                strplan = ','.join([str(p) for p in pp])
                to_save.append((strplan, plan_f1, plan_precision, plan_recall))
                if plan_f1 >= self.f1_threshold:
                    filtered_plans.append((pp, plan_f1, plan_precision, plan_recall))
                if i % 10 == 0:
                    logging.warn('Optimizer->checkpointing f1 at %d of %d' % (i+1, len(self.plans)))
                    save_f1_scores(self.keys['f1'], to_save)
                    to_save = [] # reset checkpoint
            save_f1_scores(self.keys['f1'], to_save)
        else:
            filtered_plans = [(pp, 1.0, 1.0, 1.0) for pp in self.plans]

        logging.warn('Optimizer->estimating plan costs.')
        lowest_running_cost = MAX_COST if self._costminmax == 'min' else 0

        # assuming we're always only using first GPU
        if self._use_gpu:
            device_name = torch.cuda.get_device_name(0)
            if 'T4' in device_name:
                gpu_name = 'T4'
            elif 'V100' in device_name:
                gpu_name = 'V100'
            else:
                logging.warn(f'No cost set for {device_name}, using T4 cost')
                gpu_name = 'T4'
        # build both keys for easier searching and showing both plans
        # Initialize based on costminmax
        for pp, f1, precision, recall in filtered_plans:
            costs = self._estimate_plan_cost(pp, lowest_running_cost, early_exit)
            for platform in costs:
                curr_cost = costs[platform]
                if curr_cost not in self.cost_plans:
                    cp = {}
                    cp['cost'] = curr_cost
                    cp['plan'] = pp
                    cp['f1'] = f1
                    cp['precision'] = precision
                    cp['recall'] = recall
                    cp['platform'] = platform
                    cp['device_name'] = gpu_name if platform == 'gpu' else 'cpu'
                    if self._use_gpu and platform == 'gpu':
                        cp['price'] = PRICES[platform][gpu_name] * curr_cost
                    else:
                        cp['price'] = PRICES[platform] * curr_cost
                    self.price_plans[cp['price']] = cp
                    self.cost_plans[curr_cost] = cp
                # used to clip early_exit
                if curr_cost < lowest_running_cost:
                    lowest_running_cost = curr_cost
                # logging.warn(f'{pp}: cost={round(curr_cost,rnd)} romeros, platform={platform}, f1={f1}, precision={precision}, recall={recall}')

        end_p_o = now()
        self._log_times['optimizer'] = end_p_o - start_p_o

    def get_optimal_plan(self) -> Tuple[List[Type[Node]], Dict]:
        if not self.cost_plans and not self.price_plans:
            # early exit on if we don't care about final cost numbers for other plans
            # NOTE: needs to be off if searching for cost effective plans because
            # needs to compute the cost to get the price
            early_exit = True if self.opt_target == 'performance' else False
            self._find_optimal_plan(early_exit)

        best_plan = {}
        if self.opt_target == 'performance':
            fastest_plan = self.cost_plans[min(self.cost_plans)]
            slowest_plan = self.cost_plans[max(self.cost_plans)]
            if self._costminmax == 'min':
                best_plan = fastest_plan
            else:
                best_plan = slowest_plan
        elif self.opt_target == 'cost':
            cheapest_plan = self.price_plans[min(self.price_plans)]
            costliest_plan = self.price_plans[max(self.price_plans)]
            if self._costminmax == 'min':
                best_plan = cheapest_plan
            else:
                best_plan = costliest_plan

        logging.warn(f'Optimizer->plans considered: {len(self.cost_plans)}.')
        logging.warn(f'Optimizer->{self.opt_target} plan: {best_plan}')

        return best_plan
