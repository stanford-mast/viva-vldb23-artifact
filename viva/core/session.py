import os
import sys
import logging
from queue import Queue
from timeit import default_timer as now
from typing import List, Dict, Set, Any
import pandas as pd
import pyspark.sql.dataframe as ppd
from pyspark.sql.functions import col

from viva.nodes.filters import proxy_quality_filter, explode_preds
from viva.utils.config import viva_setup, ConfigManager

class VIVA:
    def __init__(self, log_time: Dict = {}, caching: bool = False):
        self.do_logging = False if not log_time else True
        self.log_time = log_time

        # {Key: model, Value: produced DataFrame}
        self.do_caching = caching
        self.df_cache = {}
        self.spark = viva_setup()
        self.config = ConfigManager()

        # Accuracy cache
        self._acc_cache = {}

    def update_log(self, key: str, val: Any) -> None:
        self.log_time[key] = val

    def print_logs(self, query_name: str, plan: str, file_suffix: str, logname: str = None) -> None:
        pdf_lat = pd.DataFrame(self.log_time, index=[0])
        pdf_lat_csv = pdf_lat.to_csv(index=False)

        print(pdf_lat_csv)

        # Write to file if requested
        write_to_file = self.config.get_value('logging', 'writetofile')
        if write_to_file:
            base_output_dir = self.config.get_value('logging', 'output')
            output_dir = os.path.join(base_output_dir, query_name)

            # Create directory for query if needed
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # File format: lat_query_csvordering[_s_suffix].txt
            base_filename = '%s_%s' % (query_name, plan)
            if file_suffix != '':
                base_filename += '_s_%s' % file_suffix
            base_filename += '.txt'
            latencyfile = os.path.join(output_dir, 'lat_' + base_filename)
            if logname is not None:
                latencyfile = os.path.join(output_dir, logname)

            lat_fd = open(latencyfile, 'w')
            lat_fd.write(pdf_lat_csv)
            lat_fd.close()

            print('Logs written to %s' % (latencyfile))

    def get_df_cache(self):
        return self.df_cache

    def reset_cache(self):
        self.df_cache = {}
        self._acc_cache = {}

    def _get_col_mapping(self, hint):
        from viva.nodes.node_mappings import NodeMappings
        left_col, right_col = hint.get_args()
        if left_col not in NodeMappings:
            print(left_col, 'not in NodeMappings for proxy mapping!')
            sys.exit(1)
        if right_col not in NodeMappings:
            print(right_col, 'not in NodeMappings for proxy mapping!')
            sys.exit(1)
        left_col_long = NodeMappings[left_col].out_column
        right_col_long = NodeMappings[right_col].out_column

        return left_col_long, right_col_long

    def _gen_proxy_mapping(self, proxy_hints: List):
        general_proxy_map = {}
        proxy_conf_map = {}
        for p in proxy_hints:
            left_col_long, right_col_long = self._get_col_mapping(p)
            general_proxy_map[right_col_long] = left_col_long
            proxy_conf_map[left_col_long] = p.get_confidence()
        return general_proxy_map, proxy_conf_map

    # Check to see if the general model from a proxy hint has an equals equivalent
    # There are two cases we are searching for: (1) if equals directly relates
    # the column from proxy, and (2) if equals indirectly relates the column from
    # proxy (e.g., A EQ B, B EQ C)
    def _search_equals_hints(self, search_col: str, equals_hints: List, proxy_general: Set) -> str:
        unique_eq = set() # Don't visit the same equals hint more than once
        eq_queue = Queue()
        eq_queue.put(search_col)

        while not eq_queue.empty():
            curr_search_col = eq_queue.get()
            for e in equals_hints:
                if (str(e) not in unique_eq):
                    left_col_long, right_col_long = self._get_col_mapping(e)
                    if (left_col_long == curr_search_col):
                        if (right_col_long in proxy_general):
                            return right_col_long
                        else:
                            eq_queue.put(right_col_long)
                            unique_eq.add(str(e))
                    elif (right_col_long == curr_search_col):
                        if (left_col_long in proxy_general):
                            return left_col_long
                        else:
                            eq_queue.put(left_col_long)
                            unique_eq.add(str(e))

        return None

    # If canary_name is not None, we are measuring accuracy (and will use cached results)
    def run(self, input_df: ppd.DataFrame, plan: List, hints: Dict = {},
            canary_name: str = None) -> ppd.DataFrame:

        logging.warn(f'Session->{plan}.')
        # Generate proxy mapping for tracking later
        # general_proxy_map: {Key: general model (right col), Value: proxy model (left col)
        # proxy_conf_map: {Key: proxy model, Value: confidence}
        general_proxy_map, proxy_conf_map = self._gen_proxy_mapping(hints.get('proxys', []))

        # {Key: proxy_model, Value: DataFrame to join with output of general_model}
        proxy_df_map = {}

        df = input_df

        # With img2vec, we need to cache columns (id, img2vec) regardless of path,
        # and join if a TASTI op runs at some point that isn't right after img2vec
        cached_dfs = {'img2vec': None, 'dfprefixembed': None}

        start_count = filt_count = df.select('id').distinct().count()
        cache_path = []
        plen = len(plan)
        self.log_time['total'] = 0
        for idx, n in enumerate(plan):
            logging.warn(f'Session->plan ({idx+1}/{plen}, {filt_count}/{start_count}): {n}.')
            model = str(n)
            model_for_proxy = ''.join(model[:model.find('->')]) if '->' in model else model

            # If caching is enabled, check whether we can reuse results
            cache_path.append(model)
            cache_str = '->'.join(cache_path)
            if self.do_caching and cache_str in self.df_cache:
                df = self.df_cache[cache_str]
            else:
                if canary_name is None:
                    # If this is a TASTI/model sharing op and the embedding was not
                    # the last op run, join it with the current df
                    if ('tasti' in model) and ('img2vec' not in df.columns):
                        # Because we already validated TASTI plans prior to running,
                        # it is guaranteed img2vec_cached_df will not be NONE here
                        logging.warn(f'{model} about to run and needs img2vec, joining')
                        df = df.join(cached_dfs['img2vec'], df.id == cached_dfs['img2vec'].id_img2vec, 'leftouter')\
                               .drop('id_img2vec')

                    elif ('deepfaceSuffix' in model) and ('dfprefixembed' not in df.columns):
                        logging.warn(f'{model} about to run DeepFace Model and needs DFPrefixEmbed, joining [Embed sz: %i]'%(cached_dfs['dfprefixembed'].count()))
                        df = df.join(cached_dfs['dfprefixembed'], df.id == cached_dfs['dfprefixembed'].id_dfprefixembed, 'leftouter')\
                               .drop('id_dfprefixembed')

                if canary_name is not None:
                    if model_for_proxy in self._acc_cache:
                        df_temp = self._acc_cache[model_for_proxy]
                    else:
                        canary_path = self.config.get_value('storage', 'canary')
                        next_path = os.path.join(canary_path, 'canary_results_cache', canary_name)
                        path_name = os.path.join(next_path, '%s_%s.parquet' % (model_for_proxy, canary_name))
                        df_temp = self.spark.read.parquet(path_name)

                        # Cache result so we don't need to re-read
                        self._acc_cache[model_for_proxy] = df_temp.cache()

                    df = df_temp.join(df, df_temp.id ==  df.id, "leftsemi")

                if canary_name is None:
                    # Remove duplicates to (a) not process double, and (b) not
                    # double-count in filters with aggregations.
                    df = df.dropDuplicates(['id'])

                    # Sort (needed for ops like action detect)
                    df = df.sort('id')

                    # Run op if we are not in accuracy measurement mode
                    df = n.apply_op(df)

            # If caching is enabled, save the DataFrame as-is for possible reuse later
            if self.do_caching:
                # Always overwrite if in the cache already
                self.df_cache[cache_str] = df.cache()

            if 'img2vec' in model:
                cached_dfs['img2vec'] = df.select(col('id').alias('id_img2vec'), 'img2vec')

            elif 'dfprefixembed' in model:
                cached_dfs['dfprefixembed'] = df.select(col('id').alias('id_dfprefixembed'), 'dfprefixembed')

            if self.do_logging:
                start_op = now()
                df.count()
                end_op = now()
                self.log_time[f'{model}_op'] = end_op - start_op
                self.log_time['total'] += self.log_time[f'{model}_op']

            # If this is a proxy model, first run explode_preds and proxy quality filter
            if model_for_proxy in proxy_conf_map:
                df_t = explode_preds(df, model_for_proxy)
                df_t = proxy_quality_filter(df_t, proxy_conf_map[model_for_proxy])

                # Keep first for next node
                df = df_t[0]

                # Pop explode_preds from filter list
                n.pop_front_filter()

                # Run on second and save output for joining later
                proxy_df_map[model_for_proxy] = n.apply_filters(df_t[1])
            else:
                df = n.apply_filters(df)

            # If the node is in this map, check if a proxy ran before it
            equals_match = self._search_equals_hints(model_for_proxy, hints.get('equals', []), set(general_proxy_map))
            if (model_for_proxy in general_proxy_map) or (equals_match is not None):
                if equals_match is not None:
                    model_for_proxy = equals_match
                proxy_name = general_proxy_map[model_for_proxy]
                # If true, a proxy ran before it. If false, the proxy plan was not used
                if proxy_name in proxy_df_map:
                    # Merge the results if so
                    df = df.union(proxy_df_map[proxy_name])

            if self.do_logging:
                start_filt = now()
                filt_count = df.select('id').distinct().count()
                end_filt = now()
                self.log_time[f'{model}_filter'] = end_filt - start_filt
                self.log_time['total'] += self.log_time[f'{model}_filter']

        logging.warn(f'Session->plan ({plen}/{plen}): final frames {filt_count}/{start_count}.')
        return df
