import os
from copy import deepcopy

from viva.utils.config import ConfigManager
config = ConfigManager()
proxy_confidence_thresh = config.get_value('execution', 'proxy_confidence_thresh')

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint
from viva.core.planner import Planner

from viva.nodes.filters import explode_preds
from viva.plans.plan_filters import object_filter, faces_lebron_expand, similarity_filter

DunkHints = {
    'equals': [],
    'supersets': [],
    'proxys': []
}

DunkHintFilters = {
}

# Check if experiment file is present; if so, use those hints/filters instead
if os.path.exists(os.path.join(os.path.dirname(__file__), 'experiment_hints.py')):
    from viva.plans.experiment_hints import DunkHints, DunkHintFilters

DunkTree = {
    'val': 'root',
    'children': [
        {
            'val': [explode_preds, faces_lebron_expand],
            'children': [
                {
                    'val': 'fd',
                    'children': []
                }
            ]
        },
        {
            'val': [explode_preds, (object_filter, ['dunking basketball'])],
            'children': [
                {
                    'val': 'ad',
                    'children': []
                }
            ]
        }
    ]
}

DunkPlan = Planner(DunkTree, DunkHintFilters, DunkHints)
DunkCanaryTree = deepcopy(DunkTree)
DunkCanaryTree['val'] = 'skip' # skip generating multiple trees
DunkCanaryPlan = Planner(DunkCanaryTree)
