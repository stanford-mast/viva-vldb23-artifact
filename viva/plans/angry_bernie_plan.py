import os
from copy import deepcopy

from viva.utils.config import ConfigManager
config = ConfigManager()
proxy_confidence_thresh = config.get_value('execution', 'proxy_confidence_thresh')

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint
from viva.core.planner import Planner

from viva.nodes.filters import explode_preds, quality_filter
from viva.plans.plan_filters import two_people, faces_and, object_filter, similarity_filter

BernieHints = {
    'equals': [
        EqualsHint('odx', 'od'),
        EqualsHint('od', 'odn'),
        EqualsHint('ed', 'edc')
    ],
    'supersets': [],
    'proxys': []
}

# Only add in here if a superset or proxy
BernieHintFilters = {
}

# Check if experiment file is present; if so, use those hints/filters instead
if os.path.exists(os.path.join(os.path.dirname(__file__), 'experiment_hints.py')):
    from viva.plans.experiment_hints import BernieHints, BernieHintFilters

AngryBernieTree = {
    'val': 'root',
    'children': [
        {
            'val': [explode_preds, faces_and],
            'children': [
                {
                    'val': 'fd',
                    'children': []
                }
            ]
        },
        {
            'val': [explode_preds, quality_filter, two_people],
            'children': [
                {
                    'val': 'odx',
                    'children': []
                }
            ]
        },
        {
            'val': [explode_preds, (object_filter, ['angry'])],
            'children': [
                {
                    'val': 'ed',
                    'children': []
                }
            ]
        }
    ]
}

AngryBerniePlan = Planner(AngryBernieTree, BernieHintFilters, BernieHints)
AngryBernieCanaryTree = deepcopy(AngryBernieTree)
AngryBernieCanaryTree['val'] = 'skip' # skip generating multiple trees
AngryBernieCanaryPlan = Planner(AngryBernieCanaryTree)
