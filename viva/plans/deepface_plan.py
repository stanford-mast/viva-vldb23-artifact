import os
from copy import deepcopy
from pyspark.sql.functions import col, ltrim, count, struct, collect_set, array_contains

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint
from viva.core.planner import Planner

from viva.nodes.filters import explode_preds
from viva.plans.plan_filters import object_filter,age_old_filter, objects_or

DeepFaceHints = {
    'equals': [
        EqualsHint('dfage', 'dfsuffixage'),
        EqualsHint('dfgender', 'dfsuffixgender'),
        EqualsHint('dfrace', 'dfsuffixrace')
    ],
    'supersets': [
    ],
    'proxys': []
}

DeepFaceHintFilters = {
    'dfsuffixage': [explode_preds, age_old_filter],
    'dfsuffixgender': [explode_preds, (object_filter, ['Woman'])],
    'dfsuffixrace': [explode_preds, (objects_or, [['asian', 'indian', 'black','middle eastern', 'latino hispanic']])]
}

if os.path.exists(os.path.join(os.path.dirname(__file__), 'experiment_hints.py')):
    from viva.plans.experiment_hints import DeepFaceHints, DeepFaceHintFilters

DeepFaceTree = {
    'val': 'root',
    'children': [
        {
            'val': [ explode_preds, age_old_filter],
            'children': [
                {
                    'val': 'dfage',
                    'children': []
                }
            ]
        },
        {
            'val': [ explode_preds,(object_filter, ['Woman'])],
            'children': [
                {
                    'val': 'dfgender',
                    'children': []
                }
            ]
        },
        {
            'val': [ explode_preds, (objects_or, [['asian', 'indian', 'black','middle eastern', 'latino hispanic']])],
            'children': [
                {
                    'val': 'dfrace',
                    'children': []
                }
            ]
        }
    ]

}

DeepFacePlan = Planner(DeepFaceTree, DeepFaceHintFilters, DeepFaceHints) 
DeepFaceCanaryTree = deepcopy(DeepFaceTree)
DeepFaceCanaryTree['val'] = 'skip' # skip generating multiple trees
DeepFaceCanaryPlan = Planner(DeepFaceCanaryTree)
