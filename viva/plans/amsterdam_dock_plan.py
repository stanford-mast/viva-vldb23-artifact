import os
from copy import deepcopy

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint
from viva.core.planner import Planner

from viva.nodes.filters import explode_preds, quality_filter
from viva.plans.plan_filters import (objects_or, object_filter, add_field_to_struct,
                                     calculate_movement, remove_stationary, find_directions,
                                     left_turns)

AmsterdamHints = {
    'equals'   : [
        EqualsHint('odx', 'od'),
        EqualsHint('od', 'odn'),
        EqualsHint('bi', 'svm'),
    ],
    'supersets': [
        SupersetHint('qcl', 'odx')
    ],
    'proxys': []
}

AmsterdamHintFilters = {
    'odn' : [(add_field_to_struct, ['objectdetect_nano'])],
    'od'  : [(add_field_to_struct, ['objectdetect'])],
    'qcl' : [explode_preds, (objects_or, [['passenger car', 'streetcar', 'sports car', 'minivan', 'motor scooter', 'moped']])],
}

# Check if experiment file is present; if so, use those hints/filters instead
if os.path.exists(os.path.join(os.path.dirname(__file__), 'experiment_hints.py')):
    from viva.plans.experiment_hints import AmsterdamHints, AmsterdamHintFilters

AmsterdamDockTree = {
    'val': 'root',
    'children': [
        {
            'val': [explode_preds, quality_filter, (objects_or, [['car', 'person']]), \
                    calculate_movement, remove_stationary, find_directions],
            'children': [
                {
                    'val': 'ot',
                    'children': [
                        {
                            'val': [(add_field_to_struct, ['objectdetect_xlarge'])],
                            'children': [
                                {
                                    'val': 'odx',
                                    'children': []
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        {
            'val': [explode_preds, (object_filter, ['night'])],
            'children': [
                {
                    'val': 'svm',
                    'children': []
                }
            ]
        }
    ]
}

AmsterdamDockPlan = Planner(AmsterdamDockTree, AmsterdamHintFilters, AmsterdamHints)
AmsterdamCanaryTree = deepcopy(AmsterdamDockTree)
AmsterdamCanaryTree['val'] = 'skip' # skip generating multiple trees
AmsterdamCanaryPlan = Planner(AmsterdamDockTree)
