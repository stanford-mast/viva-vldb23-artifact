from viva.utils.config import ConfigManager
config = ConfigManager()
proxy_confidence_thresh = config.get_value('execution', 'proxy_confidence_thresh')

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint

from viva.nodes.filters import explode_preds, quality_filter
from viva.plans.plan_filters import (objects_or, add_field_to_struct, calculate_movement,
                                     remove_stationary, find_directions, left_turns)

AmsterdamHints = {
    'equals'   : [
    ],
    'supersets': [
    ],
    'proxys': [
        ProxyHint('tod', 'odx', proxy_confidence_thresh)
    ]
}

AmsterdamHintFilters = {
    'tod' : [(add_field_to_struct, ['objectdetect_xlarge'])]
}
