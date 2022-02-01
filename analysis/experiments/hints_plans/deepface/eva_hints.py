from viva.utils.config import ConfigManager
config = ConfigManager()

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint

from viva.nodes.filters import explode_preds, quality_filter
from viva.plans.plan_filters import object_filter, age_old_filter, objects_or

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
    'dfsuffixage': [explode_preds, age_old_filter], #
    'dfsuffixgender': [explode_preds, (object_filter, ['Woman'])], #
    'dfsuffixrace': [explode_preds, (objects_or, [['asian', 'indian', 'black','middle eastern', 'latino hispanic']])] #
}
