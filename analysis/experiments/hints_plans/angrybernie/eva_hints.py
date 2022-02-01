from viva.utils.config import ConfigManager
config = ConfigManager()
proxy_confidence_thresh = config.get_value('execution', 'proxy_confidence_thresh')

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint

from viva.nodes.filters import explode_preds, quality_filter
from viva.plans.plan_filters import two_people, faces_and, object_filter

BernieHints = {
    'equals': [
        EqualsHint('odx', 'od'),
        EqualsHint('ed', 'edc')
    ],
    'supersets': [
    ],
    'proxys': [
    ]
}

# Only add in here if a superset or proxy
BernieHintFilters = {
}
