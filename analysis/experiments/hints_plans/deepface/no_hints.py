from viva.utils.config import ConfigManager
config = ConfigManager()

from viva.hints.superset_hint import SupersetHint
from viva.hints.equals_hint import EqualsHint
from viva.hints.proxy_hint import ProxyHint

from viva.nodes.filters import explode_preds, quality_filter
from viva.plans.plan_filters import two_people, faces_and, object_filter

DeepFaceHints = {
    'equals': [],
    'supersets': [],
    'proxys': []
}

DeepFaceHintFilters = {}
