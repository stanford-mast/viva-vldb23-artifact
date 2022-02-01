import os
import sys
import json
from typing import Dict

import viva.plans.plan_filters as pf
from viva.core.planner import Planner
from viva.nodes.filters import explode_preds

""" Currently assumes plans are single-level """
def generate_plan_from_query(query_plan: Dict):
    PlanTree = {'val': 'root'}

    # Iterate over model/filters from list
    all_children = []
    for p in query_plan:
        curr_filters = [explode_preds]
        for f in p['filters']:
            func = getattr(pf, f['func'])
            args = f.get('args', None)
            if args is None:
                curr_filters.append(func)
            else:
                curr_filters.append((func, args))
        model = p['model']

        model_child = [{'val': model, 'children': []}]
        model_filter_child = {'val': curr_filters, 'children': model_child}
        all_children.append(model_filter_child)

    PlanTree['children'] = all_children
    NewPlan = Planner(PlanTree)

    return NewPlan

