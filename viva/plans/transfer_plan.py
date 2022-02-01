# A plan used for profiling ops
from viva.core.planner import Planner
from viva.nodes.filters import explode_preds, quality_filter
from viva.plans.plan_filters import two_people, faces_and, object_filter, similarity_filter, add_field_to_struct

TransferTree = {
    'val': 'skip',
    'children': [
        {
            'val': 'overhead',
            'children': []
        },
        {
            'val': 'transfer',
            'children': []
        },
    ]
}

TransferTree = {
    'val': 'skip',
    'children': [
        {
            # 'val': 'odn',
            # 'val': 'fd',
            'val': 'edc',
            'children': []
        }
    ]
}
# TransferTree = {
#     'val': 'skip',
#     'children': [
#         {
#             'val': 'ot',
#             'children': [
#                 {
#                     'val': [(add_field_to_struct, ['objectdetect_nano'])],
#                     'children': [
#                         {
#                             'val': 'od',
#                             'children': []
#                         }
#                     ]
#                 }
#             ]
#         }
#     ]
# }

TransferPlan = Planner(TransferTree)
