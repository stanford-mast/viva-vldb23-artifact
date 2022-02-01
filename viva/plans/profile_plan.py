# A plan used for profiling ops
from viva.core.planner import Planner

from viva.plans.plan_filters import add_field_to_struct

ProfileTree = {
    'val': 'skip',
    'children': [
        {
            'val': 'ts',
            'children': []
        },
        {
            'val': 'si',
            'children': []
        },
        {
            'val': 'bi',
            'children': []
        },
        {
            'val': 'svm',
            'children': []
        },
        {
            'val': 'md',
            'children': []
        },
        {
            'val': 'fd',
            'children': []
        },
        {
            'val': 'ed',
            'children': []
        },
        {
            'val': 'edc',
            'children': []
        },
        {
            'val': 'cl',
            'children': []
        },
        {
            'val': 'qcl',
            'children': []
        },
        {
            'val': 'pc',
            'children': []
        },
        {
            'val': 'ad',
            'children': []
        },
        {
            'val': 'odx',
            'children': []
        },
        {
            'val': 'odm',
            'children': []
        },
        {
            'val': 'odl',
            'children': []
        },
        {
            'val': 'odn',
            'children': []
        },
        {
            'val': 'ot',
            'children': [
                {
                    'val': [(add_field_to_struct, ['objectdetect'])],
                    'children': [
                        {
                            'val': 'od',
                            'children': []
                        }
                    ]
                }
            ]
        },
        {
            'val': 'tod',
            'children': [
                {
                    'val': 'i2v',
                    'children': []
                }
            ]
        },
        {
            'val': 'ted',
            'children': [
                {
                    'val': 'i2v',
                    'children': []
                }
            ]
        },
        {
            'val': 'tfd',
            'children': [
                {
                    'val': 'i2v',
                    'children': []
                }
            ]
        },
        {
            'val': 'tad',
            'children': [
                {
                    'val': 'i2v',
                    'children': []
                }
            ]
        },
        {
            'val': 'dfage',
            'children': []
        },
        {
            'val': 'dfgender',
            'children': []
        },
        {
            'val': 'dfrace',
            'children': []
        },
        {
            'val': 'dfsuffixage',
            'children': [
                {
                    'val': 'dfprefix',
                    'children': []
                }
            ]
        },
        {
            'val': 'dfsuffixgender',
            'children': [
                {
                    'val': 'dfprefix',
                    'children': []
                }
            ]
        },
        {
            'val': 'dfsuffixrace',
            'children': [
                {
                    'val': 'dfprefix',
                    'children': []
                }
            ]
        },
    ]
}

ProfilePlan = Planner(ProfileTree)
