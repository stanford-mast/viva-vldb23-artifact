import sys
import logging
import itertools
from copy import deepcopy
from typing import Dict, List, Type
from timeit import default_timer as now

from viva.hints.equals_hint import EqualsHint
from viva.nodes.node import Node
from viva.nodes.filters import proxy_quality_filter
from viva.plans.tests.test_utils import correctness
from viva.core.planner_utils import (
    _print_trees, str_traversal, leaf2tree, collect,
    uniqueify, deuniqueify
)

class Planner:
    def __init__(self, tree: Dict, hint_filters: Dict = {}, hints: Dict = {}):
        self.tree = tree
        self.hints = hints
        self._hint_filters = hint_filters
        self._all_trees = []
        self._all_plans = []

    def __repr__(self):
        to_print = self._all_trees if self._all_trees else [self.tree]
        all_str = 'Trees:\n'
        for tree in to_print:
            all_str += str_traversal(tree) + '\n'
        all_str += '\n'

        all_str += 'Hints:\n'
        for k in self.hints:
            all_str += str(self.hints[k])
        return all_str

    def print_trees(self):
        to_print = self._all_trees if self._all_trees else [self.tree]
        _print_trees(to_print)

    def get_hints(self, typ: str):
        return self.hints.get(typ, [])

    @property
    def all_trees(self):
        if self._all_trees:
            return self._all_trees
        else:
            self.generate_valid_trees()
            return self._all_trees

    @property
    def all_plans(self):
        if self._all_plans:
            return self._all_plans
        elif self._all_trees:
            self.generate_valid_plans()
            return self._all_plans
        else:
            self.generate_valid_trees()
            self.generate_valid_plans()
            return self._all_plans

    def generate_valid_trees(self):
        # skip generating trees when root of base tree is a skip
        logging.warn('Planner->generating trees.')
        if self.tree['val'] == 'skip':
            self._all_trees = [self.tree]
            return

        tmaps = {
            'umap': {},
            'vmap': {},
            'valid': [],
        }
        gt_tree_number = correctness(self.tree, self.hints)
        unique_tree = uniqueify(self.tree, tmaps)

        # generate all possible trees, subtrees and apply hints
        all_candidates = collect(unique_tree, self.hints, tmaps, {}, [])

        candidates = []
        # build all potential trees and subtrees
        for cand in all_candidates:
            root = cand['root']
            swaps = cand['swaps']
            for swap in swaps:
                root['children'] = list(swap)
                candidates.append(deepcopy(root))

        # prune until all trees are unique
        while True:
            candidate_plans = candidates + self._all_trees
            all_plan_combos = list(itertools.permutations(candidate_plans, 2))
            done = True
            for tree, leaf in all_plan_combos:
                merged = leaf2tree(tree, leaf, {}, [])
                if len(merged) == 0 and leaf in candidates:
                    # remove leaf from original candidates
                    candidates.remove(leaf)
                    done = False
                elif merged in self._all_trees:
                    # leaf is a result that is subsumed by a tree
                    if leaf in self._all_trees:
                        self._all_trees.remove(leaf)
                    elif leaf in candidates:
                        candidates.remove(leaf)
                    done = False
                elif merged not in self._all_trees and merged != []:
                    # new plan
                    self._all_trees.append(merged)
                    done = False
            if done is True:
                break

        num_unique_trees = len(self._all_trees)
        # _print_trees(self._all_trees)

        self._all_trees = deuniqueify(self._all_trees, tmaps['umap'])

        if gt_tree_number != num_unique_trees:
            print(f'Planner->unique trees: {num_unique_trees}, expected: {gt_tree_number}, final: {len(self._all_trees)}')
        else:
            logging.warn(f'Planner->number of trees: {len(self._all_trees)}.')

    def _print_plan_nodes(self, plan: List[Type[Node]], msg: str = None) -> None:
        if msg is not None:
            print(msg)
        print([str(pp) for pp in plan])

    def generate_valid_plans(self):
        # Track plans for duplicate checking. Key is concatenated string of node names.
        all_plan_str = set()
        logging.warn('Planner->generating valid plans.')

        for c in self._all_trees:
            # Confirm that the first value is root
            if c['val'] not in ['root', 'skip']:
                print('Invalid tree: root is not at the top!')
                sys.exit(1)
            # Recursively go through the children and add nodes to list
            node_list = []
            for child in c['children']:
                node_list = self._generate_valid_plans_helper(child, node_list)

            # We've completed a plan. Remove duplicates and check whether it's
            # a duplicate before adding to all_plans
            node_list_clean = self._remove_duplicates(deepcopy(node_list))

            # need to add objectdetect as an incol to objecttrack
            self._update_ot_in_cols(node_list_clean)

            plan_name = ''.join([n.out_column for n in node_list_clean])
            if (plan_name not in all_plan_str):
                self._all_plans.append(node_list_clean)
                all_plan_str.add(plan_name)

        logging.warn(f'Planner->number of valid plans: {len(self._all_plans)}.')

    def _generate_valid_plans_helper(self, child_to_eval: Dict, curr_plan_list: List) -> List:
        from viva.nodes.node_mappings import NodeMappings
        val = child_to_eval['val']
        children = child_to_eval['children']

        # Stop when my children's list is empty
        if not children:
            # We've reached the bottom
            # Create the node and add to the back of the curr_plan_list
            if val not in NodeMappings:
                print(val, 'not found in node mapping!')
                sys.exit(1)

            next_node = self._get_next_node(val)
            # insert img2vec or prefix in node list before adding the required
            # node
            if 'tasti' in next_node.out_column:
                tasti_node = self._get_next_node('i2v')
                curr_plan_list.append(tasti_node)
            elif 'deepfaceSuffix' in next_node.out_column:
                prefix_node = self._get_next_node('dfprefix')
                curr_plan_list.append(prefix_node)

            curr_plan_list.append(next_node)
            return curr_plan_list

        # Recursively iterate over my children
        for child in children:
            curr_plan_list = self._generate_valid_plans_helper(child, curr_plan_list)

            # If type string, we have a node dependency. Create the node and add to the back of the curr_plan_list
            if isinstance(val, str):
                if val not in NodeMappings:
                    print(val, 'not found in node mapping!')
                    sys.exit(1)

                next_node = self._get_next_node(val)
                curr_plan_list.append(next_node)
            # check before going through val if filters are empty. if they aren't means hints added some
            # at this point aren't planning to combine hint_filters and
            # original ones so can safely do this logic
            elif len(curr_plan_list[-1].filters) == 0:
                for func in val:
                    if isinstance(func, tuple):  # args has to be a list
                        func, args = func
                        curr_plan_list[-1].add_filter(func, args)
                    elif func.__name__ == 'explode_preds':
                        # Special case for explode_preds: we add the out_column argument
                        curr_plan_list[-1].add_filter(func, [curr_plan_list[-1].out_column])
                    else:
                        curr_plan_list[-1].add_filter(func)

        return curr_plan_list

    def _get_next_node(self, node_name: str) -> Type[Node]:
        from viva.nodes.node_mappings import NodeMappings
        next_node = deepcopy(NodeMappings[node_name])

        def add_filters():
            for func in self._hint_filters[node_name]:
                if isinstance(func, tuple): # args has to be a list
                    func, args = func
                    next_node.add_filter(func, args)
                elif func.__name__ == 'explode_preds':
                    # Special case for explode_preds: we add the out_column argument
                    next_node.add_filter(func, [next_node.out_column])
                else:
                    next_node.add_filter(func)

        # For all add corresponding filters with nodes
        for hint_type, list_hints in self.hints.items():
            for h in list_hints:
                left_arg, right_arg = h.get_args()
                left_arg_col = NodeMappings[left_arg].out_column
                last_added_node_col = next_node.out_column
                if left_arg_col == last_added_node_col: # or right_arg_col == last_added_node_col:
                    if node_name not in self._hint_filters:
                        if hint_type != 'equals':
                            print(node_name, 'not found in hint filters!')
                            sys.exit(1)
                        else:
                            continue
                    elif node_name in self._hint_filters and hint_type == 'equals':
                        next_node._filters = []
                    add_filters()

        return next_node

    def _update_ot_in_cols(self, plan: List[Type[Node]]):
        """
        object track depends on objectdetect which in a tree comes always before
        make sure the in columns for spark are updated and the filters are added
        """
        odmap = {
            'objectdetect': 'od',
            'objectdetect_large': 'odl',
            'objectdetect_medium': 'odm',
            'objectdetect_nano': 'odn',
            'objectdetect_xlarge': 'odx'
        }
        from viva.nodes.node_mappings import inp_col_list
        found_track = False
        for p in plan:
            if p.out_column == 'objecttrack':
                found_track = True
        if found_track == False:
            return

        for idx,p in enumerate(plan):
            if p.out_column in odmap and odmap[p.out_column] in self._hint_filters:
                node_name = odmap[p.out_column]
                p._filters = []
                for func in self._hint_filters[node_name]:
                    if isinstance(func, tuple):
                        func, args = func
                        p.add_filter(func, args)
                    elif func.__name__ == 'explode_preds':
                        # Special case for explode_preds: we add the out_column argument
                        p.add_filter(func, [p.out_column])
                    else:
                        p.add_filter(func)
            elif p.out_column == 'objecttrack':
                p._in_columns = inp_col_list + [plan[idx-1].out_column]

    # Remove duplicate node calls by merging all filters to first call of a node
    def _remove_duplicates(self, plan: List[Type[Node]]) -> List[Type[Node]]:
        # Track nodes and all filters
        # {Key: node_name, Value: [filter0, filter1, ...]}
        node_filter_map = {}

        # First pass: collect all nodes and their respective filters
        for p in plan:
            model = p.out_column
            if model not in node_filter_map:
                node_filter_map[model] = p.filters
            else:
                # Iterate rather than extend to exclude explode_preds
                for f in p.filters:
                    if f[0].__name__ != 'explode_preds':
                        node_filter_map[model].append(f)

        # Second pass: if the length of the filter list in node_filter_map is
        # longer than a node's filter, it means a merge is possible.
        # If a node is not in the map, it means we should remove it from the plan
        new_plan_list = []
        for p in plan:
            model = p.out_column
            if model in node_filter_map:
                filter_list = node_filter_map[model]
                if len(p.filters) < len(filter_list):
                    # Append all new filters
                    p.filters.extend(filter_list[len(p.filters):])
                new_plan_list.append(p)

                # Remove from node_filter_map to prune duplicates
                del node_filter_map[model]

        return new_plan_list
