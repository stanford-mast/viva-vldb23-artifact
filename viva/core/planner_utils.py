import sys
import itertools
from copy import deepcopy
from uuid import uuid1
import logging

DEBUG = False

def _myprint(text, debug=DEBUG):
    if debug:
        print(text)

def _val_to_str(val):
    if isinstance(val, list):
        val = [v.__name__ if callable(v) else str(v) for v in val]
    elif callable(val):
        val = val.__name__

    return str(val)

def fluff(rep, depth):
    rep += '\n'
    if depth > 0:
        rep += '|    '
        sp = ['     ']
        rep += ''.join(sp*(depth-1)) + '+--- '
    else:
        rep += '+--- '
    return rep

def str_traversal(tree, rep = '', depth = -1):
    val = _val_to_str(tree['val'])
    rep += f'{val}'
    if 'children' in tree and len(tree['children']) > 0:
        depth += 1
        rep = fluff(rep, depth)
        for idx, child in enumerate(tree['children']):
            rep = str_traversal(child, rep, depth)
            if idx+1 != len(tree['children']):
                rep = fluff(rep, depth)
        return rep
    return rep

def _print_trees(trees):
    if not isinstance(trees, list):
        trees = [trees]
    all_str = ''
    for idx, tree in enumerate(trees):
        all_str += str_traversal(tree)
        if idx+1 != len(trees):
            all_str += '\n'
    print(all_str)

def range_permutations(elts, start = 1):
    return [
        x for l in range(start, len(elts)+1) for x in itertools.permutations(elts, l)
    ]

def swaps(elts):
    return list(itertools.permutations(elts))

def leaf2tree(tree, leaf, plan, leaf_bool):
    """
    given a leaf, find it's parent
    if the leaf doesn't have a parent, return empty
    leaf_bool is a list because of the recursion. sets flag '1' if found and
    merged to its parent.
    """
    plan['val'] = tree['val']
    plan['children'] = []
    for idx, child in enumerate(tree['children']):
        if child['val'] == leaf['val'] and leaf not in plan['children']:
            plan['children'].append(leaf)
            leaf_bool.append(1)
        else:
            new_child = {}
            leaf2tree(child, leaf, new_child, leaf_bool)
            plan['children'].append(new_child)
            leaf_bool.append(0)

    return plan if sum(leaf_bool) else []

def umap_insert(umap, val):
    rep = _val_to_str(val)
    unique = f'{rep}_{len(umap.keys())}'
    umap[unique] = val
    return unique

def insert(maps, hint_val, tree_val):
    def vmap_insert(vmap, tree_val, hint_val, unique_hint_val):
        """
        tracks provenance of equals values that can be traced back to the
        original value.
        Helps find if a unique value is a result of a hint or an original tree
        value.
        """
        def _add_val(tree_val, hint_val, unique_hint_val):
            vmap[tree_val]['vals'].append(hint_val)
            vmap[tree_val]['unique_vals'].append(unique_hint_val)

        tree_val = _val_to_str(tree_val)
        # Option 1: this is an original value and has its own track list. Need
        # to check this first because an original value can also show up in
        # another's track list if an EQ hint was already applied to it.
        if tree_val in vmap:
            _add_val(tree_val, hint_val, unique_hint_val)
            return

        # Option 2: check if this value is hint generated and in the track list
        # of original values
        for t in vmap:
            if tree_val in vmap[t]['vals']:
                _add_val(t, hint_val, unique_hint_val)
                return

        # Option 3: this is a new val. init the track list of where values came
        # from.
        if tree_val not in vmap:
            vmap[tree_val] = {
                'vals': [],
                'unique_vals': []
            }
            _add_val(tree_val, hint_val, unique_hint_val)
            return

        logging.error('Planner->vmap_insert: should never happen')
        sys.exit(1)

    unique = umap_insert(maps['umap'], hint_val)
    vmap_insert(maps['vmap'], tree_val, hint_val, unique)
    return unique

def apply_eq_unique(candidates, eqhints, maps, leafs):
    """
    if an equals is found, check it hasn't been found yet and generate it's
    parent to reconstruct the tree.
    executes until no new hints plans are generated
    """
    # mapping A_0 -> A, A_1 -> A
    umap = maps['umap']
    # order: A = [A_0, A_1]
    valid = maps['valid']

    def find_parents(candidates, val, hint_val):
        """
        given a valid eqhint, search candidates where the original value shows up,
        copy and replace with the hint. this creates the relationship to rebuild a
        tree with the hint.
        """
        parents = []
        cand_seen = []
        for cand in candidates:
            swaps = cand['swaps']
            for idx, swap in enumerate(swaps):
                for tidx, tup in enumerate(swap):
                    if tup['val'] == val and cand not in cand_seen:
                        parent = deepcopy(cand)
                        parent['swaps'][idx][tidx]['val'] = hint_val
                        parents.append(parent)
                        cand_seen.append(cand)
        return parents

    def _test_eq_hint(tree_val, unique_val, hint):
        # Can't apply the same hint to a hint generated value
        # Can apply different hints to the same value or non hint generated value
        unique_hint_val = None
        if tree_val == hlc and (hint, unique_val) not in valid:
            hint_val = hrc
            unique_hint_val = insert(maps, hint_val, tree_val)
            # this hint was applied to this unique_value
            valid.append((hint, unique_val))
            # if its hint generated need to add it for the next iteration
            # paired with this hint
            valid.append((hint, unique_hint_val))
        elif tree_val == hrc and (hint, unique_val) not in valid:
            hint_val = hlc
            unique_hint_val = insert(maps, hint_val, tree_val)
            # this hint was applied to this unique_value
            valid.append((hint, unique_val))
            # if its hint generated need to add it for the next iteration
            # paired with this hint
            valid.append((hint, unique_hint_val))

        return unique_hint_val

    all_cands = candidates
    while True:
        done = True
        all_cands.extend(leafs)
        for hint, candidate in itertools.product(eqhints, all_cands):
            hlc, hrc = hint.get_args()
            unique_val = candidate['root']['val']
            # tree_val is original tree value
            tree_val = umap[unique_val]
            unique_hint_val = _test_eq_hint(tree_val, unique_val, hint)
            if unique_hint_val == None:
                continue

            _myprint(f'node: {tree_val} hint: {hint} {unique_val}->{unique_hint_val}')
            leaf = {
                'root': {
                    'val': unique_hint_val,
                    'children': []
                },
                'swaps': deepcopy(candidate['swaps'])
            }
            if leaf not in all_cands+leafs:
                leafs.append(leaf)
                all_cands.extend([
                    p for p in find_parents(all_cands, unique_val, unique_hint_val)
                    if p not in all_cands+leafs
                ])
                done = False
        # no more options
        if done:
            return leafs

def _gen_parent(cand, unique_hint_val):
    parent = deepcopy(cand)
    parent['root']['children'] = []
    parent['swaps'] = [
        ({'val': unique_hint_val, 'children': list(swap)},)
        for swap in parent['swaps']
    ]
    return parent

def _ss_follow_hint(all_cands, vmap, hint, unique_hint_val, valid):
    """
    UPDATE DS
    """

    hlc, hrc = hint.get_args()

    generated = []
    for cand in all_cands:
        if unique_hint_val in [tup['val'] for swap in cand['swaps'] for tup in swap]:
            # hint value already in the tuples, don't need to generate it again
            continue

        unique_val = cand['root']['val']
        # the supersetted value is an original value that has a list in the vmap
        if hrc in vmap and \
           unique_val in vmap[hrc]['unique_vals'] and \
           (hint, unique_val) not in valid and \
           unique_val not in generated:
            parent = _gen_parent(cand, unique_hint_val)
            if parent not in all_cands:
                all_cands.append(parent)
                _myprint(f'  found: {hrc}->{unique_hint_val} SS {unique_val}')
                valid.append((hint, unique_val))
                valid.append((hint, unique_hint_val))
            generated.append(unique_val)
        else:
            # the supersetted value (hrc) is generated from a hint so it's
            # in one of the lists in vmap
            for k in vmap:
                if hrc in vmap[k]['vals'] and \
                   unique_val in vmap[k]['unique_vals'] and \
                   (hint, unique_val) not in valid and \
                   unique_val not in generated:
                    parent = _gen_parent(cand, unique_hint_val)
                    if parent not in all_cands:
                        all_cands.append(parent)
                        _myprint(f'  found: {hrc}->{unique_hint_val} SS {unique_val}')
                        valid.append((hint, unique_val))
                        valid.append((hint, unique_hint_val))
                    generated.append(unique_val)

def apply_insert_unique(candidates, hints, maps, leafs):
    """
    find LC SS/PX RC where RC in plan.
    Inserts LC into plan and points original node to the new child.
    """

    # mapping A_0 -> A, A_1 -> A
    umap = maps['umap']
    valid = maps['valid']
    vmap = maps['vmap']
    # pprint(umap)
    # pprint(vmap)
    # _myprint(valid)

    all_cands = candidates
    while True:
        done = True
        all_cands.extend(leafs)
        for hint, candidate in itertools.product(hints, all_cands):
            hlc, hrc = hint.get_args()
            unique_val = candidate['root']['val']
            tree_val = umap[unique_val]
            if tree_val == hrc and tree_val != hlc and (hint, unique_val) not in valid:
                hint_val = hlc
                unique_hint_val = umap_insert(umap, hint_val)
                # this hint was applied to this unique_value
                valid.append((hint, unique_val))
                # if its hint generated need to add it for the next iteration
                # paired with this hint
                valid.append((hint, unique_hint_val))
                leaf = {
                    'root': {
                        'val': unique_hint_val,
                        'children': []
                    },
                    'swaps': deepcopy(candidate['swaps'])
                }
                if leaf not in all_cands:
                    _myprint(f'node: {tree_val} hint: {hint} {unique_val}->{unique_hint_val}')
                    leafs.append(leaf)
                    parent = _gen_parent(candidate, unique_hint_val)

                    if parent not in all_cands:
                        all_cands.append(parent)
                    _ss_follow_hint(all_cands, vmap, hint, unique_hint_val, valid)
                    done = False

        # no more options
        if done:
            return leafs

def collect(tree, hints, maps, plan, candidates):
    """
    recursively traverse and generate permuations at each branch point
    traverse and find equivalent operations defined by eqhints.
    traverse and insert superset hints
    """
    eqhints = hints.get('equals', [])
    inserts = hints.get('supersets', []) + hints.get('proxys', [])

    plan['val'] = tree['val']
    plan['children'] = []
    candidates.append({
        'root': plan,
        'swaps': swaps(tree['children'])
    })

    # apply hints until no change
    cur_len = len(candidates)
    while True:
        candidates.extend(apply_eq_unique(candidates, eqhints, maps, []))
        candidates.extend(apply_insert_unique(candidates, inserts, maps, []))
        new_len = len(candidates)
        if new_len == cur_len:
            break
        cur_len = new_len

    for child in tree['children']:
        new_child = {}
        collect(child, hints, maps, new_child, candidates)
        plan['children'].append(new_child)

    return candidates

def uniqueify(tree, maps, plan = {}):
    plan['val'] = insert(maps, tree['val'], tree['val'])
    plan['children'] = []
    for child in tree['children']:
        new_child = {}
        uniqueify(child, maps, new_child)
        plan['children'].append(new_child)

    return plan

def _deuniqueify(tree, plan, umap):
    plan['val'] = umap[tree['val']]
    plan['children'] = []
    for child in tree['children']:
        new_child = {}
        _deuniqueify(child, new_child, umap)
        plan['children'].append(new_child)

    return plan

def deuniqueify(trees: list, umap):
    """
    replace values back into trees
    """
    orig = []
    uniquestr = set()
    for tree in trees:
        unique_tree = _deuniqueify(tree, {}, umap)
        unique_str_tree = str_traversal(unique_tree)
        if unique_str_tree not in uniquestr:
            uniquestr.add(unique_str_tree)
            orig.append(unique_tree)
    return orig
