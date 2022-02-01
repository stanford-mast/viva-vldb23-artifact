import sys
import itertools
from math import factorial, prod
from time import sleep
from viva.core.planner_utils import _val_to_str

def lineage(val, hints):
    sslineage = []
    eqlineage = []
    invalid = []
    while True:
        done = True
        total = [val]+sslineage+eqlineage
        for cval, hint in list(itertools.product(total, hints)):
            hlc, hrc = hint.get_args()
            if hint.name == 'EQ':
                if cval == hlc and hrc not in eqlineage and (hint, cval) not in invalid:
                    eqlineage.append(hrc)
                    done = False
                    # check if the other side of the hint would set
                    # treeval == hlc. this is invalid
                    if hlc == cval and cval == val:
                        invalid.append((hint, cval))
                elif cval == hrc and hlc not in eqlineage and (hint, cval) not in invalid:
                    eqlineage.append(hlc)
                    done = False
                    # check if the other side of the hint would set
                    # treeval == hrc. this is invalid
                    if hrc == cval and cval == val:
                        invalid.append((hint, cval))
            elif hint.name in ['SS', 'PX'] \
                    and hrc in [val]+eqlineage+sslineage \
                    and hlc not in sslineage:
                sslineage.append(hlc)
                done = False

        if done:
            return eqlineage, sslineage

def hint_validator(hints):
    allh = sum([hints[k] for k in hints],[])
    conflicts = []
    for h1, h2 in itertools.combinations(allh, 2):
        h1lc, h1rc = h1.get_args()
        h2lc, h2rc = h2.get_args()
        if h1lc == h2lc and h1rc == h2rc and h1.name != h2.name:
            conflicts.append((h1, h2))
        elif h1lc == h2rc and h1rc == h2lc \
                and h1.name in ['SS', 'PX'] \
                and h2.name in ['SS', 'PX']:
            conflicts.append((h1, h2))

    if conflicts:
        print('Different hint types cannot have the same or conflicting arguments. Conflicts:')
        print(conflicts)
        sys.exit(1)

def correctness(tree, hints = {}):
    """
    traverse and find length of all children to count number of valid plans
    test hints
    """
    allh = sum([hints[k] for k in hints],[])

    # Key: cval, Value: set(unique values that can be taken not including cval) original valueset()
    all_eq_unique = {}
    all_ss_px_unique = {}
    numPerms = 1
    def dfs(tree, plan):
        nonlocal numPerms # Bind to the outer numPerms rather than trying to reassign
        plan['val'] = tree['val']
        plan['children'] = []
        cvals = list(_val_to_str(c['val']) for c in tree['children'])

        for cval in cvals:
            eqlineage, sslineage = lineage(cval, allh)
            if cval not in all_eq_unique:
                all_eq_unique[cval] = set()
                all_ss_px_unique[cval] = set()

            all_eq_unique[cval].update(set(eqlineage))
            all_ss_px_unique[cval].update(set(sslineage))
        if len(tree['children']) > 0:
            numPerms *= factorial(len(tree['children']))

        for child in tree['children']:
            new_child = {}
            dfs(child, new_child)
            plan['children'].append(new_child)

        return plan

    hint_validator(hints)
    dfs(tree, {})

    ### Compute number of plans ###

    # Get bins for equals
    # Subtract one for each key since that includes the original one
    eq_bins = []
    for _,v in all_eq_unique.items():
        if len(v):
            eq_bins.append(len(v) - 1)

    # Get bins for superset/proxy
    ss_px_bins = []
    for _,v in all_ss_px_unique.items():
        if len(v):
            ss_px_bins.append(len(v))

    numBins = len(ss_px_bins) + len(eq_bins)
    bins = eq_bins + ss_px_bins
    numPlans = numPerms
    for i in range(1, numBins+1):
        allCombinations = itertools.combinations(bins, i)
        factorialSum = 0
        for comb in allCombinations:
            optionsProduct = prod(list(comb))
            factorialSum += numPerms * optionsProduct 
        numPlans += factorialSum

    return numPlans
