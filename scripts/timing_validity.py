"""Validity rule for time-budgeted CP-SAT cells (used by p80 audit and p81).

A cell's result depends on the CPU it received, so every record carries
cpu_share = CPU seconds / (wall seconds x workers) and its affinity mask.
A cell counts as valid when it was validated, solved on the reserved cores
0-15, and one of the following holds:
  (a) cpu_share >= 0.9;
  (b) the solve proved optimality, so the value does not depend on the CPU
      it received;
  (c) its cpu_share is no more than 0.05 below the 5th percentile of its cell
      type (folder x with/without setups). Short budgets and large setup
      models spend a fixed single-threaded phase (model build and presolve)
      inside the budget, so their share is structurally below 0.9: about
      0.73 for 1-s solves at 30x25 and 0.88 for 600-s solves on 80x25 setup
      models. Contention lowers one cell against its own type, and rule (c)
      catches that. The bar per type is computed once, over the campaign as
      it ended (re-solved outliers included), and frozen in
      or_solution/FAMILY/share_thresholds.json, so that
      re-solving the outliers does not move it.
Cell types whose share was measured both with and without other jobs on
cores 16-23 showed the same median share (10x25 at 300 s and at 30 s).
"""
import glob
import json
from collections import defaultdict

import numpy as np

SETUP = {'S', 'LS', 'SW', 'LSW'}
CORES = list(range(16))
SHARE_MIN, TYPE_SLACK = 0.9, 0.05


THRESH = 'or_solution/FAMILY/share_thresholds.json'


def _key(p):
    folder, reg = p.split('/')[-3], p.split('/')[-2]
    return f"{folder}|{'setup' if reg in SETUP else 'nosetup'}"


def type_bars(paths):
    """5th percentile of the non-optimal cpu_share per cell type."""
    by_type = defaultdict(list)
    for p in paths:
        d = json.load(open(p))
        if d.get('hint') == 'complete' and d.get('status') != 'OPTIMAL':
            by_type[_key(p)].append(d['cpu_share'])
    return {k: round(float(np.percentile(v, 5)), 4) for k, v in by_type.items()}


def classify(paths):
    """Return {path: (valid, reason)} for re-solved cells (hint == 'complete')."""
    recs = {}
    for p in paths:
        d = json.load(open(p))
        if d.get('hint') == 'complete':
            recs[p] = d
    p5 = json.load(open(THRESH))
    out = {}
    for p, d in recs.items():
        s = d.get('cpu_share', 0)
        if not d.get('validated', False):
            out[p] = (False, 'not validated')
        elif d.get('affinity') != CORES:
            out[p] = (False, f'affinity {d.get("affinity")}')
        elif s >= SHARE_MIN:
            out[p] = (True, 'share')
        elif d.get('status') == 'OPTIMAL':
            out[p] = (True, 'optimal')
        elif _key(p) not in p5:
            out[p] = (None, 'pending: the type bar is frozen when its campaign closes')
        elif s >= p5[_key(p)] - TYPE_SLACK:
            out[p] = (True, 'type')
        else:
            out[p] = (False, f'share {s} < type p5 {p5[_key(p)]:.3f} - {TYPE_SLACK}')
    return out


def classify_folders(root, folders):
    paths = []
    for f in folders:
        paths += glob.glob(f'{root}/{f}/*/*.json')
    return classify(paths)
