"""Validity rule for time-budgeted CP-SAT cells (used by p80 audit and p81).

A cell's result depends on the CPU it received, so every record carries
cpu_share = CPU seconds / (wall seconds x workers) and its affinity mask.
A cell counts as valid when it was validated, solved on cores reserved for its
campaign (the 16-core set 0-15 of the 30-minute campaign by default; a campaign
that runs on several slots passes its slots, for example the eight-core sets
8-15 and 16-23), and one of the following holds:
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
      or_solution/FAMILY/share_thresholds.json (or in the file the campaign
      names), so that re-solving the outliers does not move it. A type with
      no frozen bar yet is reported as pending, never as valid.
Cell types whose share was measured both with and without other jobs on
cores 16-23 showed the same median share (10x25 at 300 s and at 30 s).

The type of a cell is its folder and its setup class by default (_key). The
runner scripts/p85_cpsat_arms.py writes <size>_cparms/<out>/<arm>/<regime>/,
so its folder level is the arm alone; key_cparms adds the output folder and
the size, so that a 600-s pilot cell and an 1800-s evidence cell, or a
50-module and an 80-module cell, never share a bar.

Single-thread runs (the memetic algorithm, eval_memetic_family.py) have no
structural phase inside their timed window (instance loading, seeds and the
numba warm-up come before it), so classify_single_thread applies the plain
rule: validated, wall-clock budget, one thread on one core of a reserved
slot, and cpu_share >= 0.9.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

SETUP = {'S', 'LS', 'SW', 'LSW'}
CORES = list(range(16))
SHARE_MIN, TYPE_SLACK = 0.9, 0.05


THRESH = 'or_solution/FAMILY/share_thresholds.json'


def _key(p):
    folder, reg = p.split('/')[-3], p.split('/')[-2]
    return f"{folder}|{'setup' if reg in SETUP else 'nosetup'}"


def key_cparms(p):
    """Type of a p85 record <size>_cparms/<out>/<arm>/<regime>/<name>.json:
    output folder, arm, size and setup class."""
    parts = p.split('/')
    size = parts[-5][:-len('_cparms')] if parts[-5].endswith('_cparms') else parts[-5]
    reg = parts[-2]
    return f"{parts[-4]}|{parts[-3]}|{size}|{'setup' if reg in SETUP else 'nosetup'}"


def type_bars(paths, key=None):
    """5th percentile of the non-optimal cpu_share per cell type."""
    key = key or _key
    by_type = defaultdict(list)
    for p in paths:
        d = json.load(open(p))
        if d.get('hint') == 'complete' and d.get('status') != 'OPTIMAL':
            by_type[key(p)].append(d['cpu_share'])
    return {k: round(float(np.percentile(v, 5)), 4) for k, v in by_type.items()}


def classify_single_thread(paths, slots=None):
    """Return {path: (valid, reason)} for single-thread time-budgeted runs.

    slots: the core sets reserved for the campaign (default 0-15); a run is
    valid on any one core of them."""
    allowed = {c for s in (slots or [CORES]) for c in s}
    out = {}
    for p in paths:
        d = json.load(open(p))
        aff = d.get('affinity') or []
        threads = d.get('threads', d.get('workers'))
        if not d.get('validated', False):
            out[p] = (False, 'not validated')
        elif d.get('budget_clock', 'wall') != 'wall':
            out[p] = (False, f'budget clock {d.get("budget_clock")}')
        elif threads != 1:
            out[p] = (False, f'threads {threads}')
        elif len(aff) != 1 or aff[0] not in allowed:
            out[p] = (False, f'affinity {aff}')
        elif d.get('cpu_share', 0) >= SHARE_MIN:
            out[p] = (True, 'share')
        else:
            out[p] = (False, f'share {d.get("cpu_share")} < {SHARE_MIN}')
    return out


def classify(paths, slots=None, thresh=THRESH, key=None):
    """Return {path: (valid, reason)} for re-solved cells (hint == 'complete').

    slots: the affinity masks reserved for the campaign, each a list of cores
    (default: the single mask 0-15). thresh: the file of frozen type bars.
    key: the cell type of a path (default _key: folder and setup class).
    """
    key = key or _key
    allowed = [sorted(s) for s in (slots or [CORES])]
    recs = {}
    for p in paths:
        d = json.load(open(p))
        if d.get('hint') == 'complete':
            recs[p] = d
    p5 = json.load(open(thresh)) if os.path.exists(thresh) else {}
    out = {}
    for p, d in recs.items():
        s = d.get('cpu_share', 0)
        if not d.get('validated', False):
            out[p] = (False, 'not validated')
        elif d.get('affinity') not in allowed:
            out[p] = (False, f'affinity {d.get("affinity")}')
        elif s >= SHARE_MIN:
            out[p] = (True, 'share')
        elif d.get('status') == 'OPTIMAL':
            out[p] = (True, 'optimal')
        elif key(p) not in p5:
            out[p] = (None, 'pending: the type bar is frozen when its campaign closes')
        elif s >= p5[key(p)] - TYPE_SLACK:
            out[p] = (True, 'type')
        else:
            out[p] = (False, f'share {s} < type p5 {p5[key(p)]:.3f} - {TYPE_SLACK}')
    return out


def classify_folders(root, folders, slots=None, thresh=THRESH, key=None):
    paths = []
    for f in folders:
        paths += glob.glob(f'{root}/{f}/*/*.json')
    return classify(paths, slots=slots, thresh=thresh, key=key)
